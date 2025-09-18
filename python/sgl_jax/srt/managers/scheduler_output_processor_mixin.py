from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import jax

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.io_struct import AbortReq, BatchTokenIDOut
from sgl_jax.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.common_utils import cdiv

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import (
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ):
        skip_stream_req = None

        assert self.is_generation
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            cache_miss_count,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.cache_miss_count,
        )
        if self.enable_overlap:
            logits_output, next_token_ids, cache_miss_count = (
                self.tp_worker.resolve_last_batch_result(launch_done)
            )
        else:
            # Move next_token_ids and logprobs to cpu
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = jax.device_get(
                        logits_output.next_token_logprobs
                    ).astype(float)
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        jax.device_get(logits_output.input_token_logprobs).astype(float)
                    )

        # Check finish conditions
        logprob_pt = 0
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.is_mixed_chunk and self.enable_overlap and req.finished():
                j = len(batch.out_cache_loc) - len(batch.reqs) + i
                self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                continue

            if req.is_chunked <= 0:
                # req output_ids are set here
                req.output_ids.append(next_token_id)
                req.check_finished()

                if req.finished():
                    if precision_tracer.get_trace_active():
                        precision_tracer.set_request_status_to_completed(req.rid)
                        precision_tracer.add_completed_requests_count()
                        precision_tracer.set_end_time_and_duration(req.rid)
                        logger.info(
                            "Request trace completed (%d/%d): %s",
                            precision_tracer.get_completed_requests_count(),
                            precision_tracer.get_max_requests(),
                            req.rid,
                        )
                        if (
                            precision_tracer.get_completed_requests_count()
                            >= precision_tracer.get_max_requests()
                        ):
                            precision_tracer.stop_trace()
                    self.tree_cache.cache_finished_req(req)
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    # This updates radix so others can match
                    self.tree_cache.cache_unfinished_req(req)

                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs

                # Update grammar state after token sampling
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(int(next_token_id))
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        logger.error(
                            "Grammar accept_token failed for req %s with token %s: %s",
                            req.rid,
                            next_token_id,
                            e,
                        )

                        self.abort_request(AbortReq(rid=req.rid))
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1
                # There is only at most one request being currently chunked.
                # Because this request does not finish prefill,
                # we don't want to stream the request currently being chunked.
                skip_stream_req = req

                # Incrementally update input logprobs.
                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

        batch.cache_miss_count = cache_miss_count

        if batch.cache_miss_count > 0:
            logger.info("Prefill batch. #bid: %s, #cache_miss: %s", result.bid, cache_miss_count)

        self.set_next_batch_sampling_info_done(batch)

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req, cache_miss_count)

        batch.spec_info = result.next_draft_input

    def _resolve_spec_decode_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> list[list[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""

        next_token_ids = result.next_token_ids
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens
        for i, req in enumerate(batch.reqs):
            predict_tokens.append(next_token_ids[i * stride : i * stride + accept_lens[i]])
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_lens[i]

        return predict_tokens

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ):
        logits_output, next_token_ids, cache_miss_count = (
            result.logits_output,
            result.next_token_ids,
            result.cache_miss_count,
        )
        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            next_token_ids = self._resolve_spec_decode_token_ids(result=result, batch=batch)

            # allocate_lens_list = result.allocate_lens.tolist()
            # accept_lens_list = result.accept_lens.tolist()

        if self.spec_algorithm is None or self.spec_algorithm.is_none():
            self.num_generated_tokens += len(batch.reqs)
        else:
            for next_token_id in next_token_ids:
                self.num_generated_tokens += len(next_token_id)
                self.accept_token += len(next_token_id)
            self.spec_num_forward_ct += len(batch.reqs)
            self.draft_token += len(batch.reqs) * self.draft_worker.speculative_num_draft_tokens
        # FIXME(pc) add spec decode metrics

        if self.enable_overlap:
            logits_output, next_token_ids, cache_miss_count = (
                self.tp_worker.resolve_last_batch_result(launch_done)
            )
            next_token_logprobs = logits_output.next_token_logprobs
        else:
            # spec decoding handles output logprobs inside verify process.
            if batch.return_logprob:
                next_token_logprobs = jax.device_get(logits_output.next_token_logprobs).astype(
                    float
                )

        self.token_to_kv_pool_allocator.free_group_begin()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req
            if req.is_retracted:
                continue

            indices_to_free = None
            if self.enable_overlap and req.finished():
                if self.page_size == 1:
                    indices_to_free = batch.out_cache_loc[i : i + 1]
                else:
                    if (len(req.origin_input_ids) + len(req.output_ids) - 1) % self.page_size == 0:
                        indices_to_free = batch.out_cache_loc[i : i + 1]
                if indices_to_free is not None:
                    self.token_to_kv_pool_allocator.free(indices_to_free)
                continue



            new_accepted_len = 1
            if batch.spec_algorithm is None or batch.spec_algorithm.is_none():
                req.output_ids.append(next_token_id)
            elif self.spec_algorithm.is_eagle():
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)

            req.check_finished(new_accepted_len)
            if req.finished():
                if batch.spec_algorithm is not None and batch.spec_algorithm.is_eagle():
                    cur_allocate_len = batch.spec_info.allocate_lens[i]
                    all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
                    if self.page_size > 1:
                        all_token_len = cdiv(all_token_len, self.page_size) * self.page_size
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx,
                        all_token_len:cur_allocate_len,
                    ]
                    kv_indices = kv_indices[kv_indices != 0]
                    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

                    assert (
                        len(kv_indices) <= EagleDraftInput.ALLOC_LEN_PER_DECODE
                    ), f"redundant kv indices {len(kv_indices)=} should less than {EagleDraftInput.ALLOC_LEN_PER_DECODE=}"

                    self.token_to_kv_pool_allocator.free(kv_indices)
                # End trace for finished request
                if precision_tracer.get_trace_active():
                    precision_tracer.set_request_status_to_completed(req.rid)
                    precision_tracer.add_completed_requests_count()
                    precision_tracer.set_end_time_and_duration(req.rid)
                    logger.info(
                        "Request trace completed (%d/%d): %s",
                        precision_tracer.get_completed_requests_count(),
                        precision_tracer.get_max_requests(),
                        req.rid,
                    )
                    if (
                        precision_tracer.get_completed_requests_count()
                        >= precision_tracer.get_max_requests()
                    ):
                        precision_tracer.stop_trace()
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob and (
                batch.spec_algorithm is None or batch.spec_algorithm.is_none()
            ):
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(logits_output.next_token_top_logprobs_val[i])
                    req.output_top_logprobs_idx.append(logits_output.next_token_top_logprobs_idx[i])
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            # Update grammar state after token sampling
            if req.grammar is not None:
                try:
                    req.grammar.accept_token(int(next_token_id))
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        "Grammar accept_token failed for req %s with token %s: %s",
                        req.rid,
                        next_token_id,
                        e,
                    )

                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()
            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(logits_output.hidden_states[i])
        self.set_next_batch_sampling_info_done(batch)
        self.stream_output(batch.reqs, batch.return_logprob, cache_miss_count=cache_miss_count)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        batch.cache_miss_count = cache_miss_count

        if (
            self.forward_ct_decode % self.server_args.decode_log_interval == 0
            or batch.cache_miss_count > 0
        ):
            self.log_decode_stats(running_batch=batch)

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[logprob_pt : logprob_pt + num_input_logprobs]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(output.input_token_ids_logprobs_val[i])
            req.temp_input_token_ids_logprobs_idx.append(output.input_token_ids_logprobs_idx[i])

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0 for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]
                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                    strict=True,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: list[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(output.next_token_token_ids_logprobs_val[i])
            req.output_token_ids_logprobs_idx.append(output.next_token_token_ids_logprobs_idx[i])

        return num_input_logprobs

    def stream_output(
        self: Scheduler,
        reqs: list[Req],
        return_logprob: bool,
        skip_req: Req | None = None,
        cache_miss_count: int = None,
    ):
        """Stream the output to detokenizer."""
        assert self.is_generation
        self.stream_output_generation(reqs, return_logprob, skip_req, cache_miss_count)

    def stream_output_generation(
        self: Scheduler,
        reqs: list[Req],
        return_logprob: bool,
        skip_req: Req | None = None,
        cache_miss_count: int = None,
    ):
        rids = []
        finished_reasons: list[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        spec_accepted_tokens = []
        output_hidden_states = None

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = output_token_logprobs_val = (
                output_token_logprobs_idx
            ) = input_top_logprobs_val = input_top_logprobs_idx = output_top_logprobs_val = (
                output_top_logprobs_idx
            ) = input_token_ids_logprobs_val = input_token_ids_logprobs_idx = (
                output_token_ids_logprobs_val
            ) = output_token_ids_logprobs_idx = None

        for req in reqs:
            if req is skip_req:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                should_output = True
            else:
                if req.stream:
                    stream_interval = req.sampling_params.stream_interval or self.stream_interval
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )
                else:
                    should_output = len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = req.send_output_token_logprobs_offset
                if isinstance(req.rid, list):
                    # if rid is a list, extend the list to rids
                    rids.extend(req.rid)
                else:
                    rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                if self.skip_tokenizer_init:
                    output_ids.append(req.output_ids[send_token_offset:])
                req.send_token_offset = len(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)

                if return_logprob:
                    if req.return_logprob and not req.input_logprob_sent:
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(req.input_token_ids_logprobs_val)
                        input_token_ids_logprobs_idx.append(req.input_token_ids_logprobs_idx)
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[send_output_token_logprobs_offset:]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[send_output_token_logprobs_offset:]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[send_output_token_logprobs_offset:]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[send_output_token_logprobs_offset:]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[send_output_token_logprobs_offset:]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[send_output_token_logprobs_offset:]
                        )
                        req.send_output_token_logprobs_offset = len(req.output_token_logprobs_val)
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

        # Send to detokenizer
        if rids:
            out = BatchTokenIDOut(
                rids,
                finished_reasons,
                decoded_texts,
                decode_ids_list,
                read_offsets,
                output_ids,
                skip_special_tokens,
                spaces_between_special_tokens,
                no_stop_trim,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                input_token_logprobs_val,
                input_token_logprobs_idx,
                output_token_logprobs_val,
                output_token_logprobs_idx,
                input_top_logprobs_val,
                input_top_logprobs_idx,
                output_top_logprobs_val,
                output_top_logprobs_idx,
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
                output_token_ids_logprobs_val,
                output_token_ids_logprobs_idx,
                output_hidden_states,
                cache_miss_count,
            )
            self.send_to_detokenizer.send_pyobj(out)
