import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
    get_last_loc_large_page_size_large_top_k,
    get_last_loc_large_page_size_top_k_1,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EAGLEWorker(ModelWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None

        # Initialize dummy tensors for EAGLE operations
        self.num_new_pages_per_topk = None
        self.extend_lens = None

        # this must be put at last to make sure model state is correct
        super().__init__(
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                head.data = head.data[self.hot_token_ids]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        if model_worker_batch.forward_mode.is_extend():
            model_worker_batch.padding_model_worker_batch(
                self.precompile_token_paddings,
                self.precompile_bs_paddings,
                self.precompile_cache_loc_paddings,
            )

            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
            )
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            # draft extend for Update Draft State
            self.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            # FIXME(pc) refactor this to batch output
            batch_output = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info,
                allocate_lens=model_worker_batch.seq_lens[: model_worker_batch.real_bs],
                bid=bid,
                cache_miss_count=cache_miss_count,
            )
            return batch_output

        else:
            cur_allocate_lens = model_worker_batch.spec_info.allocate_lens
            self.draft(model_worker_batch)

            batch_output = self.verify(model_worker_batch, cur_allocate_lens)

            self.forward_draft_extend_after_decode(model_worker_batch, batch_output)

            return batch_output

    def forward_target_extend(
        self, model_worker_batch: ModelWorkerBatch, sample_meta_data: SamplingMetadata
    ) -> tuple[LogitsProcessorOutput, jax.Array, int, int, np.ndarray]:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sample_meta_data
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ):
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_forward_metadata(
            model_worker_batch, is_eagle=True
        )
        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND
        last_idx = np.cumsum(model_worker_batch.extend_seq_lens, axis=0) - 1

        logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        logits_output.truncate_logits_processor_output(model_worker_batch)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]
        # assert forward_batch.spec_info is batch.spec_info

        self.capture_for_decode(logits_output, forward_batch.spec_info)

        # has_finished, unfinished_req_index = False, []
        # for i, req in enumerate(batch.reqs):
        #     if req.finished():
        #         has_finished = True
        #     else:
        #         unfinished_req_index.append(i)
        # if has_finished:
        #     unfinished_index_device = jnp.array(
        #         unfinished_req_index,
        #         dtype=jnp.int64,
        #     )
        #     batch.spec_info.filter_batch(unfinished_index_device, has_been_filtered=False)

    @property
    def draft_model_runner(self):
        return self.get_model_runner()

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = logits_output.hidden_states

    def draft(self, model_worker_batch: ModelWorkerBatch):
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        spec_info.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        # Run forward steps
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            model_worker_batch.seq_lens,
            jnp.sum(model_worker_batch.seq_lens),
            self.topk,
            self.speculative_num_draft_tokens,
            int(self.req_to_token_pool.req_to_token.shape[1]),
            self.mesh,
        )
        # build tree
        spec_info = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            seq_lens_cpu=model_worker_batch.seq_lens,
        )
        model_worker_batch.spec_info = spec_info
        return spec_info

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        spec_info: EagleVerifyInput = model_worker_batch.spec_info
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)
        model_worker_batch.padding_model_worker_batch(
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )
        forward_metadata = self.target_worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch, is_eagle=True
        )
        custom_mask = forward_metadata.custom_mask
        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        logits_output.truncate_logits_processor_output(model_worker_batch)
        spec_info.hidden_states = logits_output.hidden_states
        (
            predict,
            verified_id,
            accept_length,
            accept_index,
        ) = spec_info.sample(
            model_worker_batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            self.model_runner.rngs,
            self.mesh,
        )
        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        logits_output.hidden_states = logits_output.hidden_states[accept_index]
        model_worker_batch.positions = model_worker_batch.positions[accept_index]
        new_seq_lens = model_worker_batch.seq_lens[: model_worker_batch.real_bs] + accept_length
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            hidden_states=logits_output.hidden_states,
        )
        model_worker_batch.spec_info = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            # FIXME(pc) this field is for overlap
            allocate_lens=cur_allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
        )

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accepted_indices
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        # acceptance indices are the indices in a "flattened" batch.
        # dividing it to num_draft_tokens will yield the actual batch index.
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if RETURN_ORIGINAL_LOGPROB:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits, axis=-1)
        else:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits / temperatures, axis=-1)
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
            )

        logits_output.next_token_logprobs = logprobs[
            jnp.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def forward_draft_extend_after_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ):
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
        )
        forward_batch, logits_meatadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            batch_output.next_token_ids,
            self.speculative_num_draft_tokens,
            self.draft_model_runner,
            batch_output,
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )
        if forward_batch.input_ids.shape[0] <= 0:
            return
        draft_logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_meatadata,
        )
        # print(f"==============={draft_logits_output.hidden_states=}=====================")
        # print(f"==============={draft_logits_output.next_token_logits=}=====================")

        draft_logits_output.truncate_logits_processor_output(model_worker_batch)
        self.capture_for_decode(draft_logits_output, forward_batch.spec_info)
        select_index = (
            np.arange(len(model_worker_batch.seq_lens[: model_worker_batch.real_bs]))
            * self.speculative_num_draft_tokens
            + batch_output.accept_lens
            - 1
        )
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[select_index]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[select_index]
        topk_p, topk_index = topk_probs_from_logits(
            draft_logits_output.next_token_logits, self.topk
        )

        # prepare for next draft decode
        batch_output.next_draft_input.hidden_states = draft_logits_output.hidden_states[
            : model_worker_batch.real_bs
        ]
        batch_output.next_draft_input.topk_p = topk_p[: model_worker_batch.real_bs]
        batch_output.next_draft_input.topk_index = topk_index[: model_worker_batch.real_bs]

        verified_id_idx = jnp.cumsum(batch_output.accept_lens) - 1
        batch_output.next_draft_input.verified_id = batch_output.next_draft_input.verified_id[
            verified_id_idx
        ]

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = model_worker_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_ids is not None:
            topk_index = self.hot_token_ids[topk_index]
        # Return values
        score_list: list[jax.Array] = []
        token_list: list[jax.Array] = []
        parents_list: list[jax.Array] = []
        # Forward multiple steps
        scores = None
        # Save original positions to avoid buffer donation issues
        original_positions = np.repeat(model_worker_batch.seq_lens, self.topk)
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break
            # FIXME(pc) this should always be forward_batch ? because it has same shape
            model_worker_batch.input_ids = input_ids
            model_worker_batch.spec_info.hidden_states = hidden_states
            model_worker_batch.positions = original_positions + 1 + i

            # model_worker_batch.padding_model_worker_batch(
            #     self.precompile_token_paddings,
            #     self.precompile_bs_paddings,
            #     self.precompile_cache_loc_paddings,
            # )

            model_worker_batch.padding_model_worker_batch(
                self.precompile_token_paddings,
                self.precompile_bs_paddings,
                self.precompile_cache_loc_paddings,
            )
            self.draft_model_runner.attn_backend.forward_metadata = (
                self.draft_model_runner.attn_backend.get_forward_metadata(
                    model_worker_batch,
                    is_eagle=True,
                    speculative_step_id=i,
                    topk=topk_index.shape[1],
                )
            )
            if self.topk > 1:
                self.draft_model_runner.attn_backend.forward_metadata.custom_mask = (
                    build_tree_mask_for_draft_decode(
                        model_worker_batch.seq_lens,
                        topk=topk_index.shape[1],
                        speculative_step_id=i,
                        parents_list=parents_list,
                    )
                )
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
            # Run forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=LogitsMetadata.from_model_worker_batch(
                    model_worker_batch, self.draft_model_runner.mesh
                ),
            )
            topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
            if self.hot_token_ids is not None:
                topk_index = self.hot_token_ids[topk_index]
            hidden_states = logits_output.hidden_states
        for i in range(len(score_list)):
            # topk = 1
            score_list[i] = score_list[i][: model_worker_batch.real_bs]
            token_list[i] = token_list[i][: model_worker_batch.real_bs]
            parents_list[i] = parents_list[i][: model_worker_batch.real_bs]
        return score_list, token_list, parents_list

    def _draft_preprocess_idle(self, model_worker_batch: ModelWorkerBatch):
        pass

    def _draft_preprocess_decode(self, model_worker_batch: ModelWorkerBatch):
        # Parse args
        num_seqs = model_worker_batch.real_bs
        spec_info = model_worker_batch.spec_info

        # todo: add penalty

        if self.page_size == 1:
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                model_worker_batch.tree_cache,
                num_seqs * self.speculative_num_steps * self.topk,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    self.req_to_token_pool.req_to_token,
                    model_worker_batch.req_pool_indices,
                    model_worker_batch.seq_lens,
                    self.speculative_num_steps,
                )
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                # TODO(lmzheng): The current implementation is still a fake support
                # for page size > 1. In the `assign_draft_cache_locs` below,
                # we directly move the indices instead of the real kv cache.
                # This only works when the kernel backend runs with page size = 1.
                # If the kernel backend runs with page size > 1, we need to
                # duplicate the real KV cache. The overhead of duplicating KV
                # cache seems okay because the draft KV cache only has one layer.
                # see a related copy operation in MHATokenToKVPool::move_kv_cache.

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    num_new_pages_per_topk,
                    extend_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    self.req_to_token_pool.req_to_token,
                    model_worker_batch.req_pool_indices,
                    model_worker_batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )

                # TODO(lmzheng): remove this device sync
                extend_num_tokens = int(jnp.sum(extend_lens))

                # Store in instance variables for later use
                self.num_new_pages_per_topk = num_new_pages_per_topk
                self.extend_lens = extend_lens

            out_cache_loc, token_to_kv_pool_state_backup = alloc_paged_token_slots_extend(
                model_worker_batch.tree_cache,
                prefix_lens,
                seq_lens,
                last_loc,
                extend_num_tokens,
                backup_state=True,
            )

        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]

        # Update req_to_token_pool with cache locations (no reshape needed)
        # Layout: [seq0_topk0_steps, seq0_topk1_steps, seq1_topk0_steps, ...]
        for i in range(num_seqs):
            req_idx = model_worker_batch.req_pool_indices[i].item()
            start_pos = model_worker_batch.seq_lens[i].item()

            # For each topk branch
            for k in range(self.topk):
                # For each speculative step
                for step in range(self.speculative_num_steps):
                    # Calculate flat index: i * (topk * steps) + k * steps + step
                    flat_idx = (
                        i * (self.topk * self.speculative_num_steps)
                        + k * self.speculative_num_steps
                        + step
                    )
                    token_pos = start_pos + step
                    cache_loc = out_cache_loc[flat_idx].item()

                    # Update req_to_token mapping
                    if token_pos < self.req_to_token_pool.req_to_token.shape[1]:
                        self.req_to_token_pool.write((req_idx, token_pos), cache_loc)

        if self.page_size > 1 and self.topk > 1:
            # Remove padded slots
            out_cache_loc = out_cache_loc[: num_seqs * self.topk * self.speculative_num_steps]

        model_worker_batch.out_cache_loc = out_cache_loc
        model_worker_batch.seq_lens_sum = int(jnp.sum(model_worker_batch.seq_lens))
        model_worker_batch.return_hidden_states = False
        spec_info.positions = jnp.repeat(model_worker_batch.seq_lens, self.topk)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)


@functools.partial(jax.jit, static_argnames=["topk"])
def topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
    topk_logits, topk_index = jax.lax.top_k(working_logits, topk)
    logsumexp = jax.nn.logsumexp(working_logits, axis=-1, keepdims=True)
    topk_probs = jnp.exp(topk_logits - logsumexp)

    if axis != -1:
        topk_probs = jnp.moveaxis(topk_probs, -1, axis)
        topk_index = jnp.moveaxis(topk_index, -1, axis)

    return topk_probs, topk_index


def fast_topk(values, topk, axis=-1):
    working_values = jnp.moveaxis(values, axis, -1) if axis != -1 else values
    result_vals, result_indices = jax.lax.top_k(working_values, topk)

    if axis != -1:
        result_vals = jnp.moveaxis(result_vals, -1, axis)
        result_indices = jnp.moveaxis(result_indices, -1, axis)

    return result_vals, result_indices


# FIXME(pc) this should be jitted or convert as np.ndarray
def select_top_k_tokens(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = jnp.repeat(hidden_states, topk, axis=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            jnp.expand_dims(topk_p, axis=1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            jnp.tile(
                jnp.expand_dims(jnp.arange(-1, topk, dtype=jnp.float32), axis=0),
                (topk_p.shape[0], 1),
            ),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = jax.lax.mul(
            jnp.expand_dims(scores, axis=2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.reshape(expand_scores.shape[0], -1), topk, axis=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = jnp.take_along_axis(topk_index, topk_cs_index, axis=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + jnp.repeat(
                jnp.arange(0, hidden_states.shape[0], topk), topk
            )
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info
