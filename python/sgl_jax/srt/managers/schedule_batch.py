from __future__ import annotations

"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

import dataclasses
import logging
import threading
from http import HTTPStatus
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
from jax import numpy as jnp
from jax._src import mesh as mesh_lib

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.precision_tracer import (
    PrecisionTracerRequestMetadata,
    precision_tracer,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import ServerArgs

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


GLOBAL_SERVER_ARGS_KEYS = [
    "device",
    "disable_radix_cache",
    "enable_deterministic_sampling",
]

PADDING_BUCKETS = [1 << i for i in range(6, 21)]

# Put some global args for easy access
global_server_args_dict = {k: getattr(ServerArgs, k) for k in GLOBAL_SERVER_ARGS_KEYS}

logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        eos_token_ids: Optional[Set[int]] = None,
        vocab_size: Optional[int] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        # Each decode stage's output ids
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = []

        # Sampling info
        self.sampling_params = sampling_params

        # Memory pool info
        self.req_pool_idx: Optional[int] = None

        # Check finish
        self.tokenizer = None
        self.finished_reason = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop, set this to true
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_abort = False
        # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
        self.to_abort_message: str = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: np.ndarray = []
        # Number of tokens to run prefill.
        self.extend_input_len = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        self.last_node: Any = None
        self.last_host_node: Any = None

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[np.ndarray]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # For metrics
        self.has_log_time_stats: bool = False
        self.queue_time_start = None
        self.queue_time_end = None

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def extend_image_inputs(self, image_inputs):
        raise NotImplementedError()

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
    ):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
            ) = tree_cache.match_prefix(
                key=self.adjust_max_prefix_ids(),
            )
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def check_finished(self):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT(
                message=self.to_abort_message,
            )
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]
        if hasattr(last_token_id, "item"):
            last_token_id = last_token_id.item()
        last_token_id = int(last_token_id)
        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                if any(hasattr(token_id, "item") for token_id in self.eos_token_ids):
                    self.eos_token_ids = {
                        (
                            int(token_id.item())
                            if hasattr(token_id, "item")
                            else int(token_id)
                        )
                        for token_id in self.eos_token_ids
                    }
                matched_eos |= last_token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= last_token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= (
                        last_token_id in self.tokenizer.additional_stop_token_ids
                    )
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return

        if self.vocab_size is not None and (
            last_token_id > self.vocab_size or last_token_id < 0
        ):
            if self.sampling_params.stop_token_ids:
                self.output_ids[-1] = next(iter(self.sampling_params.stop_token_ids))
            elif self.eos_token_ids:
                self.output_ids[-1] = next(iter(self.eos_token_ids))
            self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
            return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def reset_for_retract(self):
        self.prefix_indices = []
        self.last_node = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.req_pool_idx = None
        self.already_computed = 0

    def set_finish_with_abort(self, error_msg: str):
        # set it to one token to skip the long prefill
        self.origin_input_ids = [0]
        self.return_logprob = False
        self.finished_reason = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.sampling_params=})"
        )


# Batch id
bid = 0


@dataclasses.dataclass
class ScheduleBatch:
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    chunked_req: Optional[Req] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: np.ndarray = None  # shape: [b], int32
    req_pool_indices: np.ndarray = None  # shape: [b], int32
    seq_lens: np.ndarray = None  # shape: [b], int32
    # The output locations of the KV cache
    out_cache_loc: np.ndarray = None  # shape: [b], int32
    output_ids: np.ndarray = None  # shape: [b], int32

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For logits and logprob post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: Optional[int] = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[np.ndarray] = None

    # Stream
    has_stream: bool = False

    # device mesh
    mesh: mesh_lib.Mesh = None

    cache_miss_count: int = 0

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Events
    launch_done: Optional[threading.Event] = None

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        enable_custom_logit_processor: bool = False,
        chunked_req: Optional[Req] = None,
        mesh: mesh_lib.Mesh = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            return_logprob=return_logprob,
            enable_overlap=enable_overlap,
            has_stream=any(req.stream for req in reqs),
            chunked_req=chunked_req,
            mesh=mesh,
            is_prefill_only=all(
                req.sampling_params.max_new_tokens == 0 for req in reqs
            ),
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def alloc_req_slots(self, num_reqs: int):
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
                "Please set a smaller number for `--max-running-requests`. "
                f"{self.req_to_token_pool.available_size()=}, "
                f"{num_reqs=}, "
            )
        return req_pool_indices

    def alloc_token_slots(self, num_tokens: int, backup_state: bool = False):
        self._evict_tree_cache_if_needed(num_tokens)

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
            error_msg = (
                f"{phase_str} out of memory. Try to lower your batch size.\n"
                f"Try to allocate {num_tokens} tokens.\n"
                f"{self._available_and_evictable_str()}"
            )
            logger.error(error_msg)
            if self.tree_cache is not None:
                self.tree_cache.pretty_print()
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

    def alloc_paged_token_slots_extend(
        self,
        prefix_lens: List[int],
        seq_lens: List[int],
        last_loc: List[int],
        extend_num_tokens: int,
        backup_state: bool = False,
    ):
        num_tokens = (
            extend_num_tokens
            + len(seq_lens) * self.token_to_kv_pool_allocator.page_size
        )
        self._evict_tree_cache_if_needed(num_tokens)

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc_extend(
            prefix_lens, seq_lens, last_loc, extend_num_tokens
        )
        if out_cache_loc is None:
            error_msg = (
                f"Prefill out of memory. Try to lower your batch size.\n"
                f"Try to allocate {extend_num_tokens} tokens.\n"
                f"{self._available_and_evictable_str()}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

    def alloc_paged_token_slots_decode(
        self,
        seq_lens: List[int],
        last_loc: List[int],
        backup_state: bool = False,
    ):
        num_tokens = len(seq_lens) * self.token_to_kv_pool_allocator.page_size

        self._evict_tree_cache_if_needed(num_tokens)

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc_decode(seq_lens, last_loc)
        if out_cache_loc is None:
            error_msg = (
                f"Decode out of memory. Try to lower your batch size.\n"
                f"Try to allocate {len(seq_lens)} tokens.\n"
                f"{self._available_and_evictable_str()}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        # Use EXTEND instead of MIXED for precompile cache hit
        self.forward_mode = ForwardMode.EXTEND
        running_bs = running_batch.batch_size()

        for i, req in enumerate(running_batch.reqs):
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = jnp.concatenate([self.input_ids, running_batch.input_ids])
        out_cache_loc = jnp.concatenate(
            [self.out_cache_loc, running_batch.out_cache_loc]
        )

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        delta = 0 if self.enable_overlap else -1
        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        self.extend_logprob_start_lens.extend([0] * running_bs)

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        # Allocate req slots
        bs = len(self.reqs)
        req_pool_indices = self.alloc_req_slots(bs)

        # Init arrays
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        req_pool_indices_cpu = np.array(req_pool_indices, dtype=np.int32)
        input_ids_cpu = np.array(sum(input_ids, []), dtype=np.int32)
        seq_lens_cpu = np.array(seq_lens, dtype=np.int32)
        prefix_lens_cpu = np.array(prefix_lens, dtype=np.int32)

        # Copy prefix and do some basic check
        extend_input_logprob_token_ids = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len

            prefix_indices = req.prefix_indices
            if pre_len > 0:
                # note: prefix_indices has to locate on device, or will meet Received incompatible devices for jitted computation
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), prefix_indices
                )

            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False

            # Compute the relative logprob_start_len in an extend batch
            if req.logprob_start_len >= pre_len:
                req.extend_logprob_start_len = min(
                    req.logprob_start_len - pre_len,
                    req.extend_input_len,
                    req.seqlen - 1,
                )
            else:
                req.extend_logprob_start_len = 0

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    len(req.fill_ids),
                )
                # Apply logprob_start_len
                if global_start_idx < req.logprob_start_len:
                    global_start_idx = req.logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = np.array(extend_input_logprob_token_ids)
        else:
            extend_input_logprob_token_ids = None

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = self.alloc_token_slots(extend_num_tokens)
        else:
            last_loc_cpu = get_last_loc(
                self.req_to_token_pool.req_to_token,
                req_pool_indices_cpu,
                prefix_lens_cpu,
            )
            out_cache_loc = self.alloc_paged_token_slots_extend(
                prefix_lens,
                seq_lens,
                last_loc_cpu.tolist(),
                extend_num_tokens,
            )

        # Set fields
        self.input_ids = input_ids_cpu
        self.req_pool_indices = req_pool_indices_cpu
        self.seq_lens = seq_lens_cpu
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Write to req_to_token_pool
        pt = 0
        for i in range(bs):
            self.req_to_token_pool.write(
                (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                out_cache_loc[pt : pt + extend_lens[i]],
            )
            pt += extend_lens[i]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def new_page_count_next_decode(self):
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return len(self.reqs)
        # In the decoding phase, the length of a request's KV cache should be
        # the total length of the request minus 1
        return (
            sum(1 for req in self.reqs if req.seqlen % page_size == 0)
            if self.enable_overlap
            else sum(1 for req in self.reqs if (req.seqlen - 1) % page_size == 0)
        )

    def check_decode_mem(self, buf_multiplier=1):
        num_tokens = (
            self.new_page_count_next_decode()
            * buf_multiplier
            * self.token_to_kv_pool_allocator.page_size
        )

        self._evict_tree_cache_if_needed(num_tokens)
        return self._is_available_size_sufficient(num_tokens)

    def retract_decode(self, server_args: ServerArgs):
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = list(range(len(self.reqs)))

        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),
                -len(self.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )

        def get_required_tokens(num_reqs: int):
            return num_reqs * global_config.retract_decode_steps

        def _get_available_size():
            return self.token_to_kv_pool_allocator.available_size()

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens
        first_iter = True
        while (
            _get_available_size() < get_required_tokens(len(sorted_indices))
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.token_to_kv_pool_allocator.available_size() > 0
                ), f"No space left for only one request, {self.token_to_kv_pool_allocator.available_size()=}"
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if isinstance(self.tree_cache, ChunkCache):
                # ChunkCache does not have eviction
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool_allocator.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
            else:
                last_uncached_pos = (
                    len(req.prefix_indices) // server_args.page_size
                ) * server_args.page_size
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool_allocator.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)

                # release the last node
                self.tree_cache.dec_lock_ref(req.last_node)

                # NOTE: we should use the newly evictable memory instantly.
                num_tokens = len(sorted_indices) * global_config.retract_decode_steps
                self._evict_tree_cache_if_needed(num_tokens)

            req.reset_for_retract()

            if len(retracted_reqs) == 0:
                # Corner case: only one request left
                raise ValueError(
                    "Failed to retract any request. No space left for only one request."
                )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = np.empty(0, np.int32)
        self.seq_lens = np.empty(0, np.int32)
        self.out_cache_loc = np.empty(0, np.int32)
        self.req_pool_indices = np.empty(0, np.int32)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)

        # Update fields
        self.input_ids = self.output_ids

        self.output_ids = None

        locs = self.seq_lens.copy()

        if self.enable_overlap:
            self.seq_lens = self.seq_lens + 1
        else:
            self.seq_lens = np.add(self.seq_lens, 1)
        self.seq_lens_sum += bs

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            self.out_cache_loc = self.alloc_token_slots(bs)
        else:
            last_loc = self.req_to_token_pool.req_to_token[
                self.req_pool_indices, self.seq_lens - 2
            ]
            self.out_cache_loc = self.alloc_paged_token_slots_decode(
                self.seq_lens.tolist(),
                last_loc.tolist(),
            )
        self.req_to_token_pool.write(
            (self.req_pool_indices, locs), self.out_cache_loc.astype(np.int32)
        )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = (
            self.output_ids[keep_indices] if self.output_ids is not None else None
        )
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)

        self.sampling_info.filter_batch(np.array(keep_indices))

    def merge_batch(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info, other.mesh)

        self.req_pool_indices = np.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = np.concat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = np.concat(
                [
                    self.output_ids[: len(self.seq_lens)],
                    other.output_ids[: len(other.seq_lens)],
                ]
            )
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.return_hidden_states |= other.return_hidden_states

    def get_model_worker_batch(
        self,
        token_paddings: list,
        bs_paddings: list,
        cache_loc_paddings: List,
        page_size: int,
    ) -> ModelWorkerBatch:
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
            token_paddings = bs_paddings
        else:
            extend_seq_lens = np.array(self.extend_lens, dtype=np.int32)
            extend_prefix_lens = np.array(self.prefix_lens, dtype=np.int32)
            bs_paddings = bs_paddings[-1:]
            cache_loc_paddings = cache_loc_paddings[-1:]
            extend_logprob_start_lens = self.extend_logprob_start_lens

        global bid
        bid += 1

        if self.input_ids is None:
            input_ids_cpu = np.empty(0, dtype=np.int32)
        else:
            input_ids_cpu = self.input_ids.flatten()

        real_input_ids_len = len(input_ids_cpu)
        out_cache_loc_cpu = self.out_cache_loc
        seq_lens_cpu = self.seq_lens
        real_bs = len(seq_lens_cpu)
        req_pool_indices_cpu = self.req_pool_indices
        token_indices_with_all_reqs = self.req_to_token_pool.req_to_token[
            self.req_pool_indices
        ]

        # padding seq
        # extend & decode: input_ids, positions, out_cache_loc, cache_loc
        padding_size = 0
        token_paddings.sort()
        for size in token_paddings:
            if size >= len(input_ids_cpu):
                padding_size = size - len(input_ids_cpu)
                break

        if padding_size > 0:
            input_ids_cpu = np.concat(
                [
                    input_ids_cpu,
                    np.array([0] * padding_size, dtype=input_ids_cpu.dtype),
                ],
                axis=0,
            )

        padded_input_ids_len = len(input_ids_cpu)
        out_cache_loc_num_to_padding = padded_input_ids_len - len(out_cache_loc_cpu)
        if out_cache_loc_num_to_padding > 0:
            out_cache_loc_cpu = np.concatenate(
                [
                    out_cache_loc_cpu,
                    np.array(
                        [-1] * out_cache_loc_num_to_padding,
                        dtype=out_cache_loc_cpu.dtype,
                    ),
                ],
                axis=0,
            )

        # Calculate positions and extend_start_loc after padding
        if self.forward_mode.is_extend():
            # For prefill: create positions for each token in sequences
            # Calculate total tokens without padding first
            total_tokens_before_padding = sum(
                [extend_len for extend_len in self.extend_lens]
            )
            positions_cpu = np.concatenate(
                [
                    np.arange(prefix_len, seq_len, dtype=seq_lens_cpu.dtype)
                    for seq_len, prefix_len in zip(seq_lens_cpu, self.prefix_lens)
                ]
            )

            # If input_ids was padded, pad positions too
            padding_size = len(input_ids_cpu) - total_tokens_before_padding
            if padding_size:
                positions_cpu = np.concatenate(
                    [positions_cpu, np.zeros(padding_size, dtype=positions_cpu.dtype)]
                )

            # Start location of each sequence in the flattened array
            extend_start_loc = np.cumsum(
                np.concatenate([np.array([0]), extend_seq_lens[:-1]]),
                dtype=seq_lens_cpu.dtype,
            )
        else:
            # For decode: each sequence contributes one token at the next position (seq_len)
            # Create positions for actual tokens (one per sequence at seq_len)
            batch_positions = seq_lens_cpu - 1
            # Create positions array matching the length of input_ids (including padding)
            positions_cpu = np.zeros(len(input_ids_cpu), dtype=batch_positions.dtype)
            # Fill in the actual positions for the real tokens
            # positions = positions.at[: len(batch_positions)].set(batch_positions)
            positions_cpu[: len(batch_positions)] = batch_positions
            # The padding tokens (if any) will have position 0, which is fine for padding
            # For decode, extend_start_loc is typically not used but we'll set it anyway
            extend_start_loc = np.arange(len(seq_lens_cpu), dtype=seq_lens_cpu.dtype)

        bs_padding_size = 0
        bs_paddings.sort()
        select_bs_index = -1
        for i, size in enumerate(bs_paddings):
            if size >= len(seq_lens_cpu):
                bs_padding_size = size - len(seq_lens_cpu)
                select_bs_index = i
                break

        cache_loc_flat = np.array([], dtype=np.int32)
        if len(seq_lens_cpu) > 0:
            # Filter out empty sequences
            valid_mask = seq_lens_cpu > 0
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_seq_lens = seq_lens_cpu[valid_mask]

                # Calculate aligned lengths for all valid sequences at once
                aligned_lengths = (
                    (valid_seq_lens + page_size - 1) // page_size
                ) * page_size
                total_aligned_length = np.sum(aligned_lengths)

                # Pre-allocate the result array
                cache_loc_flat = np.zeros(total_aligned_length, dtype=np.int32)

                # Fill the array efficiently
                offset = 0
                for i, (seq_idx, seq_len, aligned_len) in enumerate(
                    zip(valid_indices, valid_seq_lens, aligned_lengths)
                ):
                    # Copy the actual data
                    cache_loc_flat[offset : offset + seq_len] = (
                        token_indices_with_all_reqs[seq_idx, :seq_len]
                    )
                    # Padding is already zero from initialization
                    offset += aligned_len

        offset = np.sum(seq_lens_cpu[seq_lens_cpu > 0]) if len(seq_lens_cpu) > 0 else 0

        total_cache_loc_size = cache_loc_paddings[select_bs_index]
        assert total_cache_loc_size >= len(cache_loc_flat)

        cache_loc_cpu = np.empty(total_cache_loc_size, dtype=np.int32)
        if len(cache_loc_flat) > 0:
            cache_loc_cpu[: len(cache_loc_flat)] = cache_loc_flat
        # Initialize padding area to ensure multiprocess consistency
        if len(cache_loc_flat) < total_cache_loc_size:
            cache_loc_cpu[len(cache_loc_flat) :] = 0

        if bs_padding_size > 0:
            invalid_req_pool_indices = np.array(
                [-1] * bs_padding_size, dtype=req_pool_indices_cpu.dtype
            )
            req_pool_indices_cpu = np.concat(
                [
                    req_pool_indices_cpu,
                    invalid_req_pool_indices,
                ],
                axis=0,
            )
            invalid_seq_lens = np.array([0] * bs_padding_size, dtype=seq_lens_cpu.dtype)
            seq_lens_cpu = np.concat([seq_lens_cpu, invalid_seq_lens], axis=0)
            if self.forward_mode.is_extend():
                invalid_extend_start_loc = np.array(
                    [extend_start_loc[-1] + extend_seq_lens[-1]] * bs_padding_size,
                    dtype=extend_start_loc.dtype,
                )
                extend_start_loc = np.concat(
                    [extend_start_loc, invalid_extend_start_loc], axis=0
                )
                invalid_extend_prefix_lens = np.array(
                    [0] * bs_padding_size, dtype=extend_prefix_lens.dtype
                )
                extend_prefix_lens = np.concat(
                    [extend_prefix_lens, invalid_extend_prefix_lens], axis=0
                )
                invalid_extend_seq_lens = np.array(
                    [0] * bs_padding_size, dtype=extend_seq_lens.dtype
                )
                extend_seq_lens = np.concat(
                    [extend_seq_lens, invalid_extend_seq_lens], axis=0
                )
            else:
                invalid_extend_start_loc = np.array(
                    [len(seq_lens_cpu)] * bs_padding_size, dtype=extend_start_loc.dtype
                )
                extend_start_loc = np.concat(
                    [extend_start_loc, invalid_extend_start_loc], axis=0
                )

        if precision_tracer.get_trace_active():
            self._generate_trace_info(real_bs, bid)

        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=input_ids_cpu,
            real_input_ids_len=real_input_ids_len,
            req_pool_indices=req_pool_indices_cpu,
            seq_lens=seq_lens_cpu,
            out_cache_loc=out_cache_loc_cpu,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            sampling_info=self.sampling_info,
            positions=positions_cpu,
            extend_start_loc=extend_start_loc,
            cache_loc=cache_loc_cpu,
            extend_prefix_lens=(
                extend_prefix_lens if self.forward_mode == ForwardMode.EXTEND else None
            ),
            extend_seq_lens=(
                extend_seq_lens if self.forward_mode == ForwardMode.EXTEND else None
            ),
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            real_bs=real_bs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            launch_done=self.launch_done,
        )

    def _generate_trace_info(self, real_bs: int, bid: int) -> List[str]:
        for req in self.reqs[:real_bs]:

            if precision_tracer.get_trace_active():
                # for chunked prefill trace
                if req.fill_ids:
                    if self.forward_mode == ForwardMode.EXTEND:
                        input_ids_to_trace = req.fill_ids[len(req.prefix_indices) :]
                    else:
                        input_ids_to_trace = req.fill_ids
                else:
                    input_ids_to_trace = req.origin_input_ids

                precision_tracer.add_request_to_batch_requests_mapping(
                    bid,
                    PrecisionTracerRequestMetadata(
                        req.rid, input_ids_to_trace, self.forward_mode
                    ),
                )
                if self.forward_mode == ForwardMode.EXTEND:
                    precision_tracer.add_request_counter()
                    logger.info(
                        f"Starting trace for request {precision_tracer.get_request_counter()}: {req.rid}"
                    )

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            is_prefill_only=self.is_prefill_only,
        )

    def _evict_tree_cache_if_needed(
        self,
        num_tokens: int,
    ) -> None:
        if isinstance(self.tree_cache, ChunkCache):
            return

        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens)

    def _is_available_size_sufficient(self, num_tokens: int) -> bool:
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def _available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"


def align_to_size(l: list, size: int, value: int = 0) -> list:
    align_len = (len(l) + size - 1) // size * size
    return l[:] + [value] * (align_len - len(l))


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: np.ndarray
    # the length is outof padding
    real_input_ids_len: int
    # The sequence length
    seq_lens: np.ndarray
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: np.ndarray
    # The indices of requests in the req_to_token_pool
    req_pool_indices: np.ndarray
    # Sampling info
    sampling_info: SamplingBatchInfo
    # Position information [total_tokens]
    positions: np.ndarray
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: np.ndarray
    # cache_loc
    cache_loc: np.ndarray

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # For extend
    # extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[np.ndarray]
    extend_prefix_lens: Optional[np.ndarray]
    extend_logprob_start_lens: Optional[List[int]]
    extend_input_logprob_token_ids: Optional[np.ndarray]

    # For padding
    real_bs: int

    capture_hidden_mode: CaptureHiddenMode = None

    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: np.ndarray = None
    top_p_normalized_logprobs: bool = False
    top_p: np.ndarray = None

    # Events
    launch_done: Optional[threading.Event] = None

    # Pre-initialized ForwardBatch for overlap scheduling optimization
    forward_batch: Optional[Any] = None


def get_last_loc(
    req_to_token: np.ndarray,
    req_pool_indices: np.ndarray,
    prefix_lens: np.ndarray,
) -> np.ndarray:
    impl = get_last_loc_jax

    return impl(req_to_token, req_pool_indices, prefix_lens)


def get_last_loc_jax(
    req_to_token: np.ndarray,
    req_pool_indices: np.ndarray,
    prefix_lens: np.ndarray,
) -> np.ndarray:
    return np.where(
        prefix_lens > 0,
        req_to_token[req_pool_indices, prefix_lens - 1],
        np.full_like(prefix_lens, -1),
    )
