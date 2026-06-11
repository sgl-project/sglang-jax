# ruff: noqa: E402
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
import itertools
import logging
import os
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import numpy as np
from jax._src import mesh as mesh_lib

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    evict_from_tree_cache,
    release_kv_cache,
)
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.precision_tracer import (
    PrecisionTracerRequestMetadata,
    precision_tracer,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import DEFAULT_SAMPLING_SEED, SamplingParams
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var, pad_to_bucket

if TYPE_CHECKING:
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


GLOBAL_SERVER_ARGS_KEYS = [
    "device",
    "chunked_prefill_size",
    "disable_radix_cache",
    "speculative_accept_threshold_single",
    "speculative_accept_threshold_acc",
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
    def __init__(self, matched: int | list[int]):
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


class FINISHED_MATCHED_REGEX(BaseFinishReason):
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
        origin_input_ids: list[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        return_output_logprob_only: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: list[int] = None,
        stream: bool = False,
        lora_id: str | None = None,
        extra_key: str | None = None,
        dp_rank: int | None = None,
        origin_input_ids_unpadded: tuple[int] | None = None,
        eos_token_ids: set[int] | None = None,
        vocab_size: int | None = None,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        multimodal_embedding: list[list[float]] | None = None,
        deepstack_visual_embedding: list[list[float]] | None = None,
        deepstack_visual_pos_mask: list[int] | None = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        origin_input_ids = list(origin_input_ids)

        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids

        # Cache input IDs with hash-based values for multimodal placeholder tokens
        # Used for radix cache matching to differentiate different images/videos
        # If None, origin_input_ids is used for cache matching
        self.cache_input_ids: list[int] | None = None
        # Multimodal inputs (e.g., mrope positions) from tokenizer
        self.mm_inputs: dict | None = None

        # Each decode stage's output ids
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = []

        # Sampling info
        self.sampling_params = sampling_params
        self.return_hidden_states = return_hidden_states

        # Extra key for cache namespace isolation (e.g., cache_salt, lora_id)
        if lora_id is not None:
            extra_key = (extra_key or "") + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id if lora_id is not None else "0"
        self.dp_rank = dp_rank

        # PD disaggregation routing keys.
        self.bootstrap_host: str | None = None
        self.bootstrap_port: int | None = None
        self.bootstrap_room: int | None = None
        self.disagg_transfer_id: str | None = None

        # Memory pool info
        self.req_pool_idx: int | None = None
        self.recurrent_pool_idx: int | None = None

        # Check finish
        self.tokenizer = None
        self.finished_reason = None
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: BaseFinishReason | None = None
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
        # Incremental-detokenize cache: holds origin_input_ids_unpadded[surr_offset:]
        # + output_ids_through_stop, extended in place each step so we never
        # re-concatenate the full prompt (keeps it O(new tokens), not O(prompt)).
        self.surr_and_decode_ids = None
        self.cur_decode_ids_len = 0
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
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: int | None = None
        # SWA eviction: sequence positions [0, swa_evicted_seqlen) have had
        # their SWA pool slots freed (no longer in the sliding window).
        self.swa_evicted_seqlen: int = 0
        # The number of extend/decode batches this request has already gone through.
        # These counters gate overlap-safe SWA reclaim timing.
        self.extend_batch_idx: int = 0
        self.decode_batch_idx: int = 0
        # The prefix length of the last prefix matching
        self.last_matched_prefix_len: int = 0

        # For req-level memory management (single release entry point).
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        # Page-aligned tree-tracked prefix length; differs from
        # len(prefix_indices) when an unaligned tail is owned by the req
        # but not by the tree (page_size > 1 chunked prefill).
        self.cache_protected_len = 0

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
        self.return_output_logprob_only = return_output_logprob_only
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: list[float] | None = None
        self.input_token_logprobs_idx: list[int] | None = None
        self.input_top_logprobs_val: list[float] | None = None
        self.input_top_logprobs_idx: list[int] | None = None
        self.input_token_ids_logprobs_val: list[float] | None = None
        self.input_token_ids_logprobs_idx: list[int] | None = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: list[tuple[int]] | None = None
        self.temp_input_top_logprobs_val: list[np.ndarray] | None = None
        self.temp_input_top_logprobs_idx: list[int] | None = None
        self.temp_input_token_ids_logprobs_val: list[float] | None = None
        self.temp_input_token_ids_logprobs_idx: list[int] | None = None

        if return_logprob or return_output_logprob_only:
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
        self.hidden_states: list[list[float]] = []

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0
        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0

        # The number of accepted tokens in speculative decoding for this request.
        # This is used to compute the acceptance rate and average acceptance length per request.
        self.spec_accepted_tokens = 0

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

        # Grammar for constrained decoding
        self.grammar = None  # BaseGrammarObject | Future | None
        self.grammar_key: tuple[str, str] | None = None  # Cache key for grammar
        self.grammar_wait_ct = 0  # Counter for grammar compilation wait time

        # capture routed experts
        self.return_routed_experts = return_routed_experts
        self.routed_experts: np.ndarray | None = None  # shape (seqlen, topk)
        # latest_bid is used to improve return_routed_expert performance
        self.latest_bid: int = None

        # For deepstack
        self.multimodal_embedding = multimodal_embedding
        self.apply_for_deepstack = False
        self.deepstack_visual_pos_mask = deepstack_visual_pos_mask
        self.deepstack_visual_embedding = deepstack_visual_embedding

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
        tree_cache: BasePrefixCache | None = None,
    ):
        self.fill_ids = (
            self.origin_input_ids + self.output_ids if self.output_ids else self.origin_input_ids
        )
        # PD decode-side: KV was written externally; skip tree_cache.match_prefix
        # for this req's first decode iter.
        if getattr(self, "_pd_skip_prefix_match", False):
            self._pd_skip_prefix_match = False
            root = getattr(tree_cache, "root_node", None) if tree_cache is not None else None
            self.last_node = root
            self.last_host_node = root
            self.host_hit_length = 0
            self.last_matched_prefix_len = len(self.prefix_indices)
            self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)
            return
        # PD req with empty output_ids: skip match_prefix to avoid
        # stale radix cache hits.
        if getattr(self, "bootstrap_room", None) is not None and not self.output_ids:
            self.prefix_indices = []
            self.last_matched_prefix_len = 0
            self.extend_input_len = len(self.fill_ids)
            root = getattr(tree_cache, "root_node", None) if tree_cache is not None else None
            self.last_node = root
            self.last_host_node = root
            self.host_hit_length = 0
            return
        if tree_cache is not None:
            if getattr(tree_cache, "disable", False):
                self.prefix_indices = np.empty((0,), dtype=np.int32)
                self.last_node = tree_cache.root_node
                self.last_host_node = tree_cache.root_node
                self.host_hit_length = 0
            else:
                (
                    self.prefix_indices,
                    self.last_node,
                    self.last_host_node,
                    self.host_hit_length,
                ) = tree_cache.match_prefix(
                    key=RadixKey(self.adjust_max_prefix_ids(), self.extra_key, self.dp_rank),
                )
            self.last_matched_prefix_len = len(self.prefix_indices)
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def adjust_max_prefix_ids(self):
        self.fill_ids = (
            self.origin_input_ids + self.output_ids if self.output_ids else self.origin_input_ids
        )
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

    def pop_committed_kv_cache(self) -> int:
        assert (
            not self.kv_committed_freed
        ), f"Committed KV cache already freed ({self.kv_committed_len=})"
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self) -> tuple[int, int]:
        assert not self.kv_overallocated_freed, (
            "Overallocated KV cache already freed, "
            f"{self.kv_committed_len=}, {self.kv_allocated_len=}"
        )
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len

    @property
    def output_ids_through_stop(self) -> list[int]:
        # Truncate at the stop position so detokenize never emits ids past the
        # finish point (e.g. the extra delayed token under the overlap schedule).
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
            # Build the surrounding+decode buffer once; subsequent calls only
            # append the newly generated tail in place, so we never rebuild the
            # full (prompt + output) list — the high-concurrency hotspot.
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def check_finished(self, new_accepted_len: int = 1):
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(length=self.sampling_params.max_new_tokens)
            return

        # Check grammar termination
        if self.grammar is not None and self.grammar.is_terminated():
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
            return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]
        # if hasattr(last_token_id, "item"):
        #     last_token_id = last_token_id.item()
        # last_token_id = int(last_token_id)
        if self._check_token_based_finish(new_accepted_tokens=new_accepted_tokens):
            return

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish():
            return

    def _check_vocab_boundary_finish(self, new_accepted_tokens: list[int]) -> bool:
        for i, token_id in enumerate(new_accepted_tokens):
            if self.vocab_size is not None and (token_id > self.vocab_size or token_id < 0):
                if self.sampling_params.stop_token_ids:
                    self.output_ids[-1] = next(iter(self.sampling_params.stop_token_ids))
                elif self.eos_token_ids:
                    self.output_ids[-1] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                return True
        return False

    def _check_token_based_finish(self, new_accepted_tokens: list[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False
        matched_eos = False
        # Check stop token ids
        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                if any(hasattr(token_id, "item") for token_id in self.eos_token_ids):
                    self.eos_token_ids = {
                        (int(token_id.item()) if hasattr(token_id, "item") else int(token_id))
                        for token_id in self.eos_token_ids
                    }
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _check_str_based_finish(self):
        # Check stop strings
        stop_strs = self.sampling_params.stop_strs or []
        if len(stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return True
        return False

    def reset_for_retract(self):
        self.prefix_indices = []
        self.last_node = None
        self.swa_uuid_for_lock = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0
        self.routed_experts = None
        self.latest_bid = None
        self.cache_protected_len = 0
        self.recurrent_pool_idx = None

    def set_finish_with_abort(self, error_msg: str):
        # set it to one token to skip the long prefill
        self.origin_input_ids = [0]
        self.grammar = None
        self.return_logprob = False
        self.to_finish = FINISH_ABORT(error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError")

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.sampling_params=})"
        )


# Batch id
bid = 0


def get_global_bid():
    global bid
    return bid


def acc_global_bid():
    global bid
    bid += 1
    return bid


@dataclasses.dataclass
class ScheduleReqsInfo:
    """Store per-DP information for a batch of requests."""

    # Requests assigned to this DP rank
    reqs: list[Req] = None
    chunked_req: Req | None = None

    # Sampling info for this DP rank
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner (per DP)
    input_ids: np.ndarray = None  # shape: [b_per_dp], int32
    req_pool_indices: np.ndarray = None  # shape: [b_per_dp], int32
    seq_lens: np.ndarray = None  # shape: [b_per_dp], int32
    out_cache_loc: np.ndarray = None  # shape: [b_per_dp], int32
    output_ids: np.ndarray = None  # shape: [b_per_dp], int32

    # The sum of all sequence lengths for this DP rank
    seq_lens_sum: int = 0

    # For processing logprobs (per DP)
    top_logprobs_nums: list[int] | None = None
    token_ids_logprobs: list[list[int]] | None = None

    # For extend and mixed chunked prefill (per DP)
    prefix_lens: list[int] = None
    extend_lens: list[int] = None
    extend_num_tokens: int | None = None
    decoding_reqs: list[Req] = None
    extend_logprob_start_lens: list[int] = None
    extend_input_logprob_token_ids: np.ndarray | None = None

    # Speculative decoding info (per DP)
    spec_info: EagleDraftInput | EagleVerifyInput | None = None

    # Whether this DP rank's batch is full
    batch_is_full: bool = False

    # Recurrent state indices for hybrid recurrent models (per DP)
    recurrent_indices: np.ndarray | None = None


@dataclasses.dataclass
class ScheduleBatch:
    """Store all information of a batch on the scheduler.

    For DP > 1, per-DP request information is stored in reqs_info list.
    Global/shared state is stored directly in this class.
    """

    # Per-DP request information (list of length dp_size)
    reqs_info: list[ScheduleReqsInfo] = None
    bid: int = None

    # Memory pool and cache (shared across all DP ranks)
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None
    is_hybrid: bool = False
    is_hybrid_recurrent: bool = False

    # Batch configs (shared)
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False

    # For processing logprobs (shared settings)
    return_logprob: bool = False
    return_output_logprob_only: bool = False

    # For logits and logprob post processing (shared settings)
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # Stream (shared)
    has_stream: bool = False

    # Grammar-constrained decoding (shared)
    has_grammar: bool = False

    # device mesh (shared)
    mesh: mesh_lib.Mesh = None

    cache_miss_count: int = 0

    # Speculative decoding algorithm (shared)
    spec_algorithm: SpeculativeAlgorithm = None

    # Whether to return hidden states (shared)
    return_hidden_states: bool = False

    # Whether this batch is prefill-only (no token generation needed) (shared)
    is_prefill_only: bool = False

    # next batch sampling info for overlap scheduling
    next_batch_sampling_info: ModelWorkerSamplingInfo | None = None

    # Events (shared)
    launch_done: threading.Event | None = None

    # Whether to return captured experts
    return_routed_experts: bool = False

    # Deepstack
    apply_for_deepstack: bool = False
    input_embedding: list[list[list[float]]] | None = None
    deepstack_visual_embedding: list[list[list[float]]] | None = None
    deepstack_visual_pos_mask: list[list[int]] | None = None
    # Data Parallelism size
    dp_size: int = 1
    # Padded batch size per DP rank (set during get_model_worker_batch)
    per_dp_bs_size: int = 0

    @classmethod
    def init_new(
        cls,
        reqs: list[list[Req]],  # Per-DP requests: list of length dp_size
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        dp_size: int,
        spec_algorithm: SpeculativeAlgorithm = None,
        enable_custom_logit_processor: bool = False,
        chunked_reqs: (
            list[Req | None] | None
        ) = None,  # Per-DP chunked requests: list of length dp_size
        mesh: mesh_lib.Mesh = None,
    ):
        # Validate input
        assert len(reqs) == dp_size, f"reqs length {len(reqs)} != dp_size {dp_size}"
        if chunked_reqs is not None:
            assert (
                len(chunked_reqs) == dp_size
            ), f"chunked_reqs length {len(chunked_reqs)} != dp_size {dp_size}"
        else:
            chunked_reqs = [None] * dp_size

        # Flatten all reqs for global checks
        all_reqs = [req for dp_reqs in reqs for req in dp_reqs]

        return_logprob = any(req.return_logprob for req in all_reqs)
        return_output_logprob_only = all(req.return_output_logprob_only for req in all_reqs)
        is_hybrid = False
        if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            assert tree_cache is None or isinstance(
                tree_cache, (SWARadixCache, ChunkCache)
            ), "SWARadixCache or ChunkCache is required for SWATokenToKVPoolAllocator"
            is_hybrid = True

        is_hybrid_recurrent = isinstance(req_to_token_pool, HybridReqToTokenPool)

        # Initialize reqs_info based on dp_size with pre-distributed reqs
        reqs_info = []
        for dp_rank in range(dp_size):
            info = ScheduleReqsInfo()
            info.reqs = reqs[dp_rank]
            info.chunked_req = chunked_reqs[dp_rank]
            reqs_info.append(info)

        return cls(
            reqs_info=reqs_info,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            is_hybrid=is_hybrid,
            is_hybrid_recurrent=is_hybrid_recurrent,
            model_config=model_config,
            return_logprob=return_logprob,
            return_output_logprob_only=return_output_logprob_only,
            enable_overlap=enable_overlap,
            has_stream=any(req.stream for req in all_reqs),
            has_grammar=any(req.grammar for req in all_reqs),
            mesh=mesh,
            spec_algorithm=spec_algorithm,
            is_prefill_only=all(req.sampling_params.max_new_tokens == 0 for req in all_reqs),
            return_routed_experts=any(req.return_routed_experts for req in all_reqs),
            dp_size=dp_size,
        )

    # dp=1 spec-decode compat: pre-#939 spec code reads these as flat
    # ScheduleBatch attrs; passthrough to reqs_info[0] until the DP-aware
    # spec data contract (#1053 P1-5) replaces the callers.
    _SPEC_DP1_COMPAT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        f.name for f in dataclasses.fields(ScheduleReqsInfo)
    ) - frozenset({"batch_is_full"})

    def __getattr__(self, name: str):
        if name in ScheduleBatch._SPEC_DP1_COMPAT_FIELDS:
            assert self.dp_size == 1, (
                f"ScheduleBatch.{name} flat access is a dp=1 spec-decode shim; "
                f"dp={self.dp_size}>1 must use reqs_info[dp_rank].{name} (#1053 P1-5)"
            )
            return getattr(self.reqs_info[0], name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        if (
            name in ScheduleBatch._SPEC_DP1_COMPAT_FIELDS
            and "reqs_info" in self.__dict__
            and self.__dict__["reqs_info"]
        ):
            setattr(self.__dict__["reqs_info"][0], name, value)
        else:
            super().__setattr__(name, value)

    @property
    def batch_is_full(self) -> bool:
        return all(info.batch_is_full for info in self.reqs_info)

    def batch_size(self) -> int:
        """Get total number of requests across all DP ranks."""
        return sum(len(info.reqs) if info.reqs else 0 for info in self.reqs_info)

    def batch_size_per_dp(self, dp_rank: int) -> int:
        """Get number of requests for a specific DP rank."""
        if self.reqs_info[dp_rank].reqs is None:
            return 0
        return len(self.reqs_info[dp_rank].reqs)

    def is_empty(self) -> bool:
        """Check if batch is empty (no requests in any DP rank)."""
        return self.batch_size() == 0

    def alloc_req_slots(self, reqs: list[Req]):
        req_pool_indices = self.req_to_token_pool.alloc(reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
                "Please set a smaller number for `--max-running-requests`. "
                f"{self.req_to_token_pool.available_size()=}, "
                f"{len(reqs)=}, "
            )
        return req_pool_indices

    def alloc_paged_token_slots_decode(
        self,
        seq_lens: list[int],
        last_loc: list[int],
        backup_state: bool = False,
        dp_rank: int = 0,
    ):
        num_tokens = len(seq_lens) * self.token_to_kv_pool_allocator.page_size

        self._evict_tree_cache_if_needed({dp_rank: num_tokens})

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc_decode(
            seq_lens, last_loc, dp_rank=dp_rank
        )
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

    def mix_with_running(self, running_batch: ScheduleBatch):
        # Use EXTEND instead of MIXED for precompile cache hit
        self.forward_mode = ForwardMode.EXTEND
        if self.dp_size != running_batch.dp_size:
            raise ValueError(
                "mix_with_running requires matching dp_size, "
                f"got {self.dp_size} vs {running_batch.dp_size}"
            )

        # Snapshot per-DP merged tensor fields before merge_batch() clears out_cache_loc.
        merged_input_ids_per_dp: dict[int, np.ndarray] = {}
        merged_out_cache_loc_per_dp: dict[int, np.ndarray] = {}
        added_prefix_lens_per_dp: dict[int, list[int]] = {}

        delta = 0 if self.enable_overlap else -1
        for dp_rank in range(self.dp_size):
            self_info = self.reqs_info[dp_rank]
            running_info = running_batch.reqs_info[dp_rank]
            running_reqs = running_info.reqs or []
            if not running_reqs:
                continue

            for req in running_reqs:
                req.fill_ids = req.origin_input_ids + req.output_ids
                req.extend_input_len = 1

            self_input_ids = (
                np.asarray(self_info.input_ids, dtype=np.int32)
                if self_info.input_ids is not None
                else np.empty((0,), dtype=np.int32)
            )
            running_input_ids = (
                np.asarray(running_info.input_ids, dtype=np.int32)
                if running_info.input_ids is not None
                else np.empty((0,), dtype=np.int32)
            )
            merged_input_ids_per_dp[dp_rank] = np.concatenate(
                [self_input_ids, running_input_ids]
            ).astype(np.int32, copy=False)

            self_out_cache_loc = (
                np.asarray(self_info.out_cache_loc, dtype=np.int32)
                if self_info.out_cache_loc is not None
                else np.empty((0,), dtype=np.int32)
            )
            running_out_cache_loc = (
                np.asarray(running_info.out_cache_loc, dtype=np.int32)
                if running_info.out_cache_loc is not None
                else np.empty((0,), dtype=np.int32)
            )
            merged_out_cache_loc_per_dp[dp_rank] = np.concatenate(
                [self_out_cache_loc, running_out_cache_loc]
            ).astype(np.int32, copy=False)

            added_prefix_lens_per_dp[dp_rank] = [
                len(r.origin_input_ids) + len(r.output_ids) + delta for r in running_reqs
            ]

        self.merge_batch(running_batch)

        for dp_rank in range(self.dp_size):
            if dp_rank not in merged_input_ids_per_dp:
                continue

            info = self.reqs_info[dp_rank]
            added_prefix_lens = added_prefix_lens_per_dp[dp_rank]
            added_count = len(added_prefix_lens)

            info.input_ids = merged_input_ids_per_dp[dp_rank]
            info.out_cache_loc = merged_out_cache_loc_per_dp[dp_rank]

            if info.prefix_lens is None:
                info.prefix_lens = []
            info.prefix_lens.extend(added_prefix_lens)

            if info.extend_lens is None:
                info.extend_lens = []
            info.extend_lens.extend([1] * added_count)

            if info.extend_logprob_start_lens is None:
                info.extend_logprob_start_lens = []
            info.extend_logprob_start_lens.extend([0] * added_count)

            if self.is_hybrid_recurrent:
                info.recurrent_indices = self.req_to_token_pool.get_linear_recurrent_indices(
                    info.req_pool_indices
                )

            info.extend_num_tokens = (info.extend_num_tokens or 0) + added_count

    def prepare_for_extend(self):
        """Prepare for extend phase (unified for all dp_size >= 1).

        Process each DP rank independently. For dp_size=1, this is equivalent to
        the old single-rank logic but with cleaner structure.
        """
        self.forward_mode = ForwardMode.EXTEND

        # Process each DP rank
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            reqs = info.reqs

            # Skip empty DP ranks
            if not reqs:
                # Clear fields to avoid stale data
                info.input_ids = None
                info.seq_lens = None
                info.prefix_lens = None
                info.extend_lens = None
                info.out_cache_loc = None
                info.seq_lens_sum = 0
                info.extend_num_tokens = None
                continue

            # Allocate req slots
            req_pool_indices = self.alloc_req_slots(reqs)

            # Init arrays
            seq_lens = [len(r.fill_ids) for r in reqs]
            prefix_lens = [len(r.prefix_indices) for r in reqs]
            extend_lens = [r.extend_input_len for r in reqs]
            extend_num_tokens = sum(extend_lens)

            req_pool_indices_cpu = np.array(req_pool_indices, dtype=np.int32)
            input_ids_cpu = np.fromiter(
                itertools.chain.from_iterable(
                    r.fill_ids[pre_len:] for r, pre_len in zip(reqs, prefix_lens)
                ),
                dtype=np.int32,
                count=extend_num_tokens,
            )
            seq_lens_cpu = np.array(seq_lens, dtype=np.int32)
            prefix_lens_cpu = np.array(prefix_lens, dtype=np.int32)

            # Copy prefix and do some basic check
            extend_input_logprob_token_ids = []

            for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
                assert seq_len - pre_len == req.extend_input_len

                req.kv_committed_len = seq_len
                req.kv_allocated_len = seq_len
                req.cache_protected_len = pre_len

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

            # Allocate memory for this DP rank
            if self.token_to_kv_pool_allocator.page_size == 1:
                out_cache_loc = alloc_token_slots(
                    self.tree_cache, extend_num_tokens, dp_rank=dp_rank
                )
            else:
                last_loc_cpu = get_last_loc(
                    self.req_to_token_pool.req_to_token,
                    req_pool_indices_cpu,
                    prefix_lens_cpu,
                )
                out_cache_loc = alloc_paged_token_slots_extend(
                    self.tree_cache,
                    prefix_lens,
                    seq_lens,
                    last_loc_cpu.tolist(),
                    extend_num_tokens,
                    dp_rank=dp_rank,
                )

            for req in reqs:
                req.extend_batch_idx += 1

            # Set fields for this DP rank's info
            info.input_ids = input_ids_cpu
            info.req_pool_indices = req_pool_indices_cpu
            info.seq_lens = seq_lens_cpu
            info.out_cache_loc = out_cache_loc
            info.seq_lens_sum = sum(seq_lens)

            if self.return_logprob:
                info.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
                info.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

            info.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
            info.extend_num_tokens = extend_num_tokens
            info.prefix_lens = prefix_lens
            info.extend_lens = extend_lens
            info.extend_input_logprob_token_ids = extend_input_logprob_token_ids

            if self.is_hybrid_recurrent:
                info.recurrent_indices = self.req_to_token_pool.get_linear_recurrent_indices(
                    req_pool_indices_cpu
                )

            # Write to req_to_token_pool
            pt = 0
            for i in range(len(reqs)):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )

                pt += extend_lens[i]

            info.sampling_info = SamplingBatchInfo.from_schedule_batch(
                info,
                self.model_config.vocab_size,
                batch=self,
            )

        # Evict SWA tokens outside sliding window
        self.maybe_evict_swa()

    def new_tokens_required_next_decode(
        self,
        dp_rank: int,
        selected_indices: list[int] | None = None,
    ) -> int:
        """Calculate tokens required for next decode for a specific DP rank.

        Follows upstream sglang: uses kv_committed_len to determine page
        boundary crossings.

        Args:
            dp_rank: DP rank to calculate for
            selected_indices: Optional local indices within this DP rank

        Returns:
            Number of new tokens needed for this DP rank.
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        info = self.reqs_info[dp_rank]

        if not info.reqs:
            return 0

        requests = (
            info.reqs if selected_indices is None else [info.reqs[i] for i in selected_indices]
        )

        new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
        return new_pages * page_size

    def check_decode_mem(self, selected_indices: dict[int, list[int]] | None = None):
        """Check if all DP ranks have sufficient memory for next decode step.

        Follows upstream sglang: compute tokens needed, evict, check available.

        Args:
            selected_indices: Optional per-DP indices to check
                              Format: {dp_rank: [local_index_0, local_index_1, ...]}
                              If None, checks all requests in all DP ranks

        Returns:
            False if any DP rank has insufficient memory.
        """
        num_tokens_per_dp = {}
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            if not info.reqs:
                continue
            indices = selected_indices.get(dp_rank) if selected_indices else None
            num_tokens_per_dp[dp_rank] = self.new_tokens_required_next_decode(dp_rank, indices)

        self._evict_tree_cache_if_needed(num_tokens_per_dp)
        return self._is_available_size_sufficient(num_tokens_per_dp)

    def retract_decode(self, server_args: ServerArgs):
        """Retract requests when memory insufficient.

        Each DP rank independently:
        - Checks its own memory
        - Sorts its requests by priority
        - Retracts until sufficient

        Returns:
            retracted_reqs: All retracted requests
            new_estimate_ratio: Updated estimate
            reqs_to_abort: Requests aborted due to OOM
        """

        # Helper function: check if memory is sufficient for given DP rank
        def has_sufficient_memory(dp_rank: int, indices: list[int]) -> bool:
            num_tokens = self.new_tokens_required_next_decode(dp_rank, indices)

            evict_from_tree_cache(self.tree_cache, num_tokens, dp_rank=dp_rank)

            if self.is_hybrid:
                full_ok = (
                    self.token_to_kv_pool_allocator.full_available_size(dp_rank=dp_rank)
                    >= num_tokens
                )
                swa_ok = (
                    self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
                    >= num_tokens
                )
                return full_ok and swa_ok
            else:
                return self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank) >= num_tokens

        retracted_reqs = []
        reqs_to_abort = []
        keep_indices_per_dp = {}

        # Process each DP rank independently
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            # Skip empty ranks
            if not info.reqs or len(info.reqs) == 0:
                keep_indices_per_dp[dp_rank] = []
                continue

            # Sort requests by priority (local to this rank)
            sorted_indices = list(range(len(info.reqs)))
            sorted_indices.sort(
                key=lambda i: (
                    len(info.reqs[i].output_ids),
                    -len(info.reqs[i].origin_input_ids),
                ),
                reverse=True,
            )

            # Retract until sufficient for this rank
            first_iter = True
            while first_iter or (not has_sufficient_memory(dp_rank, sorted_indices)):
                if len(sorted_indices) == 1:
                    # Keep at least one request in the loop; handle OOM below.
                    break

                first_iter = False
                retract_idx = sorted_indices.pop()
                req = info.reqs[retract_idx]
                retracted_reqs.append(req)

                # Release the request using its local index within this DP rank
                self.release_req(retract_idx, dp_rank, len(sorted_indices), server_args)

            # If the last remaining request still can't fit, abort it gracefully
            # instead of crashing the scheduler (follows upstream sglang).
            if len(sorted_indices) <= 1 and not has_sufficient_memory(dp_rank, sorted_indices):
                last_idx = sorted_indices.pop()
                last_req = info.reqs[last_idx]
                last_req.to_finish = FINISH_ABORT(
                    f"Out of memory in DP rank {dp_rank} even after retracting all other requests "
                    "in the decode batch. Aborting the last request.",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "InternalServerError",
                )
                reqs_to_abort.append(last_req)
                self.release_req(last_idx, dp_rank, 0, server_args)
                logger.warning(
                    "retract_decode: aborted last request %s in DP rank %d due to OOM",
                    last_req.rid,
                    dp_rank,
                )

            keep_indices_per_dp[dp_rank] = sorted_indices

        # Apply filtering
        self.filter_batch(keep_indices=keep_indices_per_dp)

        # Calculate global estimate ratio by aggregating across all DP ranks
        all_reqs = [req for info in self.reqs_info for req in (info.reqs if info.reqs else [])]
        total_decoded_tokens = sum(len(r.output_ids) for r in all_reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in all_reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(all_reqs)
        ) / (
            total_max_new_tokens + 1
        )  # +1 to avoid zero division when all reqs aborted
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    def release_req(self, idx: int, dp_rank: int, remaing_req_count: int, server_args: ServerArgs):
        info = self.reqs_info[dp_rank]
        req = info.reqs[idx]

        release_kv_cache(req, self.tree_cache, is_insert=False)

        num_tokens = remaing_req_count * global_config.retract_decode_steps
        self._evict_tree_cache_if_needed({dp_rank: num_tokens})

        req.reset_for_retract()

    def retract_all(self, server_args: ServerArgs):
        retracted_reqs = [
            req for info in self.reqs_info for req in (info.reqs if info.reqs else [])
        ]
        for dp_rank, dp_reqs in enumerate(self.reqs_info):
            for idx, _ in enumerate(dp_reqs.reqs):
                self.release_req(idx, dp_rank, len(dp_reqs.reqs) - 1, server_args)

        self.filter_batch(keep_indices={dp_rank: [] for dp_rank in range(self.dp_size)})
        return retracted_reqs

    def prepare_for_idle(self):
        """Prepare for idle phase (unified for all dp_size >= 1).

        Initialize empty arrays for each DP rank.
        """
        self.forward_mode = ForwardMode.IDLE

        # Process each DP rank
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            # Set empty arrays for this DP rank
            info.input_ids = np.empty(0, np.int32)
            info.seq_lens = np.empty(0, np.int32)
            info.out_cache_loc = np.empty(0, np.int32)
            info.req_pool_indices = np.empty(0, np.int32)
            info.seq_lens_sum = 0
            info.extend_num_tokens = 0

            info.sampling_info = SamplingBatchInfo.from_schedule_batch(
                info,
                self.model_config.vocab_size,
                batch=self,
            )

    def maybe_evict_swa(self, sliding_window_size=None):
        """Evict SWA pool slots outside the sliding window for all requests."""
        if not self.is_hybrid:
            return
        if sliding_window_size is None:
            sliding_window_size = getattr(self.model_config, "sliding_window", None)
        if sliding_window_size is None or sliding_window_size <= 0:
            return
        page_size = getattr(
            self.token_to_kv_pool_allocator,
            "_page_size",
            getattr(self.token_to_kv_pool_allocator, "page_size", 1),
        )

        if self.forward_mode is not None and self.forward_mode.is_decode():
            multiplier = float(os.environ.get("SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER", "1.0"))
            evict_interval = max(page_size, int(sliding_window_size * multiplier))
            evict_interval = (evict_interval // page_size) * page_size
            for dp_rank, info in enumerate(self.reqs_info):
                if not info.reqs:
                    continue
                for req in info.reqs:
                    if isinstance(self.tree_cache, ChunkCache):
                        # ChunkCache/SWAChunkCache: no tree-node overlap concern,
                        # evict on every decode step to prevent SWA exhaustion.
                        # TODO(PD-disagg): evicting at decode_batch_idx==0 may
                        # conflict with KV transfer in a future PD disaggregation
                        # pipeline; revisit when implementing PD.
                        if req.decode_batch_idx % evict_interval == 0:
                            self._evict_swa(
                                req, req.seqlen - 1, sliding_window_size, page_size, dp_rank
                            )
                    else:
                        # SWARadixCache: skip decode_batch_idx==0 in overlap mode
                        # because the previous extend batch may still be running.
                        if req.decode_batch_idx % evict_interval == 1:
                            self._evict_swa(
                                req, req.seqlen - 1, sliding_window_size, page_size, dp_rank
                            )
            return

        if self.forward_mode is None or not self.forward_mode.is_extend():
            return

        # For SWARadixCache with active tree, extend-time SWA ownership stays
        # with the tree — eviction is deferred to tree insert / pressure handling.
        # ChunkCache and SWAChunkCache need direct per-request eviction.
        if not isinstance(self.tree_cache, ChunkCache):
            return

        chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
        for dp_rank, info in enumerate(self.reqs_info):
            if not info.reqs or not info.prefix_lens:
                continue
            for idx, req in enumerate(info.reqs):
                pre_len = info.prefix_lens[idx]
                if self.enable_overlap:
                    # In overlap mode, the previous extend batch is still running
                    # when we schedule/evict, so skip the first two extend batches.
                    if req.extend_batch_idx < 2:
                        continue
                    if chunked_prefill_size is not None and chunked_prefill_size > 0:
                        pre_len -= chunked_prefill_size
                self._evict_swa(req, pre_len, sliding_window_size, page_size, dp_rank)

    def _evict_swa(
        self,
        req: Req,
        pre_len: int,
        sliding_window_size: int,
        page_size: int,
        dp_rank: int = 0,
    ):
        """Free SWA pool slots for tokens outside the sliding window."""
        if isinstance(self.tree_cache, SWARadixCache):
            self.tree_cache.evict_req_swa(req, pre_len, dp_rank=dp_rank)
            return

        new_evicted = max(req.swa_evicted_seqlen, pre_len - sliding_window_size - page_size)
        if page_size > 1:
            new_evicted = (new_evicted // page_size) * page_size
        if new_evicted <= req.swa_evicted_seqlen:
            return
        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_evicted
        ]
        self.token_to_kv_pool_allocator.free_swa(free_slots, dp_rank=dp_rank)
        req.swa_evicted_seqlen = new_evicted

    def prepare_for_decode(self):
        """Prepare for decode phase (unified for all dp_size >= 1).

        Process each DP rank independently. For dp_size=1, this is equivalent to
        the old single-rank logic but with cleaner structure.
        """
        self.forward_mode = ForwardMode.DECODE

        self.maybe_evict_swa()

        # prepare_for_decode requires cross-rank-flat allocate_lens
        # (asserts shape[0] == batch_size); rebuild via _concat, run it, then
        # split allocate_lens back to per-rank.
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            for info in self.reqs_info:
                if not info.reqs:
                    info.input_ids = None
                    info.output_ids = None
                    info.seq_lens = None
                    info.out_cache_loc = None
                    info.seq_lens_sum = 0
            flat_spec = self._concat_spec_info_per_rank([info.spec_info for info in self.reqs_info])
            flat_spec.prepare_for_decode(self)
            real_bs_per_dp = [len(info.reqs) if info.reqs else 0 for info in self.reqs_info]
            per_rank_spec = self._split_spec_info_per_rank(flat_spec, real_bs_per_dp)
            for r, s in enumerate(per_rank_spec):
                self.reqs_info[r].spec_info = s
            return

        # Process each DP rank
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            reqs = info.reqs

            # Skip empty DP ranks
            if not reqs:
                # Clear fields to avoid stale data
                info.input_ids = None
                info.output_ids = None
                info.seq_lens = None
                info.out_cache_loc = None
                info.seq_lens_sum = 0
                continue

            bs = len(reqs)

            if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
                continue  # Skip to next DP rank

            # Handle penalizer if required
            if info.sampling_info.penalizer_orchestrator.is_required:
                if self.enable_overlap:
                    # TODO: this can be slow, optimize this.
                    delayed_output_ids = np.array(
                        [
                            (
                                req.output_ids[-1]
                                if len(req.output_ids)
                                else req.origin_input_ids[-1]
                            )
                            for req in reqs
                        ],
                        dtype=np.int64,
                    )
                    info.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                        delayed_output_ids
                    )
                else:
                    info.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                        info.output_ids
                    )

            # Update fields
            info.input_ids = info.output_ids
            info.output_ids = None
            locs = info.seq_lens.copy()

            if self.enable_overlap:
                info.seq_lens = info.seq_lens + 1
            else:
                info.seq_lens = np.add(info.seq_lens, 1)
            info.seq_lens_sum += bs

            if self.is_hybrid_recurrent:
                info.recurrent_indices = self.req_to_token_pool.get_linear_recurrent_indices(
                    info.req_pool_indices
                )

            # Allocate memory for this DP rank
            if self.token_to_kv_pool_allocator.page_size == 1:
                info.out_cache_loc = alloc_token_slots(self.tree_cache, bs, dp_rank=dp_rank)
            else:
                last_loc = self.req_to_token_pool.req_to_token[
                    info.req_pool_indices, info.seq_lens - 2
                ]
                info.out_cache_loc = self.alloc_paged_token_slots_decode(
                    info.seq_lens.tolist(),
                    last_loc.tolist(),
                    dp_rank=dp_rank,
                )

            # Write to pool
            self.req_to_token_pool.write(
                (info.req_pool_indices, locs), info.out_cache_loc.astype(np.int32)
            )
            for req in reqs:
                req.decode_batch_idx += 1
                req.kv_committed_len += 1
                req.kv_allocated_len += 1

    def filter_batch(
        self,
        chunked_req_to_exclude: dict[int, Req] | None = None,
        keep_indices: dict[int, list[int]] | None = None,
    ):
        """Filter completed requests from batch.

        Args:
            chunked_req_to_exclude: Optional dict mapping dp_rank -> chunked request to exclude for that rank
            keep_indices: Optional dict mapping dp_rank -> list of indices to keep for that rank.
                         If None, automatically calculates based on finished() status.
        """
        # Normalize exclusion dict
        if chunked_req_to_exclude is None:
            chunked_req_to_exclude = {}

        # Unified DP filtering logic (works for all dp_size including 1).
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            # Skip if this DP rank has no requests
            if not info.reqs or len(info.reqs) == 0:
                continue

            # Get chunked request to exclude for this rank
            chunked_req = chunked_req_to_exclude.get(dp_rank)

            # Get keep_indices for this DP rank
            if keep_indices is not None:
                # Use provided keep_indices for this rank
                keep_indices_dp = keep_indices.get(dp_rank, [])
                if keep_indices_dp is None:
                    keep_indices_dp = []
            else:
                keep_indices_dp = [
                    i
                    for i in range(len(info.reqs))
                    if not info.reqs[i].finished()
                    and (chunked_req is None or info.reqs[i] != chunked_req)
                ]

            # Early exit: Clear all if nothing to keep
            if len(keep_indices_dp) == 0:
                info.reqs = []
                info.req_pool_indices = None
                info.seq_lens = None
                info.output_ids = None
                info.out_cache_loc = None
                info.seq_lens_sum = 0
                info.top_logprobs_nums = None
                info.token_ids_logprobs = None
                info.sampling_info = None
                info.spec_info = None
                continue

            # Early exit: No filtering needed if all requests kept
            if len(keep_indices_dp) == len(info.reqs):
                continue

            # Filter reqs list
            info.reqs = [info.reqs[i] for i in keep_indices_dp]

            # Filter memory pool indices
            if info.req_pool_indices is not None:
                info.req_pool_indices = info.req_pool_indices[keep_indices_dp]

            # Filter sequence data
            if info.seq_lens is not None:
                info.seq_lens = info.seq_lens[keep_indices_dp]

            if info.output_ids is not None:
                info.output_ids = info.output_ids[keep_indices_dp]

            # Reset cache location (will be recomputed)
            info.out_cache_loc = None

            # Recalculate seq_lens_sum for this DP rank
            if info.seq_lens is not None:
                info.seq_lens_sum = info.seq_lens.sum().item()
            else:
                info.seq_lens_sum = 0

            # Filter logprob lists
            if info.top_logprobs_nums is not None:
                info.top_logprobs_nums = [info.top_logprobs_nums[i] for i in keep_indices_dp]
                info.token_ids_logprobs = [info.token_ids_logprobs[i] for i in keep_indices_dp]

            # Filter sampling_info
            if info.sampling_info is not None:
                info.sampling_info.filter_batch(np.array(keep_indices_dp))

            # Filter spec_info (per-rank EagleDraftInput; handles all 5 spec arrays).
            if info.spec_info is not None:
                if chunked_req_to_exclude is not None and len(chunked_req_to_exclude) > 0:
                    has_been_filtered = False
                else:
                    has_been_filtered = True
                info.spec_info.filter_batch(
                    new_indices=keep_indices_dp, has_been_filtered=has_been_filtered
                )

        # Recalculate global batch flags from all remaining requests
        all_reqs = [req for info in self.reqs_info for req in (info.reqs if info.reqs else [])]

        if len(all_reqs) > 0:
            self.return_logprob = any(req.return_logprob for req in all_reqs)
            self.return_output_logprob_only = all(
                req.return_output_logprob_only for req in all_reqs
            )
            self.has_stream = any(req.stream for req in all_reqs)
            self.has_grammar = any(req.grammar for req in all_reqs)
        else:
            self.return_logprob = False
            self.return_output_logprob_only = False
            self.has_stream = False
            self.has_grammar = False

    def merge_batch(self, other: ScheduleBatch):
        """Merge another batch into this batch (unified for all dp_size >= 1).

        Merge each DP rank independently.
        """
        # Ensure both batches have same dp_size
        assert (
            self.dp_size == other.dp_size
        ), f"Cannot merge batches with different dp_size: {self.dp_size} vs {other.dp_size}"

        # Merge each DP rank independently
        for dp_rank in range(self.dp_size):
            self_info = self.reqs_info[dp_rank]
            other_info = other.reqs_info[dp_rank]

            # Skip if other batch has no requests for this DP rank
            if not other_info.reqs or len(other_info.reqs) == 0:
                continue

            # Initialize self_info if it's empty
            if not self_info.reqs or len(self_info.reqs) == 0:
                # Copy everything from other_info
                self_info.reqs = other_info.reqs.copy()
                self_info.sampling_info = other_info.sampling_info
                # reqs_info is a weakref (orchestrator.py) still pointing at the
                # transient other_info; rebind to the surviving self_info so
                # reqs() stays live, else penalizers desync -> IndexError crash.
                if (
                    self_info.sampling_info is not None
                    and self_info.sampling_info.penalizer_orchestrator is not None
                ):
                    self_info.sampling_info.penalizer_orchestrator.reqs_info = self_info
                self_info.req_pool_indices = other_info.req_pool_indices
                self_info.seq_lens = other_info.seq_lens
                self_info.out_cache_loc = other_info.out_cache_loc
                self_info.seq_lens_sum = other_info.seq_lens_sum
                self_info.output_ids = other_info.output_ids
                self_info.top_logprobs_nums = other_info.top_logprobs_nums
                self_info.token_ids_logprobs = other_info.token_ids_logprobs
                self_info.spec_info = other_info.spec_info
                continue

            # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
            # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
            # needs to be called with pre-merged Batch.reqs.
            if self_info.sampling_info and other_info.sampling_info:
                self_info.sampling_info.merge_batch(other_info.sampling_info)

            # Merge arrays
            self_info.req_pool_indices = np.concat(
                [self_info.req_pool_indices, other_info.req_pool_indices]
            )
            self_info.seq_lens = np.concat([self_info.seq_lens, other_info.seq_lens])
            self_info.out_cache_loc = None  # Will be recomputed
            self_info.seq_lens_sum += other_info.seq_lens_sum

            # Merge output_ids
            if self_info.output_ids is not None and other_info.output_ids is not None:
                self_info.output_ids = np.concat(
                    [
                        self_info.output_ids[: len(self_info.seq_lens) - len(other_info.seq_lens)],
                        other_info.output_ids[: len(other_info.seq_lens)],
                    ]
                )

            # Merge logprobs
            if self.return_logprob and other.return_logprob:
                if (
                    self_info.top_logprobs_nums is not None
                    and other_info.top_logprobs_nums is not None
                ):
                    self_info.top_logprobs_nums.extend(other_info.top_logprobs_nums)
                    self_info.token_ids_logprobs.extend(other_info.token_ids_logprobs)
            elif self.return_logprob and self_info.top_logprobs_nums is not None:
                self_info.top_logprobs_nums.extend([0] * len(other_info.reqs))
                self_info.token_ids_logprobs.extend([None] * len(other_info.reqs))
            elif other.return_logprob and other_info.top_logprobs_nums is not None:
                self_info.top_logprobs_nums = [0] * len(
                    self_info.reqs
                ) + other_info.top_logprobs_nums
                self_info.token_ids_logprobs = [None] * len(
                    self_info.reqs
                ) + other_info.token_ids_logprobs

            # Merge reqs list
            self_info.reqs.extend(other_info.reqs)

            # Merge spec_info
            if self_info.spec_info and other_info.spec_info:
                self_info.spec_info.merge_batch(other_info.spec_info)

        # Update global flags
        self.return_logprob |= other.return_logprob
        self.return_output_logprob_only |= other.return_output_logprob_only
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states

    def _compute_global_padding_sizes(
        self,
        token_paddings: list,
        bs_paddings: list,
    ) -> tuple[int, int, int, int]:
        """Compute global padding sizes across all DP ranks.

        Returns:
            (per_dp_token_padding, total_token_size, per_dp_bs_padding, total_bs)
        """
        # Find max token count and batch size across all DP ranks
        max_tokens_per_dp = 0
        max_bs_per_dp = 0

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            if info.input_ids is not None:
                max_tokens_per_dp = max(max_tokens_per_dp, len(info.input_ids))
            if info.seq_lens is not None:
                max_bs_per_dp = max(max_bs_per_dp, len(info.seq_lens))

        token_padding, _ = pad_to_bucket(max_tokens_per_dp * self.dp_size, token_paddings)
        bs_padding, _ = pad_to_bucket(max_bs_per_dp * self.dp_size, bs_paddings)

        return (
            token_padding // self.dp_size,
            token_padding,
            bs_padding // self.dp_size,
            bs_padding,
        )

    def _merge_input_and_positions(
        self,
        per_dp_token_size: int,
        total_token_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Merge input_ids, positions, and out_cache_loc from all DP ranks.

        Returns:
            (input_ids, positions, out_cache_loc, real_input_ids_len)
        """
        input_ids_cpu = np.zeros(total_token_size, dtype=np.int32)
        positions_cpu = np.zeros(total_token_size, dtype=np.int32)
        out_cache_loc_cpu = np.full(total_token_size, -1, dtype=np.int32)

        offset = 0
        real_input_ids_len = 0

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.input_ids is None or len(info.input_ids) == 0:
                # Empty DP rank, just add padding
                offset += per_dp_token_size
                continue

            # Get data from this DP rank
            dp_input_ids = info.input_ids
            dp_len = len(dp_input_ids)
            real_input_ids_len += dp_len

            # Copy data to merged array
            input_ids_cpu[offset : offset + dp_len] = dp_input_ids

            # Build positions for this DP rank
            if self.forward_mode.is_extend():
                # For extend: positions are [prefix_len, prefix_len+1, ..., seq_len-1] for each request
                pt = offset
                for seq_len, prefix_len in zip(info.seq_lens, info.prefix_lens):
                    next_pt = pt + (seq_len - prefix_len)
                    positions_cpu[pt:next_pt] = np.arange(prefix_len, seq_len, dtype=np.int32)
                    pt = next_pt
            else:
                # For decode: positions are [seq_len-1] for each request
                dp_positions = info.seq_lens - 1
                positions_cpu[offset : offset + len(dp_positions)] = dp_positions

            # Copy out_cache_loc if available
            if info.out_cache_loc is not None:
                out_len = min(len(info.out_cache_loc), dp_len)
                out_cache_loc_cpu[offset : offset + out_len] = info.out_cache_loc[:out_len]

            # Move to next DP rank's section (with padding)
            offset += per_dp_token_size

        return input_ids_cpu, positions_cpu, out_cache_loc_cpu, real_input_ids_len

    def _merge_multimodal(
        self,
        per_dp_token_size: int,
        total_token_size: int,
    ) -> dict:
        """Assemble all per-token multimodal tensors in one DP-interleaved pass.

        Single traversal of ``reqs_info[*].reqs`` that produces, on the same
        rank-offset layout as ``_merge_input_and_positions`` (per-rank slot
        stride ``per_dp_token_size``; within a rank the per-req EXTEND window
        ``[prefix_len, seq_len)``), all three multimodal tensors at once:

        - ``input_embedding`` ``[total_token_size, hidden]`` -- per-req merged
          embedding sliced to its extend window.
        - ``mrope_positions`` ``[3, total_token_size]`` -- 3-D mRoPE positions;
          extend slices ``mm_positions[:, prefix:prefix+ext]`` (delta / arange
          fallback), decode advances ``seq_len-1 (+delta)``.
        - ``deepstack_visual_embedding`` ``[num_layers, total_token_size,
          hidden]`` -- sparse visual rows densified into the batched layout with
          non-visual rows zero, plus the derived ``apply_for_deepstack``.

        Data stays on ``Req`` (no new ScheduleReqsInfo fields); this only reads it. Collapses the
        three previously separate rank-offset loops so the layout logic lives in
        exactly one place. Each field is ``None`` / ``False`` when no request
        carries it, keeping pure-text / non-multimodal paths unchanged (0-diff).
        """
        is_extend = self.forward_mode.is_extend()
        is_decode = self.forward_mode.is_decode()

        has_mrope = any(
            _extract_mm_value(getattr(req, "mm_inputs", None), "mrope_positions") is not None
            or _extract_mm_value(getattr(req, "mm_inputs", None), "mrope_position_delta")
            is not None
            for info in self.reqs_info
            if info.reqs
            for req in info.reqs
        )

        # input_embedding / deepstack are extend-only; mrope also refreshes on
        # decode. Nothing to assemble otherwise -> all None/False (0-diff).
        emb = None
        mrope = np.zeros((3, total_token_size), dtype=np.int32) if has_mrope else None
        dense = None
        if not is_extend and mrope is None:
            return {
                "input_embedding": None,
                "mrope_positions": None,
                "apply_for_deepstack": False,
                "deepstack_visual_embedding": None,
            }

        offset = 0
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            if not info.reqs or info.seq_lens is None or len(info.seq_lens) == 0:
                offset += per_dp_token_size
                continue
            local = 0

            if is_decode:
                # Decode: one token per request; only mrope advances (embedding
                # and deepstack are extend-only and stay None/False).
                if mrope is not None:
                    for req, seq_len in zip(info.reqs, info.seq_lens):
                        base_pos = int(seq_len) - 1
                        delta = _extract_mm_value(
                            getattr(req, "mm_inputs", None), "mrope_position_delta"
                        )
                        if delta is not None:
                            base_pos += _as_int_scalar(delta)
                        mrope[:, offset + local] = base_pos
                        local += 1
                offset += per_dp_token_size
                continue

            # Extend: write each req's [prefix_len, seq_len) window.
            for req, seq_len, prefix_len in zip(info.reqs, info.seq_lens, info.prefix_lens):
                ext_len = int(seq_len) - int(prefix_len)
                if ext_len <= 0:
                    continue
                start = int(prefix_len or 0)
                end = start + ext_len

                # input_embedding: per-req merged embedding, extend window.
                mm_emb = getattr(req, "multimodal_embedding", None)
                if mm_emb is not None:
                    mm_full = np.asarray(mm_emb)
                    chunk = mm_full[start:end]
                    if emb is None:
                        emb = np.zeros((total_token_size, mm_full.shape[1]), dtype=mm_full.dtype)
                    emb[offset + local : offset + local + chunk.shape[0]] = chunk

                # mrope_positions: 3-D positions, slice with fallback.
                if mrope is not None:
                    mm_positions = _extract_mm_value(
                        getattr(req, "mm_inputs", None), "mrope_positions"
                    )
                    if mm_positions is None:
                        # Text-only req in a mixed mrope batch: 1-D positions
                        # broadcast to 3 rows (T==H==W), matching the model's
                        # non-mrope fallback for these tokens.
                        base = np.arange(start, start + ext_len, dtype=np.int32)
                        mchunk = np.broadcast_to(base.reshape(1, -1), (3, ext_len))
                    else:
                        mchunk = np.asarray(mm_positions)[:, start : start + ext_len]
                        if mchunk.size == 0:
                            delta = _extract_mm_value(
                                getattr(req, "mm_inputs", None), "mrope_position_delta"
                            )
                            base = np.arange(start, start + ext_len, dtype=np.int32)
                            if delta is not None:
                                base = base + _as_int_scalar(delta)
                            mchunk = np.broadcast_to(base.reshape(1, -1), (3, ext_len))
                    mrope[:, offset + local : offset + local + ext_len] = mchunk

                # deepstack: densify sparse visual rows into batched layout,
                # non-visual rows stay zero (so the model can add to all tokens).
                ds_emb = getattr(req, "deepstack_visual_embedding", None)
                ds_mask = getattr(req, "deepstack_visual_pos_mask", None)
                if (
                    getattr(req, "apply_for_deepstack", False)
                    and ds_emb is not None
                    and ds_mask is not None
                ):
                    full_mask = np.asarray(ds_mask).astype(bool)
                    emb_arr = np.asarray(ds_emb)  # (num_layers, num_visual, hidden)
                    # Only valid when the per-req mask spans the full prompt
                    # (skips the audio-only dummy [1]-length fallback).
                    if full_mask.shape[0] >= end and emb_arr.ndim == 3:
                        window_mask = full_mask[start:end]
                        nvis = int(window_mask.sum())
                        if nvis > 0:
                            vstart = int(full_mask[:start].sum())
                            window_emb = emb_arr[:, vstart : vstart + nvis, :]
                            if dense is None:
                                dense = np.zeros(
                                    (emb_arr.shape[0], total_token_size, emb_arr.shape[2]),
                                    dtype=emb_arr.dtype,
                                )
                            vis_pos = offset + local + np.nonzero(window_mask)[0]
                            dense[:, vis_pos, :] = window_emb

                local += ext_len
            offset += per_dp_token_size

        return {
            "input_embedding": emb,
            "mrope_positions": mrope,
            "apply_for_deepstack": dense is not None,
            "deepstack_visual_embedding": dense,
        }

    def _merge_batch_metadata(
        self,
        per_dp_bs_size: int,
        total_bs: int,
    ):
        """Merge batch-level metadata from all DP ranks.

        Returns:
            (req_pool_indices, seq_lens, extend_prefix_lens,
             extend_seq_lens, extend_logprob_start_lens, logits_indices, real_bs,
             real_bs_per_dp, logits_indices_selector)

        logits_indices_selector maps "original request order"
        (i.e., DP-rank-then-req flat order) to the DP-interleaved padded
        slot in the global batch. It lets host-side code reorder per-req
        outputs (e.g. logprobs) back to original order with one numpy
        gather, instead of rederiving per-rank offsets at every callsite.
        """
        req_pool_indices_cpu = np.full(total_bs, -1, dtype=np.int32)
        seq_lens_cpu = np.zeros(total_bs, dtype=np.int32)

        if self.forward_mode.is_extend():
            extend_prefix_lens = np.zeros(total_bs, dtype=np.int32)
            extend_seq_lens = np.zeros(total_bs, dtype=np.int32)
            extend_logprob_start_lens = np.zeros(total_bs, dtype=np.int32)
            logits_indices = np.full(total_bs, 0, dtype=np.int32)
        else:
            extend_prefix_lens = None
            extend_seq_lens = None
            extend_logprob_start_lens = None
            logits_indices = None

        offset_bs = 0
        real_bs = 0
        real_bs_per_dp = [0] * self.dp_size
        selector_chunks: list[np.ndarray] = []

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.seq_lens is None or len(info.seq_lens) == 0:
                # Empty DP rank
                offset_bs += per_dp_bs_size
                continue

            # Get data from this DP rank
            dp_bs = len(info.seq_lens)
            real_bs += dp_bs
            real_bs_per_dp[dp_rank] = dp_bs

            # Copy batch metadata
            req_pool_indices_cpu[offset_bs : offset_bs + dp_bs] = info.req_pool_indices
            seq_lens_cpu[offset_bs : offset_bs + dp_bs] = info.seq_lens

            if self.forward_mode.is_extend():
                # Copy extend-specific metadata
                extend_prefix_lens[offset_bs : offset_bs + dp_bs] = info.prefix_lens
                extend_seq_lens[offset_bs : offset_bs + dp_bs] = info.extend_lens
                dp_extend_lens = np.array(info.extend_lens, dtype=np.int32)
                local_last = np.cumsum(dp_extend_lens, dtype=np.int32) - 1
                logits_indices[offset_bs : offset_bs + dp_bs] = local_last

                # Copy extend_logprob_start_lens if available
                if (
                    hasattr(info, "extend_logprob_start_lens")
                    and info.extend_logprob_start_lens is not None
                ):
                    extend_logprob_start_lens[offset_bs : offset_bs + dp_bs] = (
                        info.extend_logprob_start_lens
                    )

            selector_chunks.append(np.arange(offset_bs, offset_bs + dp_bs, dtype=np.int32))
            offset_bs += per_dp_bs_size

        if selector_chunks:
            logits_indices_selector = np.concatenate(selector_chunks)
        else:
            logits_indices_selector = np.empty(0, dtype=np.int32)

        return (
            req_pool_indices_cpu,
            seq_lens_cpu,
            extend_prefix_lens,
            extend_seq_lens,
            extend_logprob_start_lens,
            logits_indices,
            real_bs,
            real_bs_per_dp,
            logits_indices_selector,
        )

    def _merge_cache_loc(
        self,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
        per_dp_bs_size: int,
    ) -> np.ndarray:
        """Merge cache_loc from all DP ranks with page alignment.

        Returns:
            cache_loc array
        """
        # Calculate total cache_loc size needed
        total_cache_loc_size = 0
        if self.forward_mode.is_extend():
            total_cache_loc_size = cache_loc_paddings[-1]  # Use largest padding
        else:
            # For decode mode, use the cache_loc_padding that corresponds to the bs bucket.
            total_bs = per_dp_bs_size * self.dp_size
            _, bs_index = pad_to_bucket(total_bs, bs_paddings)
            total_cache_loc_size = cache_loc_paddings[bs_index]

        per_dp_cache_loc_size = total_cache_loc_size // self.dp_size
        # View into the persistent buffer; intentionally NOT re-zeroed per step.
        # Safe because:
        #  - padding slots are never read on-device: attention kernels (RPA v3 /
        #    MLA v2 / native) bound page reads by cu_kv_lens / seq_lens, and every
        #    real-request page slot lands on a written position.
        #  - every buffer value is a valid in-bounds KV slot index (init is
        #    np.zeros + only valid slots are ever written), so even SWA's
        #    host-side mapping[cache_loc] lookup (flashattention_backend) can't go
        #    OOB. This REQUIRES the init buffer to be np.zeros, not np.empty.
        cache_loc_host_buf = self.req_to_token_pool.cache_loc_host_buf
        assert (
            cache_loc_host_buf is not None and cache_loc_host_buf.shape[0] >= total_cache_loc_size
        ), (
            "cache_loc_host_buf is not initialized or too small: "
            f"capacity={0 if cache_loc_host_buf is None else cache_loc_host_buf.shape[0]}, "
            f"required={total_cache_loc_size}"
        )
        cache_loc_cpu = cache_loc_host_buf[:total_cache_loc_size]

        offset_bs = 0
        req_to_token = self.req_to_token_pool.req_to_token

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.seq_lens is None or len(info.seq_lens) == 0:
                offset_bs += per_dp_cache_loc_size
                continue

            seq_lens = info.seq_lens
            req_pool_indices = info.req_pool_indices

            n_reqs = len(seq_lens)
            if n_reqs > 0:
                # Page-aligned offsets per request
                aligned_lens = ((seq_lens + page_size - 1) // page_size) * page_size
                offsets = np.empty(n_reqs, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(aligned_lens[:-1], out=offsets[1:])

                # Per-req contiguous slice copy from req_to_token directly.
                # Avoids:
                #  - the 8MB-per-DP intermediate `req_to_token[req_pool_indices]`
                #    full-row gather (only first seq_len of each row is used)
                #  - the 1M-element fancy-index scatter, which numpy serialises
                #    at Python level rather than as a contiguous memcpy.
                # Measured ~40x speedup at BSZ=64 OSL=16K decode vs the
                # vectorised fancy-index version (~12ms -> ~0.3ms).
                # Byte-for-byte identical output (verified with 14 edge cases
                # incl. BSZ in {1,8,32,64,512}, empty DPs, page boundaries).
                for r in range(n_reqs):
                    sl = int(seq_lens[r])
                    dest_start = int(offsets[r]) + offset_bs
                    cache_loc_cpu[dest_start : dest_start + sl] = req_to_token[
                        int(req_pool_indices[r]), :sl
                    ]

            # Move to next DP rank's section (fixed stride)
            offset_bs += per_dp_cache_loc_size

        return cache_loc_cpu

    def _merge_sampling_info(
        self,
        per_dp_bs_size: int,
        total_bs: int,
    ) -> SamplingBatchInfo:
        """Merge sampling info from all DP ranks.

        Returns:
            Merged SamplingBatchInfo
        """
        # Collect all requests for grammar support
        all_reqs = []
        for info in self.reqs_info:
            if info.reqs:
                all_reqs.extend(info.reqs)

        # Initialize merged arrays (with padding)
        temperatures = np.ones((total_bs, 1), dtype=np.float32)
        top_ps = np.ones(total_bs, dtype=np.float32)
        top_ks = np.ones(total_bs, dtype=np.int32)
        min_ps = np.zeros(total_bs, dtype=np.float32)
        sampling_seeds = None
        linear_penalty = None  # lazily allocated only if any DP rank has penalties

        offset_bs = 0
        has_sampling_seeds = False
        vocab_size = 0
        is_all_greedy = True

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.sampling_info is None or info.seq_lens is None or len(info.seq_lens) == 0:
                offset_bs += per_dp_bs_size
                continue

            dp_bs = len(info.seq_lens)
            dp_sampling = info.sampling_info
            if vocab_size == 0:
                vocab_size = dp_sampling.vocab_size

            if not info.sampling_info.is_all_greedy:
                is_all_greedy = False

            # Copy sampling parameters
            temperatures[offset_bs : offset_bs + dp_bs] = dp_sampling.temperatures[:dp_bs]
            top_ps[offset_bs : offset_bs + dp_bs] = dp_sampling.top_ps[:dp_bs]
            top_ks[offset_bs : offset_bs + dp_bs] = dp_sampling.top_ks[:dp_bs]
            min_ps[offset_bs : offset_bs + dp_bs] = dp_sampling.min_ps[:dp_bs]

            if dp_sampling.sampling_seeds is not None:
                if sampling_seeds is None:
                    sampling_seeds = np.full(total_bs, DEFAULT_SAMPLING_SEED, dtype=np.int64)
                    has_sampling_seeds = True
                sampling_seeds[offset_bs : offset_bs + dp_bs] = dp_sampling.sampling_seeds[:dp_bs]

            # Compute per-DP penalties (no-op if not required) and stitch into the
            # merged buffer using the same per-DP slot offset as the other arrays.
            dp_sampling.update_penalties()
            if dp_sampling.linear_penalty is not None and dp_sampling.linear_penalty.size > 0:
                if linear_penalty is None:
                    linear_penalty = np.zeros(
                        (total_bs, dp_sampling.linear_penalty.shape[1]),
                        dtype=dp_sampling.linear_penalty.dtype,
                    )
                linear_penalty[offset_bs : offset_bs + dp_bs] = dp_sampling.linear_penalty[:dp_bs]

            # Move to next DP rank's slot (fixed slot size)
            offset_bs += per_dp_bs_size

        return ModelWorkerSamplingInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=is_all_greedy,
            sampling_seeds=sampling_seeds if has_sampling_seeds else None,
            linear_penalty=linear_penalty,
            grammars=[req.grammar for req in all_reqs] if self.has_grammar else None,
        )

    def _get_spec_decode_mwb_dp(
        self, bs_paddings: list, enable_static_lora: bool, draft_token_num: int = 1
    ) -> ModelWorkerBatch:
        """DP-aware spec-decode ModelWorkerBatch (#1053 P1-5b).

        Reuses the nospec ``_merge_*`` helpers for per-rank seq_lens /
        req_pool_indices / sampling_info. ``spec_info`` is global (DP-padded
        order, see ``EagleDraftInput`` docstring) and lives only on
        ``reqs_info[0]``. ``input_ids``/``positions``/``cache_loc`` are
        placeholders — ``EagleDraftWorker.padding_for_decode`` rebuilds them.
        """
        # Pin total_bs to the largest precompile bucket so every cell shares
        # one jit cache entry regardless of runtime bs. Without this, each
        # smaller bucket (bs_paddings[i] < bs_paddings[-1]) triggers a fresh
        # trace the first time it's hit. precompile is expected to include a
        # largest bucket that is a multiple of dp_size; falling back to a
        # smaller bucket would split the cache key, so assert instead.
        if not bs_paddings:
            total_bs = self.dp_size
        else:
            max_bs_per_dp = 0
            for dp_rank in range(self.dp_size):
                info = self.reqs_info[dp_rank]
                if info.reqs is not None:
                    max_bs_per_dp = max(max_bs_per_dp, len(info.reqs))
            max_bs_per_dp = max(max_bs_per_dp, 1)
            total_bs, _ = pad_to_bucket(max_bs_per_dp * self.dp_size, bs_paddings)
            assert total_bs % self.dp_size == 0, (
                f"padded total_bs={total_bs} is not divisible by dp_size="
                f"{self.dp_size}; bs_paddings={bs_paddings}"
            )
        per_dp_bs = total_bs // self.dp_size
        self.per_dp_bs_size = per_dp_bs
        (
            req_pool_indices_cpu,
            seq_lens_cpu,
            _ext_prefix,
            _ext_seq,
            _ext_logprob,
            _logits_idx,
            real_bs,
            real_bs_per_dp,
            logits_indices_selector,
        ) = self._merge_batch_metadata(per_dp_bs, total_bs)
        sampling_info = self._merge_sampling_info(per_dp_bs, total_bs)
        # Concat per-rank spec_info into a cross-rank-flat EagleDraftInput,
        # then scatter into DP-padded (total_bs, ...) slots so spec_info[i]
        # aligns with seq_lens[i]. Returns a new object — does not mutate
        # the per-rank cross-round state on reqs_info[r].spec_info.
        flat_spec = self._concat_spec_info_per_rank([info.spec_info for info in self.reqs_info])
        spec_info = self._scatter_spec_info_to_dp_slots(
            flat_spec, logits_indices_selector, total_bs
        )
        # Per-rank out_cache_loc chunks (set in spec prepare_for_decode) have
        # variable length (∝ accept_len). DP-segment: pad each to max_len with
        # -1 so the P("data") shard in ForwardBatch.init_new gives rank r its
        # own slots (fa_backend doesn't use it, but native_backend would).
        ocl_chunks = [
            (
                np.asarray(i.out_cache_loc, dtype=np.int32)
                if i.out_cache_loc is not None and len(i.out_cache_loc) > 0
                else np.empty(0, dtype=np.int32)
            )
            for i in self.reqs_info
        ]
        # Pad each rank's out_cache_loc to per_dp_bs * draft_token_num so the
        # merged shape is stable across runtime bs. max_chunk_len defensive.
        max_chunk_len = max((len(c) for c in ocl_chunks), default=0)
        target_per_rank_ocl = max(per_dp_bs * draft_token_num, max_chunk_len)
        out_cache_loc = (
            np.concatenate(
                [
                    np.pad(c, (0, target_per_rank_ocl - len(c)), constant_values=-1)
                    for c in ocl_chunks
                ]
            )
            if target_per_rank_ocl > 0
            else np.empty(0, dtype=np.int32)
        )
        return ModelWorkerBatch(
            bid=acc_global_bid(),
            forward_mode=self.forward_mode,
            input_ids=np.empty(0, dtype=np.int32),
            real_input_ids_len=0,
            req_pool_indices=req_pool_indices_cpu,
            seq_lens=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            return_logprob=self.return_logprob,
            return_output_logprob_only=self.return_output_logprob_only,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            sampling_info=sampling_info,
            positions=np.empty(0, dtype=np.int32),
            cache_loc=np.empty(0, dtype=np.int32),
            extend_prefix_lens=None,
            extend_seq_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            logits_indices=None,
            lora_ids=(
                ["0"] * total_bs
                if enable_static_lora
                else [r.lora_id for i in self.reqs_info for r in (i.reqs or [])]
                + ["0"] * (total_bs - real_bs)
            ),
            real_bs=real_bs,
            real_bs_per_dp=real_bs_per_dp,
            dp_size=self.dp_size,
            per_dp_bs_size=per_dp_bs,
            logits_indices_selector=logits_indices_selector,
            capture_hidden_mode=getattr(spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL),
            launch_done=self.launch_done,
            spec_info_padded=spec_info,
            spec_algorithm=self.spec_algorithm,
            tree_cache=self.tree_cache,
            mrope_positions=None,
        )

    @staticmethod
    def _scatter_spec_info_to_dp_slots(flat, selector: np.ndarray, total_bs: int):
        """Scatter global-flat spec_info arrays into DP-padded ``(total_bs, …)``.

        ``selector[k]`` is the DP-padded slot of the k-th global-flat req
        (== ``logits_indices_selector``). Returns a new ``EagleDraftInput``;
        the cross-round flat state on ``reqs_info[r].spec_info`` is unchanged.
        """

        def _scatter1(arr):
            if arr is None:
                return None
            a = np.asarray(arr)
            out = np.zeros((total_bs,) + a.shape[1:], dtype=a.dtype)
            out[selector] = a
            return out

        return type(flat)(
            topk_p=_scatter1(flat.topk_p),
            topk_index=_scatter1(flat.topk_index),
            hidden_states=_scatter1(flat.hidden_states),
            verified_id=_scatter1(flat.verified_id),
            allocate_lens=_scatter1(flat.allocate_lens),
            capture_hidden_mode=flat.capture_hidden_mode,
            accept_length=flat.accept_length,
            accept_length_cpu=flat.accept_length_cpu,
        )

    @staticmethod
    def _split_spec_info_per_rank(flat, real_bs_per_dp: list[int]) -> list:
        """Slice a cross-rank-flat EagleDraftInput into per-rank EagleDraftInputs.

        ``flat`` layout is ``[rank0 reqs ++ rank1 reqs ++ …]`` with total length
        ``sum(real_bs_per_dp)``. Empty ranks (``real_bs == 0``) yield ``None``.
        Used at forward output boundary to write back ``reqs_info[r].spec_info``.
        """
        if flat is None:
            return [None] * len(real_bs_per_dp)

        flat._ensure_host()

        per_req_fields = (
            "topk_p",
            "topk_index",
            "hidden_states",
            "verified_id",
            "allocate_lens",
            "accept_length",
            "accept_length_cpu",
        )

        out = []
        offset = 0
        for n in real_bs_per_dp:
            if n == 0:
                out.append(None)
                continue
            kwargs = {"capture_hidden_mode": flat.capture_hidden_mode}
            for f in per_req_fields:
                v = getattr(flat, f, None)
                kwargs[f] = None if v is None else v[offset : offset + n]
            out.append(type(flat)(**kwargs))
            offset += n
        return out

    @staticmethod
    def _concat_spec_info_per_rank(per_rank: list):
        """Concat per-rank EagleDraftInputs into a single cross-rank-flat one.

        ``None`` entries are skipped. Returns ``None`` if every entry is ``None``.
        Used at forward input boundary (``_get_spec_decode_mwb_dp``) to build the
        flat shape ``_scatter_spec_info_to_dp_slots`` expects.
        """
        nonempty = [s for s in per_rank if s is not None]
        if not nonempty:
            return None

        per_req_fields = (
            "topk_p",
            "topk_index",
            "hidden_states",
            "verified_id",
            "allocate_lens",
            "accept_length",
            "accept_length_cpu",
        )

        kwargs = {"capture_hidden_mode": nonempty[0].capture_hidden_mode}
        for f in per_req_fields:
            vals = [getattr(s, f, None) for s in nonempty]
            nonnull = [v for v in vals if v is not None]
            if not nonnull:
                kwargs[f] = None
                continue
            # All nonempty ranks should agree on which optional fields they
            # carry — they came from the same per-rank verify split. A partial
            # mix means the concat length would silently drift from
            # ``sum(real_bs_per_dp)``; fail loudly instead.
            assert len(nonnull) == len(nonempty), (
                f"_concat_spec_info_per_rank: field {f!r} is None on "
                f"{len(nonempty) - len(nonnull)}/{len(nonempty)} nonempty rank(s); "
                "all-or-nothing required"
            )
            if len(nonnull) == 1:
                kwargs[f] = nonnull[0]
                continue
            if isinstance(nonnull[0], np.ndarray):
                kwargs[f] = np.concatenate(nonnull, axis=0)
            else:
                nonnull = [np.asarray(v) for v in nonnull]
                kwargs[f] = np.concatenate(nonnull, axis=0)
        return type(nonempty[0])(**kwargs)

    def get_model_worker_batch(
        self,
        token_paddings: list,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
        enable_static_lora: bool = False,
    ) -> ModelWorkerBatch:
        if self.forward_mode.is_decode_or_idle():
            token_paddings = bs_paddings
        else:
            bs_paddings = bs_paddings[-1:]
            cache_loc_paddings = cache_loc_paddings[-1:]

        bid = acc_global_bid()

        # Step 1: Compute global padding sizes across all DP ranks
        per_dp_token_padding, total_token_size, per_dp_bs_padding, total_bs = (
            self._compute_global_padding_sizes(token_paddings, bs_paddings)
        )

        # Save per_dp_bs_size for later use (e.g., in process_batch_result_decode)
        self.per_dp_bs_size = per_dp_bs_padding

        # Step 2: Merge input_ids, positions, and out_cache_loc from all DP ranks
        input_ids_cpu, positions_cpu, out_cache_loc_cpu, real_input_ids_len = (
            self._merge_input_and_positions(per_dp_token_padding, total_token_size)
        )

        # Step 3: Merge batch-level metadata from all DP ranks
        (
            req_pool_indices_cpu,
            seq_lens_cpu,
            extend_prefix_lens,
            extend_seq_lens,
            extend_logprob_start_lens,
            logits_indices,
            real_bs,
            real_bs_per_dp,
            logits_indices_selector,
        ) = self._merge_batch_metadata(per_dp_bs_padding, total_bs)

        # Step 4: Merge cache_loc from all DP ranks
        cache_loc_cpu = self._merge_cache_loc(
            bs_paddings, cache_loc_paddings, page_size, per_dp_bs_padding
        )

        # Step 5: Merge sampling info from all DP ranks
        sampling_info = self._merge_sampling_info(per_dp_bs_padding, total_bs)

        # Step 5.5: Merge recurrent_indices from all DP ranks
        recurrent_indices_cpu = None
        if any(info.recurrent_indices is not None for info in self.reqs_info):
            recurrent_indices_cpu = np.zeros(total_bs, dtype=np.int32)
            offset_bs = 0
            for dp_rank in range(self.dp_size):
                info = self.reqs_info[dp_rank]
                if info.seq_lens is not None and len(info.seq_lens) > 0:
                    dp_bs = len(info.seq_lens)
                    if info.recurrent_indices is not None:
                        recurrent_indices_cpu[offset_bs : offset_bs + dp_bs] = (
                            info.recurrent_indices
                        )
                offset_bs += per_dp_bs_padding

        # Step 5.6: has_initial_state[i] = True iff slot i already holds
        # prior KV/recurrent state (extend with prefix, or any decode slot).
        has_initial_state_cpu = np.ones(total_bs, dtype=np.bool_)
        if self.forward_mode.is_extend():
            offset_bs = 0
            for dp_rank in range(self.dp_size):
                dp_bs = real_bs_per_dp[dp_rank]
                if dp_bs > 0:
                    has_initial_state_cpu[offset_bs : offset_bs + dp_bs] = (
                        extend_prefix_lens[offset_bs : offset_bs + dp_bs] > 0
                    )
                offset_bs += per_dp_bs_padding

        # Step 6: Generate trace info if needed
        if precision_tracer.get_trace_active():
            self._generate_trace_info(real_bs, bid)

        # Step 7: Collect lora_ids from all requests
        all_reqs = []
        for info in self.reqs_info:
            if info.reqs:
                all_reqs.extend(info.reqs)

        if enable_static_lora:
            lora_ids = ["0"] * total_bs
        else:
            lora_ids = [req.lora_id for req in all_reqs[:real_bs]]
            # Pad to total_bs
            lora_ids = lora_ids + ["0"] * (total_bs - real_bs)

        # Assemble all per-token multimodal tensors (input_embedding,
        # mrope_positions, deepstack) in a single DP-interleaved pass over
        # reqs_info[*].reqs; see ScheduleBatch._merge_multimodal. Each is
        # None/False for pure-text batches, so non-multimodal paths stay
        # unchanged.
        _mm = self._merge_multimodal(per_dp_token_padding, total_token_size)
        input_embedding = _mm["input_embedding"]
        mrope_positions = _mm["mrope_positions"]
        apply_for_deepstack = _mm["apply_for_deepstack"]
        deepstack_visual_embedding = _mm["deepstack_visual_embedding"]

        # Merge per-DP top_logprobs_nums / token_ids_logprobs with the same
        # offset_bs += per_dp_bs_padding padding scheme used in _merge_batch_metadata.
        if self.return_logprob:
            top_logprobs_nums = [0] * total_bs
            token_ids_logprobs: list[list[int] | None] = [None] * total_bs
            offset_bs = 0
            for dp_rank in range(self.dp_size):
                info = self.reqs_info[dp_rank]
                if info.seq_lens is not None and len(info.seq_lens) > 0:
                    dp_bs = len(info.seq_lens)
                    if info.top_logprobs_nums is not None:
                        top_logprobs_nums[offset_bs : offset_bs + dp_bs] = info.top_logprobs_nums
                    if info.token_ids_logprobs is not None:
                        token_ids_logprobs[offset_bs : offset_bs + dp_bs] = info.token_ids_logprobs
                offset_bs += per_dp_bs_padding
        else:
            top_logprobs_nums = None
            token_ids_logprobs = None

        # extend+logprob always uses the padded path: the legacy fallback slices
        # hidden_states per req under P("data","tensor") and crashes when the row
        # count isn't divisible by dp (dp>1). The padded path supports top_logprobs /
        # token_ids / overlap at any dp. return_hidden_states still falls back.
        use_padded_input_logprob = self.forward_mode.is_extend() and not self.return_hidden_states
        input_logprob_indices = None
        merged_extend_input_logprob_token_ids = None
        if self.return_logprob:
            if use_padded_input_logprob:
                input_logprob_indices = np.zeros(total_token_size, dtype=np.int32)
                merged_extend_input_logprob_token_ids = np.zeros(total_token_size, dtype=np.int32)
                token_offset = 0
                for info in self.reqs_info:
                    out_pt = token_offset
                    local_pt = 0
                    token_id_pt = 0
                    starts = info.extend_logprob_start_lens or []
                    token_ids = info.extend_input_logprob_token_ids
                    for extend_len, start_len in zip(info.extend_lens or [], starts):
                        num_logprobs = max(extend_len - start_len, 0)
                        if num_logprobs > 0:
                            end_pt = out_pt + num_logprobs
                            input_logprob_indices[out_pt:end_pt] = np.arange(
                                local_pt + start_len,
                                local_pt + extend_len,
                                dtype=np.int32,
                            )
                            if token_ids is not None:
                                merged_extend_input_logprob_token_ids[out_pt:end_pt] = token_ids[
                                    token_id_pt : token_id_pt + num_logprobs
                                ]
                            out_pt = end_pt
                            token_id_pt += num_logprobs
                        local_pt += extend_len
                    token_offset += per_dp_token_padding
            else:
                # Only extend batches have prompt-token logprobs. Overlap also routes
                # speculated DECODE batches here carrying a stale extend residual;
                # sharding it over P("data") crashes when len % dp != 0 (dp>=4). Decode
                # never reads the field, so gate the merge on is_extend().
                chunks = [
                    info.extend_input_logprob_token_ids
                    for info in self.reqs_info
                    if getattr(info, "extend_input_logprob_token_ids", None) is not None
                ]
                if chunks and self.forward_mode.is_extend():
                    merged_extend_input_logprob_token_ids = np.concatenate(chunks)

        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=input_ids_cpu,
            real_input_ids_len=real_input_ids_len,
            req_pool_indices=req_pool_indices_cpu,
            seq_lens=seq_lens_cpu,
            out_cache_loc=out_cache_loc_cpu,
            return_logprob=self.return_logprob,
            return_output_logprob_only=self.return_output_logprob_only,
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
            sampling_info=sampling_info,
            positions=positions_cpu,
            mrope_positions=mrope_positions,
            cache_loc=cache_loc_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=merged_extend_input_logprob_token_ids,
            input_logprob_indices=input_logprob_indices,
            logits_indices=logits_indices,
            lora_ids=lora_ids,
            real_bs=real_bs,
            real_bs_per_dp=real_bs_per_dp,
            logits_indices_selector=logits_indices_selector,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                or (self.spec_algorithm is not None and not self.spec_algorithm.is_none())
                else CaptureHiddenMode.NULL
            ),
            dp_size=self.dp_size,
            per_dp_bs_size=per_dp_bs_padding,
            launch_done=self.launch_done,
            input_embedding=input_embedding,
            apply_for_deepstack=apply_for_deepstack,
            deepstack_visual_embedding=deepstack_visual_embedding,
            recurrent_indices=recurrent_indices_cpu,
            has_initial_state=has_initial_state_cpu,
            spec_algorithm=self.spec_algorithm,
        )

    def get_spec_model_worker_batch(
        self,
        token_paddings: list,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
        enable_static_lora: bool = False,
        draft_token_num: int = 1,
    ) -> ModelWorkerBatch:
        assert (
            self.forward_mode.is_decode_or_idle()
        ), "spec extend must use get_model_worker_batch, only decode reaches here"
        return self._get_spec_decode_mwb_dp(bs_paddings, enable_static_lora, draft_token_num)

    def _generate_trace_info(self, real_bs: int, bid: int) -> list[str]:
        """Generate trace information for requests (unified for all dp_size >= 1)."""
        if not precision_tracer.get_trace_active():
            return

        # Collect all requests from all DP ranks
        all_reqs = []
        for info in self.reqs_info:
            if info.reqs:
                all_reqs.extend(info.reqs)

        # Process first real_bs requests (real_bs limits to actual requests before padding)
        for req in all_reqs[:real_bs]:
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
                PrecisionTracerRequestMetadata(req.rid, input_ids_to_trace, self.forward_mode),
            )
            if self.forward_mode == ForwardMode.EXTEND:
                precision_tracer.add_request_counter()
                logger.info(
                    "Starting trace for request %d: %s",
                    precision_tracer.get_request_counter(),
                    req.rid,
                )

    def copy(self):
        """Create a shallow copy of this batch.

        Only contain fields that will be used by process_batch_result.
        """
        # Copy reqs_info for each DP rank
        copied_reqs_info = []
        for info in self.reqs_info:
            # Create a new ScheduleReqsInfo with shallow copies of necessary fields
            new_info = ScheduleReqsInfo()
            new_info.reqs = info.reqs  # Shallow copy (list reference)
            new_info.out_cache_loc = info.out_cache_loc
            new_info.decoding_reqs = info.decoding_reqs
            # process_batch_result compacts per-DP padded input logprobs via
            # _input_logprob_lens_per_dp, which reads these.
            new_info.extend_lens = info.extend_lens
            new_info.extend_logprob_start_lens = info.extend_logprob_start_lens
            copied_reqs_info.append(new_info)

        return ScheduleBatch(
            reqs_info=copied_reqs_info,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            return_logprob=self.return_logprob,
            return_output_logprob_only=self.return_output_logprob_only,
            is_prefill_only=self.is_prefill_only,
            bid=self.bid,
            dp_size=self.dp_size,
            per_dp_bs_size=self.per_dp_bs_size,
        )

    def _evict_tree_cache_if_needed(self, num_tokens_per_dp: dict[int, int]) -> None:
        """Evict from tree cache if needed for any DP rank.

        Per-DP aware implementation. Tree cache is global, eviction affects all DP ranks.

        Args:
            num_tokens_per_dp: Dict mapping dp_rank to tokens needed for that rank
        """
        if isinstance(self.tree_cache, ChunkCache):
            return

        # Per-DP loop
        for dp_rank, num_tokens in num_tokens_per_dp.items():
            if self.is_hybrid:
                full_available = self.token_to_kv_pool_allocator.full_available_size(
                    dp_rank=dp_rank
                )
                swa_available = self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)

                if (full_available < num_tokens or swa_available < num_tokens) and self.tree_cache:
                    full_num = max(0, num_tokens - full_available)
                    swa_num = max(0, num_tokens - swa_available)
                    self.tree_cache.evict(full_num, swa_num, dp_rank=dp_rank)
            else:
                available = self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                if available < num_tokens and self.tree_cache:
                    self.tree_cache.evict(num_tokens, dp_rank=dp_rank)

    def _is_available_size_sufficient(self, num_tokens_per_dp: dict[int, int]) -> bool:
        """Check if sufficient memory available across all DP ranks.

        Per-DP aware implementation. Returns False if ANY DP rank has insufficient memory.

        Args:
            num_tokens_per_dp: Dict mapping dp_rank to tokens needed for that rank

        Returns:
            True if all DP ranks have sufficient memory, False otherwise.
        """
        for dp_rank, num_tokens in num_tokens_per_dp.items():
            if self.is_hybrid:
                full_ok = (
                    self.token_to_kv_pool_allocator.full_available_size(dp_rank=dp_rank)
                    >= num_tokens
                )
                swa_ok = (
                    self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
                    >= num_tokens
                )
                if not (full_ok and swa_ok):
                    return False
            else:
                if self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank) < num_tokens:
                    return False

        return True

    def _available_and_evictable_str(self) -> str:
        """Get debug string for available and evictable memory (unified for all dp_size >= 1)."""
        # Collect all DP ranks that have requests
        dp_ranks_with_reqs = set()
        for dp_rank, info in enumerate(self.reqs_info):
            if info.reqs and len(info.reqs) > 0:
                dp_ranks_with_reqs.add(dp_rank)

        # If no requests at all, default to showing dp_rank 0
        if not dp_ranks_with_reqs:
            dp_ranks_with_reqs.add(0)

        result_strs = []
        for dp_rank in sorted(dp_ranks_with_reqs):
            # Prefix for multi-DP output
            prefix = f"[DP rank {dp_rank}] " if self.dp_size > 1 else ""

            if self.is_hybrid:
                full_available_size = self.token_to_kv_pool_allocator.full_available_size(
                    dp_rank=dp_rank
                )
                swa_available_size = self.token_to_kv_pool_allocator.swa_available_size(
                    dp_rank=dp_rank
                )
                full_evictable_size = self.tree_cache.full_evictable_size(dp_rank=dp_rank)
                swa_evictable_size = self.tree_cache.swa_evictable_size(dp_rank=dp_rank)
                result_strs.append(
                    f"{prefix}Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
                    f"{prefix}Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
                    f"{prefix}Full LRU list evictable size: {self.tree_cache.full_lru_list_evictable_size()}\n"
                    f"{prefix}SWA LRU list evictable size: {self.tree_cache.swa_lru_list_evictable_size()}\n"
                )
            else:
                available_size = self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                evictable_size = self.tree_cache.evictable_size(dp_rank=dp_rank)
                result_strs.append(
                    f"{prefix}Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
                )

        return "".join(result_strs)


def align_to_size(lst: list, size: int, value: int = 0) -> list:
    align_len = (len(lst) + size - 1) // size * size
    return lst[:] + [value] * (align_len - len(lst))


def _extract_mm_value(mm_inputs: Any, key: str):
    if mm_inputs is None:
        return None
    if isinstance(mm_inputs, dict):
        return mm_inputs.get(key)
    return getattr(mm_inputs, key, None)


def _as_int_scalar(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (np.ndarray, jax.Array)):
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    return int(value)


@dataclasses.dataclass
class ModelWorkerSamplingInfo:
    """Unified sampling information for a generation batch."""

    def __len__(self) -> int:
        return len(self.temperatures)

    def filter_batch(self, indices) -> None:
        self.temperatures = self.temperatures[indices]
        self.top_ps = self.top_ps[indices]
        self.top_ks = self.top_ks[indices]
        self.min_ps = self.min_ps[indices]

    # Basic batched sampling params
    temperatures: np.ndarray
    top_ps: np.ndarray
    top_ks: np.ndarray
    min_ps: np.ndarray

    vocab_size: int

    # Whether all requests use greedy sampling
    is_all_greedy: bool = False

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool = False

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool = False

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool = False

    # An event used for overlap schedule
    sampling_info_done: threading.Event | None = None

    sampling_seeds: np.ndarray | None = None

    linear_penalty: np.ndarray | None = None

    penalizer_orchestrator: Any | None = None

    grammars: list | None = None  # list[BaseGrammarObject | None]
    vocab_mask: np.ndarray | None = None  # Shape: [batch_size, vocab_size // 32]

    @classmethod
    def generate_for_precompile(
        cls,
        bs: int,
        vocab_size: int = 32000,
    ):
        temperatures = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([0.9 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([30 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING"):
            sampling_seeds = np.array([DEFAULT_SAMPLING_SEED for _ in range(bs)], dtype=np.int32)
        else:
            sampling_seeds = None

        num_int32_per_vocab = (vocab_size + 31) // 32
        vocab_mask = np.zeros((bs, num_int32_per_vocab), dtype=np.int32)

        ret = cls(
            temperatures=temperatures.reshape(-1, 1),
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=True,
            sampling_info_done=None,
            sampling_seeds=sampling_seeds,
            penalizer_orchestrator=None,
            linear_penalty=None,
            vocab_mask=vocab_mask,
        )
        return ret

    @classmethod
    def generate_for_precompile_all_greedy(
        cls, bs: int, vocab_size: int = 32000, do_penalties: bool = False
    ):
        temperatures = np.array([0.0 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([1 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING"):
            sampling_seeds = np.array([DEFAULT_SAMPLING_SEED for _ in range(bs)], dtype=np.int32)
        else:
            sampling_seeds = None

        num_int32_per_vocab = (vocab_size + 31) // 32
        vocab_mask = np.zeros((bs, num_int32_per_vocab), dtype=np.int32)

        ret = cls(
            temperatures=temperatures.reshape(-1, 1),
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=True,
            sampling_info_done=None,
            sampling_seeds=sampling_seeds,
            penalizer_orchestrator=None,
            linear_penalty=None,
            vocab_mask=vocab_mask,
        )
        return ret

    def update_penalties(self):
        # No-op: linear_penalty is pre-computed during ScheduleBatch._merge_sampling_info.
        # Kept for API parity with SamplingBatchInfo.update_penalties (called by overlap thread).
        return

    def update_grammar_vocab_mask(self):
        """Update vocabulary masks from grammars before sampling."""
        if not self.grammars:
            self.vocab_mask = None
            return

        first_grammar = next((g for g in self.grammars if g), None)
        if first_grammar is None:
            self.vocab_mask = None
            return

        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
        )

        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)


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
    sampling_info: ModelWorkerSamplingInfo
    # Position information [total_tokens]
    positions: np.ndarray
    # cache_loc
    cache_loc: np.ndarray

    # For logprob
    return_logprob: bool
    return_output_logprob_only: bool
    top_logprobs_nums: list[int] | None
    token_ids_logprobs: list[list[int]] | None

    # For extend
    # extend_num_tokens: Optional[int]
    extend_seq_lens: np.ndarray | None
    extend_prefix_lens: np.ndarray | None
    extend_logprob_start_lens: list[int] | None
    extend_input_logprob_token_ids: np.ndarray | None
    logits_indices: np.ndarray | None

    # For padding
    real_bs: int
    real_bs_per_dp: list[int]

    # Maps "original request order" (DP-rank-then-req flat order) to the
    # DP-interleaved padded slot in the global batch. Host code applies
    # `arr[selector]` once after device_get to put per-req outputs back
    # into original order, removing the need for per-rank index math.
    logits_indices_selector: np.ndarray | None = None

    # Pre-bucketed per-token gather indices for the padded logprob path; None on
    # the legacy variable-shape path and on non-extend batches.
    input_logprob_indices: np.ndarray | None = None

    # For Data Parallelism
    dp_size: int = 1
    per_dp_bs_size: int = 0  # Batch size per DP rank (with padding)

    # For LoRA
    lora_ids: list[str] | None = None
    lora_scalings: np.ndarray | None = None
    lora_token_indices: np.ndarray | None = None
    lora_ranks: np.ndarray | None = None

    capture_hidden_mode: CaptureHiddenMode = None

    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: np.ndarray = None
    top_p_normalized_logprobs: bool = False
    top_p: np.ndarray = None

    # Events
    launch_done: threading.Event | None = None

    # Pre-initialized ForwardBatch for overlap scheduling optimization
    forward_batch: Any | None = None

    # Cross-rank-flat (or scatter-padded under dp>1) spec input at forward
    # entry. Forward internals may mutate it through EagleVerifyInput before
    # returning a fresh EagleDraftInput. Scheduler-persisted per-rank spec
    # state lives on ScheduleBatch.reqs_info[r].spec_info.
    spec_info_padded: EagleDraftInput | EagleVerifyInput | None = None
    spec_algorithm: SpeculativeAlgorithm = None
    speculative_num_steps: int = 0
    speculative_eagle_topk: int = 0
    speculative_num_draft_tokens: int = 0
    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None

    tree_cache: BasePrefixCache = None

    input_embedding: np.ndarray | None = None
    apply_for_deepstack: bool = False
    deepstack_visual_embedding: np.ndarray | None = None

    # MRoPE position information [3, total_tokens]
    mrope_positions: np.ndarray | None = None

    # Recurrent state indices for hybrid recurrent models
    recurrent_indices: np.ndarray | None = None

    # Whether each request has prior recurrent state (lazy zero-on-read)
    has_initial_state: np.ndarray | None = None

    def get_original_input_len(self):
        """
        return unpadded tokens number for prefill and real batch size for decode
        """
        if self.forward_mode.is_decode():
            return self.real_bs
        elif self.forward_mode.is_extend():
            return self.real_input_ids_len
        else:
            raise ValueError(f"{self.forward_mode} is not support to get original token or bs num")


def get_last_loc(
    req_to_token: np.ndarray,
    req_pool_indices: np.ndarray,
    prefix_lens: np.ndarray,
) -> np.ndarray:
    return np.where(
        prefix_lens > 0,
        req_to_token[req_pool_indices, prefix_lens - 1],
        np.full_like(prefix_lens, -1),
    )
