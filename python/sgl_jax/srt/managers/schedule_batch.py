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
import logging
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import numpy as np
from jax import numpy as jnp
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
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
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
from sgl_jax.srt.utils.common_utils import get_bool_env_var

if TYPE_CHECKING:
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


GLOBAL_SERVER_ARGS_KEYS = [
    "device",
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
        self.return_hidden_states = return_hidden_states

        # Extra key for cache namespace isolation (e.g., cache_salt, lora_id)
        if lora_id is not None:
            extra_key = (extra_key or "") + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id if lora_id is not None else "0"
        self.dp_rank = dp_rank

        # Memory pool info
        self.req_pool_idx: int | None = None

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
        # The prefix length of the last prefix matching
        self.last_matched_prefix_len: int = 0

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
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
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
            self.surr_offset = max(self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0)

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

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
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
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
        self.req_pool_idx = None
        self.already_computed = 0

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


@dataclasses.dataclass
class ScheduleBatch:
    """Store all information of a batch on the scheduler.

    For DP > 1, per-DP request information is stored in reqs_info list.
    Global/shared state is stored directly in this class.
    """

    # Per-DP request information (list of length dp_size)
    reqs_info: list[ScheduleReqsInfo] = None

    # Memory pool and cache (shared across all DP ranks)
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None
    is_hybrid: bool = False

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
            model_config=model_config,
            return_logprob=return_logprob,
            return_output_logprob_only=return_output_logprob_only,
            enable_overlap=enable_overlap,
            has_stream=any(req.stream for req in all_reqs),
            has_grammar=any(req.grammar for req in all_reqs),
            mesh=mesh,
            spec_algorithm=spec_algorithm,
            is_prefill_only=all(req.sampling_params.max_new_tokens == 0 for req in all_reqs),
            dp_size=dp_size,
        )

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
        running_bs = running_batch.batch_size()

        # Collect all requests from all DP ranks
        all_running_reqs = []
        for info in running_batch.reqs_info:
            if info.reqs:
                all_running_reqs.extend(info.reqs)

        for i, req in enumerate(all_running_reqs):
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = jnp.concatenate([self.input_ids, running_batch.input_ids])
        out_cache_loc = jnp.concatenate([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        delta = 0 if self.enable_overlap else -1
        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [len(r.origin_input_ids) + len(r.output_ids) + delta for r in all_running_reqs]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        self.extend_logprob_start_lens.extend([0] * running_bs)

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
            bs = len(reqs)
            req_pool_indices = self.alloc_req_slots(bs)

            # Init arrays
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

            # Write to req_to_token_pool
            pt = 0
            for i in range(bs):
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

    def new_page_count_next_decode(
        self,
        dp_rank: int,
        selected_indices: list[int] | None = None,
    ) -> int:
        """Calculate new page count for next decode for a specific DP rank.

        Args:
            dp_rank: DP rank to calculate for
            selected_indices: Optional local indices within this DP rank

        Returns:
            Number of new pages needed for this DP rank.
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        info = self.reqs_info[dp_rank]

        if not info.reqs:
            return 0

        requests = (
            info.reqs if selected_indices is None else [info.reqs[i] for i in selected_indices]
        )

        if page_size == 1:
            return len(requests)

        return (
            sum(1 for req in requests if req.seqlen % page_size == 0)
            if self.enable_overlap
            else sum(1 for req in requests if (req.seqlen - 1) % page_size == 0)
        )

    def check_decode_mem(
        self, buf_multiplier=1, selected_indices: dict[int, list[int]] | None = None
    ):
        """Check if all DP ranks have sufficient memory for next decode step.

        Args:
            buf_multiplier: Buffer multiplier for memory calculation
            selected_indices: Optional per-DP indices to check
                              Format: {dp_rank: [local_index_0, local_index_1, ...]}
                              If None, checks all requests in all DP ranks

        Returns:
            False if any DP rank has insufficient memory.
        """
        # Calculate tokens needed per DP rank
        num_tokens_per_dp = {}
        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]
            if not info.reqs:
                continue

            # Get indices for this specific DP rank
            indices = selected_indices.get(dp_rank) if selected_indices else None

            # Calculate tokens needed for THIS specific DP rank
            num_pages = self.new_page_count_next_decode(dp_rank, indices)
            num_tokens_per_dp[dp_rank] = (
                num_pages * buf_multiplier * self.token_to_kv_pool_allocator.page_size
            )

        # Try to evict if needed, then check if sufficient
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
        """

        # Helper function: check if memory is sufficient for given DP rank
        def has_sufficient_memory(dp_rank: int, indices: list[int]) -> bool:
            num_pages = self.new_page_count_next_decode(dp_rank, indices)
            num_tokens = num_pages * self.token_to_kv_pool_allocator.page_size

            # Evict if needed
            if not isinstance(self.tree_cache, ChunkCache) and self.tree_cache:
                if self.is_hybrid:
                    full_avail = self.token_to_kv_pool_allocator.full_available_size(
                        dp_rank=dp_rank
                    )
                    swa_avail = self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
                    if full_avail < num_tokens or swa_avail < num_tokens:
                        self.tree_cache.evict(
                            max(0, num_tokens - full_avail),
                            max(0, num_tokens - swa_avail),
                            dp_rank=dp_rank,
                        )
                else:
                    avail = self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                    if avail < num_tokens:
                        self.tree_cache.evict(num_tokens, dp_rank=dp_rank)

            # Check if sufficient
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
            while not has_sufficient_memory(dp_rank, sorted_indices) or first_iter:
                if len(sorted_indices) == 1:
                    # Assert some space available
                    if self.is_hybrid:
                        full_avail = self.token_to_kv_pool_allocator.full_available_size(
                            dp_rank=dp_rank
                        )
                        swa_avail = self.token_to_kv_pool_allocator.swa_available_size(
                            dp_rank=dp_rank
                        )
                        assert (
                            full_avail > 0 and swa_avail > 0
                        ), f"[DP {dp_rank}] No space for single request (hybrid)"
                    else:
                        avail = self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                        assert avail > 0, f"[DP {dp_rank}] No space for single request"
                    break

                first_iter = False
                retract_idx = sorted_indices.pop()
                req = info.reqs[retract_idx]
                retracted_reqs.append(req)

                # Release the request using its local index within this DP rank
                self.release_req(retract_idx, dp_rank, len(sorted_indices), server_args)

            keep_indices_per_dp[dp_rank] = sorted_indices

        # Apply filtering
        self.filter_batch(keep_indices=keep_indices_per_dp)

        # Calculate global estimate ratio by aggregating across all DP ranks
        all_reqs = [req for info in self.reqs_info for req in (info.reqs if info.reqs else [])]
        total_decoded_tokens = sum(len(r.output_ids) for r in all_reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in all_reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(all_reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def release_req(self, idx: int, dp_rank: int, remaing_req_count: int, server_args: ServerArgs):
        """Release a request and free its resources.

        Args:
            idx: Index of the request within the DP rank's request list
            dp_rank: The DP rank this request belongs to
            remaing_req_count: Number of remaining requests
            server_args: Server arguments
        """
        info = self.reqs_info[dp_rank]
        req = info.reqs[idx]
        seq_lens_cpu = info.seq_lens

        if isinstance(self.tree_cache, ChunkCache):
            # ChunkCache does not have eviction
            token_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : seq_lens_cpu[idx]
            ]
            self.token_to_kv_pool_allocator.free(token_indices, req.dp_rank)
            self.req_to_token_pool.free(req.req_pool_idx)
        else:
            last_uncached_pos = (
                len(req.prefix_indices) // server_args.page_size
            ) * server_args.page_size
            token_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
            ]
            self.token_to_kv_pool_allocator.free(token_indices, req.dp_rank)
            self.req_to_token_pool.free(req.req_pool_idx)

            # release the last node
            if self.is_hybrid:
                self.tree_cache.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
            else:
                self.tree_cache.dec_lock_ref(req.last_node)

            num_tokens = remaing_req_count * global_config.retract_decode_steps
            self._evict_tree_cache_if_needed({req.dp_rank: num_tokens})

        req.reset_for_retract()

    def retract_all(self, server_args: ServerArgs):
        retracted_reqs = self.reqs
        for idx in range(len(self.reqs)):
            self.release_req(idx, len(self.reqs) - idx, server_args)

        self.filter_batch(retracted_reqs)
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

    def prepare_for_decode(self):
        """Prepare for decode phase (unified for all dp_size >= 1).

        Process each DP rank independently. For dp_size=1, this is equivalent to
        the old single-rank logic but with cleaner structure.
        """
        self.forward_mode = ForwardMode.DECODE

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

            # Handle spec decoding if enabled
            if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
                # if spec decoding is used, the decode batch is prepared inside
                # `forward_batch_speculative_generation` after running draft models.
                draft_input: EagleDraftInput = info.spec_info
                draft_input.prepare_for_decode(self)

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

        # Unified DP filtering logic (works for all dp_size including 1)
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

            # Filter speculative decoding arrays manually (if present)
            if info.spec_info is not None and info.spec_info.topk_p is not None:
                keep_indices_jax = jnp.asarray(keep_indices_dp, dtype=jnp.int32)
                info.spec_info.topk_p = info.spec_info.topk_p[keep_indices_jax]
                info.spec_info.topk_index = info.spec_info.topk_index[keep_indices_jax]
                info.spec_info.hidden_states = info.spec_info.hidden_states[keep_indices_jax]
                info.spec_info.verified_id = info.spec_info.verified_id[keep_indices_jax]
                info.spec_info.allocate_lens = info.spec_info.allocate_lens[keep_indices_jax]

            # Filter logprob lists
            if info.top_logprobs_nums is not None:
                info.top_logprobs_nums = [info.top_logprobs_nums[i] for i in keep_indices_dp]
                info.token_ids_logprobs = [info.token_ids_logprobs[i] for i in keep_indices_dp]

            # Filter sampling_info
            if info.sampling_info is not None:
                info.sampling_info.filter_batch(np.array(keep_indices_dp))

            # Filter spec_info (method call)
            if info.spec_info is not None:
                # Note: has_been_filtered logic matches original implementation
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

        token_padding, _ = find_padding_size(max_tokens_per_dp * self.dp_size, token_paddings)
        bs_padding, _ = find_padding_size(max_bs_per_dp * self.dp_size, bs_paddings)

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
                dp_positions = []
                for seq_len, prefix_len in zip(info.seq_lens, info.prefix_lens):
                    dp_positions.extend(range(prefix_len, seq_len))
                positions_cpu[offset : offset + len(dp_positions)] = dp_positions
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

    def _merge_batch_metadata(
        self,
        per_dp_bs_size: int,
        total_bs: int,
    ):
        """Merge batch-level metadata from all DP ranks.

        Returns:
            (req_pool_indices, seq_lens, extend_prefix_lens,
             extend_seq_lens, extend_logprob_start_lens, logits_indices, real_bs)
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

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.seq_lens is None or len(info.seq_lens) == 0:
                # Empty DP rank
                offset_bs += per_dp_bs_size
                continue

            # Get data from this DP rank
            dp_bs = len(info.seq_lens)
            real_bs += dp_bs

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

            offset_bs += per_dp_bs_size

        return (
            req_pool_indices_cpu,
            seq_lens_cpu,
            extend_prefix_lens,
            extend_seq_lens,
            extend_logprob_start_lens,
            logits_indices,
            real_bs,
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
            # This matches the precompile logic in tp_worker.precompile_decode() which uses:
            #   aligned_cache_loc_size = (bs * max_req_len + page_size - 1) // page_size * page_size
            # and precompile_cache_loc_paddings is defined as:
            #   [bs * ((max_req_len + page_size - 1) // page_size * page_size) for bs in bs_paddings]
            # We select cache_loc_padding based on bs bucket to match precompiled kernels.
            total_bs = per_dp_bs_size * self.dp_size
            _, bs_index = find_padding_size(total_bs, bs_paddings)
            total_cache_loc_size = cache_loc_paddings[bs_index]

        per_dp_cache_loc_size = total_cache_loc_size // self.dp_size
        cache_loc_cpu = np.zeros(total_cache_loc_size, dtype=np.int32)

        offset_bs = 0
        offset_cache = 0

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.seq_lens is None or len(info.seq_lens) == 0:
                offset_bs += per_dp_cache_loc_size
                continue

            # Get token indices from req_to_token pool for this DP rank
            token_indices = self.req_to_token_pool.req_to_token[info.req_pool_indices]

            # Ensure we start writing at the correct offset for this DP rank
            offset_cache = offset_bs

            # Build cache_loc for each request in this DP rank
            for i, seq_len in enumerate(info.seq_lens):
                if seq_len > 0:
                    # Calculate page-aligned length
                    aligned_len = ((seq_len + page_size - 1) // page_size) * page_size

                    # Safety check to prevent overflow
                    if offset_cache + seq_len > offset_bs + per_dp_cache_loc_size:
                        raise RuntimeError(
                            f"Cache loc overflow in DP rank {dp_rank}. "
                            f"Offset {offset_cache} + len {seq_len} > limit {offset_bs + per_dp_cache_loc_size}. "
                            f"Chosen bucket size: {per_dp_cache_loc_size}"
                        )

                    # Copy actual token indices
                    tokens_to_copy = token_indices[i, :seq_len]
                    cache_loc_cpu[offset_cache : offset_cache + seq_len] = tokens_to_copy

                    # Move to next page-aligned position
                    offset_cache += aligned_len

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

        offset_bs = 0
        has_sampling_seeds = False
        vocab_size = 0

        for dp_rank in range(self.dp_size):
            info = self.reqs_info[dp_rank]

            if info.sampling_info is None or info.seq_lens is None or len(info.seq_lens) == 0:
                offset_bs += per_dp_bs_size
                continue

            dp_bs = len(info.seq_lens)
            dp_sampling = info.sampling_info
            if vocab_size == 0:
                vocab_size = dp_sampling.vocab_size

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

            # Move to next DP rank's slot (fixed slot size)
            offset_bs += per_dp_bs_size

        return ModelWorkerSamplingInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            sampling_seeds=sampling_seeds if has_sampling_seeds else None,
            grammars=[req.grammar for req in all_reqs] if self.has_grammar else None,
        )

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

        global bid
        bid += 1

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
        ) = self._merge_batch_metadata(per_dp_bs_padding, total_bs)

        # Step 4: Merge cache_loc from all DP ranks
        cache_loc_cpu = self._merge_cache_loc(
            bs_paddings, cache_loc_paddings, page_size, per_dp_bs_padding
        )

        # Step 5: Merge sampling info from all DP ranks
        sampling_info = self._merge_sampling_info(per_dp_bs_padding, total_bs)

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
            top_logprobs_nums=None,  # TODO: @Brian pad logprob info
            token_ids_logprobs=None,  # TODO
            sampling_info=sampling_info,
            positions=positions_cpu,
            cache_loc=cache_loc_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=None,
            logits_indices=logits_indices,
            lora_ids=lora_ids,
            real_bs=real_bs,
            dp_size=self.dp_size,
            per_dp_bs_size=per_dp_bs_padding,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            launch_done=self.launch_done,
        )

    def get_spec_model_worker_batch(
        self,
        token_paddings: list,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
        enable_static_lora: bool = False,
    ) -> ModelWorkerBatch:
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = np.array(self.extend_lens, dtype=np.int32)
            extend_prefix_lens = np.array(self.prefix_lens, dtype=np.int32)
            extend_logprob_start_lens = self.extend_logprob_start_lens
        logits_indices = None
        if self.forward_mode.is_extend() and extend_seq_lens is not None:
            if len(extend_seq_lens) > 0:
                logits_indices = np.cumsum(extend_seq_lens, dtype=np.int32) - 1
            else:
                logits_indices = np.array([], dtype=np.int32)

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
        token_indices_with_all_reqs = self.req_to_token_pool.req_to_token[self.req_pool_indices]
        # FIXME @pc, move this to eagle_worker
        # If enable spec inference, use positions in spec info firstly
        if self.spec_info is not None and getattr(self.spec_info, "positions", None) is not None:
            positions_cpu = self.spec_info.positions
        else:
            positions_cpu = None

        # Calculate positions after padding
        if self.forward_mode.is_extend():
            # For prefill: create positions for each token in sequences
            # Calculate total tokens without padding first
            if positions_cpu is None:
                lengths = seq_lens_cpu - self.prefix_lens
                if len(lengths) > 0:
                    repeats = lengths
                    total_len = np.sum(repeats)
                    # Generate range [0, 1, ... len-1] for each sequence
                    block_starts = np.concatenate(([0], np.cumsum(repeats)[:-1]))
                    shifts = np.repeat(block_starts, repeats)
                    ranges = np.arange(total_len) - shifts
                    # Add prefix_len to each range
                    positions_cpu = np.repeat(self.prefix_lens, repeats) + ranges
                    positions_cpu = positions_cpu.astype(seq_lens_cpu.dtype)
                else:
                    positions_cpu = np.array([], dtype=seq_lens_cpu.dtype)
        else:
            if positions_cpu is None:
                # For decode: each sequence contributes one token at the next position (seq_len)
                # Create positions for actual tokens (one per sequence at seq_len)
                batch_positions = np.maximum(0, seq_lens_cpu - 1)
                # Create positions array matching the length of input_ids (including padding)
                positions_cpu = np.zeros(len(batch_positions), dtype=batch_positions.dtype)
                # Fill in the actual positions for the real tokens
                # positions = positions.at[: len(batch_positions)].set(batch_positions)
                positions_cpu[: len(batch_positions)] = batch_positions
                # The padding tokens (if any) will have position 0, which is fine for padding

        cache_loc_flat = np.array([], dtype=np.int32)

        if len(seq_lens_cpu) > 0:
            seq_lens = seq_lens_cpu
            if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
                if self.forward_mode == ForwardMode.TARGET_VERIFY:
                    seq_lens = seq_lens_cpu + self.spec_info.draft_token_num
                elif self.forward_mode == ForwardMode.DECODE:
                    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

                    seq_lens = seq_lens_cpu + EagleDraftInput.ALLOC_LEN_PER_DECODE
            # Filter out empty sequences
            valid_mask = seq_lens > 0
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_seq_lens = seq_lens[valid_mask]
                # Calculate aligned lengths for all valid sequences at once
                if (
                    self.forward_mode == ForwardMode.DECODE
                    and not self.spec_algorithm.is_none()
                    and self.spec_info.allocate_lens is not None
                ):
                    # Explicitly convert to numpy to avoid JAX device synchronization overhead
                    allocated_len_cpu = np.array(self.spec_info.allocate_lens)
                    allocated_len = allocated_len_cpu[: len(self.reqs)]
                    aligned_lengths = ((allocated_len + page_size - 1) // page_size) * page_size
                    alread_allocated_lens = allocated_len
                else:
                    aligned_lengths = ((valid_seq_lens + page_size - 1) // page_size) * page_size
                    alread_allocated_lens = valid_seq_lens
                total_aligned_length = np.sum(aligned_lengths)

                # Pre-allocate the result array
                cache_loc_flat = np.zeros(total_aligned_length, dtype=np.int32)

                # Vectorized filling of cache_loc_flat
                # Calculate destination offsets for each block (where each block starts in cache_loc_flat)
                dst_offsets = np.concatenate(([0], np.cumsum(aligned_lengths)[:-1]))

                # We need to copy 'alread_allocated_lens' elements for each request
                repeats = alread_allocated_lens
                total_elements = np.sum(repeats)

                if total_elements > 0:
                    # 1. Generate Source Indices (row, col) for token_indices_with_all_reqs
                    row_indices = np.repeat(valid_indices, repeats)

                    # Generate col indices: 0..len-1 for each row
                    # Using the shift trick: global_range - block_start_offsets
                    block_starts = np.concatenate(([0], np.cumsum(repeats)[:-1]))
                    shifts = np.repeat(block_starts, repeats)
                    col_indices = np.arange(total_elements) - shifts

                    # Extract source data
                    source_data = token_indices_with_all_reqs[row_indices, col_indices]

                    # 2. Generate Destination Indices for cache_loc_flat
                    # Base offset for each block + local col index
                    dst_base_offsets = np.repeat(dst_offsets, repeats)
                    dst_indices = dst_base_offsets + col_indices

                    # Assign
                    cache_loc_flat[dst_indices] = source_data

        if precision_tracer.get_trace_active():
            self._generate_trace_info(real_bs, bid)
        # Extract lora_ids from requests
        lora_ids = [req.lora_id for req in self.reqs]

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
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            sampling_info=self.sampling_info,
            positions=positions_cpu,
            cache_loc=cache_loc_flat,
            extend_prefix_lens=(extend_prefix_lens if self.forward_mode.is_extend() else None),
            extend_seq_lens=(extend_seq_lens if self.forward_mode.is_extend() else None),
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            logits_indices=logits_indices,
            lora_ids=lora_ids,
            real_bs=real_bs,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL)
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            launch_done=self.launch_done,
            spec_info=self.spec_info,
            spec_algorithm=self.spec_algorithm,
            tree_cache=self.tree_cache,
        )

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
            copied_reqs_info.append(new_info)

        return ScheduleBatch(
            reqs_info=copied_reqs_info,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            return_logprob=self.return_logprob,
            return_output_logprob_only=self.return_output_logprob_only,
            is_prefill_only=self.is_prefill_only,
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
                full_evictable_size = self.tree_cache.full_evictable_size()
                swa_evictable_size = self.tree_cache.swa_evictable_size()
                result_strs.append(
                    f"{prefix}Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
                    f"{prefix}Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
                    f"{prefix}Full LRU list evictable size: {self.tree_cache.full_lru_list_evictable_size()}\n"
                    f"{prefix}SWA LRU list evictable size: {self.tree_cache.swa_lru_list_evictable_size()}\n"
                )
            else:
                available_size = self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                evictable_size = self.tree_cache.evictable_size()
                result_strs.append(
                    f"{prefix}Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
                )

        return "".join(result_strs)


def align_to_size(lst: list, size: int, value: int = 0) -> list:
    align_len = (len(lst) + size - 1) // size * size
    return lst[:] + [value] * (align_len - len(lst))


def find_padding_size(current_len: int, size_buckets: list[int]) -> tuple[int, int]:
    """Find appropriate padding size from sorted size buckets.

    Args:
        current_len: Current length of the array
        size_buckets: Sorted list of size buckets

    Returns:
        (target_size, index): Target size (bucket size) and index in sorted size_buckets

    Raises:
        AssertionError: If current_len exceeds all available buckets
    """
    size_buckets_sorted = sorted(size_buckets)
    for index, size in enumerate(size_buckets_sorted):
        if size >= current_len:
            return size, index

    # No suitable bucket found - this is a configuration error
    raise AssertionError(
        f"No suitable padding bucket found for length {current_len}. "
        f"Available buckets: {size_buckets_sorted}. "
        f"Please increase bucket sizes in configuration."
    )


def pad_array_to_size(
    arr: np.ndarray, target_size: int, pad_value: int = 0, dtype: np.dtype = None
) -> np.ndarray:
    """Pad numpy array to target size.

    Args:
        arr: Array to pad
        target_size: Target size after padding
        pad_value: Value to use for padding
        dtype: Data type for padding values (defaults to arr.dtype)

    Returns:
        Padded array
    """
    if len(arr) >= target_size:
        return arr

    if dtype is None:
        dtype = arr.dtype

    padding_size = target_size - len(arr)
    padding = np.array([pad_value] * padding_size, dtype=dtype)
    return np.concatenate([arr, padding], axis=0)


@dataclasses.dataclass
class ModelWorkerSamplingInfo:
    """Unified sampling information for a generation batch."""

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
        )
        return ret

    def update_penalties(self):
        self.linear_penalty = None


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

    spec_info: EagleDraftInput | EagleVerifyInput | None = None
    spec_algorithm: SpeculativeAlgorithm = None
    speculative_num_steps: int = 0
    speculative_eagle_topk: int = 0
    speculative_num_draft_tokens: int = 0
    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None

    tree_cache: BasePrefixCache = None


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
