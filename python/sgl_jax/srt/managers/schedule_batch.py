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

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import dataclasses
import logging
from http import HTTPStatus
from typing import Any, List, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import mesh as mesh_lib
from jax.experimental.multihost_utils import process_allgather

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.jax_utils import device_array

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


GLOBAL_SERVER_ARGS_KEYS = [
    "device",
    "disable_radix_cache",
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

        # For retraction
        self.is_retracted = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
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
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: jax.Array = None  # shape: [b], int32
    input_embeds: jax.Array = None  # shape: [b, hidden_size], float32
    req_pool_indices: jax.Array = None  # shape: [b], int32
    seq_lens: jax.Array = None  # shape: [b], int32
    # The output locations of the KV cache
    out_cache_loc: jax.Array = None  # shape: [b], int32
    output_ids: jax.Array = None  # shape: [b], int32

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

    enable_dp_attention: bool = False

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_custom_logit_processor: bool = False,
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
            has_stream=any(req.stream for req in reqs),
            mesh=mesh,
        )

    @property
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

        return out_cache_loc

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

        req_pool_indices_device = jnp.array(req_pool_indices, dtype=jnp.int32)
        input_ids_device = jnp.array(sum(input_ids, []), dtype=jnp.int32)
        seq_lens_device = jnp.array(seq_lens, dtype=jnp.int32)

        # Copy prefix and do some basic check
        extend_input_logprob_token_ids = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len
            # note: req.prefix_indices is located on CPU, so we have to extract values then device_put
            prefix_indices_device = jnp.array(np.asarray(req.prefix_indices))
            if pre_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), prefix_indices_device
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
        assert self.token_to_kv_pool_allocator.page_size == 1
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)

        # Set fields
        self.input_ids = input_ids_device
        self.req_pool_indices = req_pool_indices_device
        self.seq_lens = seq_lens_device
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
        assert (
            self.token_to_kv_pool_allocator.page_size == 1
        ), "token_to_kv_pool_allocator.page_size must be 1"
        return len(self.reqs)

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

        def get_required_tokens(num_reqs: int):
            return num_reqs * global_config.retract_decode_steps

        def _get_available_size():
            return self.token_to_kv_pool_allocator.available_size()

        retracted_reqs = []
        seq_lens_cpu = jax.device_get(self.seq_lens)
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

            # TODO: apply more fine-grained retraction
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

            # NOTE(lsyin): we should use the newly evictable memory instantly.
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
        self.input_ids = jnp.empty(0, jnp.int32)
        self.seq_lens = jnp.empty(0, jnp.int32)
        self.out_cache_loc = jnp.empty(0, jnp.int32)
        self.req_pool_indices = jnp.empty(0, jnp.int32)

        # (self.input_ids, self.seq_lens, self.out_cache_loc, self.req_pool_indices) = device_array(
        #     self.mesh,
        #     (
        #         jnp.empty(0, jnp.int32),
        #         jnp.empty(0, jnp.int32),
        #         jnp.empty(0, jnp.int32),
        #         jnp.empty(0, jnp.int32),
        #     ),
        # )
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)

        # note: is_required = False
        # if self.sampling_info.penalizer_orchestrator.is_required:
        #     self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
        #         self.output_ids.astype(jnp.int32)
        #     )

        # Update fields
        self.input_ids = self.output_ids
        self.output_ids = None

        locs = self.seq_lens.copy()

        self.seq_lens = jnp.add(self.seq_lens, 1)
        self.seq_lens_sum += bs

        # Allocate memory
        assert (
            self.token_to_kv_pool_allocator.page_size == 1
        ), "token_to_kv_pool_allocator page_size must be 1"
        self.out_cache_loc = self.alloc_token_slots(bs)

        self.req_to_token_pool.write(
            (self.req_pool_indices, locs), self.out_cache_loc.astype(jnp.int32)
        )

    def filter_batch(
        self,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i for i in range(len(self.reqs)) if not self.reqs[i].finished()
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = device_array(
            self.mesh, jnp.array(keep_indices, dtype=jnp.int32)
        )

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[keep_indices_device]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)

    def merge_batch(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info, other.mesh)

        self.req_pool_indices = jnp.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = jnp.concat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = jnp.concat(
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

    def get_model_worker_batch(
        self,
        max_running_requests: int,
        max_total_num_tokens: int,
        bs_paddings: list,
        token_paddings: list,
    ) -> ModelWorkerBatch:
        logging.info(
            f"schedule_batch.get_model_worker_batch: {self.forward_mode.name} input_ids.shape={self.input_ids.shape if self.input_ids is not None else None}, out_cache_loc.shape={self.out_cache_loc.shape if self.out_cache_loc is not None else None}, seq_lens.shape={self.seq_lens.shape if self.seq_lens is not None else None}, req_pool_indices.shape={self.req_pool_indices.shape if self.req_pool_indices is not None else None}, output_ids.shape={self.output_ids.shape if self.output_ids is not None else None}"
        )
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
            token_paddings = bs_paddings
        else:
            extend_seq_lens = np.array(self.extend_lens, dtype=np.int32)
            extend_prefix_lens = np.array(self.prefix_lens, dtype=np.int32)
            bs_paddings = [1, max_running_requests]

        global bid
        bid += 1

        input_ids_cpu = (
            jax.device_get(self.input_ids.flatten())
            if self.input_ids is not None and len(self.input_ids) > 0
            else []
        )
        real_input_ids_len = len(input_ids_cpu)
        out_cache_loc_cpu = (
            jax.device_get(self.out_cache_loc)
            if self.out_cache_loc is not None and len(self.out_cache_loc) > 0
            else []
        )
        seq_lens_cpu = (
            jax.device_get(self.seq_lens)
            if self.seq_lens is not None and len(self.seq_lens) > 0
            else []
        )
        real_bs = len(seq_lens_cpu)
        req_pool_indices_cpu = (
            jax.device_get(self.req_pool_indices)
            if self.req_pool_indices is not None and len(self.req_pool_indices) > 0
            else []
        )
        token_indices_with_all_reqs = (
            jax.device_get(self.req_to_token_pool.req_to_token[self.req_pool_indices])
            if self.req_to_token_pool.req_to_token is not None
            and len(self.req_to_token_pool.req_to_token[self.req_pool_indices]) > 0
            else []
        )

        # All-gather padding sizes from all devices to ensure consistent padding
        local_token_size = len(input_ids_cpu)
        local_bs_size = len(seq_lens_cpu)
        local_cache_size = sum(seq_lens_cpu)
        logging.info(
            f"schedule_batch.get_model_worker_batch: {self.forward_mode.name} local_token_size={local_token_size}, local_bs_size={local_bs_size}, local_cache_size={local_cache_size}"
        )
        # Gather sizes from all devices
        if self.enable_dp_attention:
            # Collect local sizes into arrays for all-gather
            local_sizes = jnp.array(
                [local_token_size, local_bs_size, local_cache_size], dtype=jnp.int32
            )
            local_sizes = np.array(local_sizes_list, dtype=np.int32)
            # All-gather to get sizes from all devices
            all_sizes = process_allgather(local_sizes)
            # Calculate global max sizes
            global_max_token_size = jnp.max(all_sizes[:, 0]).item()
            global_max_bs_size = jnp.max(all_sizes[:, 1]).item()
            global_max_cache_size = jnp.max(all_sizes[:, 2]).item()
        else:
            global_max_token_size = local_token_size
            global_max_bs_size = local_bs_size
            global_max_cache_size = local_cache_size
        logging.info(
            f"schedule_batch.get_model_worker_batch: {self.forward_mode.name} global_max_token_size={global_max_token_size}, global_max_bs_size={global_max_bs_size}, global_max_cache_size={global_max_cache_size}"
        )
        # padding seq
        # extend & decode: input_ids, positions, out_cache_loc, cache_loc
        padding_size = 0
        token_paddings.sort()
        # Use global max token size for consistent padding
        for size in token_paddings:
            if size >= global_max_token_size:
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
            positions = np.concatenate(
                [
                    np.arange(prefix_len, seq_len, dtype=seq_lens_cpu.dtype)
                    for seq_len, prefix_len in zip(seq_lens_cpu, self.prefix_lens)
                ]
            )

            # If input_ids was padded, pad positions too
            padding_size = len(input_ids_cpu) - total_tokens_before_padding
            if padding_size:
                positions = np.concatenate(
                    [positions, np.zeros(padding_size, dtype=positions.dtype)]
                )

            # Start location of each sequence in the flattened array
            extend_start_loc = np.cumsum(
                np.concatenate([np.array([0]), extend_seq_lens[:-1]]),
                dtype=seq_lens_cpu.dtype,
            )
        else:
            # For decode: each sequence contributes one token at the next position (seq_len)
            # Create positions for actual tokens (one per sequence at seq_len)
            batch_positions = seq_lens_cpu  # Next position is current seq_len
            # Create positions array matching the length of input_ids (including padding)
            positions = np.zeros(len(input_ids_cpu), dtype=batch_positions.dtype)
            # Fill in the actual positions for the real tokens
            # positions = positions.at[: len(batch_positions)].set(batch_positions)
            positions[: len(batch_positions)] = batch_positions
            # The padding tokens (if any) will have position 0, which is fine for padding
            # For decode, extend_start_loc is typically not used but we'll set it anyway
            extend_start_loc = np.arange(len(seq_lens_cpu), dtype=seq_lens_cpu.dtype)

        # padding bs: req_pool_indices, seq_lens, extend_start_loc, extend_prefix_lens, extend_seq_lens
        bs_padding_size = 0
        bs_paddings.sort()
        # Use global max batch size for consistent padding
        for size in bs_paddings:
            if size >= global_max_bs_size:
                bs_padding_size = size - len(seq_lens_cpu)
                break

        total_cache_size = sum(seq_lens_cpu)

        cache_loc_flat = np.zeros(total_cache_size, dtype=np.int32)

        offset = 0
        for seq_idx in range(len(seq_lens_cpu)):
            seq_len = seq_lens_cpu[seq_idx]
            if seq_len > 0:  # Only process non-empty sequences
                cache_loc_flat[offset : offset + seq_len] = token_indices_with_all_reqs[
                    seq_idx, :seq_len
                ]
                offset += seq_len

        # Use global max cache size for consistent cache_loc padding
        total_cache_loc_size = max(max_total_num_tokens, global_max_cache_size)
        if total_cache_loc_size > len(cache_loc_flat):
            cache_loc_cpu = np.pad(
                cache_loc_flat,
                (0, total_cache_loc_size - len(cache_loc_flat)),
                constant_values=0,
            )

        # seq_lens_padding = self.seq_lens
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

        logging.info(
            f"schedule_batch.get_model_worker_batch: {self.forward_mode.name} input_ids_cpu.shape={input_ids_cpu.shape if input_ids_cpu is not None else None}, out_cache_loc_cpu.shape={out_cache_loc_cpu.shape if out_cache_loc_cpu is not None else None}, seq_lens_cpu.shape={seq_lens_cpu.shape if seq_lens_cpu is not None else None}, req_pool_indices_cpu.shape={req_pool_indices_cpu.shape if req_pool_indices_cpu is not None else None}, positions.shape={positions.shape if positions is not None else None}, extend_start_loc.shape={extend_start_loc.shape if extend_start_loc is not None else None}, cache_loc_cpu.shape={cache_loc_cpu.shape if cache_loc_cpu is not None else None}, extend_prefix_lens.shape={extend_prefix_lens.shape if extend_prefix_lens is not None else None}, extend_seq_lens.shape={extend_seq_lens.shape if extend_seq_lens is not None else None}"
        )

        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=device_array(
                self.mesh,
                input_ids_cpu,
            ),
            real_input_ids_len=real_input_ids_len,
            real_bs=real_bs,
            req_pool_indices=device_array(self.mesh, req_pool_indices_cpu),
            seq_lens=device_array(self.mesh, seq_lens_cpu),
            out_cache_loc=device_array(self.mesh, out_cache_loc_cpu),
            return_logprob=self.return_logprob,
            sampling_info=self.sampling_info,
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            positions=device_array(self.mesh, positions),
            extend_start_loc=device_array(self.mesh, extend_start_loc),
            cache_loc=device_array(self.mesh, cache_loc_cpu),
            extend_prefix_lens=(
                device_array(self.mesh, extend_prefix_lens)
                if self.forward_mode == ForwardMode.EXTEND
                else None
            ),
            extend_seq_lens=(
                device_array(self.mesh, extend_seq_lens)
                if self.forward_mode == ForwardMode.EXTEND
                else None
            ),
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
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            is_extend_in_batch=self.is_extend_in_batch,
        )

    def _evict_tree_cache_if_needed(
        self,
        num_tokens: int,
    ) -> None:
        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens)

    def _is_available_size_sufficient(self, num_tokens: int) -> bool:
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def _available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"

    # def __str__(self):
    #     return (
    #         f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
    #         f"#req={(len(self.reqs))})"
    #     )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: jax.Array
    # the length is outof padding
    real_input_ids_len: int
    real_bs: int
    # The sequence length
    seq_lens: jax.Array
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: jax.Array
    # The indices of requests in the req_to_token_pool
    req_pool_indices: jax.Array
    # Sampling info
    sampling_info: SamplingBatchInfo
    # Position information [total_tokens]
    positions: jax.Array
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array
    # cache_loc
    cache_loc: jax.Array

    # For logprob
    return_logprob: bool = False
    extend_input_logprob_token_ids: Optional[jax.Array] = None

    # For extend
    extend_prefix_lens: Optional[jax.Array] = None

    extend_seq_lens: Optional[jax.Array] = None

    def print_array(self):
        print(f"========ModelWorkerBatch")
        print(f"{self.input_ids.shape=}, {self.input_ids.sharding=}")
        print(f"{self.seq_lens.shape=}, {self.seq_lens.sharding=}")
        print(f"{self.out_cache_loc.shape=}, {self.out_cache_loc.sharding=}")
        print(f"{self.req_pool_indices.shape=}, {self.req_pool_indices.sharding=}")
        print(f"{self.positions.shape=}, {self.positions.sharding=}")
        print(f"{self.extend_start_loc.shape=}, {self.extend_start_loc.sharding=}")
        print(f"{self.cache_loc.shape=}, {self.cache_loc.sharding=}")
