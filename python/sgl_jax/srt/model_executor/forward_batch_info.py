"""
Store information about a forward batch.

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

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, List, Optional

import jax

logger = logging.getLogger(__name__)

from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.model_runner import ModelRunner

from jax.tree_util import register_pytree_node_class


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self.is_extend()

    def is_extend(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.TARGET_VERIFY
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self):
        return self == ForwardMode.DRAFT_EXTEND

    def is_extend_or_draft_extend_or_mixed(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.MIXED
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE


@total_ordering
class CaptureHiddenMode(IntEnum):
    # Do not capture anything.
    NULL = 0
    # Capture a hidden state of the last token.
    LAST = 1
    # Capture hidden states of all tokens.
    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

    def __lt__(self, other):
        return self.value < other.value


@register_pytree_node_class
@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids [total_tokens]
    input_ids: jax.Array
    # The indices of requests in the req_to_token_pool
    req_pool_indices: jax.Array
    # The sequence length for each request [batch_size]
    seq_lens: jax.Array
    # decode token position in kv cache
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None

    # kv cache
    token_to_kv_pool: KVCache = None
    attn_backend: AttentionBackend = None

    cache_loc: jax.Array = None

    # For extend
    extend_prefix_lens: Optional[jax.Array] = None
    extend_seq_lens: Optional[jax.Array] = None

    trace_request_ids: Optional[List[str]] = None
    trace_request_objects: Optional[List] = None

    def tree_flatten(self):
        children = (
            self.input_ids,
            self.req_pool_indices,
            self.seq_lens,
            self.out_cache_loc,
            self.positions,
            self.extend_start_loc,
            self.token_to_kv_pool,
            self.attn_backend,
            self.cache_loc,
            self.extend_prefix_lens,
            self.extend_seq_lens,
        )

        aux_data = {
            "forward_mode": self.forward_mode,
            "batch_size": self.batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.forward_mode = aux_data["forward_mode"]
        obj.batch_size = aux_data["batch_size"]
        obj.trace_request_ids = None
        obj.trace_request_objects = None

        obj.input_ids = children[0]
        obj.req_pool_indices = children[1]
        obj.seq_lens = children[2]
        obj.out_cache_loc = children[3]
        obj.positions = children[4]
        obj.extend_start_loc = children[5]
        obj.token_to_kv_pool = children[6]
        obj.attn_backend = children[7]
        obj.cache_loc = children[8]
        obj.extend_prefix_lens = children[9]
        obj.extend_seq_lens = children[10]

        return obj

    def __repr__(self) -> str:
        jax_array_fields = []

        for field_name in [
            "input_ids",
            "req_pool_indices",
            "seq_lens",
            "out_cache_loc",
            "positions",
            "extend_start_loc",
            "cache_loc",
            "extend_prefix_lens",
            "extend_seq_lens",
        ]:
            value = getattr(self, field_name, None)
            if value is not None and isinstance(value, jax.Array):
                jax_array_fields.append(f"{field_name}={value.shape}")

        jax_arrays_str = ", ".join(jax_array_fields)
        return f"ForwardBatch(forward_mode={self.forward_mode}, batch_size={self.batch_size}, {jax_arrays_str})"

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        (
            input_ids,
            seq_lens,
            out_cache_loc,
            positions,
            extend_start_loc,
            req_pool_indices,
            cache_loc,
            extend_prefix_lens,
            extend_seq_lens,
        ) = device_array(
            (
                batch.input_ids,
                batch.seq_lens,
                batch.out_cache_loc,
                batch.positions,
                batch.extend_start_loc,
                batch.req_pool_indices,
                batch.cache_loc,
                batch.extend_prefix_lens,
                batch.extend_seq_lens,
            ),
            sharding=(
                NamedSharding(model_runner.mesh, PartitionSpec())
                if jax.process_count() == 1
                else None
            ),
        )

        obj = cls(
            bid=batch.bid,
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=input_ids,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            extend_start_loc=extend_start_loc,
            req_pool_indices=req_pool_indices,
            cache_loc=cache_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
        )

        return obj
