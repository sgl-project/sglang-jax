"""Host metadata types and query/capacity bucketing shared by HCA backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import HCAKernelSchedule
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackendMetadata


@register_pytree_node_class
@dataclass
class HCABackendMetadata(AttentionBackendMetadata):
    """Per-forward HCA metadata plus static framework/optimization choices."""

    kernel: HCAMetadata | None = None
    schedule: HCAKernelSchedule | None = None
    use_uniform_prefill_fast_path: bool = False

    def tree_flatten(self):
        return (self.kernel,), (self.schedule, self.use_uniform_prefill_fast_path)

    @classmethod
    def tree_unflatten(cls, static_options, children):
        schedule, use_uniform_prefill_fast_path = static_options
        return cls(
            kernel=children[0],
            schedule=schedule,
            use_uniform_prefill_fast_path=use_uniform_prefill_fast_path,
        )


# Minimum padded capacities: small floors stop tiny batches re-bucketing every
# few steps, and the page-table floors hold one large batch's tables.
_DECODE_IDS_FLOOR = 8
_BOUNDARY_FLOOR = 8
_WINDOW_TABLE_FLOOR = 512
_COMPRESSED_TABLE_FLOOR = 64


def _query_schedule(cu_q_lens: np.ndarray, query_block_size: int):
    """Build execution-only query blocks after the platform schedule is known.

    Returns possibly-empty arrays; ``get_forward_metadata`` pads them to stable
    capacities."""
    q_lens = np.diff(cu_q_lens).astype(np.int32)
    block_counts = np.where(
        q_lens == 1,
        0,
        (q_lens + query_block_size - 1) // query_block_size,
    )
    request_ids = np.repeat(np.arange(q_lens.size, dtype=np.int32), block_counts)
    offsets = np.concatenate(
        [
            (
                np.arange(0, int(q_len), query_block_size, dtype=np.int32)
                if q_len != 1
                else np.empty((0,), np.int32)
            )
            for q_len in q_lens
        ]
    )
    return request_ids, offsets.astype(np.int32), np.flatnonzero(q_lens == 1).astype(np.int32)


def _pad_capacity(values: np.ndarray, capacity: int, fill) -> np.ndarray:
    """Right-pad to an exact batch-shape-derived capacity with an inert fill."""
    values = np.asarray(values, np.int32)
    if values.shape[0] > capacity:
        raise ValueError(f"HCA metadata length {values.shape[0]} exceeds capacity {capacity}")
    # np.pad costs ~20 us per call; this runs several times per decode tick.
    padded = np.full((capacity,) + values.shape[1:], fill, np.int32)
    padded[: values.shape[0]] = values
    return padded


def _bucket_capacity(length: int, floor: int, bound: int | None = None) -> int:
    """Smallest power-of-two capacity covering ``length``, capped at ``bound``."""
    capacity = floor
    while capacity < length:
        capacity *= 2
    return capacity if bound is None else min(capacity, bound)


def _bucket_max_queries(max_queries: int, floor: int) -> int:
    """Bucket the per-request query capacity to a bounded ladder.

    Decode (1) keeps its dedicated value; longer chunks round up onto powers of
    two interleaved with 1.5x steps (..., 128, 192, 256, 384, ...), so a chunk
    just past a power of two pays 1.5x KV staging instead of 2x.
    """
    if max_queries <= 1:
        return max_queries
    power = 1 << max((max_queries - 1).bit_length() - 1, 0)
    bucket = power * 3 // 2 if max_queries <= power * 3 // 2 else power * 2
    return max(floor, bucket)
