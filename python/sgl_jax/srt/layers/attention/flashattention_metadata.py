"""Shared FlashAttention metadata inputs and device-side builders."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.utils import cdiv


@register_pytree_node_class
@dataclass(frozen=True)
class PagedKVLayout:
    """Physical KV pages uploaded by the host for device-side metadata building."""

    page_indices: jax.Array
    swa_page_indices: jax.Array | None = None

    def tree_flatten(self):
        return ((self.page_indices, self.swa_page_indices), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(page_indices=children[0], swa_page_indices=children[1])


def pad_page_indices(
    page_indices: np.ndarray,
    max_num_seqs: int,
    fixed_capacity: int | None = None,
) -> np.ndarray:
    """Pad page ids to a fixed capacity or a per-sequence power-of-two bucket."""
    page_indices = np.asarray(page_indices, dtype=np.int32)
    if fixed_capacity is not None:
        target_len = int(fixed_capacity)
        if target_len < len(page_indices):
            raise ValueError(
                "page_indices exceed fixed capacity: "
                f"required={len(page_indices)}, capacity={target_len}"
            )
    elif max_num_seqs > 0 and len(page_indices) > 0:
        current_pps = cdiv(len(page_indices), max_num_seqs)
        bucketed_pps = max(16, 1 << max(0, (current_pps - 1)).bit_length())
        target_len = max_num_seqs * bucketed_pps
    else:
        return page_indices

    if len(page_indices) < target_len:
        page_indices = np.pad(
            page_indices,
            (0, target_len - len(page_indices)),
            constant_values=0,
        )
    return page_indices


def _reshape_per_dp_rows(values, dp_size: int):
    per_dp_size = values.shape[0] // dp_size
    rows = values.reshape((dp_size, per_dp_size))
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        rows = jax.sharding.reshard(rows, NamedSharding(sharding.mesh, P("data", None)))
    return rows


def _per_dp_cumsum(lens, dp_size: int):
    per_dp_bs = lens.shape[0] // dp_size
    lens_2d = _reshape_per_dp_rows(lens, dp_size)
    zeros = jnp.zeros_like(lens_2d[:, :1], dtype=jnp.int32)
    result = jnp.concatenate(
        [zeros, jnp.cumsum(lens_2d, axis=1, dtype=jnp.int32)],
        axis=1,
    ).reshape((dp_size * (per_dp_bs + 1),))
    sharding = jax.typeof(lens).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        result = jax.sharding.reshard(result, sharding)
    return result


def _repack_page_indices(
    page_indices,
    allocated_lens,
    metadata_seq_lens,
    *,
    page_size: int,
    dp_size: int,
):
    pages_per_dp = page_indices.shape[0] // dp_size

    allocated_pages = ((allocated_lens + page_size - 1) // page_size).astype(jnp.int32)
    needed_pages = ((metadata_seq_lens + page_size - 1) // page_size).astype(jnp.int32)
    allocated_pages = _reshape_per_dp_rows(allocated_pages, dp_size)
    needed_pages = _reshape_per_dp_rows(needed_pages, dp_size)

    src_offsets = jnp.cumsum(allocated_pages, axis=1, dtype=jnp.int32) - allocated_pages
    dst_offsets = jnp.cumsum(needed_pages, axis=1, dtype=jnp.int32) - needed_pages

    local_page_ids = jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :, None]
    in_req = (local_page_ids >= dst_offsets[:, None, :]) & (
        local_page_ids < (dst_offsets + needed_pages)[:, None, :]
    )
    slot_ids = jnp.argmax(in_req.astype(jnp.int32), axis=2).astype(jnp.int32)
    valid = jnp.any(in_req, axis=2)

    dp_ids = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    offsets_sharding = jax.typeof(src_offsets).sharding
    offsets_out_sharding = offsets_sharding if isinstance(offsets_sharding, NamedSharding) else None
    src_slot_offsets = src_offsets.at[dp_ids, slot_ids].get(out_sharding=offsets_out_sharding)
    dst_slot_offsets = dst_offsets.at[dp_ids, slot_ids].get(out_sharding=offsets_out_sharding)
    gather_src = (
        dp_ids * pages_per_dp
        + src_slot_offsets
        + (jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :] - dst_slot_offsets)
    )
    page_sharding = jax.typeof(page_indices).sharding
    out_sharding = page_sharding if isinstance(page_sharding, NamedSharding) else None
    gathered = (
        page_indices.at[gather_src.reshape(-1)]
        .get(
            mode="fill",
            fill_value=0,
            out_sharding=out_sharding,
        )
        .reshape((dp_size, pages_per_dp))
    )
    if isinstance(page_sharding, NamedSharding) and not page_sharding.mesh.empty:
        gathered = jax.sharding.reshard(
            gathered,
            NamedSharding(page_sharding.mesh, P("data", None)),
        )
    gathered_sharding = jax.typeof(gathered).sharding
    if isinstance(gathered_sharding, NamedSharding) and not gathered_sharding.mesh.empty:
        valid = jax.sharding.reshard(valid, gathered_sharding)
    return jnp.where(valid, gathered, jnp.zeros_like(gathered)).reshape(page_indices.shape)


def _build_metadata_from_paged_layout(
    layout,
    *,
    query_lens=None,
    cu_q_lens=None,
    seq_lens,
    allocated_lens,
    distribution,
    page_size: int,
    dp_size: int,
):
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    if cu_q_lens is None:
        if query_lens is None:
            raise ValueError("query_lens or cu_q_lens must be provided")
        cu_q_lens = _per_dp_cumsum(query_lens, dp_size)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    cu_kv_lens = _per_dp_cumsum(aligned_seq_lens, dp_size)
    page_indices = _repack_page_indices(
        layout.page_indices,
        allocated_lens,
        seq_lens,
        page_size=page_size,
        dp_size=dp_size,
    )
    swa_page_indices = None
    if layout.swa_page_indices is not None:
        swa_page_indices = _repack_page_indices(
            layout.swa_page_indices,
            allocated_lens,
            seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    data_sharding = jax.typeof(seq_lens).sharding
    if isinstance(data_sharding, NamedSharding) and not data_sharding.mesh.empty:
        cu_q_lens = jax.sharding.reshard(cu_q_lens, data_sharding)
        cu_kv_lens = jax.sharding.reshard(cu_kv_lens, data_sharding)
        page_indices = jax.sharding.reshard(page_indices, data_sharding)
        seq_lens = jax.sharding.reshard(seq_lens, data_sharding)
        distribution = jax.sharding.reshard(distribution, data_sharding)
        if swa_page_indices is not None:
            swa_page_indices = jax.sharding.reshard(swa_page_indices, data_sharding)

    return FlashAttentionMetadata(
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        swa_page_indices=swa_page_indices,
        seq_lens=seq_lens,
        distribution=distribution,
    )


def build_target_verify_metadata(
    layout: PagedKVLayout,
    prefix_lens,
    allocated_lens,
    *,
    active_mask=None,
    draft_width: int,
    page_size: int,
    dp_size: int,
):
    """Build complete target-verify metadata from a physical page layout."""
    valid = prefix_lens > 0 if active_mask is None else active_mask.astype(jnp.bool_)
    query_lens = jnp.where(
        valid,
        jnp.full_like(prefix_lens, draft_width),
        jnp.zeros_like(prefix_lens),
    )
    seq_lens = prefix_lens + query_lens
    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))
    return _build_metadata_from_paged_layout(
        layout,
        query_lens=query_lens,
        seq_lens=seq_lens,
        allocated_lens=allocated_lens,
        distribution=distribution,
        page_size=page_size,
        dp_size=dp_size,
    )


def build_draft_extend_metadata(
    layout,
    seq_lens,
    allocated_lens,
    *,
    query_lens=None,
    page_size: int,
    dp_size: int,
):
    """Build complete speculative extend metadata from page and length inputs."""
    valid = seq_lens > 0
    if query_lens is None:
        cu_q_lens = getattr(layout, "cu_q_lens", None)
        if cu_q_lens is None:
            raise ValueError("query_lens are required for a PagedKVLayout")
    else:
        cu_q_lens = None

    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))
    return _build_metadata_from_paged_layout(
        layout,
        query_lens=query_lens,
        cu_q_lens=cu_q_lens,
        seq_lens=seq_lens,
        allocated_lens=allocated_lens,
        distribution=distribution,
        page_size=page_size,
        dp_size=dp_size,
    )


def build_draft_forward_metadata(
    layout: PagedKVLayout,
    seq_lens,
    allocated_lens,
    *,
    page_size: int,
    dp_size: int,
):
    """Build metadata for one token-forward step of the draft model."""
    valid = seq_lens > 0
    # Decode keeps one input row for every bucket slot, including padding.
    # Padded slots still have zero KV length and are excluded by distribution.
    query_lens = jnp.ones_like(seq_lens, dtype=jnp.int32)
    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), jnp.zeros_like(local_num_seqs), local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))
    return _build_metadata_from_paged_layout(
        layout,
        query_lens=query_lens,
        seq_lens=seq_lens,
        allocated_lens=allocated_lens,
        distribution=distribution,
        page_size=page_size,
        dp_size=dp_size,
    )
