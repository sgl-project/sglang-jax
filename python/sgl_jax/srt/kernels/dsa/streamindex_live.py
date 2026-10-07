"""Decode indexer with per-request live-context dispatch.

Pallas reads only live KV blocks. A runtime switch chooses a score-buffer
bucket per request, so the exit-stage top-k also avoids the global capacity.
This is independent of the packed page-table stride and preserves the full
context limit. No short-context shortcut is used: every active query is scored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk


def _page_buckets(pages_per_seq: int, pages_per_block: int) -> tuple[int, ...]:
    """Powers of two blocks, capped at the caller's actual page-table capacity."""
    if pages_per_seq < 1 or pages_per_block < 1:
        raise ValueError("Page capacity and block size must be positive")
    buckets = []
    pages = pages_per_block
    while pages < pages_per_seq:
        buckets.append(pages)
        pages *= 2
    buckets.append(pages_per_seq)
    return tuple(buckets)


@jax.jit(static_argnames=("k", "pages_per_seq", "num_kv_pages_per_block", "compression_ratio"))
def streamindex_topk_live(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    num_kv_pages_per_block: int = 64,
    compression_ratio: int = 1,
) -> jax.Array:
    """Return int32[B,k] seq-local token indices, with trailing -1 padding.

    ``compression_ratio > 1`` (KPool): the cache holds one pooled entry per ``ratio``
    tokens (``page_size`` entries per page = ``page_size * ratio`` tokens), seq_lens and
    cu_kv_lens stay in token units, and the result is in pooled-entry space.

    Decode ABI matches ``streamindex_topk_ref(one_token_per_seq=True)``:
    one query per sequence, packed page table, cu_kv_lens in TOKEN units,
    and an uncompressed BF16 cache [physical_pages,page_size,head_dim].
    The caller writes current keys before invoking this function.

    Active rows are sorted by their live-context bucket and processed in
    same-bucket groups of up to four. Do not vmap the bucket switch: that would
    evaluate every bucket and restore full-capacity work.

    Scoring follows the existing Pallas kernel; selection is exact top-k
    (subject to floating-point scoring and tie ordering), unlike the reference's
    approximate candidate reduction. Tests compare against an exact score oracle.
    Oversized rows use exact chunk-local top-k followed by a merge, avoiding
    a full-row sort when the score bucket exceeds SparseCore VMEM.
    Live sequence lengths must not exceed pages_per_seq * page_size.
    """
    del cu_q_lens  # In decode, query i belongs to sequence i.
    batch = q.shape[0]
    if q.ndim != 3 or batch != seq_lens.shape[0]:
        raise ValueError("Live indexer requires one query per sequence")
    if cache_kv.ndim != 3 or q.shape[-1] != cache_kv.shape[-1]:
        raise ValueError("Expected [pages,page_size,head_dim] indexer cache")
    if q.shape[-1] % 128:
        raise ValueError("Pallas indexer requires a head dimension divisible by 128")
    if k < 1:
        raise ValueError("k must be positive")
    page_size = cache_kv.shape[1]
    if page_indices.shape[0] < pages_per_seq:
        raise ValueError("Page table is smaller than pages_per_seq")
    if page_size * num_kv_pages_per_block % 128:
        raise ValueError("KV block must contain a multiple of 128 tokens")
    # Keep 2K-token buckets at page_size=128, while larger contexts amortize
    # paged-DMA/control overhead over up to 64 pages per scoring block.
    buckets = _page_buckets(pages_per_seq, min(16, num_kv_pages_per_block))
    limits = jnp.asarray([pages * page_size * compression_ratio for pages in buckets], jnp.int32)
    # A view of the existing cache, not a gathered copy. Packing is part of the
    # Pallas cache ABI: two BF16 elements per 32-bit word.
    if cache_kv.dtype != jnp.bfloat16 or page_size % 2:
        raise ValueError("Live indexer requires BF16 cache and an even page size")
    cache4d = cache_kv.reshape(cache_kv.shape[0], page_size // 2, 2, cache_kv.shape[-1])
    if batch == 0:
        return jnp.full((0, k), -1, jnp.int32)

    group_size = min(4, batch)
    # Leave one group's lookahead after the real rows: a bucket run can end
    # at any row, so the final partial group still needs a full static slice.
    padded_batch = ((batch + group_size - 1) // group_size) * group_size + group_size - 1
    if padded_batch != batch:
        q = jnp.pad(q, ((0, padded_batch - batch), (0, 0), (0, 0)))
        weights = jnp.pad(weights, ((0, padded_batch - batch), (0, 0)))
        seq_lens = jnp.pad(seq_lens, ((0, padded_batch - batch),))
        page_starts = jnp.pad(
            cu_kv_lens[:-1] // (page_size * compression_ratio), ((0, padded_batch - batch),)
        )
    else:
        page_starts = cu_kv_lens[:-1] // (page_size * compression_ratio)

    row_ids = jnp.arange(padded_batch, dtype=jnp.int32)
    active = (row_ids < distribution[0]) & (seq_lens > 0)
    row_buckets = jnp.sum(seq_lens[:, None] > limits[None, :], axis=1, dtype=jnp.int32)
    row_buckets = jnp.where(active, row_buckets, len(buckets))
    order = jnp.argsort(row_buckets, stable=True)
    sorted_buckets = row_buckets[order]
    active_count = jnp.sum(active, dtype=jnp.int32)

    # cu_q_lens is an ignored ABI slot (decode: query i belongs to sequence i).
    one_cuq = jnp.arange(group_size + 1, dtype=jnp.int32)
    one_dist = jnp.asarray([group_size, group_size, group_size], jnp.int32)
    bucket_branches = []

    for bucket_pages in buckets:

        def score_and_select(args, bucket_pages=bucket_pages):
            q_group, w_group, lengths_group, starts_group, group_order, out = args
            offsets = starts_group[:, None] + jnp.arange(bucket_pages, dtype=jnp.int32)[None, :]
            offsets = jnp.minimum(offsets, page_indices.shape[0] - 1)
            pages = page_indices[offsets].reshape(-1)
            selected = streamindex_topk(
                q_group,
                w_group,
                cache4d,
                lengths_group,
                pages,
                one_cuq,
                one_dist,
                k=k,
                compression_ratio=compression_ratio,
                num_kv_pages_per_block=min(bucket_pages, num_kv_pages_per_block),
                num_queries_per_block=1,
                decode_req_batch_size=group_size,
            )
            old = out[group_order]
            return out.at[group_order].set(jnp.where(lengths_group[:, None] > 0, selected, old))

        bucket_branches.append(score_and_select)

    def condition(carry):
        cursor, _ = carry
        return cursor < active_count

    def group_step(carry):
        cursor, out = carry
        group_order = jax.lax.dynamic_slice_in_dim(order, cursor, group_size)
        group_buckets = jax.lax.dynamic_slice_in_dim(sorted_buckets, cursor, group_size)
        bucket = group_buckets[0]
        in_bucket = group_buckets == bucket
        run_size = jnp.sum(in_bucket, dtype=jnp.int32)
        q_group = q[group_order]
        w_group = weights[group_order]
        lengths_group = jnp.where(in_bucket, seq_lens[group_order], 0)
        starts_group = page_starts[group_order]
        out = jax.lax.switch(
            bucket,
            tuple(bucket_branches),
            (q_group, w_group, lengths_group, starts_group, group_order, out),
        )
        return cursor + run_size, out

    out = jnp.full((padded_batch, k), -1, jnp.int32)
    _, out = jax.lax.while_loop(condition, group_step, (jnp.asarray(0, jnp.int32), out))
    return out[:batch]
