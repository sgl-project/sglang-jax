"""Blocked (query-batched) sparse-MLA prefill — P1 of the DSA attend path.

The deployed per-query kernel (``sparse_mla_prefill.py``) runs one program per
query: each of the ``S`` queries independently DMAs its ``K`` selected pages. At
long context the selections overlap heavily (sinks + local windows), so the same
physical page is re-fetched by thousands of programs — ~298x redundant HBM
traffic per layer at 110k — and the score matmul feeds only ``H`` (~4) sublane
rows, idling the MXU.

The blocked kernel amortises both: ``QB`` queries share one program, the block's
selected-page **union** is DMA'd once per page, and per-query selection is
restored with a ``-inf`` membership bias (bitmap AND causal AND kv_len) inside a
flash-softmax over ``QB*H`` score rows.

This module hosts (in build order):

1. ``build_block_unit_tables`` — jnp preprocessing (outside the kernel): turn
   ``topk_units [T, K]`` into per-block union tables + membership bitmaps.
2. the Pallas query-block kernel itself (v1: single-sequence flat).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_SENTINEL = jnp.iinfo(jnp.int32).max


def build_block_unit_tables(
    topk_units,  # [T, K] int32 selected unit ids per query, -1 padded
    *,
    query_block: int,  # QB: queries per block
    u_max: int,  # static cap on the per-block union size
):
    """Per-query-block union + membership tables for the blocked kernel.

    Returns ``(blk_units, blk_member, blk_counts)``:

    * ``blk_units [nQB, u_max]`` int32 — the block's selected units, sorted
      ascending, ``-1`` padded. Uniques beyond ``u_max`` are dropped.
    * ``blk_member [nQB, query_block, u_max]`` int8 — 1 where query ``q`` of the
      block selected ``blk_units[b, u]`` (the kernel's -inf bias source).
    * ``blk_counts [nQB]`` int32 — the TRUE union size, **uncapped**: a value
      ``> u_max`` means the block overflowed and its tables are incomplete; the
      caller must gate (pick ``u_max >= min(query_block*K, num_units)`` to make
      overflow impossible).

    ``T`` need not divide ``query_block``; trailing queries are padded with
    ``-1`` rows (all-zero membership). Unit ids are opaque keys: callers with
    packed multi-request (ragged) inputs lift seq-local ids to global keys
    (e.g. ``base + page``) before calling, so blocks may straddle requests.
    """
    T, K = topk_units.shape
    qb = query_block
    n_blk = -(-T // qb)
    pad = n_blk * qb - T
    tk = jnp.pad(topk_units.astype(jnp.int32), ((0, pad), (0, 0)), constant_values=-1)
    flat = tk.reshape(n_blk, qb * K)

    # sort with -1/padding mapped to a +inf sentinel so invalids sink to the end
    vals = jnp.where(flat < 0, _SENTINEL, flat)
    svals = jnp.sort(vals, axis=1)
    is_new = jnp.concatenate([jnp.ones((n_blk, 1), bool), svals[:, 1:] != svals[:, :-1]], axis=1)
    uniq = is_new & (svals != _SENTINEL)
    blk_counts = uniq.sum(axis=1).astype(jnp.int32)

    # compact: scatter each first-occurrence to its rank; overflow ranks drop
    rank = jnp.cumsum(uniq, axis=1) - 1
    rank = jnp.where(uniq, rank, u_max)  # non-unique/sentinel -> out of range
    rows = jnp.broadcast_to(jnp.arange(n_blk, dtype=jnp.int32)[:, None], rank.shape)
    blk_units = jnp.full((n_blk, u_max), -1, jnp.int32)
    blk_units = blk_units.at[rows, rank].set(svals, mode="drop")

    # membership: binary-search every (query, k) selection in the compacted
    # table (valid prefix ascending; -1 pad remapped to the sentinel so the
    # array is globally sorted). O(T*K*log u_max) and no [.., K, u_max]
    # broadcast materialisation.
    search_tbl = jnp.where(blk_units < 0, _SENTINEL, blk_units)
    pos = jax.vmap(jnp.searchsorted)(search_tbl, vals)  # [n_blk, qb*K]
    pos_c = jnp.minimum(pos, u_max - 1)
    hit = (
        (vals != _SENTINEL)
        & (pos < u_max)
        & (jnp.take_along_axis(search_tbl, pos_c, axis=1) == vals)
    )
    b_idx = jnp.broadcast_to(jnp.arange(n_blk, dtype=jnp.int32)[:, None, None], (n_blk, qb, K))
    q_idx = jnp.broadcast_to(jnp.arange(qb, dtype=jnp.int32)[None, :, None], (n_blk, qb, K))
    u_idx = jnp.where(hit, pos, u_max).reshape(n_blk, qb, K)  # miss -> dropped
    blk_member = jnp.zeros((n_blk, qb, u_max), jnp.int8)
    blk_member = blk_member.at[b_idx, q_idx, u_idx].set(1, mode="drop")
    return blk_units, blk_member, blk_counts
