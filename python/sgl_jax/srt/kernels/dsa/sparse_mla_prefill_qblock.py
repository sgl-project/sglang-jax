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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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


def _qblock_kernel(
    q_ref,  # [1, 1, QBHp, Dk_pad] VMEM  (q*H+h rows; zero-padded)
    units_ref,  # [1, 1, 1, U_pad]  SMEM  block union unit ids (-1 pad)
    cnt_ref,  # [1, 1, 1, 1]        SMEM  block union size (loop bound)
    qpos_ref,  # [1, 1, 1, QBHp]    VMEM  per-row query position (-1 on pad rows)
    mem_ref,  # [1, 1, U_pad, QBHp] VMEM  int8 membership, unit-major
    kv_hbm,  # [B, T(+RBF), Dk_pad] HBM   flat latent (DMA-gathered)
    o_ref,  # [1, 1, QBHp, Dv]
    kv_scratch,  # [RBF, Dk_pad] VMEM     one gathered unit
    sem,  # DMA semaphore
    *,
    sm_scale: float,
    Dv: int,
    RB: int,
    RBF: int,
):
    """Query-block sparse-MLA kernel, v1 (flat single-buffer form).

    Score layout is **key-major**: the unit's ``RBF`` keys sit on the sublane
    axis and the block's ``QB*H`` (query, head) rows sit on the *lane* axis
    (``s = kv · qᵀ -> [RBF, QBHp]``). That orientation lets the per-unit
    membership bias come straight from a dynamic **sublane** row read of
    ``mem_ref`` (lane-axis dynamic slicing is not needed anywhere), and the
    flash-softmax state (``m``/``l``) lives as plain lane vectors.
    """
    b = pl.program_id(0)
    QBHp = q_ref.shape[2]
    q = q_ref[0, 0]  # [QBHp, Dk_pad]
    cnt = cnt_ref[0, 0, 0, 0]
    qpos_row = qpos_ref[0, 0]  # [1, QBHp] int32
    rows = jax.lax.broadcasted_iota(jnp.int32, (RBF, 1), 0)  # key row within unit

    m0 = jnp.full((QBHp,), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((QBHp,), dtype=jnp.float32)
    acc0 = jnp.zeros((QBHp, Dv), dtype=jnp.float32)

    def unit_body(j, carry):
        m_i, l_i, acc = carry
        u = units_ref[0, 0, 0, j]  # scalar unit id (>= 0 for j < cnt)
        pltpu.make_async_copy(kv_hbm.at[b, pl.ds(u * RB, RBF), :], kv_scratch.at[...], sem).start()

        # membership row for this unit: dynamic sublane read -> [QBHp] lanes
        mem_row = mem_ref[0, 0, j, :]
        kp = u * RB + rows  # [RBF, 1] key positions
        valid = (mem_row[None, :] > 0) & (kp <= qpos_row)  # [RBF, QBHp]
        if RBF > RB:
            valid &= rows < RB  # drop sublane over-fetch rows
        bias = jnp.where(valid, 0.0, -jnp.inf)  # [RBF, QBHp] fp32

        pltpu.make_async_copy(kv_hbm.at[b, pl.ds(u * RB, RBF), :], kv_scratch.at[...], sem).wait()
        kv_blk = kv_scratch[...]  # [RBF, Dk_pad]

        # score: [RBF,Dk]·[QBHp,Dk] -> [RBF, QBHp] (keys sublane, queries lane)
        s = (
            jax.lax.dot_general(
                kv_blk, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            * sm_scale
        )
        s = s + bias

        # flash-softmax over keys = axis 0 (the sublane axis)
        m_cur = jnp.max(s, axis=0)  # [QBHp]
        m_new = jnp.maximum(m_i, m_cur)
        corr = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        p = jnp.exp(s - corr[None, :])  # [RBF, QBHp]
        alpha = jnp.where(jnp.isneginf(m_new), 1.0, jnp.exp(m_i - m_new))
        l_new = l_i * alpha + jnp.sum(p, axis=0)
        v_blk = kv_blk[:, :Dv]  # [RBF, Dv]
        # acc[qh, dv] += sum_r p[r, qh] * v[r, dv]  (contract the key axis)
        acc = acc * alpha[:, None] + jax.lax.dot_general(
            p.astype(kv_blk.dtype),
            v_blk,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return (m_new, l_new, acc)

    m_i, l_i, acc = jax.lax.fori_loop(0, cnt, unit_body, (m0, l0, acc0))
    out = acc / jnp.where(l_i == 0.0, 1.0, l_i)[:, None]
    o_ref[0, 0] = out.astype(o_ref.dtype)


def sparse_mla_attention_qblock(
    q,  # [B, S, H, Dk]        Dk = kv_lora_rank + qk_rope_head_dim
    kv,  # [B, T, Dk]           flat MLA latent cache (value = first Dv cols)
    indices,  # [B, S, K] int32  selected unit ids (unit == read_block tokens)
    positions,  # [B, S] int32   query positions (causal bound)
    *,
    kv_lora_rank: int = 512,
    read_block: int = 128,
    query_block: int = 64,
    u_max: int | None = None,  # per-block union cap; default makes overflow impossible
    sm_scale: float,
    interpret: bool = False,
):
    """Blocked sparse MLA-latent attention (query-batching kernel, v1: flat KV).

    Semantics match :func:`sparse_mla_prefill.sparse_mla_attention` (same inputs,
    same masked-softmax math); the difference is purely execution shape: queries
    are processed ``query_block`` at a time and each block DMAs its selected-page
    *union* once instead of per query. Returns ``[B, S, H, kv_lora_rank]`` fp32.
    """
    B, S, H, Dk = q.shape
    K = indices.shape[2]
    Dv = kv_lora_rank
    RB = read_block
    QB = query_block
    T = kv.shape[1]
    num_units = -(-T // RB)
    if u_max is None:
        u_max = min(QB * K, num_units)

    # ── preprocessing (jnp, outside the kernel) ────────────────────────────
    blk_u, bm, bc = jax.vmap(
        functools.partial(build_block_unit_tables, query_block=QB, u_max=u_max)
    )(indices)
    # blk_u [B,nQB,u_max] bm [B,nQB,QB,u_max] bc [B,nQB]
    nQB = blk_u.shape[1]
    U_pad = ((u_max + 31) // 32) * 32  # int8 sublane tile
    QBH = QB * H
    QBHp = ((QBH + 127) // 128) * 128  # lane axis of the score

    # membership: -> unit-major [B,nQB,U_pad,QBHp] int8, rows H-expanded so the
    # kernel's lane r = q*H + h reads member[q] directly.
    memt = jnp.repeat(bm.transpose(0, 1, 3, 2), H, axis=3)  # [B,nQB,u_max,QBH]
    memt = jnp.pad(memt, ((0, 0), (0, 0), (0, U_pad - u_max), (0, QBHp - QBH)))
    units4 = jnp.pad(blk_u, ((0, 0), (0, 0), (0, U_pad - u_max)), constant_values=-1)
    units4 = units4.reshape(B, nQB, 1, U_pad)
    counts4 = bc.reshape(B, nQB, 1, 1)

    # q -> [B, nQB, QBHp, Dk_pad] (row = q*H + h), zero pad rows/features
    Dk_pad = ((Dk + 127) // 128) * 128
    Sp = nQB * QB
    q4 = jnp.pad(q, ((0, 0), (0, Sp - S), (0, 0), (0, Dk_pad - Dk)))
    q4 = q4.reshape(B, nQB, QBH, Dk_pad)
    q4 = jnp.pad(q4, ((0, 0), (0, 0), (0, QBHp - QBH), (0, 0)))

    # per-row positions (-1 on padded rows => nothing valid => zero output row)
    pos_p = jnp.pad(positions.astype(jnp.int32), ((0, 0), (0, Sp - S)), constant_values=-1)
    pos_rows = jnp.repeat(pos_p.reshape(B, nQB, QB), H, axis=2)  # [B,nQB,QBH]
    pos_rows = jnp.pad(pos_rows, ((0, 0), (0, 0), (0, QBHp - QBH)), constant_values=-1)
    pos_rows = pos_rows.reshape(B, nQB, 1, QBHp)

    # flat KV: pad features to Dk_pad and rows by RBF (tail-unit over-fetch guard)
    RBF = ((RB + 15) // 16) * 16
    kv = jnp.pad(kv, ((0, 0), (0, RBF), (0, Dk_pad - Dk)))

    kernel = functools.partial(_qblock_kernel, sm_scale=sm_scale, Dv=Dv, RB=RB, RBF=RBF)
    smem = pltpu.SMEM
    out = pl.pallas_call(
        kernel,
        grid=(B, nQB),
        in_specs=[
            pl.BlockSpec((1, 1, QBHp, Dk_pad), lambda b, n: (b, n, 0, 0)),  # q
            pl.BlockSpec((1, 1, 1, U_pad), lambda b, n: (b, n, 0, 0), memory_space=smem),
            pl.BlockSpec((1, 1, 1, 1), lambda b, n: (b, n, 0, 0), memory_space=smem),
            pl.BlockSpec((1, 1, 1, QBHp), lambda b, n: (b, n, 0, 0)),  # qpos rows
            pl.BlockSpec((1, 1, U_pad, QBHp), lambda b, n: (b, n, 0, 0)),  # membership
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv (untiled, DMA-gathered)
        ],
        out_specs=pl.BlockSpec((1, 1, QBHp, Dv), lambda b, n: (b, n, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, nQB, QBHp, Dv), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((RBF, Dk_pad), kv.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=interpret,
    )(q4, units4, counts4, pos_rows, memt, kv)

    out = out[:, :, :QBH, :].reshape(B, Sp, H, Dv)
    return out[:, :S]
