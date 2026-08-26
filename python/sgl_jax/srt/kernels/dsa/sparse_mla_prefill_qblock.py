"""Blocked (query-batched) sparse-MLA prefill — P1 of the DSA attend path.

The deployed per-query kernel (``sparse_mla_prefill.py``) runs one program per
query: each of the ``S`` queries independently DMAs its ``K`` selected pages. At
long context the selections overlap heavily (sinks + local windows), so the same
physical page is re-fetched by thousands of programs — ~298x redundant HBM
traffic per layer at 110k — and the score matmul feeds only ``H`` (~4) sublane
rows, idling the MXU.

The blocked kernel amortises both: ``QB`` queries share one program, the block's
selected-page **union** is DMA'd once per page, and per-query selection is
restored with a ``-inf`` membership bias (bitmap AND causal) inside a
flash-softmax over ``QB*H`` score rows.

This module hosts:

1. ``build_block_unit_tables`` — jnp preprocessing (outside the kernel): turn
   ``topk_units [T, K]`` into per-block union lists + a **by-unit-id** membership
   bitmap. Deliberately scatter-free: an earlier compacted-slot bitmap built with
   a 3D ``.at[].set`` scatter cost ~25 ms/layer on TPU at the 110k shape; the
   broadcast-compare + sort form is ~0.6 ms.
2. the Pallas query-block kernel (v1: single-sequence flat KV) with an
   ``NBUF``-deep ring of unit DMAs — the per-unit fetches are latency-bound
   (~1.5 µs each), so the next units are prefetched while the current one is
   scored.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SENTINEL = jnp.iinfo(jnp.int32).max

_NBUF = 4  # DMA ring depth (prefetch distance)


def build_block_unit_tables(
    topk_units,  # [T, K] int32 selected unit ids per query, -1 padded
    *,
    query_block: int,  # QB: queries per block
    num_units: int,  # total unit-id space (ids are in [0, num_units) or -1)
    u_max: int,  # static cap on the per-block union size
):
    """Per-query-block union + membership tables for the blocked kernel.

    Returns ``(blk_units, blk_member, blk_counts)``:

    * ``blk_units [nQB, u_max]`` int32 — the block's selected units, sorted
      ascending, ``-1`` padded. Uniques beyond ``u_max`` are dropped.
    * ``blk_member [nQB, query_block, num_units]`` int8 — 1 where query ``q`` of
      the block selected unit id ``p`` (indexed **by unit id**, not by union
      slot — the kernel's -inf bias source).
    * ``blk_counts [nQB]`` int32 — the TRUE union size, **uncapped**: a value
      ``> u_max`` means ``blk_units`` is incomplete and the caller must gate
      (pick ``u_max >= min(query_block*K, num_units)`` to make overflow
      impossible). ``blk_member`` is by-id and unaffected by overflow.

    ``T`` need not divide ``query_block``; trailing queries are padded with
    ``-1`` rows (all-zero membership). Callers with packed multi-request
    (ragged) inputs lift seq-local ids to global keys (e.g. ``base + page``)
    before calling, so blocks may straddle requests.
    """
    T, K = topk_units.shape
    qb = query_block
    n_blk = -(-T // qb)
    pad = n_blk * qb - T
    tk = jnp.pad(topk_units.astype(jnp.int32), ((0, pad), (0, 0)), constant_values=-1)
    tk = tk.reshape(n_blk, qb, K)

    # membership by unit id: broadcast compare + reduce over K (fuses on TPU;
    # negative/padded selections match nothing).
    ids = jnp.arange(num_units, dtype=jnp.int32)
    sel = (tk[:, :, :, None] == ids[None, None, None, :]).any(axis=2)  # [nQB,qb,NU]
    blk_member = sel.astype(jnp.int8)

    # union list: presence per block, compacted by sorting masked unit ids
    present = sel.any(axis=1)  # [n_blk, NU]
    blk_counts = present.sum(axis=1).astype(jnp.int32)
    masked = jnp.where(present, ids[None, :], _SENTINEL)
    srt = jax.lax.sort(masked, dimension=1)
    srt = (
        srt[:, :u_max]
        if u_max < num_units
        else jnp.pad(srt, ((0, 0), (0, u_max - num_units)), constant_values=_SENTINEL)
    )
    blk_units = jnp.where(srt == _SENTINEL, -1, srt)
    return blk_units, blk_member, blk_counts


def _qblock_kernel(
    q_ref,  # [1, 1, QBHp, Dk_pad] VMEM  (q*H+h rows; zero-padded)
    units_ref,  # [1, 1, 1, U_pad]  SMEM  block union unit ids (-1 pad)
    cnt_ref,  # [1, 1, 1, 1]        SMEM  block union size (loop bound)
    qpos_ref,  # [1, 1, 1, QBHp]    VMEM  per-row query position (-1 on pad rows)
    mem_ref,  # [1, 1, NU_pad, QBHp] VMEM int8 membership, indexed by unit id
    kv_hbm,  # [B, T(+RBF), Dk_pad] HBM   flat latent (DMA-gathered)
    o_ref,  # [1, 1, QBHp, Dv]
    kv_scratch,  # [NBUF, RBF, Dk_pad] VMEM  DMA ring
    sem,  # DMA semaphores (NBUF,)
    *,
    sm_scale: float,
    Dv: int,
    RB: int,
    RBF: int,
):
    """Query-block sparse-MLA kernel (flat KV), ring-prefetched.

    Score layout is **key-major**: the unit's ``RBF`` keys sit on the sublane
    axis and the block's ``QB*H`` (query, head) rows sit on the *lane* axis
    (``s = kv · qᵀ -> [RBF, QBHp]``). That orientation lets the per-unit
    membership bias come from an aligned-window sublane read of ``mem_ref``
    (no lane-axis dynamic slicing anywhere), and the flash-softmax state
    (``m``/``l``) lives as plain lane vectors.

    Unit fetches are pipelined through an ``NBUF``-slot VMEM ring: iteration
    ``j`` waits on slot ``j % NBUF`` while slots for ``j+1 .. j+NBUF-1`` are in
    flight — without this the loop is bound by per-DMA latency (~1.5 µs), not
    bandwidth.
    """
    b = pl.program_id(0)
    QBHp = q_ref.shape[2]
    q = q_ref[0, 0]  # [QBHp, Dk_pad]
    cnt = cnt_ref[0, 0, 0, 0]
    qpos_row = qpos_ref[0, 0]  # [1, QBHp] int32
    rows = jax.lax.broadcasted_iota(jnp.int32, (RBF, 1), 0)  # key row within unit

    def _copy(j, slot):
        u = jnp.maximum(units_ref[0, 0, 0, j], 0)
        return pltpu.make_async_copy(
            kv_hbm.at[b, pl.ds(u * RB, RBF), :], kv_scratch.at[slot], sem.at[slot]
        )

    # prologue: fill the ring (d=d: bind the loop var per iteration, B023)
    for d in range(_NBUF - 1):

        @pl.when(d < cnt)
        def _(d=d):
            _copy(d, d).start()

    m0 = jnp.full((QBHp,), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((QBHp,), dtype=jnp.float32)
    acc0 = jnp.zeros((QBHp, Dv), dtype=jnp.float32)

    def unit_body(j, carry):
        m_i, l_i, acc = carry
        u = units_ref[0, 0, 0, j]  # scalar unit id (>= 0 for j < cnt)
        slot = jax.lax.rem(j, _NBUF)

        # keep the ring full: issue the fetch NBUF-1 ahead
        nxt = j + _NBUF - 1

        @pl.when(nxt < cnt)
        def _():
            _copy(nxt, jax.lax.rem(nxt, _NBUF)).start()

        # membership row for this unit id. A direct mem_ref[.., u, :] is not
        # Mosaic-lowerable (int8 tile is (32, 128): a dynamic sublane index
        # must be provably %32), so read the aligned 32-row window containing
        # u — (u // 32) * 32 is provably aligned, the same idiom as the paged
        # r8*8 trick — and select row u within it via an iota compare + max.
        u32 = (u // 32) * 32
        mem_win = mem_ref[0, 0, pl.ds(u32, 32), :]  # [32, QBHp] int8
        rowsel = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) == (u - u32)
        mem_row = jnp.max(
            jnp.where(rowsel, mem_win.astype(jnp.int32), 0), axis=0, keepdims=True
        )  # [1, QBHp]
        kp = u * RB + rows  # [RBF, 1] key positions
        valid = (mem_row > 0) & (kp <= qpos_row)  # [RBF, QBHp]
        if RBF > RB:
            valid &= rows < RB  # drop sublane over-fetch rows
        bias = jnp.where(valid, 0.0, -jnp.inf)  # [RBF, QBHp] fp32

        _copy(j, slot).wait()
        kv_blk = kv_scratch[slot]  # [RBF, Dk_pad]

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
    blk_u, blk_m, blk_c = jax.vmap(
        functools.partial(build_block_unit_tables, query_block=QB, num_units=num_units, u_max=u_max)
    )(indices)
    # blk_u [B,nQB,u_max] blk_m [B,nQB,QB,NU] blk_c [B,nQB]
    nQB = blk_u.shape[1]
    U_pad = ((u_max + 31) // 32) * 32
    NU_pad = ((num_units + 31) // 32) * 32  # int8 sublane tile
    QBH = QB * H
    QBHp = ((QBH + 127) // 128) * 128  # lane axis of the score

    # membership: -> id-major [B,nQB,NU_pad,QBHp] int8, lanes H-expanded so the
    # kernel's lane r = q*H + h reads member[q] directly.
    memt = jnp.repeat(blk_m.transpose(0, 1, 3, 2), H, axis=3)  # [B,nQB,NU,QBH]
    memt = jnp.pad(memt, ((0, 0), (0, 0), (0, NU_pad - num_units), (0, QBHp - QBH)))
    units4 = jnp.pad(blk_u, ((0, 0), (0, 0), (0, U_pad - u_max)), constant_values=-1)
    units4 = units4.reshape(B, nQB, 1, U_pad)
    # clamp: if a block overflowed u_max (impossible with the default cap), only
    # the retained prefix of the union is walked.
    counts4 = jnp.minimum(blk_c, u_max).reshape(B, nQB, 1, 1)

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
            pl.BlockSpec((1, 1, NU_pad, QBHp), lambda b, n: (b, n, 0, 0)),  # membership
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv (untiled, DMA-gathered)
        ],
        out_specs=pl.BlockSpec((1, 1, QBHp, Dv), lambda b, n: (b, n, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, nQB, QBHp, Dv), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((_NBUF, RBF, Dk_pad), kv.dtype),
            pltpu.SemaphoreType.DMA((_NBUF,)),
        ],
        interpret=interpret,
    )(q4, units4, counts4, pos_rows, memt, kv)

    out = out[:, :, :QBH, :].reshape(B, Sp, H, Dv)
    return out[:, :S]
