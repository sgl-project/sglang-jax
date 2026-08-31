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

Implementation notes (each earned on the TPU):

* Preprocessing is **scatter-free**: a 3D ``.at[b, q, u].set`` membership
  scatter cost ~25 ms/layer at the 110k shape; the broadcast-compare form is
  ~1 ms. The membership bitmap is indexed by **union slot** (not unit id), so
  its footprint is bounded by ``u_max`` — independent of the packed page-table
  width (a by-id bitmap explodes at high concurrency).
* Unit fetches are pipelined through an ``NBUF``-slot VMEM ring: the per-unit
  DMAs are latency-bound (~1.5 µs each), so without prefetch the loop runs at
  DMA latency, not bandwidth (measured 27 ms -> 7.6 ms at the 110k shape).
* Score layout is **key-major** ([RBF keys, QB*H rows]): the membership bias
  comes from an aligned-window sublane read (int8 tile is (32, 128), so a
  dynamic sublane index must be provably %32 — read the ``(j//32)*32`` window
  and select the row with an iota compare), and no lane-axis dynamic slicing
  or in-kernel transpose is needed anywhere.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

logger = logging.getLogger(__name__)

_SENTINEL = jnp.iinfo(jnp.int32).max

_NBUF = 4  # DMA ring depth (prefetch distance)


@functools.cache
def _warn_umax_truncation(u_max: int, u_safe: int) -> None:
    # functools.cache => one warning per distinct (u_max, u_safe), not per trace
    logger.warning(
        "sparse_mla_attention_qblock: u_max=%d < %d (the lossless bound "
        "min(query_block*K, num_units)); blocks whose selected-page union "
        "exceeds the cap silently drop the tail of the union, so outputs may "
        "lose accuracy vs the per-query kernel.",
        u_max,
        u_safe,
    )


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
      block selected ``blk_units[b, u]`` (indexed by union **slot**; the
      kernel's -inf bias source).
    * ``blk_counts [nQB]`` int32 — the TRUE union size, **uncapped**: a value
      ``> u_max`` means the tables are incomplete and the caller must gate
      (pick ``u_max >= min(query_block*K, num_units)`` to make overflow
      impossible).

    ``T`` need not divide ``query_block``; trailing queries are padded with
    ``-1`` rows (all-zero membership). Unit ids are opaque keys: callers with
    packed multi-request (ragged) inputs lift seq-local ids to global keys
    (e.g. ``page_table_base + page``) before calling, so blocks may straddle
    requests.
    """
    T, K = topk_units.shape
    qb = query_block
    n_blk = -(-T // qb)
    pad = n_blk * qb - T
    tk = jnp.pad(topk_units.astype(jnp.int32), ((0, pad), (0, 0)), constant_values=-1)
    tk = tk.reshape(n_blk, qb, K)
    flat = tk.reshape(n_blk, qb * K)

    # union list: sort with -1 mapped to a +inf sentinel, mark first occurrences,
    # compact by rank (2D scatter — cheap, unlike a 3D membership scatter).
    vals = jnp.where(flat < 0, _SENTINEL, flat)
    svals = jnp.sort(vals, axis=1)
    is_new = jnp.concatenate([jnp.ones((n_blk, 1), bool), svals[:, 1:] != svals[:, :-1]], axis=1)
    uniq = is_new & (svals != _SENTINEL)
    blk_counts = uniq.sum(axis=1).astype(jnp.int32)
    rank = jnp.cumsum(uniq, axis=1) - 1
    rank = jnp.where(uniq, rank, u_max)  # non-unique/sentinel -> dropped
    rows = jnp.broadcast_to(jnp.arange(n_blk, dtype=jnp.int32)[:, None], rank.shape)
    blk_units = jnp.full((n_blk, u_max), -1, jnp.int32)
    blk_units = blk_units.at[rows, rank].set(svals, mode="drop")

    # membership by slot: broadcast compare against the compacted union (fuses
    # into a compare+reduce on TPU; -1 pads on either side never match).
    blk_member = (
        ((tk[:, :, :, None] == blk_units[:, None, None, :]) & (tk[:, :, :, None] >= 0))
        .any(axis=2)
        .astype(jnp.int8)
    )
    return blk_units, blk_member, blk_counts


def _qblock_kernel(
    q_ref,  # [1, 1, QBHp, Dk_pad] VMEM  (q*H+h rows; zero-padded)
    units_ref,  # [1, 1, 1, U_pad]  SMEM  block union unit ids (-1 pad)
    cnt_ref,  # [1, 1, 1, 1]        SMEM  block union size (loop bound)
    qpos_ref,  # [1, 1, 1, QBHp]    VMEM  per-row query position (-1 on pad rows)
    kvlen_ref,  # [1, 1, 1, QBHp]   VMEM  per-row kv length bound
    base_ref,  # [1, 1, 1, QBHp]    VMEM  per-row page-table base, in TOKENS
    mem_ref,  # [1, 1, U_pad, QBHp] VMEM  int8 membership by union slot
    kv_hbm,  # flat: [B, T(+RBF), Dk_pad]; paged: [1, num_pages*PS, Dk_pad] HBM
    pt_ref,  # [1, 1, 1, PTW]      SMEM  packed page table (paged only)
    o_ref,  # [1, 1, QBHp, Dv]
    kv_scratch,  # [NBUF, RBF, Dk_pad] VMEM  DMA ring
    sem,  # DMA semaphores (NBUF,)
    *,
    sm_scale: float,
    Dv: int,
    RB: int,
    RBF: int,
    paged: bool,
    PS: int,  # page size (paged only; == RB in v1 paged mode)
    PTW: int,
):
    """Query-block sparse-MLA kernel, ring-prefetched (flat or packed-paged KV).

    In paged mode a unit id is a **global key** = position in the packed
    ``page_indices`` table (the caller lifts seq-local pages by the request's
    base), so the DMA source is ``page_indices[u] * PS`` and the key's
    seq-local position is ``u*RB + r - base_tokens[row]`` — a plain 2D
    broadcast, which is what lets one block hold queries from different
    requests.
    """
    b = pl.program_id(0)
    QBHp = q_ref.shape[2]
    q = q_ref[0, 0]  # [QBHp, Dk_pad]
    cnt = cnt_ref[0, 0, 0, 0]
    qpos_row = qpos_ref[0, 0]  # [1, QBHp] int32
    kvlen_row = kvlen_ref[0, 0]  # [1, QBHp] int32
    base_row = base_ref[0, 0]  # [1, QBHp] int32 (tokens; 0 when flat)
    rows = jax.lax.broadcasted_iota(jnp.int32, (RBF, 1), 0)  # key row within unit

    def _copy(j, slot):
        u = jnp.maximum(units_ref[0, 0, 0, j], 0)
        if paged:
            # whole-page unit: physical row 0 of page pt[u]. The token axis is
            # sublane-tiled (8): express the offset as r8*8 so divisibility is
            # provable (PS % 8 == 0), the same idiom as the dense paged kernel.
            pp = pt_ref[0, 0, 0, jnp.minimum(u, PTW - 1)]
            r8 = pp * (PS // 8)
            src = kv_hbm.at[0, pl.ds(r8 * 8, RBF), :]
        else:
            src = kv_hbm.at[b, pl.ds(u * RB, RBF), :]
        return pltpu.make_async_copy(src, kv_scratch.at[slot], sem.at[slot])

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

        # membership row for this slot: aligned 32-row window + iota select
        # (a direct dynamic sublane index is not provably %32 — Mosaic E2003).
        j32 = (j // 32) * 32
        mem_win = mem_ref[0, 0, pl.ds(j32, 32), :]  # [32, QBHp] int8
        rowsel = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) == (j - j32)
        mem_row = jnp.max(
            jnp.where(rowsel, mem_win.astype(jnp.int32), 0), axis=0, keepdims=True
        )  # [1, QBHp]

        # seq-local key positions per (key row, query row): kp = u*RB + r - base.
        # For queries of a different request than the unit's owner this is
        # garbage, but their membership bit is 0 so the lane is -inf anyway.
        kp = (u * RB + rows) - base_row  # [RBF, QBHp]
        valid = (mem_row > 0) & (kp <= qpos_row) & (kp < kvlen_row)
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
    kv,  # [B, T, Dk] flat latent | packed 4D paged cache (ragged mode)
    indices,  # [B, S, K] int32  selected unit ids (unit == read_block tokens)
    positions,  # [B, S] int32   query positions (causal bound)
    *,
    kv_lora_rank: int = 512,
    read_block: int = 128,
    query_block: int = 64,
    u_max: int | None = None,  # per-block union cap; default makes overflow impossible
    sm_scale: float,
    interpret: bool = False,
    # ── packed-ragged mode (multi-request extend; mirrors sparse_mla_attention) ──
    page_size: int | None = None,
    q_seq_id=None,  # [total_tokens] int32  token -> request id (enables ragged mode)
    seq_lens=None,  # [num_seqs] int32       per-request kv length (causal bound)
    cu_kv_lens=None,  # [num_seqs+1] int32    page-aligned kv offsets
    page_indices=None,  # [total_pages] int32 packed physical page ids
):
    """Blocked sparse MLA-latent attention (query-batching kernel).

    Semantics match :func:`sparse_mla_prefill.sparse_mla_attention` (same
    masked-softmax math); the difference is purely execution shape: queries are
    processed ``query_block`` at a time and each block DMAs its selected-page
    *union* once instead of per query. Returns ``[B, S, H, kv_lora_rank]`` fp32.

    Modes:
    * flat (default): ``kv`` is ``[B, T, Dk]``; ``indices`` are unit ids over it.
    * ragged (``q_seq_id`` set): ``kv`` is the packed 4D paged MLA cache and
      ``indices`` are **seq-local page ids** (as produced by the indexer);
      they are lifted to global page-table keys here. Requires
      ``read_block == page_size`` (``% 16 == 0``) and ``B == 1``.

    ``u_max`` caps the per-block union table. The default,
    ``min(query_block * K, num_units)``, makes overflow impossible and is
    lossless. An explicitly smaller cap trades accuracy for preprocessing
    footprint: a block whose true union exceeds it has the tail of its sorted
    union silently dropped (those keys are never attended), so outputs may
    diverge from :func:`sparse_mla_prefill.sparse_mla_attention`. A one-time
    warning is logged when such a cap is in effect.
    """
    B, S, H, Dk = q.shape
    K = indices.shape[2]
    Dv = kv_lora_rank
    RB = read_block
    QB = query_block
    ragged = q_seq_id is not None
    if ragged:
        if B != 1:
            raise ValueError(f"ragged mode packs all requests into B=1 (got B={B})")
        if seq_lens is None or cu_kv_lens is None or page_indices is None:
            raise ValueError("ragged mode requires seq_lens, cu_kv_lens and page_indices")
        if page_size is None or page_size != RB:
            raise ValueError("qblock ragged mode requires read_block == page_size")
        if RB % 16 != 0:
            # same contract as the per-query paged kernel: _copy resolves the
            # page start as (PS//8)*8 and reads RBF = align_up(RB, 16) rows, so
            # a non-%16 page size would offset wrong / cross a page silently.
            raise ValueError(
                f"qblock ragged mode requires read_block % 16 == 0 (got {RB}): "
                "exact PS//8 sublane offset, no cross-page over-fetch"
            )

    Dk_pad = ((Dk + 127) // 128) * 128
    RBF = ((RB + 15) // 16) * 16

    if ragged:
        ps = page_size
        PTW = page_indices.shape[0]
        num_units = PTW
        # lift seq-local pages to global page-table keys; resolve per-row bounds
        qsid = jnp.clip(q_seq_id, 0, seq_lens.shape[0] - 1).astype(jnp.int32)
        base_pages = (cu_kv_lens[qsid] // ps).astype(jnp.int32)  # [S]
        idx_g = jnp.where(indices >= 0, indices + base_pages.reshape(B, S, 1), -1)
        kvlen_tok = seq_lens[qsid].astype(jnp.int32)  # [S]
        base_tok = base_pages * ps  # [S]
        if kv.shape[-1] != Dk_pad:
            raise ValueError(f"paged cache last dim {kv.shape[-1]} != Dk_pad {Dk_pad}")
        num_pages = kv.shape[0]
        kv2 = kv.reshape(1, num_pages * ps, Dk_pad)
        pt_arg = page_indices.reshape(1, 1, 1, PTW).astype(jnp.int32)
    else:
        ps = 0
        PTW = 1
        T = kv.shape[1]
        num_units = -(-T // RB)
        idx_g = indices
        kvlen_tok = jnp.full((S,), T, jnp.int32)
        base_tok = jnp.zeros((S,), jnp.int32)
        # flat KV: pad features to Dk_pad and rows by RBF (tail-unit over-fetch)
        kv2 = jnp.pad(kv, ((0, 0), (0, RBF), (0, Dk_pad - Dk)))
        pt_arg = jnp.zeros((B, 1, 1, 1), jnp.int32)

    u_safe = min(QB * K, num_units)
    if u_max is None:
        u_max = u_safe
    elif u_max < u_safe:
        _warn_umax_truncation(u_max, u_safe)

    # ── preprocessing (jnp, outside the kernel) ────────────────────────────
    blk_u, blk_m, blk_c = jax.vmap(
        functools.partial(build_block_unit_tables, query_block=QB, u_max=u_max)
    )(idx_g)
    # blk_u [B,nQB,u_max] blk_m [B,nQB,QB,u_max] blk_c [B,nQB]
    nQB = blk_u.shape[1]
    U_pad = ((u_max + 31) // 32) * 32  # int8 sublane tile
    QBH = QB * H
    QBHp = ((QBH + 127) // 128) * 128  # lane axis of the score

    # membership -> slot-major [B,nQB,U_pad,QBHp] int8, lanes H-expanded so the
    # kernel's lane r = q*H + h reads member[q] directly.
    memt = jnp.repeat(blk_m.transpose(0, 1, 3, 2), H, axis=3)  # [B,nQB,u_max,QBH]
    memt = jnp.pad(memt, ((0, 0), (0, 0), (0, U_pad - u_max), (0, QBHp - QBH)))
    units4 = jnp.pad(blk_u, ((0, 0), (0, 0), (0, U_pad - u_max)), constant_values=-1)
    units4 = units4.reshape(B, nQB, 1, U_pad)
    # clamp: if a block overflowed u_max (impossible with the default cap), only
    # the retained prefix of the union is walked.
    counts4 = jnp.minimum(blk_c, u_max).reshape(B, nQB, 1, 1)

    # q -> [B, nQB, QBHp, Dk_pad] (row = q*H + h), zero pad rows/features
    Sp = nQB * QB
    q4 = jnp.pad(q, ((0, 0), (0, Sp - S), (0, 0), (0, Dk_pad - Dk)))
    q4 = q4.reshape(B, nQB, QBH, Dk_pad)
    q4 = jnp.pad(q4, ((0, 0), (0, 0), (0, QBHp - QBH), (0, 0)))

    def _rows(vec, fill):
        # [B, S] -> [B, nQB, 1, QBHp] with H-expansion and `fill` on pad rows
        v = jnp.pad(vec.reshape(B, S), ((0, 0), (0, Sp - S)), constant_values=fill)
        v = jnp.repeat(v.reshape(B, nQB, QB), H, axis=2)
        v = jnp.pad(v, ((0, 0), (0, 0), (0, QBHp - QBH)), constant_values=fill)
        return v.reshape(B, nQB, 1, QBHp)

    # pad rows: qpos=-1 => no key can satisfy kp <= qpos => zero output row
    pos_rows = _rows(positions.astype(jnp.int32), -1)
    kvlen_rows = _rows(jnp.broadcast_to(kvlen_tok, (B, S)), 0)
    base_rows = _rows(jnp.broadcast_to(base_tok, (B, S)), 0)

    kernel = functools.partial(
        _qblock_kernel,
        sm_scale=sm_scale,
        Dv=Dv,
        RB=RB,
        RBF=RBF,
        paged=ragged,
        PS=ps,
        PTW=PTW,
    )
    smem = pltpu.SMEM
    row_spec = pl.BlockSpec((1, 1, 1, QBHp), lambda b, n: (b, n, 0, 0))
    out = pl.pallas_call(
        kernel,
        grid=(B, nQB),
        in_specs=[
            pl.BlockSpec((1, 1, QBHp, Dk_pad), lambda b, n: (b, n, 0, 0)),  # q
            pl.BlockSpec((1, 1, 1, U_pad), lambda b, n: (b, n, 0, 0), memory_space=smem),
            pl.BlockSpec((1, 1, 1, 1), lambda b, n: (b, n, 0, 0), memory_space=smem),
            row_spec,  # qpos rows
            row_spec,  # kvlen rows
            row_spec,  # base rows (tokens)
            pl.BlockSpec((1, 1, U_pad, QBHp), lambda b, n: (b, n, 0, 0)),  # membership
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv (untiled, DMA-gathered)
            pl.BlockSpec((1, 1, 1, PTW), lambda b, n: (b, 0, 0, 0), memory_space=smem),
        ],
        out_specs=pl.BlockSpec((1, 1, QBHp, Dv), lambda b, n: (b, n, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, nQB, QBHp, Dv), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((_NBUF, RBF, Dk_pad), kv2.dtype),
            pltpu.SemaphoreType.DMA((_NBUF,)),
        ],
        interpret=interpret,
    )(q4, units4, counts4, pos_rows, kvlen_rows, base_rows, memt, kv2, pt_arg)

    out = out[:, :, :QBH, :].reshape(B, Sp, H, Dv)
    return out[:, :S]


def prefill_write_and_attend_ragged_qblock(
    ql,  # [total_tokens, H, kv_lora_rank]   absorbed latent query (nope)
    qpe,  # [total_tokens, H, rope]           rope query
    kvc,  # [total_tokens, kv_lora_rank]      new c_kv to write
    kpe,  # [total_tokens, rope]              new k_rope to write
    cache,  # [P, ps//pk, pk, Dk_pad]         paged fused latent cache
    topk_pages,  # [total_tokens, K] int32    seq-local page ids (-1 padded)
    positions,  # [total_tokens] int32        absolute query positions (causal bound)
    loc,  # [total_tokens] int32              physical flat slot per token
    seq_lens,  # [num_seqs] int32             per-request kv length
    cu_q_lens,  # [num_seqs+1] int32          per-request query offsets
    cu_kv_lens,  # [num_seqs+1] int32         page-aligned kv offsets
    page_indices,  # [total_pages] int32      packed physical page ids
    *,
    kv_lora_rank: int,
    page_size: int,
    sm_scale: float,
    query_block: int = 64,
    interpret: bool = False,
):
    """Packed-ragged self-write + **blocked** sparse-MLA prefill.

    Drop-in replacement for
    :func:`sparse_mla_prefill.prefill_write_and_attend_ragged` (same signature
    plus ``query_block``); only the attend execution shape differs.
    """
    T, H, Dv = ql.shape
    rope = qpe.shape[-1]
    ps = page_size
    Pn, pspk, pk, Dk_pad = cache.shape
    S = seq_lens.shape[0]

    q_sparse = jnp.concatenate([ql, qpe], axis=-1)  # [T, H, Dv+rope]

    # self-write: per-token scatter to out_cache_loc (identical to the
    # per-query wrapper; mode="drop"+wrap_negative_indices=False drops -1 pads).
    row = jnp.zeros((T, Dk_pad), cache.dtype)
    row = row.at[:, :Dv].set(kvc.astype(cache.dtype))
    row = row.at[:, Dv : Dv + rope].set(kpe.reshape(T, rope).astype(cache.dtype))
    # Scatter directly into the 4D cache. The flat-view round-trip
    # (reshape [Pn*ps, D] -> scatter -> reshape back) materialises TWO full
    # relayout copies of the cache per call (the 4D pool and the 2D view get
    # different tilings) — measured 13.8% of 110k prefill device time. 3D
    # index math on ``loc`` is free; padded loc == -1 stays dropped (negative
    # page index with wrap_negative_indices=False).
    cache_new = paged_write_back(
        cache,
        row,
        loc.astype(jnp.int32),
        page_size=ps,
        r_cap=T // ps + S + 34,
        interpret=interpret,
    )

    # token -> request id (same convention as the per-query ragged wrapper)
    t = jnp.arange(T, dtype=jnp.int32)
    q_seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, S - 1).astype(
        jnp.int32
    )

    out = sparse_mla_attention_qblock(
        q_sparse.reshape(1, T, H, q_sparse.shape[2]),
        cache_new,
        topk_pages.reshape(1, T, -1),
        positions.reshape(1, T),
        kv_lora_rank=Dv,
        read_block=ps,
        query_block=query_block,
        sm_scale=float(sm_scale),
        page_size=ps,
        q_seq_id=q_seq_id,
        seq_lens=seq_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        interpret=interpret,
    )
    return out.reshape(T, H, Dv), cache_new


# ── pallas paged write-back (self-write without the XLA scatter) ─────────────
#
# ``flat.at[loc].set(row)`` (and its 4D-index form) makes XLA canonicalise the
# scatter through a flat view of the pool: two full-cache relayout copies per
# call (the 4D pool and the 2D view tile differently) — measured 13.8% of 110k
# prefill device time, plus the scatter itself (~4%). This kernel writes the
# rows in place instead: the pool is input/output-aliased, and the new rows are
# DMA'd HBM->HBM in maximal contiguous runs (out_cache_loc is page-contiguous,
# so a chunked single-request prefill is ONE run; ragged batches get ~one run
# per page). kv_packing word alignment is handled by splitting each run into
# head-tokens / aligned-words / tail-tokens entries at trace time.


def _build_write_runs(loc, *, kv_packing: int, r_cap: int):
    """Run table for :func:`paged_write_back`.

    Returns ``(table [3*r_cap, 4] int32, n_raw_runs)``; entry = (kind, src,
    dst, n) where kind 0 = word run (src/dst in kv_packing-word indices) and
    kind 1 = token run (src/dst in token indices). Entries with n == 0 are
    no-ops. ``n_raw_runs > r_cap`` means the table overflowed and the caller
    must fall back.
    """
    T = loc.shape[0]
    pk = kv_packing
    R = r_cap
    loc = loc.astype(jnp.int32)
    valid = loc >= 0
    contig = jnp.concatenate(
        [jnp.zeros((1,), bool), (loc[1:] == loc[:-1] + 1) & valid[1:] & valid[:-1]]
    )
    is_start = valid & ~contig
    rid = jnp.cumsum(is_start) - 1  # run id per token (where valid)
    n_raw = is_start.sum().astype(jnp.int32)

    t_idx = jnp.arange(T, dtype=jnp.int32)
    sidx = jnp.where(is_start, rid, R)
    src0 = jnp.zeros((R,), jnp.int32).at[sidx].set(t_idx, mode="drop")
    dst0 = jnp.zeros((R,), jnp.int32).at[sidx].set(loc, mode="drop")
    lens = jnp.zeros((R,), jnp.int32).at[jnp.where(valid, rid, R)].add(1, mode="drop")

    # split by word phase: aligned middle goes as one word run, edges per token
    phase_ok = (src0 % pk) == (dst0 % pk)
    head = jnp.where(phase_ok, (-dst0) % pk, 0)
    head = jnp.minimum(head, lens)
    words = jnp.where(phase_ok, (lens - head) // pk, 0)
    tail = lens - head - words * pk

    zero = jnp.zeros((R,), jnp.int32)
    one = jnp.ones((R,), jnp.int32)
    e_kind = jnp.stack([one, zero, one], axis=1)  # [R, 3]
    e_src = jnp.stack([src0, (src0 + head) // pk, src0 + head + words * pk], axis=1)
    e_dst = jnp.stack([dst0, (dst0 + head) // pk, dst0 + head + words * pk], axis=1)
    e_n = jnp.stack([head, words, tail], axis=1)
    table = jnp.stack([e_kind, e_src, e_dst, e_n], axis=2).reshape(3 * R, 4)
    return table, n_raw


def _write_back_kernel(tbl_ref, row_ref, cache_in_ref, out_ref, wdst_ref, wsrc_ref, sem):
    del cache_in_ref  # aliased with out_ref; all access goes through out_ref
    Pn, pspk, pk, D = out_ref.shape
    # word view: dim0 (words) is untiled, so dynamic word offsets are legal.
    # A flat [Pn*ps, D] token view is NOT: it inherits the (pk, ...) pairing
    # tile on dim0 and single-token slices at odd offsets fail Mosaic
    # alignment. Edge tokens therefore go through a word-granular RMW below.
    cache_w = out_ref.reshape(Pn * pspk, pk, D)
    E = tbl_ref.shape[0]

    def entry(e, carry):
        kind = tbl_ref[e, 0]
        src = tbl_ref[e, 1]
        dst = tbl_ref[e, 2]
        n = tbl_ref[e, 3]

        @pl.when((kind == 0) & (n > 0))
        def _():
            cp = pltpu.make_async_copy(row_ref.at[pl.ds(src, n)], cache_w.at[pl.ds(dst, n)], sem)
            cp.start()
            cp.wait()

        @pl.when((kind == 1) & (n > 0))
        def _():
            # per-token read-modify-write at word granularity: fetch the dst
            # word (preserving the neighbour token), overwrite one row via
            # statically-unrolled phase branches (pk is static), write back.
            def tok(i, c):
                st = src + i
                dt = dst + i
                sw = st // pk
                dw = dt // pk
                cpa = pltpu.make_async_copy(cache_w.at[pl.ds(dw, 1)], wdst_ref, sem)
                cpa.start()
                cpa.wait()
                cpb = pltpu.make_async_copy(row_ref.at[pl.ds(sw, 1)], wsrc_ref, sem)
                cpb.start()
                cpb.wait()
                for spv in range(pk):
                    for dpv in range(pk):

                        @pl.when((st % pk == spv) & (dt % pk == dpv))
                        def _(spv=spv, dpv=dpv):
                            wdst_ref[0, dpv, :] = wsrc_ref[0, spv, :]

                cpc = pltpu.make_async_copy(wdst_ref, cache_w.at[pl.ds(dw, 1)], sem)
                cpc.start()
                cpc.wait()
                return c

            jax.lax.fori_loop(0, n, tok, 0)

        return carry

    jax.lax.fori_loop(0, E, entry, 0)


def paged_write_back(
    cache,  # [Pn, ps//pk, pk, D] paged pool
    row,  # [T, D] new rows (already cache dtype / padded feature dim)
    loc,  # [T] int32 flat slot per token; -1 = dropped
    *,
    page_size: int,
    r_cap: int | None = None,
    interpret: bool = False,
):
    """In-place paged self-write: ``cache[loc[t]] = row[t]`` for ``loc >= 0``.

    Bit-identical to ``cache.reshape(-1, D).at[loc].set(row, mode="drop",
    wrap_negative_indices=False)`` but without the scatter's flat-view
    relayout round-trip. Falls back to that scatter when the run table
    overflows ``r_cap`` (pathologically fragmented ``loc``).
    """
    Pn, pspk, pk, D = cache.shape
    ps = page_size
    assert pspk * pk == ps, (cache.shape, page_size)
    T = row.shape[0]
    Tp = -(-T // pk) * pk
    if Tp != T:
        row = jnp.pad(row, ((0, Tp - T), (0, 0)))
        loc = jnp.pad(loc, ((0, Tp - T),), constant_values=-1)
    if r_cap is None:
        r_cap = 2 * (Tp // ps) + 130
    table, n_raw = _build_write_runs(loc, kv_packing=pk, r_cap=r_cap)
    row_w = row.reshape(Tp // pk, pk, D)

    def _pallas(cache, row_w, table):
        return pl.pallas_call(
            _write_back_kernel,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),  # run table
                pl.BlockSpec(memory_space=pltpu.HBM),  # row (word view)
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache (aliased)
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            scratch_shapes=[
                pltpu.VMEM((1, pk, D), cache.dtype),
                pltpu.VMEM((1, pk, D), cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
            input_output_aliases={2: 0},
            interpret=interpret,
        )(table, row_w, cache)

    def _scatter(cache, row_w, table):
        del table
        flat = cache.reshape(Pn * ps, D)
        flat = flat.at[loc].set(row_w.reshape(Tp, D), mode="drop", wrap_negative_indices=False)
        return flat.reshape(cache.shape)

    if interpret:
        # interpret cannot lower dynamic-size DMAs; the scatter is the
        # bit-identical reference semantics anyway.
        return _scatter(cache, row_w, table)
    return jax.lax.cond(n_raw <= r_cap, _pallas, _scatter, cache, row_w, table)
