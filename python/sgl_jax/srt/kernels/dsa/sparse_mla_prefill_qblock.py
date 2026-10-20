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


def _attend_sched_tagged(fn):
    """Prefill micro-batch PIN>=7: run the attend pallas_call inside
    ``set_xla_metadata(_scheduling_group_id=...)`` when the model set a group for this window
    (``dsa_sparse_backend.PREFILL_SCHED_GROUP``). Only the attend is tagged -- tagging the
    paged write-back too made the group span write-back -> (copies) -> attend and the TPU
    scheduler rejects groups with such gaps (gp213, 10-03)."""

    def _call(*a, **k):
        from sgl_jax.srt.layers.attention import dsa_sparse_backend as _b

        gid = getattr(_b, "PREFILL_SCHED_GROUP", None)
        if gid is None:
            return fn(*a, **k)
        from jax.experimental import xla_metadata

        with xla_metadata.set_xla_metadata(_scheduling_group_id=int(gid)):
            return fn(*a, **k)

    return _call


logger = logging.getLogger(__name__)

_SENTINEL = jnp.iinfo(jnp.int32).max

# DMA ring depth (prefetch distance). Each unit is one 128-token page (~147 KB
# at Dk_pad=576/640), far below the v7x HBM->VMEM DMA knee (t0*B ~ 1.9 MiB), so
# the loop is latency-bound at this depth: measured 27 -> 7.6 ms per layer going
# from no prefetch to 4 at the 110k shape. VMEM cost is NBUF x PPU x RBF x Dk_pad x 2 B.
_NBUF = 4
# Two independent query-row groups per unit step so the VPU softmax of one group
# overlaps the MXU work of the other (see _qblock_kernel); falls back to one group
# when the padded rows do not split into 128-lane halves.
_ROW_GROUPS = 2
# Pages (128-token units) processed per ring slot / loop step: 8 pages stack to
# M=1024 key rows per score matmul and the flash update runs once per 8 pages.
_PPU = 8
# Keep the flash state (m, l, acc [QBHp, Dv] f32 = 2 MiB at the TSA shape) in VMEM
# scratch refs instead of fori_loop carries: a carry of >= 128 vregs is spilled and
# refilled every iteration by the register allocator (static cost: 45% of the
# step's instructions are spill traffic). Numerics unchanged.
_ACC_SCRATCH = True


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
    kv_hbm,  # flat: [B, T(+RBF), Dk_pad]; paged: 4D pool or [1, Pn*PS, Dk_pad] HBM
    pt_ref,  # [1, 1, 1, PTW]      SMEM  packed page table (paged only)
    o_ref,  # [1, 1, QBHp, Dv]
    kv_scratch,  # [NBUF, PPU*RBF, Dk_pad] VMEM  DMA ring (PPU pages per slot)
    sem,  # DMA semaphores (NBUF, PPU)
    *state_refs,  # (m_ref [1, QBHp] f32, l_ref [1, QBHp] f32, acc_ref [QBHp, Dv] f32) when ACC_SCRATCH
    sm_scale: float,
    Dv: int,
    RB: int,
    RBF: int,
    paged: bool,
    PS: int,  # page size (paged only; == RB in v1 paged mode)
    PTW: int,
    PPU: int = 1,  # pages (units) per ring slot / per loop step
    ACC_SCRATCH: bool = False,
    HEAD_MAJOR: bool = False,  # q_ref/o_ref blocks are [1, H, QB, *]; rows ordered h*QB + q
    ROW_GROUPS: int = 1,  # >1: split the query rows into independent groups per unit step (see unit_body)
):
    if kv_hbm.ndim == 4:
        _pn, _pspk, _pk, _dk = kv_hbm.shape
        kv_hbm = kv_hbm.reshape(1, _pn * _pspk * _pk, _dk)
    """Query-block sparse-MLA kernel, ring-prefetched (flat or packed-paged KV).

    In paged mode a unit id is a **global key** = position in the packed
    ``page_indices`` table (the caller lifts seq-local pages by the request's
    base), so the DMA source is ``page_indices[u] * PS`` and the key's
    seq-local position is ``u*RB + r - base_tokens[row]`` — a plain 2D
    broadcast, which is what lets one block hold queries from different
    requests.

    ``PPU`` > 1 walks the union ``PPU`` units per loop step: the slot holds
    ``PPU`` pages stacked on sublanes, the score matmul sees ``PPU*RBF`` key
    rows (MXU M axis) and the PV contraction is ``PPU*RBF`` deep, and the flash
    update runs once per step. Units beyond ``cnt`` in the last step are
    ``-1``-padded in ``units_ref`` (DMA clamped to unit 0, membership row all
    zero -> -inf bias), so the math is the masked softmax over the same keys.
    """
    b = pl.program_id(0)
    if HEAD_MAJOR:
        # [1, H, QB, Dk_pad] block; merging (H, QB) into rows is layout-preserving when
        # QB % 8 == 0 (the (8,128) tiles of each head's QB rows stack contiguously).
        _, _Hh, _QBq, _Dkp = q_ref.shape
        QBHp = _Hh * _QBq
        q = q_ref[0].reshape(QBHp, _Dkp)  # rows h*QB + q
    else:
        QBHp = q_ref.shape[2]
        q = q_ref[0, 0]  # [QBHp, Dk_pad]
    cnt = cnt_ref[0, 0, 0, 0]
    nsteps = (cnt + PPU - 1) // PPU
    qpos_row = qpos_ref[0, 0]  # [1, QBHp] int32
    kvlen_row = kvlen_ref[0, 0]  # [1, QBHp] int32
    base_row = base_ref[0, 0]  # [1, QBHp] int32 (tokens; 0 when flat)
    rows = jax.lax.broadcasted_iota(jnp.int32, (RBF, 1), 0)  # key row within unit

    def _copy(j, slot, i):
        # page i of step j -> rows [i*RBF, (i+1)*RBF) of ring slot `slot`
        u = jnp.maximum(units_ref[0, 0, 0, j * PPU + i], 0)
        if paged:
            pp = pt_ref[0, 0, 0, jnp.minimum(u, PTW - 1)]
            r8 = pp * (PS // 8)
            src = kv_hbm.at[0, pl.ds(r8 * 8, RBF), :]
        else:
            src = kv_hbm.at[b, pl.ds(u * RB, RBF), :]
        return pltpu.make_async_copy(
            src, kv_scratch.at[slot, pl.ds(i * RBF, RBF), :], sem.at[slot, i]
        )

    for d in range(_NBUF - 1):

        @pl.when(d < nsteps)
        def _(d=d):
            for i in range(PPU):
                _copy(d, d, i).start()

    m0 = jnp.full((QBHp,), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((QBHp,), dtype=jnp.float32)
    acc0 = jnp.zeros((QBHp, Dv), dtype=jnp.float32)
    if ACC_SCRATCH:
        m_ref, l_ref, acc_ref = state_refs
        m_ref[...] = m0[None, :]
        l_ref[...] = l0[None, :]
        acc_ref[...] = acc0

    def _bias(j, i):
        # membership row of union slot j*PPU+i (int8 tile is (32,128): read the
        # aligned 32-row window and select the row with an iota compare)
        jj = j * PPU + i
        u = units_ref[0, 0, 0, jj]  # scalar unit id (>= 0 for jj < cnt, -1 pad)
        j32 = (jj // 32) * 32
        mem_win = mem_ref[0, 0, pl.ds(j32, 32), :]  # [32, QBHp] int8
        rowsel = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) == (jj - j32)
        mem_row = jnp.max(
            jnp.where(rowsel, mem_win.astype(jnp.int32), 0), axis=0, keepdims=True
        )  # [1, QBHp]
        kp = (u * RB + rows) - base_row  # [RBF, QBHp]
        valid = (mem_row > 0) & (kp <= qpos_row) & (kp < kvlen_row)
        if RBF > RB:
            valid &= rows < RB  # drop sublane over-fetch rows
        return jnp.where(valid, 0.0, -jnp.inf)  # [RBF, QBHp] fp32

    def unit_body(j, carry):
        if ACC_SCRATCH:
            m_i = m_ref[0]
            l_i = l_ref[0]
            acc = acc_ref[...]
        else:
            m_i, l_i, acc = carry
        slot = jax.lax.rem(j, _NBUF)
        nxt = j + _NBUF - 1

        @pl.when(nxt < nsteps)
        def _():
            for i in range(PPU):
                _copy(nxt, jax.lax.rem(nxt, _NBUF), i).start()

        for i in range(PPU):
            _copy(j, slot, i).wait()
        kv_blk = kv_scratch[slot]  # [PPU*RBF, Dk_pad]
        v_blk = kv_blk[:, :Dv]  # [PPU*RBF, Dv]

        def _qk_raw(q_rows):
            return jax.lax.dot_general(
                kv_blk, q_rows, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )

        # Issue the score matmul(s) first, build the membership/causal bias afterwards:
        # the bias is ~6.5k int32 vector-unit ops per unit step (compare/and/select over
        # [PPU*RBF, QBHp]) and, written before the DMA wait, ran with the MXU idle
        # (static cost 10-03: longest MXU-idle stretch 5050 of 10280 bundles). Behind
        # the dots it can be packed under the MXU pushes; the values are unchanged.
        s_raw = [_qk_raw(q)] if ROW_GROUPS == 1 else None
        if ROW_GROUPS != 1:
            RG = QBHp // ROW_GROUPS
            groups = [slice(g * RG, (g + 1) * RG) for g in range(ROW_GROUPS)]
            s_raw = [_qk_raw(q[sl]) for sl in groups]
        biases = [_bias(j, i) for i in range(PPU)]
        bias = biases[0] if PPU == 1 else jnp.concatenate(biases, axis=0)  # [PPU*RBF, QBHp]

        def _qk(s_raw_, bias_cols):
            return s_raw_ * sm_scale + bias_cols

        def _softmax_pv(s_, m_prev, l_prev, acc_prev):
            m_cur = jnp.max(s_, axis=0)  # [rows]
            m_new_ = jnp.maximum(m_prev, m_cur)
            corr = jnp.where(jnp.isneginf(m_new_), 0.0, m_new_)
            p_ = jnp.exp(s_ - corr[None, :])  # [PPU*RBF, rows]
            alpha = jnp.where(jnp.isneginf(m_new_), 1.0, jnp.exp(m_prev - m_new_))
            l_new_ = l_prev * alpha + jnp.sum(p_, axis=0)
            acc_new_ = acc_prev * alpha[:, None] + jax.lax.dot_general(
                p_.astype(kv_blk.dtype),
                v_blk,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            return m_new_, l_new_, acc_new_

        if ROW_GROUPS == 1:
            m_new, l_new, acc = _softmax_pv(_qk(s_raw[0], bias), m_i, l_i, acc)
            if ACC_SCRATCH:
                m_ref[0] = m_new
                l_ref[0] = l_new
                acc_ref[...] = acc
                return carry
            return (m_new, l_new, acc)

        # Row groups (flash-MLA style, 10-03): the MXU is in-order and with one group a
        # unit step is QK -> softmax (MXU idle) -> PV in series (static cost: MXU slot
        # 33%, longest MXU-idle stretch half the loop body). Issuing every group's QK
        # before any PV lets one group's VPU softmax overlap the other's matmuls; rows
        # are independent, so the per-row math -- and the result -- is unchanged.
        s_g = [_qk(sr, bias[:, sl]) for sr, sl in zip(s_raw, groups)]
        outs = []
        for sl, s_ in zip(groups, s_g):
            m_new, l_new, acc_new = _softmax_pv(s_, m_i[sl], l_i[sl], acc[sl])
            if ACC_SCRATCH:
                m_ref[0, sl] = m_new
                l_ref[0, sl] = l_new
                acc_ref[sl, :] = acc_new
            else:
                outs.append((m_new, l_new, acc_new))
        if ACC_SCRATCH:
            return carry
        return (
            jnp.concatenate([o[0] for o in outs], axis=0),
            jnp.concatenate([o[1] for o in outs], axis=0),
            jnp.concatenate([o[2] for o in outs], axis=0),
        )

    if ACC_SCRATCH:
        jax.lax.fori_loop(0, nsteps, unit_body, 0)
        m_i, l_i, acc = m_ref[0], l_ref[0], acc_ref[...]
    else:
        m_i, l_i, acc = jax.lax.fori_loop(0, nsteps, unit_body, (m0, l0, acc0))
    out = acc / jnp.where(l_i == 0.0, 1.0, l_i)[:, None]
    if HEAD_MAJOR:
        o_ref[0] = out.astype(o_ref.dtype).reshape(_Hh, _QBq, Dv)
    else:
        o_ref[0, 0] = out.astype(o_ref.dtype)


def sparse_mla_attention_qblock(
    q,  # [B, S, H, Dk]        Dk = kv_lora_rank + qk_rope_head_dim
    kv,  # [B, T, Dk] flat latent | packed 4D paged cache (ragged mode)
    indices,  # [B, S, K] int32  selected unit ids (unit == read_block tokens)
    positions,  # [B, S] int32   query positions (causal bound)
    *,
    kv_lora_rank: int = 512,
    read_block: int = 128,
    query_block: int = 256,
    u_max: int | None = None,  # per-block union cap; default makes overflow impossible
    sm_scale: float,
    interpret: bool = False,
    # ── packed-ragged mode (multi-request extend; mirrors sparse_mla_attention) ──
    page_size: int | None = None,
    q_seq_id=None,  # [total_tokens] int32  token -> request id (enables ragged mode)
    seq_lens=None,  # [num_seqs] int32       per-request kv length (causal bound)
    cu_kv_lens=None,  # [num_seqs+1] int32    page-aligned kv offsets
    page_indices=None,  # [total_pages] int32 packed physical page ids
    out_dtype=jnp.float32,  # output dtype; the kernel normalises in f32 and casts on store
    head_major: bool = False,  # q is [B, H, S, Dk] and the output [B, H, S, Dv] (see below)
):
    """Blocked sparse MLA-latent attention (query-batching kernel).

    ``head_major``: the TSA absorb path produces q with a batched dot whose natural
    output is batch(head)-major, and consumes the output with another batched dot
    that wants the same; with token-major kernel I/O XLA inserts a tile-level relayout
    on each side (gp246 4k trace: 7.95 ms + 4.55 ms per chunk). In head-major mode the
    kernel block for query block ``n`` is the ``[H, QB, Dk_pad]`` slab gathered from
    ``[B, H, nQB, QB, Dk_pad]`` (a bitcast view of ``[B, H, S, Dk_pad]`` when QB % 8 == 0),
    rows are ordered ``h*QB + q`` instead of ``q*H + h`` (the kernel is per-row
    independent, so the math is unchanged), and the output comes back ``[B, H, S, Dv]``.
    Requires ``(QB * H) % 128 == 0`` (no lane padding of the row axis).

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
    if head_major:
        B, H, S, Dk = q.shape
    else:
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
        kv2 = kv  # native 4D pool; flattened inside the kernel (HBM ref view)
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
    # int8 sublane tile; with PPU > 1 the last step may read union slots up to
    # cnt + PPU - 1 (padded -1 units / zero membership rows), keep them in range.
    U_pad = ((u_max + _PPU - 1 + 31) // 32) * 32
    QBH = QB * H
    QBHp = ((QBH + 127) // 128) * 128  # lane axis of the score

    # membership -> slot-major [B,nQB,U_pad,QBHp] int8, lanes H-expanded so the
    # kernel's lane r = q*H + h reads member[q] directly.
    if head_major:
        if QBHp != QBH:
            raise ValueError(f"head_major needs (query_block*H) % 128 == 0, got {QB}*{H}={QBH}")
        # lane r = h*QB + q reads member[q]
        memt = jnp.tile(blk_m.transpose(0, 1, 3, 2), (1, 1, 1, H))  # [B,nQB,u_max,QBH]
    else:
        memt = jnp.repeat(blk_m.transpose(0, 1, 3, 2), H, axis=3)  # [B,nQB,u_max,QBH]
    memt = jnp.pad(memt, ((0, 0), (0, 0), (0, U_pad - u_max), (0, QBHp - QBH)))
    units4 = jnp.pad(blk_u, ((0, 0), (0, 0), (0, U_pad - u_max)), constant_values=-1)
    units4 = units4.reshape(B, nQB, 1, U_pad)
    # clamp: if a block overflowed u_max (impossible with the default cap), only
    # the retained prefix of the union is walked.
    counts4 = jnp.minimum(blk_c, u_max).reshape(B, nQB, 1, 1)

    Sp = nQB * QB
    if head_major:
        # q [B, H, S, Dk] -> [B, H, nQB, QB, Dk_pad]: pad rows/features, split S (bitcast)
        q4 = jnp.pad(q, ((0, 0), (0, 0), (0, Sp - S), (0, Dk_pad - Dk)))
        q4 = q4.reshape(B, H, nQB, QB, Dk_pad)
    else:
        # q -> [B, nQB, QBHp, Dk_pad] (row = q*H + h), zero pad rows/features
        q4 = jnp.pad(q, ((0, 0), (0, Sp - S), (0, 0), (0, Dk_pad - Dk)))
        q4 = q4.reshape(B, nQB, QBH, Dk_pad)
        q4 = jnp.pad(q4, ((0, 0), (0, 0), (0, QBHp - QBH), (0, 0)))

    def _rows(vec, fill):
        # [B, S] -> [B, nQB, 1, QBHp] with H-expansion and `fill` on pad rows
        v = jnp.pad(vec.reshape(B, S), ((0, 0), (0, Sp - S)), constant_values=fill)
        if head_major:
            v = jnp.tile(v.reshape(B, nQB, QB), (1, 1, H))  # lane h*QB + q -> token q
        else:
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
        PPU=_PPU,
        ACC_SCRATCH=_ACC_SCRATCH,
        HEAD_MAJOR=head_major,
        ROW_GROUPS=_ROW_GROUPS if (QBHp % (128 * _ROW_GROUPS) == 0) else 1,
    )
    smem = pltpu.SMEM
    row_spec = pl.BlockSpec((1, 1, 1, QBHp), lambda b, n: (b, n, 0, 0))
    if head_major:
        q_spec = pl.BlockSpec((1, H, None, QB, Dk_pad), lambda b, n: (b, 0, n, 0, 0))
        o_spec = pl.BlockSpec((1, H, None, QB, Dv), lambda b, n: (b, 0, n, 0, 0))
        o_shape = (B, H, nQB, QB, Dv)
    else:
        q_spec = pl.BlockSpec((1, 1, QBHp, Dk_pad), lambda b, n: (b, n, 0, 0))
        o_spec = pl.BlockSpec((1, 1, QBHp, Dv), lambda b, n: (b, n, 0, 0))
        o_shape = (B, nQB, QBHp, Dv)
    out = pl.pallas_call(
        kernel,
        grid=(B, nQB),
        in_specs=[
            q_spec,  # q
            pl.BlockSpec((1, 1, 1, U_pad), lambda b, n: (b, n, 0, 0), memory_space=smem),
            pl.BlockSpec((1, 1, 1, 1), lambda b, n: (b, n, 0, 0), memory_space=smem),
            row_spec,  # qpos rows
            row_spec,  # kvlen rows
            row_spec,  # base rows (tokens)
            pl.BlockSpec((1, 1, U_pad, QBHp), lambda b, n: (b, n, 0, 0)),  # membership
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv (untiled, DMA-gathered)
            pl.BlockSpec((1, 1, 1, PTW), lambda b, n: (b, 0, 0, 0), memory_space=smem),
        ],
        out_specs=o_spec,
        out_shape=jax.ShapeDtypeStruct(o_shape, out_dtype),
        scratch_shapes=[
            pltpu.VMEM((_NBUF, _PPU * RBF, Dk_pad), kv2.dtype),
            pltpu.SemaphoreType.DMA((_NBUF, _PPU)),
        ]
        + (
            [
                pltpu.VMEM((1, QBHp), jnp.float32),
                pltpu.VMEM((1, QBHp), jnp.float32),
                pltpu.VMEM((QBHp, Dv), jnp.float32),
            ]
            if _ACC_SCRATCH
            else []
        ),
        interpret=interpret,
    )(q4, units4, counts4, pos_rows, kvlen_rows, base_rows, memt, kv2, pt_arg)

    if head_major:
        return out.reshape(B, H, Sp, Dv)[:, :, :S]
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
    query_block: int = 256,
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
        r_cap=T // ps + 2 * S + 2,  # <= ceil(q / ps) + 1 runs per sequence
        interpret=interpret,
    )

    # token -> request id (same convention as the per-query ragged wrapper)
    t = jnp.arange(T, dtype=jnp.int32)
    q_seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, S - 1).astype(
        jnp.int32
    )

    out = _attend_sched_tagged(sparse_mla_attention_qblock)(
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


def prefill_write_and_attend_ragged_qblock_tsa(
    ql,  # [q_tokens, H, kv_lora_rank]      local token slice, FULL heads
    qpe,  # [q_tokens, H, rope]
    kvc,  # [total_tokens, kv_lora_rank]     full chunk's new c_kv to write
    kpe,  # [total_tokens, rope]
    cache,  # [P, ps//pk, pk, Dk_pad]        paged fused latent cache
    topk_pages,  # [q_tokens, K] int32       seq-local page ids for local queries
    positions,  # [q_tokens] int32           absolute positions of local queries
    loc,  # [total_tokens] int32             physical flat slot per chunk token
    seq_lens,  # [num_seqs] int32
    cu_q_lens,  # [num_seqs+1] int32         GLOBAL per-request query offsets
    cu_kv_lens,  # [num_seqs+1] int32
    page_indices,  # [total_pages] int32
    *,
    kv_lora_rank: int,
    page_size: int,
    sm_scale: float,
    query_block: int = 16,
    q_token_offset=0,  # int or traced scalar: global index of ql's first row
    q_token_index=None,  # [q_tokens] int32: global index per local row (overrides the offset)
    interpret: bool = False,
    out_dtype=None,  # attend output dtype (None = f32); bf16 when the consumer casts to bf16 anyway
    head_major: bool = False,  # ql/qpe are [H, q_tokens, *] and the output [H, q_tokens, Dv]
):
    """Token-sharded-attention variant of the ragged qblock wrapper.

    The write side covers the FULL chunk (``kvc``/``kpe``/``loc``, replicated
    per device — the latent pool is num_kv_heads=1 so every device keeps a
    complete cache copy), while the attend side runs only this device's
    query slice with ALL heads. ``q_token_offset`` (contiguous slice) or
    ``q_token_index`` (any row order, e.g. the block-strided zigzag assignment)
    maps local query rows back to global chunk positions for the per-request
    causal metadata; the attend itself only depends on the rows' own
    positions / page-topk, so the output of a row is independent of which
    device ran it.
    """
    if head_major:
        H, Tq, Dv = ql.shape
    else:
        Tq, H, Dv = ql.shape
    rope = qpe.shape[-1]
    Tkv = kvc.shape[0]
    ps = page_size
    _, _, _, Dk_pad = cache.shape
    S = seq_lens.shape[0]

    q_sparse = jnp.concatenate([ql, qpe], axis=-1)  # [Tq, H, Dv+rope] (or [H, Tq, Dv+rope])

    row = jnp.zeros((Tkv, Dk_pad), cache.dtype)
    row = row.at[:, :Dv].set(kvc.astype(cache.dtype))
    row = row.at[:, Dv : Dv + rope].set(kpe.reshape(Tkv, rope).astype(cache.dtype))
    cache_new = paged_write_back(
        cache,
        row,
        loc.astype(jnp.int32),
        page_size=ps,
        r_cap=Tkv // ps + 2 * S + 2,
        interpret=interpret,
    )

    if q_token_index is not None:
        t = jnp.asarray(q_token_index, jnp.int32).reshape(Tq)
    else:
        t = q_token_offset + jnp.arange(Tq, dtype=jnp.int32)
    q_seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, S - 1).astype(
        jnp.int32
    )

    q_in = (
        q_sparse.reshape(1, H, Tq, q_sparse.shape[2])
        if head_major
        else q_sparse.reshape(1, Tq, H, q_sparse.shape[2])
    )
    out = _attend_sched_tagged(sparse_mla_attention_qblock)(
        q_in,
        cache_new,
        topk_pages.reshape(1, Tq, -1),
        positions.reshape(1, Tq),
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
        out_dtype=jnp.float32 if out_dtype is None else out_dtype,
        head_major=head_major,
    )
    if head_major:
        return out.reshape(H, Tq, Dv), cache_new
    return out.reshape(Tq, H, Dv), cache_new


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


def _write_back_kernel(n_ref, tbl_ref, row_ref, cache_in_ref, out_ref, wdst_ref, wsrc_ref, sem):
    del cache_in_ref  # aliased with out_ref; all access goes through out_ref
    Pn, pspk, pk, D = out_ref.shape
    # word view: dim0 (words) is untiled, so dynamic word offsets are legal.
    # A flat [Pn*ps, D] token view is NOT: it inherits the (pk, ...) pairing
    # tile on dim0 and single-token slices at odd offsets fail Mosaic
    # alignment. Edge tokens therefore go through a word-granular RMW below.
    cache_w = out_ref.reshape(Pn * pspk, pk, D)
    # Only the first n_ref[0] = 3 * n_runs table entries can be non-empty (the
    # builder zero-fills the rows of unused run ids), so the scalar loop stops
    # there instead of walking all 3 * r_cap entries: at 4 draft tokens x 64
    # requests the table has 768 entries but ~64-192 runs, and the loop is
    # scalar-bound (~0.17 us per entry).
    E = n_ref[0]

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


# Row counts up to this size get a run table of at least ``num_rows`` entries by
# default, so ``pallas_always_fits`` holds and the runtime cond is dropped. The
# table is ``[3 * r_cap, 4]`` int32 in SMEM (48 KB at 1024 rows). Decode-form
# speculative verify at 64 requests x 4 draft tokens is 256 rows; the old default
# (2 * pages + 130 = 134) did not cover it and cost two whole-pool copies per
# layer (~0.15 ms each on GLM-5.2 tp16).
STATIC_DISPATCH_MAX_ROWS = 1024
# Row counts up to this size take the 4D-native scatter when the caller opts in
# (speculative verify / draft-extend pre-write); see paged_write_back.
SCATTER_ROWS_MAX = 1024


def default_run_capacity(num_rows: int, page_size: int) -> int:
    """Default ``r_cap`` of ``paged_write_back`` for ``num_rows`` (pk-padded) rows."""
    r_cap = 2 * (num_rows // page_size) + 130
    if num_rows <= STATIC_DISPATCH_MAX_ROWS:
        r_cap = max(r_cap, num_rows)
    return r_cap


def pallas_always_fits(num_rows: int, r_cap: int) -> bool:
    """True when the run table of ``paged_write_back`` cannot overflow.

    Every row starts at most one run (``n_raw = is_start.sum() <= num_rows``),
    so ``num_rows <= r_cap`` makes the Pallas branch always valid and the
    runtime ``lax.cond`` against the scatter fallback unnecessary.
    """
    return int(num_rows) <= int(r_cap)


def paged_write_back(
    cache,  # [Pn, ps//pk, pk, D] paged pool
    row,  # [T, D] new rows (already cache dtype / padded feature dim)
    loc,  # [T] int32 flat slot per token; -1 = dropped
    *,
    page_size: int,
    r_cap: int | None = None,
    interpret: bool = False,
    small_rows_scatter: bool = False,
):
    """In-place paged self-write: ``cache[loc[t]] = row[t]`` for ``loc >= 0``.

    ``small_rows_scatter``: for row counts up to ``SCATTER_ROWS_MAX`` skip the
    run table and the Pallas kernel and issue the 4D-native XLA scatter
    directly. Measured on v7x with the production pool (2274 pages): 0.03 ms
    at 4 rows and 0.065 ms at 256 rows versus 0.04 / 0.27 ms for the kernel,
    whose per-run DMAs are latency-bound at those sizes. Opt-in so the
    prefill path is unchanged.

    Bit-identical to ``cache.reshape(-1, D).at[loc].set(row, mode="drop",
    wrap_negative_indices=False)`` but without the scatter's flat-view
    relayout round-trip. Falls back to that scatter when the run table
    overflows ``r_cap`` (pathologically fragmented ``loc``).
    """
    Pn, pspk, pk, D = cache.shape
    ps = page_size
    assert pspk * pk == ps, (cache.shape, page_size)
    T = row.shape[0]
    if loc.shape[0] != T:
        # Both branches index ``row`` by positions taken from ``loc`` (the run
        # table's src offsets / the scatter's index set); a longer ``loc`` reads
        # past ``row`` in the Pallas branch and fails to broadcast in the
        # scatter branch. Fail at trace time with the real reason instead.
        raise ValueError(f"paged_write_back: loc has {loc.shape[0]} entries but row has {T} rows")
    Tp = -(-T // pk) * pk
    if Tp != T:
        row = jnp.pad(row, ((0, Tp - T), (0, 0)))
        loc = jnp.pad(loc, ((0, Tp - T),), constant_values=-1)
    if small_rows_scatter and Tp <= SCATTER_ROWS_MAX:
        page = jnp.where(loc >= 0, loc // ps, -1)
        rem = jnp.where(loc >= 0, loc % ps, 0)
        return cache.at[page, rem // pk, rem % pk].set(
            row.astype(cache.dtype), mode="drop", wrap_negative_indices=False
        )
    if r_cap is None:
        r_cap = default_run_capacity(Tp, ps)
    table, n_raw = _build_write_runs(loc, kv_packing=pk, r_cap=r_cap)
    row_w = row.reshape(Tp // pk, pk, D)
    n_ent = (3 * jnp.minimum(n_raw, r_cap)).reshape(1).astype(jnp.int32)  # live table entries

    def _pallas(cache, row_w, table, n_ent):
        return pl.pallas_call(
            _write_back_kernel,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),  # live entry count [1]
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
            input_output_aliases={3: 0},
            interpret=interpret,
        )(n_ent, table, row_w, cache)

    def _scatter(cache, row_w, table, n_ent):
        del table, n_ent
        # 4D-native scatter: same slot semantics as the flat-view version but
        # without cache.reshape(Pn*ps, D), which on jax 0.11.1 materializes a
        # full retile of the pool inside this cond branch (branch_1_fun,
        # ~2ms/layer at 110k).
        page = jnp.where(loc >= 0, loc // ps, -1)
        rem = jnp.where(loc >= 0, loc % ps, 0)
        return cache.at[page, rem // pk, rem % pk].set(
            row.astype(cache.dtype), mode="drop", wrap_negative_indices=False
        )

    if interpret:
        # interpret cannot lower dynamic-size DMAs; the scatter is the
        # bit-identical reference semantics anyway.
        return _scatter(cache, row_w, table, n_ent)
    # No lax.cond on n_raw <= r_cap: both branches hand back a whole pool, XLA cannot alias
    # the pool in place across the conditional (spec verify: 2 x ~0.02 ms per layer; 10-03
    # gp157 4k trace: ~20 whole-pool copies per 4k prefill window, 6-9 ms). The run table
    # cannot overflow r_cap for extend-form slot layouts (page-contiguous new slots per
    # sequence; test_write_back_run_bound.py) nor for the static ``pallas_always_fits``
    # shapes, so the kernel is taken unconditionally.
    return _pallas(cache, row_w, table, n_ent)
