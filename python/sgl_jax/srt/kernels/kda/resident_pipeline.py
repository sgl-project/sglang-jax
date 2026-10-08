"""Fused resident-state KDA prefill pipeline (standalone JAX-callable handle).

This module executes the KDA chunk recurrence in a single Pallas launch with the
recurrent [K, V] state kept resident in VMEM across chunks, fusing:
  1. In-kernel gate activation and chunk-local log2 prefix sum.
  2. Factored MXU intra-chunk Aqk/L/kg construction with BC=32 MXU blocks.
  3. Folded-RHS block-preconditioned Neumann triangular solve for v_new (solving
     V=128 columns instead of V+K=256 by subtracting k_eg_beta @ state before
     the solve).
  4. Merged MXU contractions for [k_eg_beta; qg] @ state and [Aqk; kg^T] @ v_new.
  5. Multi-head batched (MB = H) execution with sublane-aligned BlockSpecs,
     packed gate parameters, bitshift boolean masks, and deferred V load.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.kda.kda import (
    _RCP_LN2,
    _align_seqs,
    _unalign_output,
    assert_shape,
    assert_shape_or_none,
    exp2,
    get_interpret,
)

_HIGHEST = jax.lax.Precision.HIGHEST
_GROUP_CHUNKS = 4


def resident_group_chunks(num_chunks: int, group_chunks: int = _GROUP_CHUNKS) -> int:
    """Largest group <= ``group_chunks`` that exactly divides the chunk count."""
    group = max(1, int(group_chunks))
    if num_chunks <= 0:
        return 1
    group = min(group, num_chunks)
    while group > 1 and num_chunks % group:
        group -= 1
    return group


def _build_bounded_intra_3d(q, k, g, beta, scale, causal_bt, strict_bt, BC, prec, safe_gate):
    """Build causal Aqk, strictly-lower L, and end-of-chunk kg via 3D batched MXU dots."""
    MB, BT, _ = q.shape
    num_blocks = BT // BC
    ref_idx = (BC // 2) if safe_gate else 0
    aqk_rows = []
    l_rows = []
    k_inv_prefix = None
    prev_ref = None

    for blk in range(num_blocks):
        s = blk * BC
        e = s + BC
        q_blk = q[:, s:e, :]
        k_blk = k[:, s:e, :]
        g_blk = g[:, s:e, :]
        ref = g_blk[:, ref_idx : ref_idx + 1, :]
        g_diff = g_blk - ref
        g_exp = exp2(g_diff)
        q_scaled = q_blk * g_exp
        k_scaled = k_blk * g_exp
        k_inv_curr = k_blk * exp2(-g_diff)
        if blk == 0:
            k_inv_prefix = k_inv_curr
        else:
            if safe_gate and BC > 16:
                mid_ref = g_blk[:, 0:1, :]
                k_inv_scaled = (k_inv_prefix * exp2(mid_ref - prev_ref)) * exp2(ref - mid_ref)
            else:
                k_inv_scaled = k_inv_prefix * exp2(ref - prev_ref)
            k_inv_prefix = jnp.concatenate([k_inv_scaled, k_inv_curr], axis=1)
        prev_ref = ref
        qk_scaled = jnp.concatenate([q_scaled, k_scaled], axis=1)
        qk_dot_valid = jax.lax.dot_general(
            qk_scaled,
            k_inv_prefix,
            (((2,), (2,)), ((0,), (0,))),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
        if e < BT:
            qk_dot = jnp.concatenate(
                [qk_dot_valid, jnp.zeros((MB, 2 * BC, BT - e), dtype=jnp.float32)],
                axis=2,
            )
        else:
            qk_dot = qk_dot_valid
        aqk_rows.append(qk_dot[:, :BC, :])
        l_rows.append(qk_dot[:, BC:, :])

    Aqk = jnp.where(causal_bt[None, :, :], jnp.concatenate(aqk_rows, axis=1) * scale, 0.0)
    L = jnp.where(strict_bt[None, :, :], jnp.concatenate(l_rows, axis=1) * beta, 0.0)
    g_last = g[:, BT - 1 : BT, :]
    if safe_gate and (BT - 1 - ((num_blocks - 1) * BC + ref_idx)) > 16:
        mid_last = g[:, BT - 16 : BT - 15, :]
        kg = (k_inv_prefix * exp2(mid_last - prev_ref)) * exp2(g_last - mid_last)
    else:
        kg = k_inv_prefix * exp2(g_last - prev_ref)
    return Aqk, L, kg


def _build_unbounded_intra_3d(q, k, g, beta, scale, causal_bt, strict_bt, BC, prec):
    """Build causal Aqk, strictly-lower L, and end-of-chunk kg for unbounded gates (3D)."""
    MB, BT, _ = q.shape
    num_blocks = BT // BC
    aqk_rows = []
    l_rows = []

    for row_blk in range(num_blocks):
        s = row_blk * BC
        e = s + BC
        q_row = q[:, s:e, :]
        k_row = k[:, s:e, :]
        g_row = g[:, s:e, :]
        ref = g_row[:, 0:1, :]
        row_decay = exp2(jnp.clip(g_row - ref, -126.0, 0.0))

        g_diff = g_row[:, :, None, :] - g_row[:, None, :, :]
        diag_decay = exp2(jnp.clip(g_diff, -126.0, 0.0))
        aqk_diag = jnp.sum(q_row[:, :, None, :] * diag_decay * k_row[:, None, :, :], axis=-1)
        l_diag = jnp.sum(k_row[:, :, None, :] * diag_decay * k_row[:, None, :, :], axis=-1)

        if s > 0:
            col_decay = exp2(jnp.clip(ref - g[:, :s, :], -126.0, 0.0))
            qk_row = jnp.concatenate([q_row * row_decay, k_row * row_decay], axis=1)
            qk_off = jax.lax.dot_general(
                qk_row,
                k[:, :s, :] * col_decay,
                (((2,), (2,)), ((0,), (0,))),
                precision=prec,
                preferred_element_type=jnp.float32,
            )
            aqk_parts = [qk_off[:, :BC, :], aqk_diag]
            l_parts = [qk_off[:, BC:, :], l_diag]
        else:
            aqk_parts = [aqk_diag]
            l_parts = [l_diag]

        if e < BT:
            zeros = jnp.zeros((MB, BC, BT - e), dtype=jnp.float32)
            aqk_parts.append(zeros)
            l_parts.append(zeros)

        aqk_rows.append(jnp.concatenate(aqk_parts, axis=2))
        l_rows.append(jnp.concatenate(l_parts, axis=2))

    Aqk = jnp.where(causal_bt[None, :, :], jnp.concatenate(aqk_rows, axis=1) * scale, 0.0)
    L = jnp.where(strict_bt[None, :, :], jnp.concatenate(l_rows, axis=1) * beta, 0.0)
    g_last = g[:, BT - 1 : BT, :]
    kg = k * exp2(jnp.clip(g_last - g, -126.0, 0.0))
    return Aqk, L, kg


def _solve_intra_3d(L, rhs, identity_bt, same_block_mask, BC_inv, use_neumann, prec):
    """Solve (I + L) x = rhs via block-diagonal Neumann preconditioner on 3D [MB, BT, *]."""
    MB, BT, _ = L.shape

    def _dot(a, b):
        return jax.lax.dot_general(
            a,
            b,
            (((2,), (1,)), ((0,), (0,))),
            precision=prec,
            preferred_element_type=jnp.float32,
        )

    L_diag = jnp.where(same_block_mask[None, :, :], L, 0.0)
    neg_Ld = -L_diag
    P = jnp.where(identity_bt[None, :, :], 1.0, neg_Ld)
    Mk = neg_Ld
    num_diag_steps = int(math.log2(BC_inv)) - 1
    for step in range(num_diag_steps):
        if step == 0 and num_diag_steps == 2:
            M2 = _dot(Mk, Mk)
            PM2_and_M4 = _dot(jnp.concatenate([P, M2], axis=1), M2)
            P = P + PM2_and_M4[:, :BT, :]
            P = P + _dot(P, PM2_and_M4[:, BT:, :])
            break
        Mk = _dot(Mk, Mk)
        P = P + _dot(P, Mk)

    NC_inv = BT // BC_inv
    if NC_inv == 1:
        return _dot(P, rhs)
    if use_neumann:
        F = jnp.where(same_block_mask[None, :, :], 0.0, L)
        v_new = _dot(P, rhs)
        half = BT // 2
        if NC_inv == 2:
            H_bot = -_dot(P[:, half:, :], F[:, :, :half])
            return jnp.concatenate(
                [
                    v_new[:, :half, :],
                    v_new[:, half:, :] + _dot(H_bot, v_new[:, :half, :]),
                ],
                axis=1,
            )
        if NC_inv == 8:
            H_bot56 = -_dot(P[:, BC_inv:, :], F)
            v_new = jnp.concatenate(
                [
                    v_new[:, :BC_inv, :],
                    v_new[:, BC_inv:, :] + _dot(H_bot56, v_new),
                ],
                axis=1,
            )
            H_full = jnp.concatenate(
                [jnp.zeros((MB, BC_inv, BT), dtype=jnp.float32), H_bot56],
                axis=1,
            )
            H_bot48 = _dot(H_bot56[:, BC_inv:, :], H_full)
            s1 = 2 * BC_inv
            v_new = jnp.concatenate(
                [
                    v_new[:, :s1, :],
                    v_new[:, s1:, :] + _dot(H_bot48, v_new),
                ],
                axis=1,
            )
            H2_col_half = jnp.concatenate(
                [jnp.zeros((MB, s1, half), dtype=jnp.float32), H_bot48[:, :, :half]],
                axis=1,
            )
            H_bot32 = _dot(H_bot48[:, s1:, :], H2_col_half)
            return jnp.concatenate(
                [
                    v_new[:, :half, :],
                    v_new[:, half:, :] + _dot(H_bot32, v_new[:, :half, :]),
                ],
                axis=1,
            )

        H_mat = -_dot(P, F)
        num_block_steps = int(math.log2(NC_inv))
        for step in range(num_block_steps):
            if step < num_block_steps - 1:
                s_row = (1 << step) * BC_inv
                v_new = jnp.concatenate(
                    [
                        v_new[:, :s_row, :],
                        v_new[:, s_row:, :] + _dot(H_mat[:, s_row:, :], v_new),
                    ],
                    axis=1,
                )
                if step == num_block_steps - 2:
                    H_mat = _dot(H_mat[:, half:, :], H_mat[:, :, :half])
                else:
                    H_mat = _dot(H_mat, H_mat)
            else:
                v_new = jnp.concatenate(
                    [
                        v_new[:, :half, :],
                        v_new[:, half:, :] + _dot(H_mat, v_new[:, :half, :]),
                    ],
                    axis=1,
                )
        return v_new

    solved_blocks = []
    for blk in range(NC_inv):
        s = blk * BC_inv
        e = s + BC_inv
        res = rhs[:, s:e, :]
        if solved_blocks:
            res = res - _dot(L[:, s:e, :s], jnp.concatenate(solved_blocks, axis=1))
        solved_blocks.append(_dot(P[:, s:e, s:e], res))
    return jnp.concatenate(solved_blocks, axis=1)


def _build_bounded_intra_2d(q, k, g, beta, scale, causal_bt, strict_bt, BC, prec, safe_gate):
    """Build causal Aqk, strictly-lower L, and end-of-chunk kg via 2D MXU dots."""
    BT = q.shape[0]
    num_blocks = BT // BC
    ref_idx = (BC // 2) if safe_gate else 0
    aqk_rows = []
    l_rows = []
    k_inv_prefix = None
    prev_ref = None

    for blk in range(num_blocks):
        s = blk * BC
        e = s + BC
        q_blk = q[s:e]
        k_blk = k[s:e]
        g_blk = g[s:e]
        ref = g_blk[ref_idx : ref_idx + 1]
        g_diff = g_blk - ref
        g_exp = exp2(g_diff)
        q_scaled = q_blk * g_exp
        k_scaled = k_blk * g_exp
        k_inv_curr = k_blk * exp2(-g_diff)
        if blk == 0:
            k_inv_prefix = k_inv_curr
        else:
            if safe_gate and BC > 16:
                mid_ref = g_blk[0:1]
                k_inv_scaled = (k_inv_prefix * exp2(mid_ref - prev_ref)) * exp2(ref - mid_ref)
            else:
                k_inv_scaled = k_inv_prefix * exp2(ref - prev_ref)
            k_inv_prefix = jnp.concatenate([k_inv_scaled, k_inv_curr], axis=0)
        prev_ref = ref
        qk_scaled = jnp.concatenate([q_scaled, k_scaled], axis=0)
        qk_dot_valid = jax.lax.dot_general(
            qk_scaled,
            k_inv_prefix,
            (((1,), (1,)), ((), ())),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
        if e < BT:
            qk_dot = jnp.concatenate(
                [qk_dot_valid, jnp.zeros((2 * BC, BT - e), dtype=jnp.float32)],
                axis=1,
            )
        else:
            qk_dot = qk_dot_valid
        aqk_rows.append(qk_dot[:BC])
        l_rows.append(qk_dot[BC:])

    Aqk = jnp.where(causal_bt, jnp.concatenate(aqk_rows, axis=0) * scale, 0.0)
    L = jnp.where(strict_bt, jnp.concatenate(l_rows, axis=0) * beta, 0.0)
    g_last = g[BT - 1 : BT, :]
    if safe_gate and (BT - 1 - ((num_blocks - 1) * BC + ref_idx)) > 16:
        mid_last = g[BT - 16 : BT - 15, :]
        kg = (k_inv_prefix * exp2(mid_last - prev_ref)) * exp2(g_last - mid_last)
    else:
        kg = k_inv_prefix * exp2(g_last - prev_ref)
    return Aqk, L, kg


def _build_unbounded_intra_2d(q, k, g, beta, scale, causal_bt, strict_bt, BC, prec):
    """Build causal Aqk, strictly-lower L, and end-of-chunk kg for unbounded gates."""
    BT = q.shape[0]
    num_blocks = BT // BC
    aqk_rows = []
    l_rows = []

    for row_blk in range(num_blocks):
        s = row_blk * BC
        e = s + BC
        q_row = q[s:e]
        k_row = k[s:e]
        g_row = g[s:e]
        ref = g_row[0:1, :]
        row_decay = exp2(jnp.clip(g_row - ref, -126.0, 0.0))

        g_diff = g_row[:, None, :] - g_row[None, :, :]
        diag_decay = exp2(jnp.clip(g_diff, -126.0, 0.0))
        aqk_diag = jnp.sum(q_row[:, None, :] * diag_decay * k_row[None, :, :], axis=-1)
        l_diag = jnp.sum(k_row[:, None, :] * diag_decay * k_row[None, :, :], axis=-1)

        if s > 0:
            col_decay = exp2(jnp.clip(ref - g[:s], -126.0, 0.0))
            qk_row = jnp.concatenate([q_row * row_decay, k_row * row_decay], axis=0)
            qk_off = jax.lax.dot_general(
                qk_row,
                k[:s] * col_decay,
                (((1,), (1,)), ((), ())),
                precision=prec,
                preferred_element_type=jnp.float32,
            )
            aqk_parts = [qk_off[:BC], aqk_diag]
            l_parts = [qk_off[BC:], l_diag]
        else:
            aqk_parts = [aqk_diag]
            l_parts = [l_diag]

        if e < BT:
            zeros = jnp.zeros((BC, BT - e), dtype=jnp.float32)
            aqk_parts.append(zeros)
            l_parts.append(zeros)

        aqk_rows.append(jnp.concatenate(aqk_parts, axis=1))
        l_rows.append(jnp.concatenate(l_parts, axis=1))

    Aqk = jnp.where(causal_bt, jnp.concatenate(aqk_rows, axis=0) * scale, 0.0)
    L = jnp.where(strict_bt, jnp.concatenate(l_rows, axis=0) * beta, 0.0)
    g_last = g[BT - 1 : BT, :]
    kg = k * exp2(jnp.clip(g_last - g, -126.0, 0.0))
    return Aqk, L, kg


def _solve_intra_2d(L, rhs, identity_bt, same_block_mask, BC_inv, use_neumann, prec):
    """Solve (I + L) x = rhs via block-diagonal Neumann preconditioner on MXU."""
    BT = L.shape[0]

    def _dot(a, b):
        return jax.lax.dot_general(
            a,
            b,
            (((1,), (0,)), ((), ())),
            precision=prec,
            preferred_element_type=jnp.float32,
        )

    L_diag = jnp.where(same_block_mask, L, 0.0)
    neg_Ld = -L_diag
    P = jnp.where(identity_bt, 1.0, neg_Ld)
    Mk = neg_Ld
    num_diag_steps = int(math.log2(BC_inv)) - 1
    for step in range(num_diag_steps):
        if step == 0 and num_diag_steps == 2:
            M2 = _dot(Mk, Mk)
            PM2_and_M4 = _dot(jnp.concatenate([P, M2], axis=0), M2)
            P = P + PM2_and_M4[:BT]
            P = P + _dot(P, PM2_and_M4[BT:])
            break
        Mk = _dot(Mk, Mk)
        P = P + _dot(P, Mk)

    NC_inv = BT // BC_inv
    if NC_inv == 1:
        return _dot(P, rhs)
    if use_neumann:
        F = jnp.where(same_block_mask, 0.0, L)
        v_new = _dot(P, rhs)
        half = BT // 2
        if NC_inv == 2:
            H_bot = -_dot(P[half:, :], F[:, :half])
            return jnp.concatenate(
                [
                    v_new[:half],
                    v_new[half:] + _dot(H_bot, v_new[:half]),
                ],
                axis=0,
            )
        if NC_inv == 8:
            H_bot56 = -_dot(P[BC_inv:, :], F)
            v_new = jnp.concatenate(
                [
                    v_new[:BC_inv],
                    v_new[BC_inv:] + _dot(H_bot56, v_new),
                ],
                axis=0,
            )
            H_full = jnp.concatenate(
                [jnp.zeros((BC_inv, BT), dtype=jnp.float32), H_bot56],
                axis=0,
            )
            H_bot48 = _dot(H_bot56[BC_inv:, :], H_full)
            s1 = 2 * BC_inv
            v_new = jnp.concatenate(
                [
                    v_new[:s1],
                    v_new[s1:] + _dot(H_bot48, v_new),
                ],
                axis=0,
            )
            H2_col_half = jnp.concatenate(
                [jnp.zeros((s1, half), dtype=jnp.float32), H_bot48[:, :half]],
                axis=0,
            )
            H_bot32 = _dot(H_bot48[s1:, :], H2_col_half)
            return jnp.concatenate(
                [
                    v_new[:half],
                    v_new[half:] + _dot(H_bot32, v_new[:half]),
                ],
                axis=0,
            )

        H_mat = -_dot(P, F)
        num_block_steps = int(math.log2(NC_inv))
        for step in range(num_block_steps):
            if step < num_block_steps - 1:
                s_row = (1 << step) * BC_inv
                v_new = jnp.concatenate(
                    [
                        v_new[:s_row],
                        v_new[s_row:] + _dot(H_mat[s_row:], v_new),
                    ],
                    axis=0,
                )
                if step == num_block_steps - 2:
                    H_mat = _dot(H_mat[half:, :], H_mat[:, :half])
                else:
                    H_mat = _dot(H_mat, H_mat)
            else:
                v_new = jnp.concatenate(
                    [
                        v_new[:half],
                        v_new[half:] + _dot(H_mat, v_new[:half]),
                    ],
                    axis=0,
                )
        return v_new

    solved_blocks = []
    for blk in range(NC_inv):
        s = blk * BC_inv
        e = s + BC_inv
        res = rhs[s:e]
        if solved_blocks:
            res = res - _dot(L[s:e, :s], jnp.concatenate(solved_blocks, axis=0))
        solved_blocks.append(_dot(P[s:e, s:e], res))
    return jnp.concatenate(solved_blocks, axis=0)


def _resident_pipeline_mb_single_seq_kernel(
    meta_ref,
    q_ref,
    k_ref,
    v_ref,
    gk_ref,
    beta_ref,
    gate_params_ref,
    h0_ref,
    o_ref,
    ht_ref,
    state_ref,
    *,
    BT: int,
    NT: int,
    MB: int,
    K: int,
    V: int,
    scale: float,
    intra_block_size: int,
    safe_gate: bool,
    lower_bound: float | None,
    USE_GATE_IN_KERNEL: bool,
    GATE_PACKED: bool,
    USE_QK_L2NORM: bool,
    NEED_CUMSUM: bool,
    BETA_BATCH_FIRST: bool,
    BATCH_FIRST_IO: bool,
    VARLEN: bool = False,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
):
    idx_c = pl.program_id(2)
    elide_initial_state = (not VARLEN) and (not USE_INITIAL_STATE) and (NT == 1)

    if not elide_initial_state:
        init_cond = (meta_ref[2 * NT + idx_c] != 0) if VARLEN else (idx_c == 0)

        @pl.when(init_cond)
        def _init_state():
            if USE_INITIAL_STATE:
                state_ref[...] = h0_ref[0].astype(jnp.float32)
            else:
                state_ref[...] = jnp.zeros([MB, K, V], dtype=jnp.float32)

    use_neumann = lower_bound is not None
    prec = jax.lax.Precision.DEFAULT if use_neumann else _HIGHEST
    qk_bc = min(32 if safe_gate else 16, BT)
    inv_bc = min(8 if use_neumann else intra_block_size, BT)
    inv_shift = int(math.log2(inv_bc))
    idx_bt = jnp.arange(BT, dtype=jnp.int32)
    row_idx = idx_bt[:, None]
    col_idx = idx_bt[None, :]
    causal_bt = row_idx >= col_idx
    strict_bt = row_idx > col_idx
    identity_bt = row_idx == col_idx
    same_block_mask = (row_idx >> inv_shift) == (col_idx >> inv_shift)
    num_cumsum_steps = int(math.log2(BT))

    if BATCH_FIRST_IO:
        q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32)
        k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32)
        g_in = gk_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    else:
        q = q_ref[0].astype(jnp.float32)
        k = k_ref[0].astype(jnp.float32)
        g_in = gk_ref[0].astype(jnp.float32)

    if BETA_BATCH_FIRST:
        beta_2d = beta_ref[0, 0].astype(jnp.float32)
        if USE_QK_L2NORM:
            beta_2d = jnp.clip(beta_2d, 0.0, 1.0)
        beta = jnp.stack([beta_2d[:, m : m + 1] for m in range(MB)], axis=0)
    else:
        beta = beta_ref[:, 0, 0, 0, :].astype(jnp.float32)[:, :, None]
        if USE_QK_L2NORM:
            beta = jnp.clip(beta, 0.0, 1.0)

    if USE_QK_L2NORM:
        q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = q.astype(jnp.bfloat16).astype(jnp.float32)
        k = k.astype(jnp.bfloat16).astype(jnp.float32)

    if USE_GATE_IN_KERNEL:
        if GATE_PACKED:
            gp = gate_params_ref[0, 0, :, :].astype(jnp.float32)
            dt_b = gp[:MB, None, :]
            a_scale = gp[MB : 2 * MB, None, :]
        else:
            gp = gate_params_ref[:, 0, :, :].astype(jnp.float32)
            a_scale = gp[:, 0:1, :]
            dt_b = gp[:, 1:2, :]
        g_val = g_in + dt_b
        if lower_bound is None:
            g_act = -a_scale * jax.nn.softplus(g_val)
        else:
            g_act = lower_bound * jax.nn.sigmoid(a_scale * g_val)
    else:
        g_act = g_in

    if not VARLEN:
        valid_3d = (idx_c * BT + jax.lax.broadcasted_iota(jnp.int32, (MB, BT, K), 1)) < meta_ref[1]
        q = jnp.where(valid_3d, q, 0.0)
        k = jnp.where(valid_3d, k, 0.0)
        g_act = jnp.where(valid_3d, g_act, 0.0)

    if NEED_CUMSUM or USE_GATE_IN_KERNEL:
        for d in range(num_cumsum_steps):
            stride = 1 << d
            g_act = jnp.concatenate(
                [g_act[:, :stride, :], g_act[:, stride:, :] + g_act[:, :-stride, :]],
                axis=1,
            )
        g = g_act * _RCP_LN2
    else:
        g = g_act

    if lower_bound is None:
        Aqk, L, kg = _build_unbounded_intra_3d(
            q, k, g, beta, scale, causal_bt, strict_bt, qk_bc, prec
        )
    else:
        Aqk, L, kg = _build_bounded_intra_3d(
            q, k, g, beta, scale, causal_bt, strict_bt, qk_bc, prec, safe_gate
        )

    if elide_initial_state:
        if BATCH_FIRST_IO:
            v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
        else:
            v = v_ref[0].astype(jnp.float32)
        rhs = v * beta
        v_new = _solve_intra_3d(L, rhs, identity_bt, same_block_mask, inv_bc, use_neumann, prec)
        if STORE_FINAL_STATE:
            o_and_ds = jax.lax.dot_general(
                jnp.concatenate([Aqk, kg.transpose(0, 2, 1)], axis=1),
                v_new,
                (((2,), (1,)), ((0,), (0,))),
                precision=prec,
                preferred_element_type=jnp.float32,
            )
            o = o_and_ds[:, :BT, :]
            ht_ref[0] = o_and_ds[:, BT:, :].astype(ht_ref.dtype)
        else:
            o = jax.lax.dot_general(
                Aqk,
                v_new,
                (((2,), (1,)), ((0,), (0,))),
                precision=prec,
                preferred_element_type=jnp.float32,
            )
        if BATCH_FIRST_IO:
            o_ref[0] = o.transpose(1, 0, 2).astype(o_ref.dtype)
        else:
            o_ref[0] = o.astype(o_ref.dtype)
        return

    g_last_exp = exp2(jnp.maximum(g[:, BT - 1, :], -126.0))[:, :, None]
    if use_neumann:
        eg = exp2(jnp.maximum(g, -126.0))
        k_qg = jnp.concatenate([k * eg * beta, q * eg], axis=1)
        state = state_ref[...]
        kq_state = jax.lax.dot_general(
            k_qg,
            state,
            (((2,), (1,)), ((0,), (0,))),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
    else:
        g_head = g[:, 0:1, :]
        eg_rel = exp2(jnp.maximum(g - g_head, -126.0))
        k_qg = jnp.concatenate([k * eg_rel * beta, q * eg_rel], axis=1)
        g_head_exp = exp2(jnp.maximum(g_head[:, 0, :], -126.0))[:, :, None]
        state = state_ref[...]
        kq_state = jax.lax.dot_general(
            k_qg,
            state * g_head_exp,
            (((2,), (1,)), ((0,), (0,))),
            precision=prec,
            preferred_element_type=jnp.float32,
        )

    if BATCH_FIRST_IO:
        v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    else:
        v = v_ref[0].astype(jnp.float32)
    rhs = v * beta - kq_state[:, :BT, :]
    o_inter = kq_state[:, BT:, :] * scale

    v_new = _solve_intra_3d(L, rhs, identity_bt, same_block_mask, inv_bc, use_neumann, prec)

    o_and_ds = jax.lax.dot_general(
        jnp.concatenate([Aqk, kg.transpose(0, 2, 1)], axis=1),
        v_new,
        (((2,), (1,)), ((0,), (0,))),
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    o = o_inter + o_and_ds[:, :BT, :]
    if BATCH_FIRST_IO:
        o_ref[0] = o.transpose(1, 0, 2).astype(o_ref.dtype)
    else:
        o_ref[0] = o.astype(o_ref.dtype)

    state_next = state * g_last_exp + o_and_ds[:, BT:, :]
    state_ref[...] = state_next

    if STORE_FINAL_STATE:
        if VARLEN:
            ht_ref[0] = state_next.astype(ht_ref.dtype)
        else:

            @pl.when(idx_c == NT - 1)
            def _store_final():
                ht_ref[0] = state_next.astype(ht_ref.dtype)


def _resident_pipeline_kernel(
    seqlens_ref,
    q_ref,
    k_ref,
    v_ref,
    gk_ref,
    beta_ref,
    gate_params_ref,
    h0_ref,
    o_ref,
    ht_ref,
    state_ref,
    *,
    BT,
    GROUP,
    K,
    V,
    N,
    scale,
    intra_block_size,
    safe_gate=True,
    lower_bound=None,
    USE_GATE_IN_KERNEL=False,
    USE_QK_L2NORM=False,
    NEED_CUMSUM=False,
    SINGLE_SEQ=False,
    HAS_PARTIAL_CHUNKS=False,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    idx_n = pl.program_id(0)
    idx_nb = pl.program_id(2)

    if not SINGLE_SEQ or HAS_PARTIAL_CHUNKS:
        bos = seqlens_ref[idx_n]
        eos = seqlens_ref[idx_n + 1]
        real_NT = (eos - bos) // BT
        valid_eos = seqlens_ref[N + 1 + idx_n]

    @pl.when(idx_nb == 0)
    def _():
        if USE_INITIAL_STATE:
            state_ref[...] = h0_ref[0, 0].astype(jnp.float32)
        else:
            state_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)

    use_neumann = lower_bound is not None
    prec = jax.lax.Precision.DEFAULT if use_neumann else _HIGHEST
    qk_bc = min(32 if safe_gate else 16, BT)
    inv_bc = min(8 if use_neumann else intra_block_size, BT)
    inv_shift = int(math.log2(inv_bc))
    idx_bt = jnp.arange(BT, dtype=jnp.int32)
    row_idx = idx_bt[:, None]
    col_idx = idx_bt[None, :]
    causal_bt = row_idx >= col_idx
    strict_bt = row_idx > col_idx
    identity_bt = row_idx == col_idx
    same_block_mask = (row_idx >> inv_shift) == (col_idx >> inv_shift)
    num_cumsum_steps = int(math.log2(BT))

    def _run_group():
        if USE_GATE_IN_KERNEL:
            gp = gate_params_ref[0, 0, :, :].astype(jnp.float32)
            A_scale = gp[0:1, :]
            dt_b = gp[1:2, :]

        for i_chunk in range(GROUP):
            rows = dslice(i_chunk * BT, BT)
            q = q_ref[0, 0, rows, :].astype(jnp.float32)
            k = k_ref[0, 0, rows, :].astype(jnp.float32)
            v = v_ref[0, 0, rows, :].astype(jnp.float32)
            g_in = gk_ref[0, 0, rows, :].astype(jnp.float32)
            beta = beta_ref[0, 0, rows, 0:1].astype(jnp.float32)

            if USE_QK_L2NORM:
                q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                q = q.astype(jnp.bfloat16).astype(jnp.float32)
                k = k.astype(jnp.bfloat16).astype(jnp.float32)
                beta = jnp.clip(beta, 0.0, 1.0)

            if USE_GATE_IN_KERNEL:
                g_val = g_in + dt_b
                if lower_bound is None:
                    g_act = -A_scale * jax.nn.softplus(g_val)
                else:
                    g_act = lower_bound * jax.nn.sigmoid(A_scale * g_val)
            else:
                g_act = g_in

            if HAS_PARTIAL_CHUNKS:
                chunk_start = bos + (idx_nb * GROUP + i_chunk) * BT
                valid_f32 = ((chunk_start + idx_bt) < valid_eos).astype(jnp.float32)[:, None]
                if NEED_CUMSUM or USE_GATE_IN_KERNEL:
                    g_act = g_act * valid_f32
                beta = beta * valid_f32

            if NEED_CUMSUM or USE_GATE_IN_KERNEL:
                for d in range(num_cumsum_steps):
                    stride = 1 << d
                    g_act = jnp.concatenate(
                        [g_act[:stride], g_act[stride:] + g_act[:-stride]], axis=0
                    )
                g = g_act * _RCP_LN2
            else:
                g = g_act

            if lower_bound is None:
                Aqk, L, kg = _build_unbounded_intra_2d(
                    q, k, g, beta, scale, causal_bt, strict_bt, qk_bc, prec
                )
            else:
                Aqk, L, kg = _build_bounded_intra_2d(
                    q, k, g, beta, scale, causal_bt, strict_bt, qk_bc, prec, safe_gate
                )

            state = state_ref[...]
            if use_neumann:
                eg = exp2(jnp.maximum(g, -126.0))
                k_eg_beta = k * eg * beta
                qg = q * eg
                kq_state = jax.lax.dot_general(
                    jnp.concatenate([k_eg_beta, qg], axis=0),
                    state,
                    (((1,), (0,)), ((), ())),
                    precision=prec,
                    preferred_element_type=jnp.float32,
                )
            else:
                g_head = g[0:1, :]
                eg_rel = exp2(jnp.maximum(g - g_head, -126.0))
                k_eg_beta = k * eg_rel * beta
                qg = q * eg_rel
                state_head = state * exp2(jnp.maximum(g_head[0], -126.0))[:, None]
                kq_state = jax.lax.dot_general(
                    jnp.concatenate([k_eg_beta, qg], axis=0),
                    state_head,
                    (((1,), (0,)), ((), ())),
                    precision=prec,
                    preferred_element_type=jnp.float32,
                )

            rhs = v * beta - kq_state[:BT]
            o_inter = kq_state[BT:] * scale

            v_new = _solve_intra_2d(L, rhs, identity_bt, same_block_mask, inv_bc, use_neumann, prec)

            o_and_ds = jax.lax.dot_general(
                jnp.concatenate([Aqk, kg.T], axis=0),
                v_new,
                (((1,), (0,)), ((), ())),
                precision=prec,
                preferred_element_type=jnp.float32,
            )
            o = o_inter + o_and_ds[:BT]
            o_ref[0, 0, rows, :] = o.astype(o_ref.dtype)

            g_last = g[BT - 1]
            state_ref[...] = state * exp2(jnp.maximum(g_last, -126.0))[:, None] + o_and_ds[BT:]

    if SINGLE_SEQ:
        _run_group()
    else:

        @pl.when(idx_nb * GROUP < real_NT)
        def _():
            _run_group()

    if STORE_FINAL_STATE:
        if SINGLE_SEQ:

            @pl.when(idx_nb == pl.num_programs(2) - 1)
            def _():
                ht_ref[0, 0] = state_ref[...].astype(ht_ref.dtype)

        else:

            @pl.when(((idx_nb + 1) * GROUP == real_NT) | ((real_NT == 0) & (idx_nb == 0)))
            def _():
                ht_ref[0, 0] = state_ref[...].astype(ht_ref.dtype)


def resident_pipeline_stage(
    q,
    k,
    v,
    gk,
    beta,
    initial_state,
    scale,
    cu_seqlens,
    *,
    chunk_size,
    group,
    intra_block_size,
    output_final_state,
    safe_gate=True,
    lower_bound=None,
    use_gate_in_kernel=False,
    use_qk_l2norm_in_kernel=False,
    need_cumsum=False,
    A_log=None,
    dt_bias=None,
    valid_eos=None,
    single_seq=False,
    has_partial_chunks=False,
):
    """One fused Pallas launch: gate + intra solve + state recurrence + output."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    N = cu_seqlens.shape[0] - 1
    h0 = None if initial_state is None else initial_state.astype(jnp.float32)

    if (
        T % BT == 0
        and (K % 128 == 0)
        and (V % 128 == 0)
        and (not single_seq or not has_partial_chunks)
    ):
        NT = T // BT
        mb = min(H, 16)
        while H % mb != 0:
            mb -= 1

        batch_first_io = (mb % 8) == 0
        if batch_first_io:
            q_in, k_in, v_in, gk_in = q, k, v, gk
            in_spec_k = pl.BlockSpec([1, BT, mb, K], index_map=lambda h, b, c, *_: (b, c, h, 0))
            in_spec_v = pl.BlockSpec([1, BT, mb, V], index_map=lambda h, b, c, *_: (b, c, h, 0))
            o_shape = jax.ShapeDtypeStruct([B, T, H, V], q.dtype)
        else:
            q_in = jnp.transpose(q, (0, 2, 1, 3))
            k_in = jnp.transpose(k, (0, 2, 1, 3))
            v_in = jnp.transpose(v, (0, 2, 1, 3))
            gk_in = jnp.transpose(gk, (0, 2, 1, 3))
            in_spec_k = pl.BlockSpec([1, mb, BT, K], index_map=lambda h, b, c, *_: (b, h, c, 0))
            in_spec_v = pl.BlockSpec([1, mb, BT, V], index_map=lambda h, b, c, *_: (b, h, c, 0))
            o_shape = jax.ShapeDtypeStruct([B, H, T, V], q.dtype)

        beta_batch_first = mb == H
        if beta_batch_first:
            beta_t = beta.reshape(B, NT, BT, H)
            beta_spec = pl.BlockSpec([1, 1, BT, mb], index_map=lambda h, b, c, *_: (b, c, 0, h))
        else:
            beta_t = beta.transpose(2, 0, 1).reshape(H, B, NT, 1, BT)
            beta_spec = pl.BlockSpec(
                [mb, 1, 1, 1, BT], index_map=lambda h, b, c, *_: (h, b, c, 0, 0)
            )

        gate_packed = beta_batch_first
        if use_gate_in_kernel:
            assert A_log is not None
            A_f32 = jnp.minimum(A_log.reshape(-1).astype(jnp.float32), 80.0)
            if A_f32.shape[0] < H:
                A_f32 = jnp.repeat(A_f32, H // A_f32.shape[0], axis=0)
            if gate_packed:
                gp_rows = max(8, ((2 * mb + 7) // 8) * 8)
                a_scale_2d = jnp.broadcast_to(jnp.exp(A_f32)[:, None], (H, K))
                gate_2d = jnp.pad(a_scale_2d, ((mb, gp_rows - 2 * mb), (0, 0)))
                if dt_bias is not None:
                    db_f32 = dt_bias.reshape(-1, K).astype(jnp.float32)
                    if db_f32.shape[0] < H:
                        db_f32 = jnp.repeat(db_f32, H // db_f32.shape[0], axis=0)
                    gate_2d = gate_2d + jnp.pad(db_f32, ((0, gp_rows - mb), (0, 0)))
                gate_params = gate_2d.reshape(1, 1, gp_rows, K)
                gate_spec = pl.BlockSpec(
                    [1, 1, gp_rows, K], index_map=lambda h, b, c, *_: (0, 0, 0, 0)
                )
            else:
                a_scale_full = jnp.broadcast_to(jnp.exp(A_f32).reshape(H, 1, 1, 1), (H, 1, 1, K))
                if dt_bias is not None:
                    db_f32 = dt_bias.reshape(-1, K).astype(jnp.float32)
                    if db_f32.shape[0] < H:
                        db_f32 = jnp.repeat(db_f32, H // db_f32.shape[0], axis=0)
                    db_full = db_f32.reshape(H, 1, 1, K)
                else:
                    db_full = jnp.zeros((H, 1, 1, K), dtype=jnp.float32)
                pad_rows = jnp.zeros((H, 1, 6, K), dtype=jnp.float32)
                gate_params = jnp.concatenate([a_scale_full, db_full, pad_rows], axis=2)
                gate_spec = pl.BlockSpec([mb, 1, 8, K], index_map=lambda h, b, c, *_: (h, 0, 0, 0))
        else:
            gate_params = None
            gate_spec = None

        if single_seq:
            in_state_spec = (
                None
                if h0 is None
                else pl.BlockSpec([1, mb, K, V], index_map=lambda h, b, c, *_: (0, h, 0, 0))
            )
            out_state_spec = (
                pl.BlockSpec([1, mb, K, V], index_map=lambda h, b, c, *_: (0, h, 0, 0))
                if output_final_state
                else None
            )
            ht_shape = (
                jax.ShapeDtypeStruct([N, H, K, V], jnp.float32) if output_final_state else None
            )
            h0_in = h0
            chunk_meta = cu_seqlens.astype(jnp.int32)
            kernel_fn = _resident_pipeline_mb_single_seq_kernel
            num_scalar_prefetch = 1
            call_args = (chunk_meta, q_in, k_in, v_in, gk_in, beta_t, gate_params, h0_in)
        else:
            chunk_pos = jnp.arange(NT, dtype=jnp.int32) * BT
            in_seg = (chunk_pos[None, :] >= cu_seqlens[:-1, None]) & (
                chunk_pos[None, :] < cu_seqlens[1:, None]
            )
            chunk_valid = jnp.any(in_seg, axis=0)
            seg_idx = jnp.argmax(in_seg, axis=0).astype(jnp.int32)
            in_seg_id = jnp.where(chunk_valid, seg_idx, N)
            seg_bos = cu_seqlens[seg_idx]
            seg_eos = cu_seqlens[seg_idx + 1]
            is_bos = jnp.where(chunk_valid & (chunk_pos == seg_bos), 1, 0).astype(jnp.int32)
            is_bos = is_bos.at[0].set(1)
            is_eos = chunk_valid & ((chunk_pos + BT) == seg_eos)
            out_seg_id = jnp.where(is_eos, seg_idx, N).astype(jnp.int32)
            chunk_meta = jnp.concatenate([in_seg_id, out_seg_id, is_bos], axis=0)

            h0_in = None if h0 is None else jnp.pad(h0, ((0, 1), (0, 0), (0, 0), (0, 0)))
            in_state_spec = (
                None
                if h0 is None
                else pl.BlockSpec(
                    [1, mb, K, V], index_map=lambda h, b, c, meta_ref: (meta_ref[c], h, 0, 0)
                )
            )
            out_state_spec = (
                pl.BlockSpec(
                    [1, mb, K, V], index_map=lambda h, b, c, meta_ref: (meta_ref[NT + c], h, 0, 0)
                )
                if output_final_state
                else None
            )
            ht_shape = (
                jax.ShapeDtypeStruct([N + 1, H, K, V], jnp.float32) if output_final_state else None
            )
            kernel_fn = _resident_pipeline_mb_single_seq_kernel
            num_scalar_prefetch = 1
            call_args = (chunk_meta, q_in, k_in, v_in, gk_in, beta_t, gate_params, h0_in)

        o_out, ht_out = pl.pallas_call(
            functools.partial(
                kernel_fn,
                BT=BT,
                NT=NT,
                MB=mb,
                K=K,
                V=V,
                scale=scale,
                intra_block_size=intra_block_size,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                USE_GATE_IN_KERNEL=use_gate_in_kernel,
                GATE_PACKED=gate_packed,
                USE_QK_L2NORM=use_qk_l2norm_in_kernel,
                NEED_CUMSUM=need_cumsum,
                BETA_BATCH_FIRST=beta_batch_first,
                BATCH_FIRST_IO=batch_first_io,
                VARLEN=not single_seq,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=output_final_state,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                grid=(H // mb, B, NT),
                in_specs=[
                    in_spec_k,
                    in_spec_k,
                    in_spec_v,
                    in_spec_k,
                    beta_spec,
                    gate_spec,
                    in_state_spec,
                ],
                out_specs=[
                    in_spec_v,
                    out_state_spec,
                ],
                scratch_shapes=[pltpu.VMEM((mb, K, V), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                disable_bounds_checks=True,
                vmem_limit_bytes=(60 * 1024 * 1024) if mb > 4 else None,
            ),
            out_shape=[o_shape, ht_shape],
            interpret=get_interpret(),
        )(*call_args)

        if not single_seq and output_final_state:
            ht_out = ht_out[:N]
            if valid_eos is not None:
                empty_seg = (valid_eos - cu_seqlens[:-1]) == 0
                fallback_h = h0 if h0 is not None else jnp.zeros_like(ht_out)
                ht_out = jnp.where(empty_seg[:, None, None, None], fallback_h, ht_out)

        if not batch_first_io:
            o_out = jnp.transpose(o_out, (0, 2, 1, 3))
        return o_out, ht_out

    GBT = BT * group
    assert T % GBT == 0, f"T={T} must be divisible by the group extent {GBT}"

    if single_seq:
        T_alloc = T

        def _pack(x):
            return jnp.transpose(x, (0, 2, 1, 3))

    else:
        T_alloc = T + GBT

        def _pack(x):
            x = jnp.pad(x, ((0, 0), (0, GBT), (0, 0), (0, 0)))
            return jnp.transpose(x, (0, 2, 1, 3))

    q_t, k_t, v_t, gk_t = _pack(q), _pack(k), _pack(v), _pack(gk)
    beta_t = _pack(beta.reshape(B, T, H, 1))

    if use_gate_in_kernel:
        assert A_log is not None
        A_f32 = jnp.minimum(A_log.reshape(-1).astype(jnp.float32), 80.0)
        if A_f32.shape[0] < H:
            A_f32 = jnp.repeat(A_f32, H // A_f32.shape[0], axis=0)
        A_scale_full = jnp.broadcast_to(jnp.exp(A_f32).reshape(1, H, 1, 1), (1, H, 1, K))
        if dt_bias is not None:
            db_f32 = dt_bias.reshape(-1, K).astype(jnp.float32)
            if db_f32.shape[0] < H:
                db_f32 = jnp.repeat(db_f32, H // db_f32.shape[0], axis=0)
            db_full = db_f32.reshape(1, H, 1, K)
        else:
            db_full = jnp.zeros((1, H, 1, K), dtype=jnp.float32)
        pad_rows = jnp.zeros((1, H, 6, K), dtype=jnp.float32)
        gate_params = jnp.concatenate([A_scale_full, db_full, pad_rows], axis=2)
        gate_spec = pl.BlockSpec([1, 1, 8, K], index_map=lambda n, h, nb, seqlens_ref: (0, h, 0, 0))
    else:
        gate_params = None
        gate_spec = None

    if valid_eos is None:
        valid_eos = cu_seqlens[1:]
    seqlens_meta = jnp.concatenate(
        [cu_seqlens.astype(jnp.int32), valid_eos.astype(jnp.int32)], axis=0
    )

    if single_seq:

        def _t_index_map(n, h, nb, seqlens_ref):
            return (0, h, nb, 0)

    else:

        def _t_index_map(n, h, nb, seqlens_ref):
            bos = pl.multiple_of(seqlens_ref[n], GBT)
            eos = pl.multiple_of(seqlens_ref[n + 1], GBT)
            block_idx = jnp.where(bos // GBT + nb < eos // GBT, bos // GBT + nb, T // GBT)
            return (0, h, block_idx, 0)

    def _state_index_map(n, h, nb, seqlens_ref):
        return (n, h, 0, 0)

    def _time_spec(width):
        return pl.BlockSpec([1, 1, GBT, width], index_map=_t_index_map)

    state_spec = pl.BlockSpec([1, 1, K, V], index_map=_state_index_map)
    o_shape = jax.ShapeDtypeStruct([B, H, T_alloc, V], q.dtype)
    ht_shape = jax.ShapeDtypeStruct([N, H, K, V], jnp.float32) if output_final_state else None

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _resident_pipeline_kernel,
            BT=BT,
            GROUP=group,
            K=K,
            V=V,
            N=N,
            scale=scale,
            intra_block_size=intra_block_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            USE_GATE_IN_KERNEL=use_gate_in_kernel,
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            NEED_CUMSUM=need_cumsum,
            SINGLE_SEQ=single_seq,
            HAS_PARTIAL_CHUNKS=has_partial_chunks,
            USE_INITIAL_STATE=h0 is not None,
            STORE_FINAL_STATE=output_final_state,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(N, H, T // GBT),
            in_specs=[
                _time_spec(K),
                _time_spec(K),
                _time_spec(V),
                _time_spec(K),
                _time_spec(1),
                gate_spec,
                None if h0 is None else state_spec,
            ],
            out_specs=[
                _time_spec(V),
                state_spec if output_final_state else None,
            ],
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        out_shape=[o_shape, ht_shape],
        interpret=get_interpret(),
    )(seqlens_meta, q_t, k_t, v_t, gk_t, beta_t, gate_params, h0)

    if single_seq:
        return jnp.transpose(o_out, (0, 2, 1, 3)), ht_out
    return jnp.transpose(o_out[:, :, :T, :], (0, 2, 1, 3)), ht_out


def resident_pipeline_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int = 64,
    intra_block_size: int = 16,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
):
    """Fused resident-state KDA prefill for variable-length (packed) sequences."""
    B, T, H, K = q.shape
    BT = chunk_size
    N = cu_seqlens.shape[-1] - 1

    assert cu_seqlens is not None, "cu_seqlens must not be None for varlen path"
    assert B == 1, f"varlen requires B=1 (packed layout), got B={B}"
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, v.shape[-1]), "v")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (N, H, K, v.shape[-1]), "initial_state")

    _orig_cu_seqlens = cu_seqlens
    T_input = T
    if N == 1:
        single_seq = True
        valid_eos = None
        if T % BT != 0:
            pad_t = BT - (T % BT)
            pad4 = ((0, 0), (0, pad_t), (0, 0), (0, 0))
            q = jnp.pad(q, pad4)
            k = jnp.pad(k, pad4)
            v = jnp.pad(v, pad4)
            g_pad = jnp.asarray(-1e30 if use_gate_in_kernel else 0.0, dtype=g.dtype)
            g = jnp.pad(g, pad4, constant_values=g_pad)
            beta = jnp.pad(beta, ((0, 0), (0, pad_t), (0, 0)))
            T = T + pad_t
        group = resident_group_chunks(T // BT)
    else:
        single_seq = False
        [q, k, v, g], [beta], cu_seqlens, _ = _align_seqs(
            [q, k, v, g],
            [beta],
            cu_seqlens,
            align=BT,
            pad_values_4d=[0, 0, 0, -1e30 if use_gate_in_kernel else 0],
        )
        T = q.shape[1]
        orig_lens = _orig_cu_seqlens[1:] - _orig_cu_seqlens[:-1]
        valid_eos = cu_seqlens[:-1] + orig_lens
        group = 1

    o, final_state = resident_pipeline_stage(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale,
        cu_seqlens,
        chunk_size=BT,
        group=group,
        intra_block_size=intra_block_size,
        output_final_state=output_final_state,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        need_cumsum=True,
        A_log=A_log,
        dt_bias=dt_bias,
        valid_eos=valid_eos,
        single_seq=single_seq,
        has_partial_chunks=False,
    )

    if N == 1:
        if o.shape[1] != T_input:
            o = o[:, :T_input]
    else:
        o = _unalign_output(o, _orig_cu_seqlens, cu_seqlens, T_input)
    return (
        o,
        final_state,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
