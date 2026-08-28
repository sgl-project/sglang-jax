# SPDX-License-Identifier: Apache-2.0
# Adapted from a private upstream Pallas-kernel repository.
# Upstream contact: pathfinder-pf.
"""Inference-only native segment-ID KDA forward Pallas kernel.

This module is intentionally separate from chunk_fwd_mega.py so inference
optimizations cannot change the training, recompute, or context-parallel paths.
"""

# This file retains the source kernel's compact mathematical notation.
# ruff: noqa: E702, E731, F841, SIM102, UP033

from __future__ import annotations

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


_RCP_LN2 = 1.0 / math.log(2)

CHUNK_KIND_ALL_PAD = 0
CHUNK_KIND_FULL_IN_SEGMENT = 1
CHUNK_KIND_BOUNDARY = 2
CHUNK_KIND_PARTIAL_PAD = 3
CHUNK_FLAG_LAST_REAL = 4


def _build_chunk_metadata(segment_ids, chunk_size):
    """Build per-chunk metadata from segment IDs with zero denoting padding."""
    orig_ndim = segment_ids.ndim
    if orig_ndim == 1:
        segment_ids = segment_ids[None, :]
    batch, tokens = segment_ids.shape
    block_tokens = int(chunk_size)
    num_blocks = tokens // block_tokens
    seg = segment_ids.astype(jnp.int32).reshape(batch, num_blocks, block_tokens)
    seg_first = seg[:, :, 0]
    valid_mask = seg > 0
    pos = jnp.arange(block_tokens, dtype=jnp.int32)
    last_valid_idx = jnp.where(valid_mask, pos[None, None, :], -1).max(axis=2)
    has_any = last_valid_idx >= 0
    safe_idx = jnp.where(has_any, last_valid_idx, 0)
    seg_last = jnp.where(
        has_any,
        jnp.take_along_axis(seg, safe_idx[:, :, None], axis=2)[:, :, 0],
        0,
    )
    seg_prev = jnp.concatenate(
        [
            jnp.zeros((batch, num_blocks, 1), dtype=jnp.int32),
            seg[:, :, :-1],
        ],
        axis=2,
    )
    is_new = (seg != seg_prev) & (seg > 0)
    distinct = is_new.sum(axis=2).astype(jnp.int32)
    has_pad = (seg == 0).any(axis=2)
    chunk_kind = jnp.where(
        distinct == 0,
        jnp.int32(CHUNK_KIND_ALL_PAD),
        jnp.where(
            distinct >= 2,
            jnp.int32(CHUNK_KIND_BOUNDARY),
            jnp.where(
                has_pad,
                jnp.int32(CHUNK_KIND_PARTIAL_PAD),
                jnp.int32(CHUNK_KIND_FULL_IN_SEGMENT),
            ),
        ),
    )
    seg_id = jnp.where(chunk_kind == CHUNK_KIND_FULL_IN_SEGMENT, seg_first, 0).astype(jnp.int32)
    if orig_ndim == 1:
        return chunk_kind[0], seg_first[0], seg_last[0], seg_id[0]
    return chunk_kind, seg_first, seg_last, seg_id


def _build_unbounded_intra_terms(q, k, g_cumsum, beta, scale, block_size, precision):
    """Build causal Aqk/L terms without positive gate exponentials.

    Kimi-Linear's unbounded gate can accumulate far below the BF16/FP32
    exponent range inside one 64-token tile.  Cross-block products use the
    first gate value of the row block as a shared reference, which keeps both
    factors in ``[0, 1]``.  The diagonal block uses the causal gate difference
    directly because no single shared reference is safe on both sides of its
    diagonal.
    """
    _, tokens, _ = q.shape
    num_blocks = tokens // block_size
    aqk_rows = []
    l_rows = []

    def _dot(lhs, rhs):
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2,), (2,)), ((0,), (0,))),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    for row_block in range(num_blocks):
        row_start = row_block * block_size
        row_end = row_start + block_size
        q_row = q[:, row_start:row_end]
        k_row = k[:, row_start:row_end]
        g_row = g_cumsum[:, row_start:row_end]
        reference = g_row[:, :1]
        row_decay = jnp.exp2(jnp.clip(g_row - reference, -126.0, 0.0))
        aqk_blocks = []
        l_blocks = []

        for col_block in range(row_block):
            col_start = col_block * block_size
            col_end = col_start + block_size
            k_col = k[:, col_start:col_end]
            g_col = g_cumsum[:, col_start:col_end]
            col_decay = jnp.exp2(jnp.clip(reference - g_col, -126.0, 0.0))
            scaled_k_col = k_col * col_decay
            aqk_blocks.append(_dot(q_row * row_decay, scaled_k_col))
            l_blocks.append(_dot(k_row * row_decay, scaled_k_col))

        gate_diff = g_row[:, :, None, :] - g_row[:, None, :, :]
        diagonal_decay = jnp.exp2(jnp.clip(gate_diff, -126.0, 0.0))
        aqk_diagonal = jnp.sum(
            q_row[:, :, None, :] * diagonal_decay * k_row[:, None, :, :],
            axis=-1,
        )
        l_diagonal = jnp.sum(
            k_row[:, :, None, :] * diagonal_decay * k_row[:, None, :, :],
            axis=-1,
        )
        row_iota = jax.lax.broadcasted_iota(jnp.int32, aqk_diagonal.shape, 1)
        col_iota = jax.lax.broadcasted_iota(jnp.int32, aqk_diagonal.shape, 2)
        aqk_blocks.append(jnp.where(row_iota >= col_iota, aqk_diagonal, 0.0))
        l_blocks.append(jnp.where(row_iota > col_iota, l_diagonal, 0.0))

        trailing = tokens - row_end
        if trailing:
            zeros = jnp.zeros((q.shape[0], block_size, trailing), dtype=jnp.float32)
            aqk_blocks.append(zeros)
            l_blocks.append(zeros)
        aqk_rows.append(jnp.concatenate(aqk_blocks, axis=2))
        l_rows.append(jnp.concatenate(l_blocks, axis=2))

    aqk = jnp.concatenate(aqk_rows, axis=1) * scale
    lower_matrix = jnp.concatenate(l_rows, axis=1) * beta[:, :, None]
    g_last = g_cumsum[:, -1:, :]
    kg = k * jnp.exp2(jnp.clip(g_last - g_cumsum, -126.0, 0.0))
    return aqk, lower_matrix, kg


def _build_bounded_intra_terms(q, k, g_cumsum, beta, scale, block_size, precision, safe_gate):
    """Build the existing fast factored Aqk/L terms for bounded K3 gates."""
    mini_batch, tokens, _ = q.shape
    num_blocks = tokens // block_size
    reference_index = block_size // 2 if safe_gate else 0
    row_iota = jax.lax.broadcasted_iota(jnp.int32, (block_size, tokens), 0)
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (block_size, tokens), 1)
    aqk_rows = []
    l_rows = []
    k_inverse_prefix = None
    previous_reference = None

    for block in range(num_blocks):
        start = block * block_size
        end = start + block_size
        q_block = q[:, start:end]
        k_block = k[:, start:end]
        g_block = g_cumsum[:, start:end]
        reference = g_block[:, reference_index : reference_index + 1]
        gate_difference = g_block - reference
        gate_exp = jnp.exp2(gate_difference)
        q_scaled = q_block * gate_exp
        k_scaled = k_block * gate_exp
        k_inverse_current = k_block * jnp.exp2(-gate_difference)
        if block == 0:
            k_inverse_prefix = k_inverse_current
        else:
            reference_decay = jnp.exp2(reference - previous_reference)
            k_inverse_prefix = jnp.concatenate(
                [k_inverse_prefix * reference_decay, k_inverse_current], axis=1
            )
        previous_reference = reference
        qk_scaled = jnp.concatenate([q_scaled, k_scaled], axis=1)
        qk_dot_valid = jax.lax.dot_general(
            qk_scaled,
            k_inverse_prefix,
            (((2,), (2,)), ((0,), (0,))),
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        if end < tokens:
            qk_dot = jnp.concatenate(
                [
                    qk_dot_valid,
                    jnp.zeros(
                        (mini_batch, 2 * block_size, tokens - end),
                        dtype=jnp.float32,
                    ),
                ],
                axis=2,
            )
        else:
            qk_dot = qk_dot_valid
        aqk_row = qk_dot[:, :block_size] * scale
        l_row = qk_dot[:, block_size:] * beta[:, start:end, None]
        in_diagonal_block = (col_iota >= start) & (col_iota < end)
        local_col = col_iota - start
        aqk_row = jnp.where((~in_diagonal_block) | (row_iota >= local_col), aqk_row, 0.0)
        l_row = jnp.where((~in_diagonal_block) | (row_iota > local_col), l_row, 0.0)
        aqk_rows.append(aqk_row)
        l_rows.append(l_row)

    g_last = g_cumsum[:, -1:, :]
    kg = k_inverse_prefix * jnp.exp2(g_last - previous_reference)
    return jnp.concatenate(aqk_rows, axis=1), jnp.concatenate(l_rows, axis=1), kg


def _solve_unbounded_block_forward(L, rhs, block_inverse, block_size, precision):
    """Solve ``(I + L) x = rhs`` with stable block forward substitution."""
    _, tokens, _ = L.shape
    solved_blocks = []

    def _dot(lhs, rhs):
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2,), (1,)), ((0,), (0,))),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    for block in range(tokens // block_size):
        start = block * block_size
        end = start + block_size
        residual = rhs[:, start:end]
        if solved_blocks:
            solved_prefix = jnp.concatenate(solved_blocks, axis=1)
            residual -= _dot(L[:, start:end, :start], solved_prefix)
        inverse = block_inverse[:, start:end, start:end]
        solved_blocks.append(_dot(inverse, residual))
    return jnp.concatenate(solved_blocks, axis=1)


def _solve_unbounded_unit_lower(L, rhs, block_size, precision):
    """Create small diagonal inverses, then run ordered block substitution."""
    tokens = L.shape[1]
    identity = jnp.eye(tokens, dtype=jnp.float32)
    indices = jnp.arange(tokens, dtype=jnp.int32)
    block_indices = indices // block_size
    same_block = (block_indices[:, None] == block_indices[None, :]).astype(jnp.float32)
    negative_diagonal_blocks = -(L.astype(jnp.float32) * same_block[None])

    def _dot(lhs, rhs):
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2,), (1,)), ((0,), (0,))),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    block_inverse = identity[None] + negative_diagonal_blocks
    power = negative_diagonal_blocks
    num_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[block_size]
    for _ in range(num_steps):
        power = _dot(power, power)
        block_inverse = _dot(block_inverse, identity[None] + power)
    return _solve_unbounded_block_forward(L, rhs, block_inverse, block_size, precision)


# =====================================================================
# Native segment_ids mega kernel -- avoids _align_seqs gather/scatter
# =====================================================================


def _fwd_mega_kernel_native_segids(
    # Scalar-prefetch metadata
    chunk_kind_meta_ref,  # [B, NT_META_PAD]
    seg_first_meta_ref,  # [B, NT_META_PAD]
    seg_last_meta_ref,  # [B, NT_META_PAD]
    # Prefetch: segment_ids
    seg_ids_ref,  # [B, 128] (128-aligned block)
    # Inputs
    q_ref,  # [MB, 1, BT, K_PAD]
    k_ref,  # [MB, 1, BT, K_PAD]
    v_ref,  # [MB, 1, BT, V_ALIGNED]
    g_ref,  # [MB, 1, BT, K_PAD]
    beta_ref,  # [MB, 1, 1, 1, BT]
    h0_ref,  # auto window or full HBM ref for manual DMA
    A_scale_ref,  # [MB, 1, 1, 1] or None
    dt_bias_ref,  # [MB, 1, 1, K_PAD] or None
    # Outputs
    o_ref,  # [MB, 1, BT, V_ALIGNED]
    ht_ref,  # all segments or streaming [2, MB, 1, K_PAD, V]
    Aqk_ref,  # [MB, 1, BT, BT]
    Akk_ref,  # [MB, 1, BT, BT]
    g_cumsum_out_ref,  # [MB, 1, BT, K_PAD]
    chunk_h_out_ref,  # [MB, 1, 1, K_PAD, V_ALIGNED] per-chunk h
    # Scratch (VMEM persists across the NT loop for the same head and batch)
    scratch_ref,  # [MB, K_PAD, V_ALIGNED] KV state
    prev_seg_ref,  # [1] previous segment ID
    final_state_dma_ref,  # [MB, K_PAD, V_ALIGNED] state write staging
    state_dma_sem,  # DMA semaphore for conditional state transfers
    *,
    NT,
    BT,
    N_max,
    scale,
    cumsum_scale,
    MB,
    K_PAD,
    V_PAD,
    OUTPUT_PRECISION,
    safe_gate,
    NORMALIZE_QK,
    use_gate_in_kernel,
    lower_bound,
    USE_NEUMANN,
    QK_BC,
    INV_BC,
    SKIP_STAGE4_MASK,
    PACK_HEAD_INV,
    CLIP_BETA_IN_KERNEL,
    PACKED_METADATA,
    HAS_H0,
    STORE_RESIDUALS,
    STORE_H,
    STORE_FINAL_STATE,
    BATCH_FIRST,
    MANUAL_H0_DMA,
    MANUAL_HT_DMA,
    OVERLAP_H0_DMA,
    OVERLAP_HT_DMA,
    RESIDUAL_CHUNK_LAYOUT,
):
    """Native segment_ids mega kernel body."""
    i_b = pl.program_id(1)
    i_c = pl.program_id(2)
    chain_start = i_c == 0
    t0 = i_c * BT

    def _load_token_segment_ids():
        seg_full = seg_ids_ref[i_b, :].astype(jnp.int32)  # [128]
        if BT == 128:
            return seg_full
        seg_first_half = seg_full[:BT]
        seg_second_half = seg_full[BT : 2 * BT]
        use_second = ((i_c * BT) % 128) >= BT
        return jnp.where(use_second, seg_second_half, seg_first_half)

    if PACKED_METADATA:
        packed_metadata = chunk_kind_meta_ref[i_b, i_c]
        encoded_kind = packed_metadata & jnp.int32(0x7)
        first_seg = (packed_metadata >> jnp.int32(3)) & jnp.int32(0x3FFF)
        last_seg = packed_metadata >> jnp.int32(17)
    else:
        encoded_kind = chunk_kind_meta_ref[i_b, i_c]
        first_seg = seg_first_meta_ref[i_b, i_c]
        last_seg = seg_last_meta_ref[i_b, i_c]
    kind = encoded_kind & jnp.int32(0x3)
    is_last_real = (encoded_kind & jnp.int32(CHUNK_FLAG_LAST_REAL)) != 0

    def _dma_state_to_ht(src_ref, segment_id):
        h_start = pl.program_id(0) * MB
        cp = pltpu.make_async_copy(
            src_ref,
            ht_ref.at[
                segment_id - 1,
                pl.ds(h_start, MB),
                i_b,
                pl.ds(None),
                pl.ds(None),
            ],
            state_dma_sem,
        )
        cp.start()
        cp.wait()

    def _read_h0(segment_id):
        return h0_ref[segment_id - 1, :, 0, :, :].astype(jnp.float32)

    # ALL_PAD ----------------------------------------------------------
    @pl.when(kind == CHUNK_KIND_ALL_PAD)
    def _all_pad():
        if BATCH_FIRST:
            o_ref[0] = jnp.zeros([BT, MB, V_PAD], dtype=o_ref.dtype)
        else:
            o_ref[:, 0] = jnp.zeros([MB, BT, V_PAD], dtype=o_ref.dtype)
        if STORE_RESIDUALS:
            if RESIDUAL_CHUNK_LAYOUT:
                Aqk_ref[:, 0, 0] = jnp.zeros([MB, BT, BT], dtype=Aqk_ref.dtype)
                Akk_ref[:, 0, 0] = jnp.zeros([MB, BT, BT], dtype=Akk_ref.dtype)
            else:
                Aqk_ref[:, 0] = jnp.zeros([MB, BT, BT], dtype=Aqk_ref.dtype)
                Akk_ref[:, 0] = jnp.zeros([MB, BT, BT], dtype=Akk_ref.dtype)

            g_cumsum_out_ref[:, 0] = jnp.zeros([MB, BT, K_PAD], dtype=g_cumsum_out_ref.dtype)

    # FULL_IN_SEGMENT fast path ---------------------------------------
    @pl.when(kind == CHUNK_KIND_FULL_IN_SEGMENT)
    def _full():
        prev_seg = prev_seg_ref[0]
        seg_changed = (first_seg != prev_seg) | chain_start

        if STORE_FINAL_STATE:

            @pl.when(seg_changed & (~chain_start))
            def _save_prev():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, prev_seg)
                else:
                    ht_ref[prev_seg - 1, ...] = scratch_ref[...].astype(ht_ref.dtype)[:, None, :, :]

        @pl.when(seg_changed)
        def _init():
            scratch_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)

        if HAS_H0:

            @pl.when(seg_changed)
            def _load_h0():
                if MANUAL_H0_DMA:
                    h_start = pl.program_id(0) * MB
                    cp = pltpu.make_async_copy(
                        h0_ref.at[
                            first_seg - 1,
                            pl.ds(h_start, MB),
                            i_b,
                            pl.ds(None),
                            pl.ds(None),
                        ],
                        scratch_ref,
                        state_dma_sem,
                    )
                    cp.start()
                    cp.wait()
                else:
                    scratch_ref[...] = _read_h0(first_seg)

        # --- Stage 1: Gate cumsum ---
        if BATCH_FIRST:
            q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            g_raw = g_ref[0].transpose(1, 0, 2).astype(jnp.float32)
        else:
            q = q_ref[:, 0, :, :].astype(jnp.float32)
            k = k_ref[:, 0, :, :].astype(jnp.float32)
            v = v_ref[:, 0, :, :].astype(jnp.float32)
            g_raw = g_ref[:, 0, :, :].astype(jnp.float32)
        beta = (
            beta_ref[0, 0].transpose(1, 0).astype(jnp.float32)
            if BATCH_FIRST and beta_ref.ndim == 4
            else beta_ref[:, 0, 0, 0, :].astype(jnp.float32)
        )
        if CLIP_BETA_IN_KERNEL:
            beta = jnp.clip(beta, 0, 1)
        if NORMALIZE_QK:
            q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            q = q.astype(jnp.bfloat16).astype(jnp.float32)
            k = k.astype(jnp.bfloat16).astype(jnp.float32)

        # Gate activation (matches cu_seqlens kernel)
        g_f32 = g_raw
        if use_gate_in_kernel:
            dt_b = dt_bias_ref[:, 0, 0]
            g_f32 = g_f32 + dt_b[:, None, :]
            A_scale = A_scale_ref[:, 0, 0, 0]
            if lower_bound is None:
                g_f32 = -A_scale[:, None, None] * jax.nn.softplus(g_f32)
            else:
                g_f32 = lower_bound * jax.nn.sigmoid(A_scale[:, None, None] * g_f32)

        g_cumsum = g_f32 * cumsum_scale
        shift = 1
        while shift < BT:
            shifted = jnp.concatenate(
                [
                    jnp.zeros_like(g_cumsum[:, :shift]),
                    g_cumsum[:, :-shift],
                ],
                axis=1,
            )
            g_cumsum = g_cumsum + shifted
            shift *= 2

        # --- Stage 2: Intra-chunk solve ---
        BC = QK_BC
        beta_f32 = beta[:, :, None]
        if lower_bound is None:
            Aqk, L, kg = _build_unbounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION
            )
        else:
            Aqk, L, kg = _build_bounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION, safe_gate
            )

        v_beta = v * beta_f32
        k_eg_beta = k * jnp.exp2(g_cumsum) * beta_f32
        I_bt = jnp.eye(BT, dtype=jnp.float32)
        _dot = lambda a, b: jax.lax.dot_general(
            a,
            b,
            (((2,), (1,)), ((0,), (0,))),
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )

        if USE_NEUMANN:
            BC_inv = INV_BC
            NC_inv = BT // BC_inv
            inv_dt = jnp.float32
            L_inv = L.astype(inv_dt)
            solve_bt = BT
            solve_I = I_bt
            if PACK_HEAD_INV:
                # Pair independent heads in the MXU matrix dimensions. A
                # 64x64 batched dot occupies a physical 128x128 tile, so two
                # heads can share that tile without changing either BC8
                # triangular system.
                pair_mb = MB // 2
                L_pair = L_inv.reshape(pair_mb, 2, BT, BT)
                z = jnp.zeros_like(L_pair[:, 0])
                L_inv = jnp.concatenate(
                    [
                        jnp.concatenate([L_pair[:, 0], z], axis=2),
                        jnp.concatenate([z, L_pair[:, 1]], axis=2),
                    ],
                    axis=1,
                )
                solve_bt = 2 * BT
                solve_I = jnp.eye(solve_bt, dtype=inv_dt)
            _idx = jnp.arange(solve_bt, dtype=jnp.int32)
            _blk = _idx // BC_inv
            _same = (_blk[:, None] == _blk[None, :]).astype(inv_dt)
            L_diag = L_inv * _same[None]
            F = L_inv - L_diag
            neg_Ld = -L_diag
            S = solve_I[None] + neg_Ld
            Mk = neg_Ld
            num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
            for _ in range(num_diag_steps):
                Mk = _dot(Mk, Mk)
                S = _dot(S, solve_I[None] + Mk)
            P = S
            rhs = jnp.concatenate([v_beta.astype(inv_dt), k_eg_beta.astype(inv_dt)], axis=-1)
            if PACK_HEAD_INV:
                rhs = rhs.reshape(MB // 2, 2 * BT, V_PAD + K_PAD)
            if NC_inv == 1:
                result = _dot(P, rhs)
            else:
                F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
                P_merged = _dot(P, F_and_rhs)
                G = P_merged[:, :, :solve_bt]
                P_rhs = P_merged[:, :, solve_bt:]
                H_mat = -G
                inv_I_G = solve_I[None] + H_mat
                Hk = H_mat
                log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
                num_horner_steps = log2_NC_inv - 1
                if num_horner_steps > 0:
                    Hk = _dot(Hk, Hk)
                    for _ in range(num_horner_steps - 1):
                        merged_lhs = jnp.concatenate([inv_I_G, Hk], axis=1)
                        merged_products = _dot(merged_lhs, Hk)
                        inv_I_G = inv_I_G + merged_products[:, :solve_bt]
                        Hk = merged_products[:, solve_bt:]
                    inv_I_G = inv_I_G + _dot(inv_I_G, Hk)
                if STORE_RESIDUALS:
                    rhs_and_inverse = jnp.concatenate([P_rhs, P], axis=-1)
                    result_and_inverse = _dot(inv_I_G, rhs_and_inverse)

                    result = result_and_inverse[:, :, : V_PAD + K_PAD]
                    A_inv = result_and_inverse[:, :, V_PAD + K_PAD :]
                else:
                    result = _dot(inv_I_G, P_rhs)
            if PACK_HEAD_INV:
                result = result.reshape(MB, BT, V_PAD + K_PAD)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]
            if NC_inv == 1:
                A_inv = P
        else:
            rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)
            result = _solve_unbounded_unit_lower(L, rhs, INV_BC, OUTPUT_PRECISION)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]

        # --- Stage 3+4: State + Output ---
        b_h = scratch_ref[...]
        if STORE_H:
            chunk_h_out_ref[:, 0, 0] = b_h.astype(chunk_h_out_ref.dtype)
        b_qg = q * jnp.exp2(jnp.maximum(g_cumsum, -126.0))
        b_v_o = jnp.matmul(
            jnp.concatenate([w, b_qg], axis=1),
            b_h,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        b_v_new = u - b_v_o[:, :BT]
        b_o = b_v_o[:, BT:] * scale
        if SKIP_STAGE4_MASK:
            b_A = Aqk.astype(jnp.float32)
        else:
            m_s = (jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]).astype(jnp.float32)
            b_A = jnp.where(m_s[None, :, :], Aqk.astype(jnp.float32), 0.0)
        b_o_h = jnp.matmul(
            jnp.concatenate([b_A, kg.transpose(0, 2, 1)], axis=1),
            b_v_new,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        b_o += b_o_h[:, :BT]
        if BATCH_FIRST:
            o_ref[0] = b_o.transpose(1, 0, 2).astype(o_ref.dtype)
        else:
            o_ref[:, 0] = b_o.astype(o_ref.dtype)

        b_gk_last = g_cumsum[:, BT - 1, :]
        b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None] + b_o_h[:, BT:]
        scratch_ref[...] = b_h_new
        if STORE_RESIDUALS:
            if RESIDUAL_CHUNK_LAYOUT:
                Aqk_ref[:, 0, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0, 0] = A_inv.astype(Akk_ref.dtype)
            else:
                Aqk_ref[:, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0] = A_inv.astype(Akk_ref.dtype)
            g_cumsum_out_ref[:, 0] = g_cumsum.astype(g_cumsum_out_ref.dtype)
        prev_seg_ref[...] = jnp.broadcast_to(first_seg, (128,))

        if STORE_FINAL_STATE:

            @pl.when(is_last_real)
            def _final():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, first_seg)
                else:
                    ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[:, None, :, :]

    # PARTIAL_PAD ------------------------------------------------------
    @pl.when(kind == CHUNK_KIND_PARTIAL_PAD)
    def _partial():
        seg = _load_token_segment_ids()
        prev_seg = prev_seg_ref[0]
        seg_changed = (first_seg != prev_seg) | chain_start

        if STORE_FINAL_STATE:

            @pl.when(seg_changed & (~chain_start))
            def _save_prev():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, prev_seg)
                else:
                    ht_ref[prev_seg - 1, ...] = scratch_ref[...].astype(ht_ref.dtype)[:, None, :, :]

        @pl.when(seg_changed)
        def _init():
            scratch_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)

        if HAS_H0:

            @pl.when(seg_changed)
            def _load_h0():
                if MANUAL_H0_DMA:
                    h_start = pl.program_id(0) * MB
                    cp = pltpu.make_async_copy(
                        h0_ref.at[
                            first_seg - 1,
                            pl.ds(h_start, MB),
                            i_b,
                            pl.ds(None),
                            pl.ds(None),
                        ],
                        scratch_ref,
                        state_dma_sem,
                    )
                    cp.start()
                    cp.wait()
                else:
                    scratch_ref[...] = _read_h0(first_seg)

        vm = (seg > 0).astype(jnp.float32)  # [BT] valid mask
        if BATCH_FIRST:
            q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
            k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
            v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
            g_raw = g_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
        else:
            q = q_ref[:, 0, :, :].astype(jnp.float32) * vm[None, :, None]
            k = k_ref[:, 0, :, :].astype(jnp.float32) * vm[None, :, None]
            v = v_ref[:, 0, :, :].astype(jnp.float32) * vm[None, :, None]
            g_raw = g_ref[:, 0, :, :].astype(jnp.float32) * vm[None, :, None]
        beta = (
            beta_ref[0, 0].transpose(1, 0).astype(jnp.float32)
            if BATCH_FIRST and beta_ref.ndim == 4
            else beta_ref[:, 0, 0, 0, :].astype(jnp.float32)
        ) * vm[None, :]
        if CLIP_BETA_IN_KERNEL:
            beta = jnp.clip(beta, 0, 1)
        if NORMALIZE_QK:
            q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            q = q.astype(jnp.bfloat16).astype(jnp.float32)
            k = k.astype(jnp.bfloat16).astype(jnp.float32)

        # Gate activation (must match FULL chunk)
        g_f32 = g_raw
        if use_gate_in_kernel:
            dt_b = dt_bias_ref[:, 0, 0]
            g_f32 = g_f32 + dt_b[:, None, :] * vm[None, :, None]
            A_scale = A_scale_ref[:, 0, 0, 0]
            if lower_bound is None:
                g_f32 = -A_scale[:, None, None] * jax.nn.softplus(g_f32)
            else:
                g_f32 = lower_bound * jax.nn.sigmoid(A_scale[:, None, None] * g_f32)
            g_f32 = g_f32 * vm[None, :, None]

        g_cumsum = g_f32 * cumsum_scale
        shift = 1
        while shift < BT:
            shifted = jnp.concatenate(
                [
                    jnp.zeros_like(g_cumsum[:, :shift]),
                    g_cumsum[:, :-shift],
                ],
                axis=1,
            )
            g_cumsum = g_cumsum + shifted
            shift *= 2

        BC = QK_BC
        beta_f32 = beta[:, :, None]
        if lower_bound is None:
            Aqk, L, _ = _build_unbounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION
            )
        else:
            Aqk, L, _ = _build_bounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION, safe_gate
            )
        v_beta = v * beta_f32
        k_eg_beta = k * jnp.exp2(g_cumsum) * beta_f32
        I_bt = jnp.eye(BT, dtype=jnp.float32)
        _dot = lambda a, b: jax.lax.dot_general(
            a,
            b,
            (((2,), (1,)), ((0,), (0,))),
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        if USE_NEUMANN:
            BC_inv = INV_BC
            NC_inv = BT // BC_inv
            inv_dt = jnp.float32
            L_inv = L.astype(inv_dt)
            _idx = jnp.arange(BT, dtype=jnp.int32)
            _blk = _idx // BC_inv
            _same = (_blk[:, None] == _blk[None, :]).astype(inv_dt)
            L_diag = L_inv * _same[None]
            F = L_inv - L_diag
            neg_Ld = -L_diag
            S = I_bt[None] + neg_Ld
            Mk = neg_Ld
            num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
            for _ in range(num_diag_steps):
                Mk = _dot(Mk, Mk)
                S = _dot(S, I_bt[None] + Mk)
            P = S
            rhs = jnp.concatenate([v_beta.astype(inv_dt), k_eg_beta.astype(inv_dt)], axis=-1)
            if NC_inv == 1:
                result = _dot(P, rhs)
            else:
                F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
                P_merged = _dot(P, F_and_rhs)
                G = P_merged[:, :, :BT]
                P_rhs = P_merged[:, :, BT:]
                H_mat = -G
                inv_I_G = I_bt[None] + H_mat
                Hk = H_mat
                log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
                num_horner_steps = log2_NC_inv - 1
                if num_horner_steps > 0:
                    Hk = _dot(Hk, Hk)
                    for _ in range(num_horner_steps - 1):
                        merged_lhs = jnp.concatenate([inv_I_G, Hk], axis=1)
                        merged_products = _dot(merged_lhs, Hk)
                        inv_I_G = inv_I_G + merged_products[:, :BT]
                        Hk = merged_products[:, BT:]
                    inv_I_G = inv_I_G + _dot(inv_I_G, Hk)
                if STORE_RESIDUALS:
                    rhs_and_inverse = jnp.concatenate([P_rhs, P], axis=-1)
                    result_and_inverse = _dot(inv_I_G, rhs_and_inverse)
                    result = result_and_inverse[:, :, : V_PAD + K_PAD]
                    A_inv = result_and_inverse[:, :, V_PAD + K_PAD :]
                else:
                    result = _dot(inv_I_G, P_rhs)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]
            if NC_inv == 1:
                A_inv = P
        else:
            rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)
            result = _solve_unbounded_unit_lower(L, rhs, INV_BC, OUTPUT_PRECISION)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]
        g_last = (g_cumsum * vm[None, :, None]).min(axis=1, keepdims=True)
        kg = k * jnp.exp2(g_last - g_cumsum) * vm[None, :, None]

        b_h = scratch_ref[...]
        if STORE_H:
            chunk_h_out_ref[:, 0, 0] = b_h.astype(chunk_h_out_ref.dtype)
        b_qg = q * jnp.exp2(jnp.maximum(g_cumsum, -126.0))
        b_v_o = jnp.matmul(
            jnp.concatenate([w, b_qg], axis=1),
            b_h,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        b_v_new = u - b_v_o[:, :BT]
        b_o = b_v_o[:, BT:] * scale
        if SKIP_STAGE4_MASK:
            b_A = Aqk.astype(jnp.float32)
        else:
            m_s = (jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]).astype(jnp.float32)
            b_A = jnp.where(m_s[None, :, :], Aqk.astype(jnp.float32), 0.0)
        b_o_h = jnp.matmul(
            jnp.concatenate([b_A, kg.transpose(0, 2, 1)], axis=1),
            b_v_new,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        b_o += b_o_h[:, :BT]
        b_o = b_o * vm[None, :, None]
        if BATCH_FIRST:
            o_ref[0] = b_o.transpose(1, 0, 2).astype(o_ref.dtype)
        else:
            o_ref[:, 0] = b_o.astype(o_ref.dtype)

        b_gk_last = g_last[:, 0, :]
        b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None] + b_o_h[:, BT:]
        scratch_ref[...] = b_h_new
        if STORE_FINAL_STATE:
            if MANUAL_HT_DMA:
                _dma_state_to_ht(scratch_ref, first_seg)
            else:
                ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[:, None, :, :]
        if STORE_RESIDUALS:
            if RESIDUAL_CHUNK_LAYOUT:
                Aqk_ref[:, 0, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0, 0] = A_inv.astype(Akk_ref.dtype)
            else:
                Aqk_ref[:, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0] = A_inv.astype(Akk_ref.dtype)
            g_cumsum_out_ref[:, 0] = g_cumsum.astype(g_cumsum_out_ref.dtype)
        prev_seg_ref[...] = jnp.broadcast_to(first_seg, (128,))

        if STORE_FINAL_STATE:

            @pl.when(is_last_real)
            def _final():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, first_seg)
                else:
                    ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[:, None, :, :]

    # BOUNDARY: two segments in one chunk -----------------------------
    @pl.when(kind == CHUNK_KIND_BOUNDARY)
    def _boundary():
        seg = _load_token_segment_ids()
        prev_seg = prev_seg_ref[0]
        seg_changed = (first_seg != prev_seg) | chain_start

        if STORE_FINAL_STATE:

            @pl.when(seg_changed & (~chain_start))
            def _save_prev():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, prev_seg)
                else:
                    ht_ref[prev_seg - 1, ...] = scratch_ref[...].astype(ht_ref.dtype)[:, None, :, :]

        @pl.when(seg_changed)
        def _init():
            scratch_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)

        if HAS_H0:

            @pl.when(seg_changed)
            def _load_h0():
                if MANUAL_H0_DMA:
                    h_start = pl.program_id(0) * MB
                    cp = pltpu.make_async_copy(
                        h0_ref.at[
                            first_seg - 1,
                            pl.ds(h_start, MB),
                            i_b,
                            pl.ds(None),
                            pl.ds(None),
                        ],
                        scratch_ref,
                        state_dma_sem,
                    )
                    cp.start()
                    cp.wait()
                else:
                    scratch_ref[...] = _read_h0(first_seg)

        if HAS_H0 and MANUAL_H0_DMA and OVERLAP_H0_DMA:
            h_start = pl.program_id(0) * MB
            h0_B_copy = pltpu.make_async_copy(
                h0_ref.at[
                    last_seg - 1,
                    pl.ds(h_start, MB),
                    i_b,
                    pl.ds(None),
                    pl.ds(None),
                ],
                final_state_dma_ref,
                state_dma_sem,
            )
            h0_B_copy.start()

        # Segment masks
        seg_A_mask = (seg == first_seg).astype(jnp.float32)  # [BT]
        seg_B_mask = (seg == last_seg).astype(jnp.float32)  # [BT]

        # --- FULL-style computation (backward-compatible) ---
        if BATCH_FIRST:
            q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
            g_raw = g_ref[0].transpose(1, 0, 2).astype(jnp.float32)
        else:
            q = q_ref[:, 0, :, :].astype(jnp.float32)
            k = k_ref[:, 0, :, :].astype(jnp.float32)
            v = v_ref[:, 0, :, :].astype(jnp.float32)
            g_raw = g_ref[:, 0, :, :].astype(jnp.float32)
        beta = (
            beta_ref[0, 0].transpose(1, 0).astype(jnp.float32)
            if BATCH_FIRST and beta_ref.ndim == 4
            else beta_ref[:, 0, 0, 0, :].astype(jnp.float32)
        )
        if CLIP_BETA_IN_KERNEL:
            beta = jnp.clip(beta, 0, 1)
        if NORMALIZE_QK:
            q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            q = q.astype(jnp.bfloat16).astype(jnp.float32)
            k = k.astype(jnp.bfloat16).astype(jnp.float32)

        # Gate activation
        g_f32 = g_raw
        if use_gate_in_kernel:
            dt_b = dt_bias_ref[:, 0, 0]
            g_f32 = g_f32 + dt_b[:, None, :]
            A_scale = A_scale_ref[:, 0, 0, 0]
            if lower_bound is None:
                g_f32 = -A_scale[:, None, None] * jax.nn.softplus(g_f32)
            else:
                g_f32 = lower_bound * jax.nn.sigmoid(A_scale[:, None, None] * g_f32)

        g_cumsum = g_f32 * cumsum_scale
        shift = 1
        while shift < BT:
            shifted = jnp.concatenate(
                [
                    jnp.zeros_like(g_cumsum[:, :shift]),
                    g_cumsum[:, :-shift],
                ],
                axis=1,
            )
            g_cumsum = g_cumsum + shifted
            shift *= 2

        # Aqk/L (FULL-style, no cross-seg masking in loop for stability)
        BC = QK_BC
        beta_f32 = beta[:, :, None]
        # Restart the log-domain prefix at the packed segment boundary. This
        # preserves segment B's initial-state contribution; within-segment
        # gate differences used by Aqk/L are unchanged by the constant shift.
        first_segment_total = jnp.min(
            jnp.where(seg_A_mask[None, :, None] > 0, g_cumsum, 0.0),
            axis=1,
            keepdims=True,
        )
        g_cumsum = g_cumsum - first_segment_total * seg_B_mask[None, :, None]
        if lower_bound is None:
            Aqk, L, _ = _build_unbounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION
            )
        else:
            Aqk, L, _ = _build_bounded_intra_terms(
                q, k, g_cumsum, beta, scale, BC, OUTPUT_PRECISION, safe_gate
            )

        # Mask L for segment-independent solve
        same_seg_L = (
            seg_A_mask[:, None] * seg_A_mask[None, :] + seg_B_mask[:, None] * seg_B_mask[None, :]
        )
        L = L * same_seg_L[None]

        # Solve (same as FULL)
        v_beta = v * beta_f32
        k_eg_beta = k * jnp.exp2(g_cumsum) * beta_f32
        I_bt = jnp.eye(BT, dtype=jnp.float32)
        _dot = lambda a, b: jax.lax.dot_general(
            a, b, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

        if USE_NEUMANN:
            BC_inv = INV_BC
            NC_inv = BT // BC_inv
            inv_dt = jnp.float32
            L_inv = L.astype(inv_dt)
            solve_bt = BT
            solve_I = I_bt
            if PACK_HEAD_INV:
                pair_mb = MB // 2
                L_pair = L_inv.reshape(pair_mb, 2, BT, BT)
                z = jnp.zeros_like(L_pair[:, 0])
                L_inv = jnp.concatenate(
                    [
                        jnp.concatenate([L_pair[:, 0], z], axis=2),
                        jnp.concatenate([z, L_pair[:, 1]], axis=2),
                    ],
                    axis=1,
                )
                solve_bt = 2 * BT
                solve_I = jnp.eye(solve_bt, dtype=inv_dt)
            _idx = jnp.arange(solve_bt, dtype=jnp.int32)
            _blk = _idx // BC_inv
            _same = (_blk[:, None] == _blk[None, :]).astype(inv_dt)
            L_diag = L_inv * _same[None]
            F = L_inv - L_diag
            neg_Ld = -L_diag
            S = solve_I[None] + neg_Ld
            Mk = neg_Ld
            num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
            for _ in range(num_diag_steps):
                Mk = _dot(Mk, Mk)
                S = _dot(S, solve_I[None] + Mk)
            P = S
            rhs = jnp.concatenate([v_beta.astype(inv_dt), k_eg_beta.astype(inv_dt)], axis=-1)
            if PACK_HEAD_INV:
                rhs = rhs.reshape(MB // 2, 2 * BT, V_PAD + K_PAD)
            if NC_inv == 1:
                result = _dot(P, rhs)
                A_inv = P
            else:
                F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
                P_merged = _dot(P, F_and_rhs)
                G = P_merged[:, :, :solve_bt]
                P_rhs = P_merged[:, :, solve_bt:]
                H_mat = -G
                inv_I_G = solve_I[None] + H_mat
                Hk = H_mat
                log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
                num_horner_steps = log2_NC_inv - 1
                if num_horner_steps > 0:
                    Hk = _dot(Hk, Hk)
                    for _ in range(num_horner_steps - 1):
                        merged_lhs = jnp.concatenate([inv_I_G, Hk], axis=1)
                        merged_products = _dot(merged_lhs, Hk)
                        inv_I_G = inv_I_G + merged_products[:, :solve_bt]
                        Hk = merged_products[:, solve_bt:]
                    inv_I_G = inv_I_G + _dot(inv_I_G, Hk)
                if STORE_RESIDUALS:
                    rhs_and_inverse = jnp.concatenate([P_rhs, P], axis=-1)
                    result_and_inverse = _dot(inv_I_G, rhs_and_inverse)
                    result = result_and_inverse[:, :, : V_PAD + K_PAD]
                    A_inv = result_and_inverse[:, :, V_PAD + K_PAD :]
                else:
                    result = _dot(inv_I_G, P_rhs)
            if PACK_HEAD_INV:
                result = result.reshape(MB, BT, V_PAD + K_PAD)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]
        else:
            rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)
            result = _solve_unbounded_unit_lower(L, rhs, INV_BC, OUTPUT_PRECISION)
            u = result[:, :, :V_PAD]
            w = result[:, :, V_PAD : V_PAD + K_PAD]

        if HAS_H0:
            if MANUAL_H0_DMA:
                if OVERLAP_H0_DMA:
                    h0_B_copy.wait()
                else:
                    h_start = pl.program_id(0) * MB
                    cp = pltpu.make_async_copy(
                        h0_ref.at[
                            last_seg - 1,
                            pl.ds(h_start, MB),
                            i_b,
                            pl.ds(None),
                            pl.ds(None),
                        ],
                        final_state_dma_ref,
                        state_dma_sem,
                    )
                    cp.start()
                    cp.wait()
            else:
                final_state_dma_ref[...] = _read_h0(last_seg)
        else:
            final_state_dma_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)
        h_B_initial = final_state_dma_ref[...]

        g_last = g_cumsum[:, BT - 1 : BT, :]
        kg = k * jnp.exp2(g_last - g_cumsum)

        # --- Output + State ---
        b_h = scratch_ref[...]
        if STORE_H:
            chunk_h_out_ref[:, 0, 0] = b_h.astype(chunk_h_out_ref.dtype)

        b_qg = q * jnp.exp2(jnp.maximum(g_cumsum, -126.0))
        b_v_o = jnp.matmul(
            jnp.concatenate([w, b_qg], axis=1),
            b_h,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        w_h = b_v_o[:, :BT]
        b_v_new = u - w_h
        state_o = b_v_o[:, BT:] * scale
        b_v_o_B = jnp.matmul(
            jnp.concatenate([w, b_qg], axis=1),
            h_B_initial,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        w_h_B = b_v_o_B[:, :BT]
        state_o_B = b_v_o_B[:, BT:] * scale

        # Output: Aqk is already same-segment masked.  Build the correct
        # per-segment v_new before the intra-chunk matmul so segment B does not
        # need two additional Aqk compensation matmuls.
        Aqk_seg = Aqk * same_seg_L[None]
        if SKIP_STAGE4_MASK:
            Aqk_stage4 = Aqk_seg.astype(jnp.float32)
        else:
            m_s = (jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]).astype(jnp.float32)
            Aqk_stage4 = jnp.where(m_s[None, :, :], Aqk_seg.astype(jnp.float32), 0.0)
        mask_A = seg_A_mask[None, :, None]
        mask_B = seg_B_mask[None, :, None]
        v_new_B = (u - w_h_B) * mask_B
        intra_rhs = b_v_new * mask_A + v_new_B
        intra_o = jnp.matmul(
            Aqk_stage4,
            intra_rhs,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )
        b_o = (state_o + intra_o) * mask_A + (state_o_B + intra_o) * mask_B
        if BATCH_FIRST:
            o_ref[0] = b_o.transpose(1, 0, 2).astype(o_ref.dtype)
        else:
            o_ref[:, 0] = b_o.astype(o_ref.dtype)

        # State: per-segment
        g_A_total = jnp.sum(g_f32 * seg_A_mask[None, :, None], axis=1, keepdims=True) * cumsum_scale
        g_B_total = jnp.sum(g_f32 * seg_B_mask[None, :, None], axis=1, keepdims=True) * cumsum_scale
        # Mask the exponent before exp2. Masking the result afterwards can
        # produce inf * 0 = NaN for tokens belonging to the other segment.
        kg_A_exp = jnp.where(mask_A != 0, g_A_total - g_cumsum, 0.0)
        kg_A = k * jnp.exp2(kg_A_exp) * mask_A
        kg_B_exp = jnp.where(mask_B != 0, g_B_total - g_cumsum, 0.0)
        kg_B = k * jnp.exp2(kg_B_exp) * mask_B

        # seg_A final state
        h_A_new = b_h * jnp.exp2(g_A_total[:, 0, :])[:, :, None] + jnp.matmul(
            kg_A.transpose(0, 2, 1),
            b_v_new * seg_A_mask[None, :, None],
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        if STORE_FINAL_STATE:
            if MANUAL_HT_DMA:
                final_state_dma_ref[...] = h_A_new
                if OVERLAP_HT_DMA:
                    h_start = pl.program_id(0) * MB
                    ht_A_copy = pltpu.make_async_copy(
                        final_state_dma_ref,
                        ht_ref.at[
                            first_seg - 1,
                            pl.ds(h_start, MB),
                            i_b,
                            pl.ds(None),
                            pl.ds(None),
                        ],
                        state_dma_sem,
                    )
                    ht_A_copy.start()
                else:
                    _dma_state_to_ht(final_state_dma_ref, first_seg)
            else:
                ht_ref[first_seg - 1, ...] = h_A_new.astype(ht_ref.dtype)[:, None, :, :]

        # seg_B starts inside this chunk and therefore uses its own h0.
        h_B_new = h_B_initial * jnp.exp2(g_B_total[:, 0, :])[:, :, None] + jnp.matmul(
            kg_B.transpose(0, 2, 1),
            v_new_B,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        scratch_ref[...] = h_B_new
        if STORE_FINAL_STATE and MANUAL_HT_DMA and OVERLAP_HT_DMA:
            ht_A_copy.wait()
        if STORE_FINAL_STATE:
            if not MANUAL_HT_DMA:
                ht_ref[last_seg - 1, ...] = h_B_new.astype(ht_ref.dtype)[:, None, :, :]

        if STORE_RESIDUALS:
            if RESIDUAL_CHUNK_LAYOUT:
                Aqk_ref[:, 0, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0, 0] = A_inv.astype(Akk_ref.dtype)
            else:
                Aqk_ref[:, 0] = Aqk.astype(Aqk_ref.dtype)
                Akk_ref[:, 0] = A_inv.astype(Akk_ref.dtype)
            g_cumsum_out_ref[:, 0] = g_cumsum.astype(g_cumsum_out_ref.dtype)
        prev_seg_ref[...] = jnp.broadcast_to(last_seg, (128,))

        if STORE_FINAL_STATE:

            @pl.when(is_last_real)
            def _final():
                if MANUAL_HT_DMA:
                    _dma_state_to_ht(scratch_ref, last_seg)
                else:
                    ht_ref[last_seg - 1, ...] = h_B_new.astype(ht_ref.dtype)[:, None, :, :]


def _fwd_mega_kernel_native_segids_packed(packed_metadata_ref, *args, **kwargs):
    return _fwd_mega_kernel_native_segids(
        packed_metadata_ref,
        packed_metadata_ref,
        packed_metadata_ref,
        *args,
        **kwargs,
    )


_NATIVE_SEGIDS_STATIC_ARGNAMES = [
    "output_final_state",
    "scale",
    "chunk_size",
    "store_h",
    "store_v_new",
    "disable_recompute",
    "only_fwd",
    "safe_gate",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "lower_bound",
    "mini_batch",
    "N_max",
    "residual_chunk_layout",
    "batch_first",
]


def _chunk_kda_fwd_native_segids_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    initial_state=None,
    output_final_state=False,
    scale=1.0,
    chunk_size=64,
    store_h=False,
    store_v_new=False,
    disable_recompute=False,
    only_fwd=False,
    safe_gate=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    mini_batch=None,
    N_max=None,
    residual_chunk_layout=False,
    batch_first=False,
):
    """Native segment_ids mega kernel wrapper. No _align_seqs."""
    if batch_first:
        B, T, H, K = q.shape
    else:
        H, B, T, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    if only_fwd and (disable_recompute or store_h or store_v_new):
        raise ValueError("only_fwd=True is incompatible with backward-intermediate storage")
    if lower_bound is None and not only_fwd:
        raise ValueError("unbounded Mega KDA currently supports inference-only forward execution")
    if disable_recompute:
        store_h = True

    # Normalize segment_ids to [B, T]
    if segment_ids.ndim == 1:
        segment_ids = segment_ids[None, :]
        if B > 1:
            segment_ids = jnp.broadcast_to(segment_ids, (B, T))

    NT = T // BT
    packed_metadata = only_fwd and os.environ.get("KDA_PACKED_METADATA", "1") == "1"
    chunk_kind, seg_first, seg_last, seg_id = _build_chunk_metadata(segment_ids, BT)
    next_kind = jnp.concatenate(
        [
            chunk_kind[:, 1:],
            jnp.zeros((B, 1), dtype=jnp.int32),
        ],
        axis=1,
    )
    is_last_real_chunk = (chunk_kind != CHUNK_KIND_ALL_PAD) & (next_kind == CHUNK_KIND_ALL_PAD)
    chunk_kind = chunk_kind | (is_last_real_chunk.astype(jnp.int32) * CHUNK_FLAG_LAST_REAL)
    if N_max is None:
        N_max = int(segment_ids.max())  # Fallback: should not happen when called via dispatch
    # The packed representation uses 3 kind/flag bits and two 14-bit segment IDs.
    # Preserve the legacy three-prefetch ABI for unusually large N_max.
    if packed_metadata and N_max >= 16384:
        packed_metadata = False

    K_PAD = int(align_up(K, 128))
    V_ALIGNED = int(align_up(V, 128))
    # Pad T to 128-aligned for segment_ids block spec (TPU requires last dim % 128 == 0)
    T_PAD_S = int(align_up(T, 128))
    NT_S = T_PAD_S // 128  # number of 128-blocks in T

    def _pad(x, t):
        p = t - x.shape[-1]
        return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, p))) if p > 0 else x

    q_t, k_t, v_t, g_t = _pad(q, K_PAD), _pad(k, K_PAD), _pad(v, V_ALIGNED), _pad(g, K_PAD)
    # Pad segment_ids to [B, T_PAD_S] (128-aligned in T dim)
    if segment_ids.shape[-1] < T_PAD_S:
        segment_ids = jnp.pad(segment_ids, ((0, 0), (0, T_PAD_S - segment_ids.shape[-1])))

    # Rebuild metadata with padded T (chunk_size still BT, but more chunks now due to padding)
    # Metadata arrays need last dim >= 128 or == array dim. Pad NT to 128 if needed.
    NT_meta = T // BT  # original number of chunks
    NT_meta_PAD = max(int(align_up(NT_meta, 128)), 128)  # at least 128
    chunk_kind_padded = jnp.pad(
        chunk_kind, ((0, 0), (0, NT_meta_PAD - NT_meta)), constant_values=CHUNK_KIND_ALL_PAD
    )
    seg_first_padded = jnp.pad(seg_first, ((0, 0), (0, NT_meta_PAD - NT_meta)))
    seg_last_padded = jnp.pad(seg_last, ((0, 0), (0, NT_meta_PAD - NT_meta)))

    env_mini_batch = os.environ.get("KDA_FWD_MB") if only_fwd else None
    if mini_batch is None and env_mini_batch:
        mini_batch = int(env_mini_batch)
    if mini_batch is not None:
        MB = mini_batch
    elif only_fwd:
        # The initial-state operand is scalar-prefetch-indexed to one segment.
        # Try the full head group for inference; the final-state output window
        # remains bounded by N_max and compilation will enforce the VMEM limit.
        MB = min(H, 32)
    else:
        MB = min(H, 8)
    while H % MB != 0:
        MB //= 2

    # The batch-first beta window is copy-free when one program owns the full
    # head dimension. Smaller head groups use the head-first layout whose last
    # dimension is the complete 64-token block and therefore TPU legal.
    beta_batch_first = batch_first and only_fwd and MB == H
    beta_t = (
        beta.reshape(B, NT, BT, H)
        if beta_batch_first
        else beta.transpose(2, 0, 1).reshape(H, B, NT, 1, BT)
    )

    # The gate family is a model-static specialization. K3's bounded gate
    # keeps the faster global Neumann composition; Kimi-Linear's unbounded
    # gate uses ordered block substitution to avoid cancellation.
    use_neumann = lower_bound is not None
    # With gate values as low as -5, a 32-token block can form masked
    # pre-causal products near 2**155 before the mask is applied. Keeping the
    # rescaling window at 16 bounds those products below the FP32/BF16 range.
    qk_bc = 16
    # A monolithic 64x64 Neumann polynomial can reach O(1e17)
    # intermediates for correlated normalized keys with near-zero decay. The
    # mathematically cancelling result then loses precision in FP32. Eight
    # 8-token diagonal blocks bound the cancellation before the block-level
    # composition on TPU MXU.
    inv_bc = 8
    # Stage 2 has already made Aqk causal in all three native segment-ID
    # chunk variants.  In inference, avoid rebuilding and applying the same
    # 64x64 mask on the Stage 4 dependency chain.  Keep the training lowering
    # unchanged because it shares this kernel body.
    skip_stage4_mask = only_fwd
    pack_head_inv = (
        use_neumann and only_fwd and MB % 2 == 0 and os.environ.get("KDA_PACK_HEAD_INV", "1") == "1"
    )
    clip_beta_in_kernel = only_fwd
    overlap_h0_dma = only_fwd and os.environ.get("KDA_OVERLAP_H0_DMA", "1") == "1"
    overlap_ht_dma = only_fwd and os.environ.get("KDA_OVERLAP_HT_DMA", "1") == "1"
    _prec = jax.lax.Precision.DEFAULT if use_neumann else jax.lax.Precision.HIGHEST

    # Prepare h0: broadcast to [1, H, 1, K_PAD, V_ALIGNED] so each grid point
    # can read its own MB-sized block via _h0_map (avoids OOB on dim1).
    has_h0 = initial_state is not None
    state_dma_override = os.environ.get("KDA_MANUAL_STATE_DMA")
    manual_state_dma = N_max >= 6 if state_dma_override is None else state_dma_override == "1"
    h0_dma_override = os.environ.get("KDA_MANUAL_H0_DMA")
    manual_h0_dma = (
        only_fwd and has_h0 and (N_max > 6 if h0_dma_override is None else h0_dma_override == "1")
    )
    if has_h0:
        h0 = initial_state
        if h0.ndim == 4:
            h0 = h0[None, ...]  # [1, N, H, K, V]
        if h0.shape[0] < B:
            h0 = jnp.broadcast_to(h0, (B,) + h0.shape[1:])

        # Pass ALL N_max initial states for per-segment loading
        # h0: [B, N_max, H, K, V] -> [N_max, H, B, K, V] -> pad -> [N_max, H, B, K_PAD, V_ALIGNED]
        h0 = h0[:, :, :, :, :].transpose(1, 2, 0, 3, 4)  # [N_max, H, B, K, V]
        if K_PAD > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PAD - K), (0, 0)))
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        h0_in = h0.astype(jnp.float32)  # [N_max, H, B, K_PAD, V_ALIGNED]
    else:
        h0_in = None

    # Prepare A_log and dt_bias for gate activation inside kernel
    if use_gate_in_kernel and A_log is not None:
        H_A = A_log.shape[0]
        n_rep = H // H_A
        A_expanded = (
            jnp.repeat(A_log.astype(jnp.float32), n_rep) if n_rep > 1 else A_log.astype(jnp.float32)
        )
        A_scale_in = jnp.exp(A_expanded).reshape(H, 1, 1, 1)
    else:
        A_scale_in = jnp.zeros((H, 1, 1, 1), dtype=jnp.float32)

    if use_gate_in_kernel and dt_bias is not None:
        H_A = A_log.shape[0]
        n_rep = H // H_A
        db_2d = dt_bias.reshape(-1)[: H_A * K].reshape(H_A, K).astype(jnp.float32)
        if n_rep > 1:
            db_2d = jnp.repeat(db_2d, n_rep, axis=0)
        db_in = db_2d[:, None, None, :].astype(jnp.float32)  # [H, 1, 1, K]
        if K_PAD > K:
            db_in = jnp.pad(db_in, ((0, 0), (0, 0), (0, 0), (0, K_PAD - K)))
        db_in = jnp.broadcast_to(db_in, (H, 1, 1, K_PAD))
        # db_in stays [H, 1, 1, K_PAD]
    else:
        db_in = jnp.zeros((H, 1, 1, K_PAD), dtype=jnp.float32)

    # Block specs -- index maps must accept all prefetch refs as args
    # Note: index map returns block offset (in units of block size), not array element offset
    def _seg_map(h, b, c, *refs):
        # segment_ids has shape [B, T_PAD_S], block size is [B, 128]
        # So we need to return (b_offset, seg_offset) where seg_offset is in units of 128
        block_128 = (c * BT) // 128
        return (0, block_128)

    def _in_map(h, b, c, *refs):
        return (b, c, h, 0) if batch_first else (h, b, c, 0)

    def _h0_map(h, b, c, *refs):
        return (0, h, b, 0, 0)

    def _out_map(h, b, c, *refs):
        return (h, b, c, 0)

    def _out_chunk_map(h, b, c, *refs):
        return (h, b, c, 0, 0)

    def _beta_map(h, b, c, *refs):
        return (b, c, 0, h) if beta_batch_first else (h, b, c, 0, 0)

    def _ht_map(h, b, c, *refs):
        return (0, h, b, 0, 0)

    seg_spec = pl.BlockSpec([B, 128], index_map=_seg_map)
    q_spec = pl.BlockSpec(
        [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
        index_map=_in_map,
    )
    k_spec = pl.BlockSpec(
        [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
        index_map=_in_map,
    )
    v_spec = pl.BlockSpec(
        [1, BT, MB, V_ALIGNED] if batch_first else [MB, 1, BT, V_ALIGNED],
        index_map=_in_map,
    )
    g_spec = pl.BlockSpec(
        [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
        index_map=_in_map,
    )
    beta_spec = (
        pl.BlockSpec([1, 1, BT, MB], index_map=_beta_map)
        if beta_batch_first
        else pl.BlockSpec([MB, 1, 1, 1, BT], index_map=_beta_map)
    )
    h0_spec = (
        (
            pl.BlockSpec(memory_space=pl.ANY)
            if manual_h0_dma
            else pl.BlockSpec(
                [N_max, MB, 1, K_PAD, V_ALIGNED],
                index_map=_h0_map,
            )
        )
        if has_h0
        else None
    )

    def _alog_map(h, b, c, *refs):
        return (h, 0, 0, 0)

    def _dtbias_map(h, b, c, *refs):
        return (h, 0, 0, 0)

    alog_spec = pl.BlockSpec([MB, 1, 1, 1], index_map=_alog_map)
    dtbias_spec = pl.BlockSpec([MB, 1, 1, K_PAD], index_map=_dtbias_map)
    if batch_first:

        def _o_map(h, b, c, *refs):
            return (b, c, h, 0)

        o_spec = pl.BlockSpec([1, BT, MB, V_ALIGNED], index_map=_o_map)
    else:
        o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_out_map)
    store_final_state = output_final_state or store_h
    ht_dma_override = os.environ.get("KDA_MANUAL_HT_DMA")
    manual_ht_dma = (
        only_fwd
        and store_final_state
        and (manual_state_dma if ht_dma_override is None else ht_dma_override == "1")
    )
    ht_spec = (
        (
            pl.BlockSpec(memory_space=pl.ANY)
            if manual_ht_dma
            else pl.BlockSpec(
                [N_max, MB, 1, K_PAD, V_ALIGNED],
                index_map=_ht_map,
            )
        )
        if store_final_state
        else None
    )

    store_residuals = not only_fwd
    store_chunk_h = store_residuals and store_h
    aqk_spec = (
        pl.BlockSpec(
            [MB, 1, 1, BT, BT] if residual_chunk_layout else [MB, 1, BT, BT],
            index_map=_out_chunk_map if residual_chunk_layout else _out_map,
        )
        if store_residuals
        else None
    )
    akk_spec = (
        pl.BlockSpec(
            [MB, 1, 1, BT, BT] if residual_chunk_layout else [MB, 1, BT, BT],
            index_map=_out_chunk_map if residual_chunk_layout else _out_map,
        )
        if store_residuals
        else None
    )
    g_cumsum_spec = (
        pl.BlockSpec([MB, 1, BT, K_PAD], index_map=_out_map) if store_residuals else None
    )
    chunk_h_spec = (
        pl.BlockSpec(
            [MB, 1, 1, K_PAD, V_ALIGNED],
            index_map=lambda h, b, c, *r: (h, b, c, 0, 0),
        )
        if store_chunk_h
        else None
    )

    aqk_shape = (
        jax.ShapeDtypeStruct(
            (H, B, NT, BT, BT) if residual_chunk_layout else (H, B, T, BT),
            q.dtype,
        )
        if store_residuals
        else None
    )
    akk_shape = (
        jax.ShapeDtypeStruct(
            (H, B, NT, BT, BT) if residual_chunk_layout else (H, B, T, BT),
            q.dtype,
        )
        if store_residuals
        else None
    )
    g_cumsum_shape = (
        jax.ShapeDtypeStruct((H, B, T, K_PAD), jnp.float32) if store_residuals else None
    )

    chunk_h_shape = (
        jax.ShapeDtypeStruct((H, B, NT, K_PAD, V_ALIGNED), jnp.float32) if store_chunk_h else None
    )

    grid = (H // MB, B, NT)
    if packed_metadata:
        native_kernel = _fwd_mega_kernel_native_segids_packed
        scalar_prefetch_count = 1
        packed_meta = (
            chunk_kind_padded.astype(jnp.int32)
            | (seg_first_padded.astype(jnp.int32) << jnp.int32(3))
            | (seg_last_padded.astype(jnp.int32) << jnp.int32(17))
        )
        scalar_inputs = (packed_meta,)
    else:
        native_kernel = _fwd_mega_kernel_native_segids
        scalar_prefetch_count = 3
        scalar_inputs = (
            chunk_kind_padded,
            seg_first_padded,
            seg_last_padded,
        )

    o_out, ht_out, Aqk_out, Akk_out, g_cumsum_out, chunk_h_out = pl.pallas_call(
        functools.partial(
            native_kernel,
            NT=NT,
            BT=BT,
            N_max=N_max,
            scale=scale,
            cumsum_scale=_RCP_LN2,
            MB=MB,
            K_PAD=K_PAD,
            V_PAD=V_ALIGNED,
            OUTPUT_PRECISION=_prec,
            safe_gate=safe_gate,
            NORMALIZE_QK=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            USE_NEUMANN=use_neumann,
            QK_BC=qk_bc,
            INV_BC=inv_bc,
            SKIP_STAGE4_MASK=skip_stage4_mask,
            PACK_HEAD_INV=pack_head_inv,
            CLIP_BETA_IN_KERNEL=clip_beta_in_kernel,
            PACKED_METADATA=packed_metadata,
            HAS_H0=has_h0,
            STORE_RESIDUALS=store_residuals,
            STORE_H=store_chunk_h,
            STORE_FINAL_STATE=store_final_state,
            BATCH_FIRST=batch_first,
            MANUAL_H0_DMA=manual_h0_dma,
            MANUAL_HT_DMA=manual_ht_dma,
            OVERLAP_H0_DMA=overlap_h0_dma,
            OVERLAP_HT_DMA=overlap_ht_dma,
            RESIDUAL_CHUNK_LAYOUT=residual_chunk_layout,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(
                (B, T, H, V_ALIGNED) if batch_first else (H, B, T, V_ALIGNED),
                v.dtype,
            ),
            (
                jax.ShapeDtypeStruct((N_max, H, B, K_PAD, V_ALIGNED), jnp.float32)
                if store_final_state
                else None
            ),
            aqk_shape,
            akk_shape,
            g_cumsum_shape,
            chunk_h_shape,
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=scalar_prefetch_count,
            grid=grid,
            in_specs=[
                seg_spec,
                q_spec,
                k_spec,
                v_spec,
                g_spec,
                beta_spec,
                h0_spec,
                alog_spec,
                dtbias_spec,
            ],
            out_specs=[
                o_spec,
                ht_spec,
                aqk_spec,
                akk_spec,
                g_cumsum_spec,
                chunk_h_spec,
            ],
            scratch_shapes=[
                pltpu.VMEM((MB, K_PAD, V_ALIGNED), jnp.float32),
                pltpu.VMEM((128,), jnp.int32),
                pltpu.VMEM((MB, K_PAD, V_ALIGNED), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
            vmem_limit_bytes=(60 * 1024 * 1024) if MB > 4 else None,
        ),
        interpret=os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true"),
    )(
        *scalar_inputs,
        segment_ids,
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        h0_in,
        A_scale_in,
        db_in,
    )

    # Match the selected contract: [B, T, H, V] or [H, B, T, V].
    o_out = o_out[..., :V]
    # ht_out: [1, H, B, K_PAD, V_ALIGNED] -> [B, 1, H, K, V]
    if ht_out is not None:
        ht_out = ht_out[:, :, :, :K, :V].transpose(2, 0, 1, 3, 4)  # [B, N_max, H, K, V]

    final_state = ht_out if store_final_state else None

    # Trim g_cumsum: [H, B, T, K_PAD] -> [H, B, T, K]
    if g_cumsum_out is not None and K_PAD > K:
        g_cumsum_out = g_cumsum_out[:, :, :, :K]

    if chunk_h_out is not None:
        chunk_h_out = chunk_h_out[:, :, :, :K, :V]
    return (
        o_out,
        final_state,
        g_cumsum_out,
        Aqk_out,
        Akk_out,
        None,
        None,
        None,
        None,
        None,
        chunk_h_out,
        None,
    )


_chunk_kda_fwd_native_segids = jax.jit(
    _chunk_kda_fwd_native_segids_impl,
    static_argnames=_NATIVE_SEGIDS_STATIC_ARGNAMES,
)


@functools.lru_cache(maxsize=None)
def _native_residual_layout_jit_for_sharding(sharding):
    """JIT variant whose ABI accepts the Pallas Aqk/Akk physical layout."""
    from jax._src.layout import Format, Layout

    pallas_layout = Layout(
        major_to_minor=(0, 1, 2, 3, 4),
        tiling=((8, 128), (2, 1)),
    )
    residual_format = Format(layout=pallas_layout, sharding=sharding)
    return jax.jit(
        _chunk_kda_fwd_native_segids_impl,
        static_argnames=_NATIVE_SEGIDS_STATIC_ARGNAMES,
        out_shardings=(
            None,
            None,
            None,
            residual_format,
            residual_format,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def _get_native_residual_layout_jit(q):
    sharding = getattr(q, "sharding", None)
    if sharding is None:
        # Direct benchmark/eager use has a concrete array. A tracer without a
        # concrete sharding falls back to the portable JIT variant.
        return _chunk_kda_fwd_native_segids
    return _native_residual_layout_jit_for_sharding(sharding)
