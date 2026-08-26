"""Flat-grid and sequence-grid KDA v2 scheduling kernels."""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.kda.kda import (
    align_up,
    assert_shape,
    assert_shape_or_none,
    exp,
    exp2,
    get_interpret,
    pad_to_multiple,
)
from sgl_jax.srt.kernels.kda.v2.fused import _fused_h_o_chunk_step


def _chunk_gated_delta_rule_fwd_kernel_flat(
    seq_ids_ref,
    is_first_ref,
    is_last_ref,
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    gk_ref,
    h0_ref,
    h_ref,
    v_new_ref,
    ht_ref,
    scratch_ref,
    *,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
):
    # Flat-chunk grid (h, nt): every step is a real chunk -- O(total_chunks)
    # instead of the previous O(N x total_chunks) where each sequence swept
    # the full global chunk range and idled through chunks it did not own.
    # Sequence boundaries come from the prefetched flags: reset the state
    # carry at a sequence's first chunk, emit its final state at the last.
    idx_nt = pl.program_id(1)

    BT = k_ref.shape[2]
    K, V = k_ref.shape[-1], v_ref.shape[-1]
    b_k = k_ref[0, 0]

    @pl.when(is_first_ref[idx_nt] == 1)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    h_ref[0, 0, 0] = scratch_ref[...].astype(h_ref.dtype)

    b_w = w_ref[0, 0]
    b_v = jnp.dot(
        b_w.astype(jnp.float32),
        scratch_ref[...],
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_u = v_ref[0, 0]
    b_v = b_u.astype(b_v.dtype) - b_v
    if SAVE_NEW_VALUE:
        v_new_ref[0, 0] = b_v.astype(v_new_ref.dtype)

    if USE_G:
        b_g = g_ref[0, 0, :, 0]
        b_g_last = g_ref[0, 0, BT - 1, 0].astype(jnp.float32)
        if USE_EXP2:
            b_v = b_v * exp2(b_g_last - b_g)[:, None]
            b_g_last = exp2(b_g_last)
        else:
            b_v = b_v * exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
        scratch_ref[...] *= b_g_last
    if USE_GK:
        b_gk_last = gk_ref[0, 0, BT - 1].astype(jnp.float32)
        if USE_EXP2:
            scratch_ref[...] *= exp2(b_gk_last)[:, None]
        else:
            scratch_ref[...] *= exp(b_gk_last)[:, None]

    scratch_ref[...] += jnp.dot(
        b_k.astype(jnp.float32).T,
        b_v.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    if STORE_FINAL_STATE:

        @pl.when(is_last_ref[idx_nt] == 1)
        def _():
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_gated_delta_rule_fwd_h_flat(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    use_exp2=True,
    cu_seqlens=None,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size

    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    assert B == 1, f"varlen mode requires B==1, got B={B}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(u, (B, T, H, V), "u")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # --- Varlen launcher (flat-chunk grid) ---
    # Runs after _align_seqs, so every sequence is BT-aligned and the packed
    # chunk list is contiguous: the grid is O(total_chunks). The previous
    # (N, H, NT_max) grid swept the FULL global chunk range once per
    # sequence, idling through foreign chunks -- a per-sequence tax measured
    # at ~0.5us x (N-1) x (T/BT) x H (e.g. ~46 ms for 8x1024 packed at
    # T=8192, H=96 on v6e).
    k = k.astype(jnp.float32)
    w = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    K_PADSIZE = int(align_up(K, 128))
    V_ALIGNED = int(align_up(V, 128))

    assert chunk_indices is not None
    NT = len(chunk_indices)
    assert NT == T // BT, "flat-chunk fwd_h requires BT-aligned packing"

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunks_per_seq = jnp.diff(cu_i32) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    flat_idx = jnp.arange(NT, dtype=jnp.int32)
    seq_ids = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1).astype(
        jnp.int32
    )
    local_ids = flat_idx - cum_chunks[seq_ids]
    is_first = (local_ids == 0).astype(jnp.int32)
    is_last = (local_ids == chunks_per_seq[seq_ids] - 1).astype(jnp.int32)

    def _padk(x):
        if K_PADSIZE > K:
            return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        return x

    k_t = jnp.transpose(_padk(k), (0, 2, 1, 3))
    w_t = jnp.transpose(_padk(w), (0, 2, 1, 3))
    v_pad = jnp.pad(u_f32, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V))) if V_ALIGNED > V else u_f32
    v_t = jnp.transpose(v_pad, (0, 2, 1, 3))

    if g is not None:
        g_fp32 = g.astype(jnp.float32).reshape(B, T, H, 1)
        g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
        g_t = jnp.transpose(g_fp32, (0, 2, 1, 3))
    else:
        g_t = None

    if gk is not None:
        gk_fp32 = gk.astype(jnp.float32)
        if K_PADSIZE > K:
            gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        gk_t = jnp.transpose(gk_fp32, (0, 2, 1, 3))
    else:
        gk_t = None

    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    g_pad_size = g_t.shape[-1] if g_t is not None else 128
    h_spec = jax.ShapeDtypeStruct([B, NT, H, K_PADSIZE, V_ALIGNED], k.dtype)
    v_new_spec = jax.ShapeDtypeStruct([B, H, T, V_ALIGNED], jnp.float32) if save_new_value else None
    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )

    def _t_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (0, h, nt, 0)

    def _h_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (0, nt, h, 0, 0)

    def _state_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (seq_ids_ref[nt], h, 0, 0)

    k_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    v_blockspec = pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map)
    w_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    g_blockspec = (
        pl.BlockSpec([1, 1, BT, g_pad_size], index_map=_t_index_map) if g is not None else None
    )
    gk_blockspec = (
        pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map) if gk is not None else None
    )
    h0_blockspec = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
        if initial_state is not None
        else None
    )

    h_blockspec_out = pl.BlockSpec([1, 1, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map)
    v_new_blockspec_out = (
        pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map) if save_new_value else None
    )
    ht_blockspec_out = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
        if output_final_state
        else None
    )

    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    grid = (H, NT)
    interpret = get_interpret()

    h_out, v_new_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_gated_delta_rule_fwd_kernel_flat,
            USE_G=(g is not None),
            USE_GK=(gk is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            USE_EXP2=use_exp2,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=grid,
            in_specs=[
                k_blockspec,
                v_blockspec,
                w_blockspec,
                g_blockspec,
                gk_blockspec,
                h0_blockspec,
            ],
            out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
            scratch_shapes=[scratch],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        out_shape=[h_spec, v_new_spec, ht_spec],
        interpret=interpret,
    )(seq_ids, is_first, is_last, k_t, v_t, w_t, g_t, gk_t, h0)

    h_out = h_out[:, :, :, :K, :V]
    v_new_out = jnp.transpose(v_new_out[:, :, :, :V], (0, 2, 1, 3)) if save_new_value else None
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None

    return h_out, v_new_out, ht_out


def _chunk_kda_fused_h_o_kernel(
    seq_id_ref,  # [NC] prefetch: owning sequence, consumed only by index_map
    start_flag_ref,  # [NC] prefetch: first chunk of a sequence; resets state
    end_flag_ref,  # [NC] prefetch: last chunk of a sequence; writes final_state
    q_ref,  # [1, 1, BT, K]
    k_ref,  # [1, 1, BT, K]   kg from stage 2 (k * exp2(g_last - g))
    v_ref,  # [1, 1, BT, V]   u from stage 2 (corrected values)
    w_ref,  # [1, 1, BT, K]
    g_ref,  # [1, 1, BT, K]   g_cumsum (fp32, log2 domain)
    A_ref,  # [1, 1, BT, BT]  Aqk from stage 2
    h0_ref,  # [1, 1, K, V] or None
    o_ref,  # [1, 1, BT, V] out
    ht_ref,  # [1, 1, K, V] out or None
    scratch_ref,  # [K, V] f32, persistent hidden state
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    """Run a flat chunk grid (flat_grid=True) with grid=(H, NC).

    The c dimension advances serially in packed order, so total work is
    O(chunks), independent of sequence count N. Start/end flags mark sequence
    boundaries. Inputs for inactive chunks are zero from K1, making their
    residual and state updates no-ops.
    """
    idx_c = pl.program_id(1)
    K, V = k_ref.shape[-1], v_ref.shape[-1]

    @pl.when(start_flag_ref[idx_c] == 1)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale)

    @pl.when(end_flag_ref[idx_c] == 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def _chunk_kda_fused_h_o_kernel_seqgrid(
    seqlens_ref,  # [N+1] prefetch: cu_seqlens
    q_ref,
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    A_ref,
    h0_ref,
    o_ref,
    ht_ref,
    scratch_ref,
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    """Run the legacy (N, H, NT_max) grid for flat_grid=False ablation.

    Every sequence scans the global chunk range. Steps where nt >= real_NT
    remain idle but still incur grid and DMA costs, producing O(N x chunks)
    scheduling overhead.
    """
    idx_n = pl.program_id(0)
    idx_nt = pl.program_id(2)

    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    BT = k_ref.shape[2]
    real_NT = (eos - bos) // BT
    K, V = k_ref.shape[-1], v_ref.shape[-1]

    @pl.when(idx_nt == 0)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    @pl.when(idx_nt < real_NT)
    def _():
        _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale)

    @pl.when(idx_nt == real_NT - 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)
