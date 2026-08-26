"""Fused KDA v2 recurrence and output kernels."""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.kda.kda import align_up, exp2, get_interpret


def _fused_step_math(q, kk, v, w, g, A, S, scale):
    """Run one fused recurrence step for one head and chunk.

    Returns o [BT,V] and S_new [K,V].
    """
    BT = q.shape[0]
    b_v = jnp.dot(
        w.astype(jnp.float32),
        S,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_v = v.astype(jnp.float32) - b_v  # [BT, V] delta-rule residual (formerly v_new)

    b_g = g.astype(jnp.float32)
    b_g_ref = b_g[0:1, :]
    b_qg = q.astype(jnp.float32) * exp2(jnp.maximum(b_g - b_g_ref, -126.0))
    b_h_scaled = S * exp2(jnp.maximum(b_g_ref[0], -126.0))[:, None]
    b_o = scale * jnp.dot(
        b_qg,
        b_h_scaled,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(m_s, A.astype(jnp.float32), 0.0)
    b_o += jnp.dot(
        b_A, b_v, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )

    S_new = S * exp2(b_g[BT - 1])[:, None] + jnp.dot(
        kk.astype(jnp.float32).T,
        b_v,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return b_o, S_new


def _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale):
    """Legacy-layout wrapper that reads [1,1,BT,*] refs and writes the result."""
    b_o, S_new = _fused_step_math(
        q_ref[0, 0],
        k_ref[0, 0],
        v_ref[0, 0],
        w_ref[0, 0],
        g_ref[0, 0],
        A_ref[0, 0],
        scratch_ref[...],
        scale,
    )
    o_ref[0, 0] = b_o.astype(o_ref.dtype)
    scratch_ref[...] = S_new


def chunk_kda_fused_h_o(
    q,  # unified_in=True: [1, H, T, K]; False: [1, T, H, K]
    kg,
    w,
    u,
    g_cumsum,
    A,
    scale,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
    unified_in=True,
    flat_grid=True,
):
    # Import locally because flat_grid reuses the fused chunk math above.
    from sgl_jax.srt.kernels.kda.v2.flat_grid import (
        _chunk_kda_fused_h_o_kernel,
        _chunk_kda_fused_h_o_kernel_seqgrid,
    )

    BT = chunk_size
    assert cu_seqlens is not None
    N = cu_seqlens.shape[-1] - 1

    if unified_in:
        B, H, T_out, K = q.shape
        V = u.shape[-1]
        assert B == 1 and T_out % BT == 0
        K_PADSIZE = int(align_up(K, 128))
        V_ALIGNED = int(align_up(V, 128))

        def _padlast(x, D, D_pad):
            if D_pad > D:
                return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, D_pad - D)))
            return x

        q_t = _padlast(q, K, K_PADSIZE)
        k_t = _padlast(kg, K, K_PADSIZE)
        w_t = _padlast(w, K, K_PADSIZE)
        v_t = _padlast(u, V, V_ALIGNED)
        g_t = _padlast(g_cumsum, K, K_PADSIZE)
        A_t = A
        if not flat_grid:
            # The legacy grid needs a trailing trash block as its clamp target.
            pad_t = lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, BT), (0, 0)))
            q_t, k_t, w_t, v_t, g_t, A_t = map(pad_t, (q_t, k_t, w_t, v_t, g_t, A_t))
    else:
        # Legacy _prep for the data-movement ablation: materialize fp32,
        # append a trailing pad, and transpose the [1,T,H,D] input.
        B, T_out, H, K = q.shape
        V = u.shape[-1]
        assert B == 1 and T_out % BT == 0
        K_PADSIZE = int(align_up(K, 128))
        V_ALIGNED = int(align_up(V, 128))

        def _prep(x, D, D_pad):
            x = x.astype(jnp.float32) if x.dtype != jnp.float32 else x
            if D_pad > D:
                x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, D_pad - D)))
            x = jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
            return jnp.transpose(x, (0, 2, 1, 3))  # [1, H, T+BT, D_pad]

        q_t = _prep(q, K, K_PADSIZE)
        k_t = _prep(kg, K, K_PADSIZE)
        w_t = _prep(w, K, K_PADSIZE)
        v_t = _prep(u, V, V_ALIGNED)
        g_t = _prep(g_cumsum, K, K_PADSIZE)
        A_t = _prep(A, BT, BT)

    T_pad = q_t.shape[2]

    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )
    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    kernel_kw = dict(
        scale=scale,
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state,
    )
    cu_i32 = cu_seqlens.astype(jnp.int32)

    if flat_grid:
        # Flat grid: O(chunks), with sequence identity in three prefetched scalars.
        NC = T_pad // BT
        chunk_bos = jnp.arange(NC, dtype=jnp.int32) * BT
        seq_id = jnp.clip(jnp.searchsorted(cu_i32[1:], chunk_bos, side="right"), 0, N - 1).astype(
            jnp.int32
        )
        start_flag = (chunk_bos == cu_i32[seq_id]).astype(jnp.int32)
        end_flag = (chunk_bos + BT == cu_i32[seq_id + 1]).astype(jnp.int32)

        _state_index_map = lambda h, c, seq_id_ref, *_: (seq_id_ref[c], h, 0, 0)
        tspec = lambda D: pl.BlockSpec([1, 1, BT, D], index_map=lambda h, c, *_: (0, h, c, 0))
        h0_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if initial_state is not None
            else None
        )
        ht_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if output_final_state
            else None
        )

        o_out, ht_out = pl.pallas_call(
            functools.partial(_chunk_kda_fused_h_o_kernel, **kernel_kw),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(H, NC),
                in_specs=[
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(V_ALIGNED),
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(BT),
                    h0_blockspec,
                ],
                out_specs=[tspec(V_ALIGNED), ht_blockspec],
                scratch_shapes=[scratch],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
            out_shape=[jax.ShapeDtypeStruct([1, H, T_pad, V_ALIGNED], jnp.float32), ht_spec],
            interpret=get_interpret(),
        )(seq_id, start_flag, end_flag, q_t, k_t, v_t, w_t, g_t, A_t, h0)
    else:
        # Legacy (N, H, NT_max) ablation grid: O(N x chunks).
        T_ref = T_pad - BT  # Logical T; block T_ref//BT is the clamp target.
        NT_max = T_ref // BT

        def _t_index_map(n, h, nt, seqlens_ref):
            bos = pl.multiple_of(seqlens_ref[n], BT)
            return (0, h, jnp.minimum(bos // BT + nt, T_ref // BT), 0)

        _state_index_map = lambda n, h, nt, *_: (n, h, 0, 0)
        tspec = lambda D: pl.BlockSpec([1, 1, BT, D], index_map=_t_index_map)
        h0_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if initial_state is not None
            else None
        )
        ht_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if output_final_state
            else None
        )

        o_out, ht_out = pl.pallas_call(
            functools.partial(_chunk_kda_fused_h_o_kernel_seqgrid, **kernel_kw),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(N, H, NT_max),
                in_specs=[
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(V_ALIGNED),
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(BT),
                    h0_blockspec,
                ],
                out_specs=[tspec(V_ALIGNED), ht_blockspec],
                scratch_shapes=[scratch],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
            out_shape=[jax.ShapeDtypeStruct([1, H, T_pad, V_ALIGNED], jnp.float32), ht_spec],
            interpret=get_interpret(),
        )(cu_i32, q_t, k_t, v_t, w_t, g_t, A_t, h0)

    o = jnp.transpose(o_out[:, :, :T_out, :V], (0, 2, 1, 3))
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None
    return o, ht_out
