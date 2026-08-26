"""Native-layout head-block KDA v2 kernels."""

from __future__ import annotations

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.kda.kda import _RCP_LN2, _intra_head_math, exp2, get_interpret


def _kda_fwd_intra_kernel_hb(
    q_ref,  # [1, BT, H, K]
    k_ref,  # [1, BT, H, K]
    g_ref,  # [1, BT, H, K]
    beta_ref,  # [1, BT, H, 1]
    v_ref,  # [1, BT, H, V]
    a_ref,  # [H, K] or None
    bias_ref,  # [H, K] or None
    u_out_ref,  # [1, BT, H, V]
    w_out_ref,  # [1, BT, H, K]
    kg_out_ref,  # [1, BT, H, K]
    Aqk_out_ref,  # [1, BT, H, BT]
    g_cum_out_ref,  # [1, BT, H, K] f32
    *,
    chunk_size,
    head_dim,
    value_dim,
    scale,
    safe_gate,
    APPLY_GATE,
    LOWER_BOUND,
    NUM_HEADS,
):
    if not safe_gate:
        # Debug path (elementwise decay plus row-wise elimination): process
        # heads serially and keep _intra_head_math as the single source.
        for h in range(NUM_HEADS):
            a_vec = a_ref[h] if APPLY_GATE else None
            bias_vec = bias_ref[h] if APPLY_GATE else None
            u, w, kg, Aqk, _, g_cum = _intra_head_math(
                q_ref[0, :, h, :],
                k_ref[0, :, h, :],
                g_ref[0, :, h, :],
                beta_ref[0, :, h, :],
                v_ref[0, :, h, :],
                a_vec,
                bias_vec,
                chunk_size=chunk_size,
                head_dim=head_dim,
                value_dim=value_dim,
                scale=scale,
                safe_gate=safe_gate,
                APPLY_GATE=APPLY_GATE,
                LOWER_BOUND=LOWER_BOUND,
                PRE_CUMSUM=False,
                WANT_AINV=False,
            )
            u_out_ref[0, :, h, :] = u.astype(u_out_ref.dtype)
            w_out_ref[0, :, h, :] = w.astype(w_out_ref.dtype)
            kg_out_ref[0, :, h, :] = kg.astype(kg_out_ref.dtype)
            Aqk_out_ref[0, :, h, :] = Aqk.astype(Aqk_out_ref.dtype)
            g_cum_out_ref[0, :, h, :] = g_cum.astype(g_cum_out_ref.dtype)
        return

    # ---- safe_gate fast path: vectorize elementwise work across H and
    # interleave MXU stages. Each head is independent. Stage-major issue
    # traverses h within a stage, so adjacent MXU instructions have no data
    # dependency and can pipeline fill/drain. Elementwise work handles all
    # heads together, reducing instruction count by H. ----
    BT = chunk_size
    H = NUM_HEADS
    # Stage 1: gate activation and prefix sum, vectorized across H.
    g_all = g_ref[0].astype(jnp.float32)  # [BT, H, K]
    if APPLY_GATE:
        a_all = a_ref[...].astype(jnp.float32)  # [H, K]
        b_all = bias_ref[...].astype(jnp.float32)
        g_all = LOWER_BOUND * jax.nn.sigmoid(a_all[None] * (g_all + b_all[None]))
    num_steps = int(math.log2(BT))
    assert (1 << num_steps) == BT
    for d in range(num_steps):
        s = 1 << d
        g_all = jnp.concatenate([g_all[:s], g_all[s:] + g_all[:-s]], axis=0)
    g_all = g_all * _RCP_LN2
    g_cum_out_ref[0] = g_all.astype(g_cum_out_ref.dtype)

    q_all = q_ref[0].astype(jnp.float32)
    k_all = k_ref[0].astype(jnp.float32)
    v_all = v_ref[0].astype(jnp.float32)
    beta_all = beta_ref[0].astype(jnp.float32)  # [BT, H, 1]

    # Build strip rows/columns vectorized across H; interleave GEMMs by (blk, h).
    SB = 16
    o_i = jnp.arange(BT, dtype=jnp.int32)
    causal = o_i[:, None] >= o_i[None, :]
    strict = o_i[:, None] > o_i[None, :]
    dn = (((1,), (1,)), ((), ()))
    aqk_parts = [[] for _ in range(H)]
    l_parts = [[] for _ in range(H)]
    for blk in range(BT // SB):
        cols = slice(blk * SB, (blk + 1) * SB)
        r_b = g_all[blk * SB + SB // 2 : blk * SB + SB // 2 + 1]  # [1, H, K]
        row_all = exp2(g_all - r_b)  # [BT, H, K]
        col_all = k_all[cols] * exp2(r_b - g_all[cols])  # [SB, H, K]
        qrow = q_all * row_all
        krow = k_all * row_all
        for h in range(H):
            aqk_parts[h].append(
                jax.lax.dot_general(
                    qrow[:, h], col_all[:, h], dn, preferred_element_type=jnp.float32
                )
            )
            l_parts[h].append(
                jax.lax.dot_general(
                    krow[:, h], col_all[:, h], dn, preferred_element_type=jnp.float32
                )
            )

    v_beta = v_all * beta_all  # [BT, H, V]
    k_eg_beta = k_all * exp2(g_all) * beta_all  # [BT, H, K]

    Aqks, Ls, zs = [], [], []
    for h in range(H):
        Aqks.append(jnp.where(causal, scale * jnp.concatenate(aqk_parts[h], -1), 0.0))
        Ls.append(jnp.where(strict, jnp.concatenate(l_parts[h], -1), 0.0) * beta_all[:, h])
        zs.append(jnp.concatenate([v_beta[:, h], k_eg_beta[:, h]], axis=-1))

    # Fused-wide Neumann factor chain with stages interleaved across H.
    zs = [z - jax.lax.dot(L, z, preferred_element_type=jnp.float32) for L, z in zip(Ls, zs)]
    Lp = list(Ls)
    for _ in range(int(math.log2(BT)) - 1):
        Lp = [jax.lax.dot(P, P, preferred_element_type=jnp.float32) for P in Lp]
        zs = [z + jax.lax.dot(P, z, preferred_element_type=jnp.float32) for P, z in zip(Lp, zs)]

    # Assemble outputs and write them back vectorized across H.
    u_all = jnp.stack([z[:, :value_dim] for z in zs], axis=1)
    w_all = jnp.stack([z[:, value_dim:] for z in zs], axis=1)
    kg_all = k_all * exp2(g_all[BT - 1 : BT] - g_all)
    u_out_ref[0] = u_all.astype(u_out_ref.dtype)
    w_out_ref[0] = w_all.astype(w_out_ref.dtype)
    kg_out_ref[0] = kg_all.astype(kg_out_ref.dtype)
    Aqk_out_ref[0] = jnp.stack(Aqks, axis=1).astype(Aqk_out_ref.dtype)


def kda_fwd_intra_hb(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    chunk_size=64,
    safe_gate=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
):
    """Run head-block K1 with native [1, T, H, D] I/O and grid=(NC,)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    assert B == 1 and T % BT == 0 and H % 8 == 0
    NC = T // BT

    beta4 = beta.reshape(B, T, H, 1)
    if use_gate_in_kernel:
        assert A_log is not None
        a_r = jnp.broadcast_to(jnp.exp(A_log.astype(jnp.float32))[:, None], (H, K))
        bias_r = (
            jnp.zeros((H, K), jnp.float32)
            if dt_bias is None
            else dt_bias.astype(jnp.float32).reshape(H, K)
        )
        gate_spec = pl.BlockSpec(block_shape=(H, K), index_map=lambda c: (0, 0))
    else:
        a_r, bias_r, gate_spec = None, None, None

    def _spec(last_dim):
        return pl.BlockSpec(block_shape=(1, BT, H, last_dim), index_map=lambda c: (0, c, 0, 0))

    dt = q.dtype
    u4, w4, kg4, Aqk4, g_cum4 = pl.pallas_call(
        functools.partial(
            _kda_fwd_intra_kernel_hb,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            safe_gate=safe_gate,
            APPLY_GATE=use_gate_in_kernel,
            LOWER_BOUND=lower_bound,
            NUM_HEADS=H,
        ),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct((1, T, H, V), dt),
            jax.ShapeDtypeStruct((1, T, H, K), dt),
            jax.ShapeDtypeStruct((1, T, H, K), dt),
            jax.ShapeDtypeStruct((1, T, H, BT), dt),
            jax.ShapeDtypeStruct((1, T, H, K), jnp.float32),
        ],
        in_specs=[_spec(K), _spec(K), _spec(K), _spec(1), _spec(V), gate_spec, gate_spec],
        out_specs=[_spec(V), _spec(K), _spec(K), _spec(BT), _spec(K)],
        grid=(NC,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(q, k, gk, beta4, v, a_r, bias_r)
    return w4, u4, kg4, Aqk4, g_cum4


def _chunk_kda_fused_h_o_kernel_hb(
    seq_id_ref,  # [NC] prefetch
    start_flag_ref,  # [NC]
    end_flag_ref,  # [NC]
    q_ref,  # [1, BT, H, K]
    k_ref,  # [1, BT, H, K]  kg
    v_ref,  # [1, BT, H, V]  u
    w_ref,  # [1, BT, H, K]
    g_ref,  # [1, BT, H, K]  g_cumsum f32
    A_ref,  # [1, BT, H, BT]
    h0_ref,  # [1, H, K, V] or None
    o_ref,  # [1, BT, H, V] out, stored directly in input dtype
    ht_ref,  # [1, H, K, V] out or None
    scratch_ref,  # [H, K, V] fp32, with all head states resident in VMEM
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    NUM_HEADS,
):
    idx_c = pl.program_id(0)

    @pl.when(start_flag_ref[idx_c] == 1)
    def _():
        scratch_ref[...] = jnp.zeros_like(scratch_ref)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0].astype(jnp.float32)

    # Vectorize elementwise work across H and interleave MXU issue by stage:
    # A computes residuals, B/C compute o, and D updates state.
    q_all = q_ref[0].astype(jnp.float32)  # [BT, H, K]
    k_all = k_ref[0].astype(jnp.float32)
    v_all = v_ref[0].astype(jnp.float32)  # [BT, H, V]
    w_all = w_ref[0].astype(jnp.float32)
    g_all = g_ref[0].astype(jnp.float32)
    A_all = A_ref[0].astype(jnp.float32)  # [BT, H, BT]
    S_all = scratch_ref[...]  # [H, K, V]

    BT = q_ref.shape[1]
    g0 = g_all[0:1]  # [1, H, K]
    qg_all = q_all * exp2(jnp.maximum(g_all - g0, -126.0))
    h_scale = exp2(jnp.maximum(g0[0], -126.0))  # [H, K]
    g_last = g_all[BT - 1]  # [H, K]
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    A_mask = jnp.where(m_s[:, None, :], A_all, 0.0)  # [BT, H, BT]

    HI = jax.lax.Precision.HIGHEST
    bv = [
        v_all[:, h]
        - jnp.dot(w_all[:, h], S_all[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    o1 = [
        scale
        * jnp.dot(
            qg_all[:, h],
            S_all[h] * h_scale[h][:, None],
            precision=HI,
            preferred_element_type=jnp.float32,
        )
        for h in range(NUM_HEADS)
    ]
    o2 = [
        jnp.dot(A_mask[:, h], bv[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    o_ref[0] = jnp.stack([a + b for a, b in zip(o1, o2)], axis=1).astype(o_ref.dtype)

    upd = [
        jnp.dot(k_all[:, h].T, bv[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    scratch_ref[...] = S_all * exp2(g_last)[:, :, None] + jnp.stack(upd, axis=0)

    @pl.when(end_flag_ref[idx_c] == 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_kda_fused_h_o_hb(
    q,
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
):
    """Run head-block K2 serially over grid=(NC,) with native [1,T,H,D] input.

    The output o is stored directly in the input dtype.
    """
    B, T, H, K = q.shape
    V = u.shape[-1]
    BT = chunk_size
    assert cu_seqlens is not None
    assert B == 1 and T % BT == 0 and H % 8 == 0
    N = cu_seqlens.shape[-1] - 1
    NC = T // BT

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunk_bos = jnp.arange(NC, dtype=jnp.int32) * BT
    seq_id = jnp.clip(jnp.searchsorted(cu_i32[1:], chunk_bos, side="right"), 0, N - 1).astype(
        jnp.int32
    )
    start_flag = (chunk_bos == cu_i32[seq_id]).astype(jnp.int32)
    end_flag = (chunk_bos + BT == cu_i32[seq_id + 1]).astype(jnp.int32)

    def _spec(last_dim):
        return pl.BlockSpec(block_shape=(1, BT, H, last_dim), index_map=lambda c, *_: (0, c, 0, 0))

    _state_map = lambda c, seq_id_ref, *_: (seq_id_ref[c], 0, 0, 0)
    h0_blockspec = (
        pl.BlockSpec([1, H, K, V], index_map=_state_map) if initial_state is not None else None
    )
    ht_blockspec = pl.BlockSpec([1, H, K, V], index_map=_state_map) if output_final_state else None
    ht_spec = jax.ShapeDtypeStruct([N, H, K, V], jnp.float32) if output_final_state else None

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_kda_fused_h_o_kernel_hb,
            scale=scale,
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            NUM_HEADS=H,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(NC,),
            in_specs=[_spec(K), _spec(K), _spec(V), _spec(K), _spec(K), _spec(BT), h0_blockspec],
            out_specs=[_spec(V), ht_blockspec],
            scratch_shapes=[pltpu.VMEM((H, K, V), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        out_shape=[jax.ShapeDtypeStruct([1, T, H, V], q.dtype), ht_spec],
        interpret=get_interpret(),
    )(seq_id, start_flag, end_flag, q, kg, u, w, g_cumsum, A, initial_state)

    return o_out, ht_out
