# Adapted from https://github.com/primatrix/pallas-kernel (rev 41431b1, release/v0.4)
# Vendored to remove external dependency after the upstream repository went private.
#
# This file merges the following modules into a single file:
#   - tops/utils.py (assert_shape, assert_shape_or_none)
#   - tops/ops/utils.py (exp, get_interpret)
#   - tops/ops/simple_gla/fused_recurrent.py (fused_recurrent_simple_gla)
#   - tops/ops/common/chunk_h.py (_build_chunk_map, _chunk_fwd_h_kernel_varlen, chunk_fwd_h_kernel_varlen)
#   - tops/ops/common/chunk_o.py (_chunk_fwd_o_kernel, _chunk_fwd_o_pl, chunk_fwd_o)
#   - tops/ops/simple_gla/chunk.py (chunk_simple_gla_fwd_varlen)
#   - tops/ops/simple_gla/__init__.py (SimpleGLAKernelMode, simple_gla_fwd)

from __future__ import annotations

import enum
import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Utilities (from tops/utils.py and tops/ops/utils.py)
# =============================================================================


def assert_shape_or_none(
    x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if x is None:
        return
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert (
                    tensor.shape == expected_shape
                ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape(
    x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert (
                tensor.shape == expected_shape
            ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def exp(x):
    return jnp.exp(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# =============================================================================
# Fused recurrent (from tops/ops/simple_gla/fused_recurrent.py)
# Pure JAX implementation using jax.lax.scan, decode-friendly.
# =============================================================================


def _scan_segment(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over one dense segment."""
    if reverse:
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        if g is not None:
            g = jnp.flip(g, axis=1)

    B, _T, H, K = q.shape
    V = v.shape[-1]
    h0 = initial_state if initial_state is not None else jnp.zeros((B, H, K, V), dtype=q.dtype)

    q_t = jnp.swapaxes(q, 0, 1)
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)
    g_t = jnp.swapaxes(g, 0, 1) if g is not None else jnp.zeros((q_t.shape[0], B, H), dtype=q.dtype)
    use_g = g is not None

    def step(h, xs):
        q_i, k_i, v_i, g_i = xs
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma[None, :]
        else:
            decay = jnp.broadcast_to(g_gamma[None, :], (B, H))

        h = h * jnp.exp(decay)[:, :, None, None]
        h = h + k_i[:, :, :, None] * v_i[:, :, None, :]
        o_i = jnp.sum(h * (q_i[:, :, :, None] * scale), axis=2)
        return h, o_i

    h_final, o_t = jax.lax.scan(step, h0, (q_t, k_t, v_t, g_t))
    o = jnp.swapaxes(o_t, 0, 1)

    if reverse:
        o = jnp.flip(o, axis=1)

    return o, h_final


def _scan_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
    cu_seqlens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over packed varlen data with one JAX scan."""
    _B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    seq_starts = cu_seqlens[:-1]
    seq_ends = cu_seqlens[1:]
    token_starts = seq_starts[seq_ids]
    token_ends = seq_ends[seq_ids]

    if reverse:
        scan_order = token_ends - 1 - (token_idx - token_starts)
        reset_mask = token_idx == (token_ends - 1)
    else:
        scan_order = token_idx
        reset_mask = token_idx == token_starts

    scan_seq_ids = seq_ids[scan_order]
    q_s = q[0, scan_order]
    k_s = k[0, scan_order]
    v_s = v[0, scan_order]
    g_s = g[0, scan_order] if g is not None else jnp.zeros((T, H), dtype=q.dtype)

    h0_all = initial_state if initial_state is not None else jnp.zeros((N, H, K, V), dtype=q.dtype)
    use_g = g is not None

    def step(carry, xs):
        h_prev, final_states = carry
        seq_id, do_reset, q_i, k_i, v_i, g_i = xs

        h = jnp.where(do_reset, h0_all[seq_id], h_prev)
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma
        else:
            decay = g_gamma

        h = h * jnp.exp(decay)[:, None, None]
        h = h + k_i[:, :, None] * v_i[:, None, :]
        o_i = jnp.sum(h * (q_i[:, :, None] * scale), axis=1)

        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_i

    init_carry = (
        jnp.zeros((H, K, V), dtype=q.dtype),
        h0_all,
    )
    (h_last, final_states), o_scan = jax.lax.scan(
        step,
        init_carry,
        (scan_seq_ids, reset_mask[scan_order], q_s, k_s, v_s, g_s),
    )
    del h_last

    inv_order = jnp.argsort(scan_order)
    o = o_scan[inv_order][None, ...]
    return o, final_states


def fused_recurrent_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Simple GLA fused recurrent forward for decode-friendly execution.

    Args:
        q: [B, T, H, K] queries.
        k: [B, T, H, K] keys.
        v: [B, T, H, V] values.
        g: [B, T, H] optional per-token log gate.
        g_gamma: [H] optional per-head constant log decay.
        scale: Optional query scaling factor. Defaults to K ** -0.5.
        initial_state: [N, H, K, V] optional recurrent state, where N=B for dense
            mode and N=len(cu_seqlens)-1 for varlen mode.
        output_final_state: Whether to return the final recurrent state.
        reverse: Whether to process each sequence in reverse time order.
        cu_seqlens: [N+1] cumulative sequence lengths for packed varlen inputs.

    Returns:
        Tuple of output [B, T, H, V] in q.dtype and optional final state
        [N, H, K, V] in the input dtype.
    """
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T,H,V], got {v.ndim}D"

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape[:3] == q.shape[:3], f"v shape {v.shape} incompatible with q"
    assert g is not None or g_gamma is not None, "At least one of g or g_gamma must be provided"
    if g is not None:
        assert g.ndim == 3 and g.shape == (B, T, H), f"g shape {g.shape} != {(B, T, H)}"
    if g_gamma is not None:
        assert (
            g_gamma.ndim == 1 and g_gamma.shape[0] == H
        ), f"g_gamma shape {g_gamma.shape} != ({H},)"
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    if initial_state is not None:
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"

    if scale is None:
        scale = K**-0.5
    scale = float(scale)

    q_f = q
    k_f = k
    v_f = v
    g_f = g
    g_gamma_f = g_gamma
    h0_f = initial_state

    if cu_seqlens is None:
        o, ht = _scan_segment(
            q_f,
            k_f,
            v_f,
            g=g_f,
            g_gamma=g_gamma_f,
            scale=scale,
            initial_state=h0_f,
            reverse=reverse,
        )
        return o, (ht if output_final_state else None)

    cu_f = jnp.asarray(cu_seqlens, dtype=jnp.int32)
    o, ht = _scan_varlen(
        q_f,
        k_f,
        v_f,
        g=g_f,
        g_gamma=g_gamma_f,
        scale=scale,
        initial_state=h0_f,
        reverse=reverse,
        cu_seqlens=cu_f,
    )
    return o, (ht if output_final_state else None)


# =============================================================================
# Chunk forward h — varlen path (from tops/ops/common/chunk_h.py)
# Pallas TPU kernel for computing hidden states with variable-length sequences.
# =============================================================================


def _build_chunk_map(cu_seqlens, T_sum, BT):
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    return seq_idx


def _chunk_fwd_h_kernel_varlen(
    k_ref,  # [1, BT, BK]
    v_ref,  # [1, BT, BV]
    h0_ref,  # [N, 1, BK, BV]
    gk_ref,  # [1, BT, BK]
    g_gamma_ref,  # [H,]
    cu_seqlens_ref,  # [num_seq+1]
    chunk_to_seq,  # [T_sum/BT]
    h_ref,  # [NS, 1, BK, BV]
    ht_ref,  # [N, 1, BK , BV]
    scratch_ref,  # [BK, BV]
    *,
    BT,
    BS,
):
    BT, BK = k_ref.shape[1], k_ref.shape[2]
    BV = v_ref.shape[2]

    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)

    i_h, _i_k, _i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)

    if g_gamma_ref is not None:
        b_g = g_gamma_ref[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    t0 = i_t * BT

    seq_idx = chunk_to_seq[i_t]

    bos = cu_seqlens_ref[seq_idx]
    eos = cu_seqlens_ref[seq_idx + 1]

    @pl.when(bos != eos)
    def _():
        # reset h state
        @pl.when(t0 == bos)
        def reset_state():
            if h0_ref is not None:
                scratch_ref[...] = h0_ref[seq_idx, 0].astype(scratch_ref.dtype)
            else:
                scratch_ref[...] = b_h_start

        # store intermediate state
        @pl.when(i_t % NTS == 0)
        def store_fn():
            s_i = i_t // NTS
            h_ref[s_i, 0] = scratch_ref[...].astype(h_ref.dtype)
            return None

        k_tile = k_ref[(0, slice(None), slice(None))]  # [BT,BK]
        v_tile = v_ref[(0, slice(None), slice(None))]  # [BT,BV]

        if g_gamma_ref is not None:
            # tpu not support scalar bf16 mul
            b_g_last = (
                g_gamma_ref[i_h].astype(jnp.float32) * jnp.minimum(BT, eos - i_t * BT)
            ).astype(g_gamma_ref.dtype)
            scratch_ref[...] *= exp(b_g_last)
            v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)

        if gk_ref is not None:
            gk_tile = gk_ref[(0, slice(None), slice(None))]  # BT * BK
            g_last = gk_tile[-1, :]
            decay = exp(g_last)
            scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
            k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

        # state update
        scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
            k_tile.astype(jnp.float32).T,
            v_tile.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        @pl.when(t0 + BT >= eos)
        def write_final():
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = scratch_ref[...].astype(jnp.float32)


@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
        "interpret",
    ],
)
def chunk_fwd_h_kernel_varlen(
    k: jax.Array,  # [B,T,H,K]
    v: jax.Array,  # [B,T,H,V]
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    interpret: bool = False,
):
    assert g is None, "g should be None."
    assert gv is None, "gv should be None."
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    T_sum = B * T
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)

    N, NS = (
        len(cu_seqlens_dev) - 1,
        T_sum // BS,
    )

    k = jnp.reshape(k, (T_sum, H, K))
    v = jnp.reshape(v, (T_sum, H, V))

    k = jnp.transpose(k, (1, 0, 2))  # (H,B*T,K)
    v = jnp.transpose(v, (1, 0, 2))  # (H,B*T,V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K))
        gk = jnp.transpose(gk, (1, 0, 2))  # (H,B*T,K)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), T_sum // BT)

    def k_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def gk_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def v_index_map(head_index, _, v_index, t_index):
        return head_index, t_index, v_index

    def h0_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def ht_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def h_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, BT, BK), k_index_map),
        pl.BlockSpec((1, BT, BV), v_index_map),
    ]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    kernel = functools.partial(
        _chunk_fwd_h_kernel_varlen,
        BT=BT,
        BS=BS,
    )
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
    )(k, v, h0, gk, g_gamma, cu_seqlens_dev, chunk_to_seq)
    if output_final_state:
        return h, ht
    return h, None


# =============================================================================
# Chunk forward o (from tops/ops/common/chunk_o.py)
# Pallas TPU kernel for computing chunk output.
# =============================================================================


def _chunk_fwd_o_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    g_ref,
    g_gamma_ref,
    scale_ref,
    o_ref,
    *,
    BT: int,
):
    """Pallas kernel for chunk_fwd_o.

    Grid: (H, total_NT, num_v_tiles)
    Refs (after block spec indexing):
      q_ref/k_ref: (1, 1, BT, K)
      v_ref: (1, 1, BT, BV)
      h_ref: (1, 1, K, BV)
      g_ref: (1, 1, BT, 128) or None  (broadcast to 4D for TPU alignment)
      g_gamma_ref: [H] via SMEM or ANY
      scale_ref: (1,) via SMEM or ANY
      o_ref: (1, 1, BT, BV)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, BV)
    b_h = h_ref[0, 0]  # (K, BV)

    b_o = jnp.dot(
        b_q,
        b_h,
        preferred_element_type=jnp.float32,
    )
    b_A = jnp.dot(
        b_q,
        b_k.T,
        preferred_element_type=jnp.float32,
    )

    if g_ref is not None:
        b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
        b_o = b_o * exp(b_g)[:, None]
        g_diff = b_g[:, None] - b_g[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff = jnp.where(fwd_mask, g_diff, 0.0)
        b_A = b_A * exp(safe_g_diff)

    if g_gamma_ref is not None:
        head_idx = pl.program_id(0)
        b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
        b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
        b_o = b_o * exp(b_g_gamma)[:, None]
        g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff = jnp.where(fwd_mask, g_gamma_diff, 0.0)
        b_A = b_A * exp(safe_g_gamma_diff)

    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(mask, b_A, 0.0)
    scale = scale_ref[0].astype(jnp.float32)

    # Keep b_A in fp32 for precision; upcast b_v instead.
    b_o = (
        b_o * scale
        + jnp.dot(
            b_A,
            b_v.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    o_ref[0, 0] = b_o.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size",),
)
def _chunk_fwd_o_pl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Pallas launcher for chunk_fwd_o on the uniform-length path."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    def _reshape_bt(x, D):
        return x.reshape(B, NT, BT, H, D).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, D)

    _q = _reshape_bt(q, K)  # (H, total_NT, BT, K)
    _k = _reshape_bt(k, K)  # (H, total_NT, BT, K)
    _v = _reshape_bt(v, V)  # (H, total_NT, BT, V)
    _h = h.reshape(B, NT, H, K, V).transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _g = None
    if g is not None:
        _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT)
        _g = jnp.broadcast_to(_g[:, :, :, None], (H, total_NT, BT, 128))  # 4D for TPU alignment

    BV = 128 if V % 128 == 0 else V
    num_v_tiles = V // BV

    if num_v_tiles > 1:
        # Split V into tiles and merge with H: (H, ..., V) -> (H*num_v_tiles, ..., BV)
        _v = (
            _v.reshape(H, total_NT, BT, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, BT, BV)
        )
        _h = (
            _h.reshape(H, total_NT, K, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, K, BV)
        )
        # g_gamma: repeat each head value for its V-tiles
        if g_gamma is not None:
            g_gamma = jnp.repeat(g_gamma, num_v_tiles)  # (H * num_v_tiles,)

    H_VT = H * num_v_tiles
    grid = (H_VT, total_NT)

    # q/k/g index by head = hv_idx // num_v_tiles; v/h index by hv_idx directly
    spec_qk = pl.BlockSpec(
        (1, 1, BT, K), index_map=lambda hv_idx, nt_idx: (hv_idx // num_v_tiles, nt_idx, 0, 0)
    )
    spec_v = pl.BlockSpec((1, 1, BT, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0))
    spec_h = pl.BlockSpec((1, 1, K, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0))
    interpret = get_interpret()
    spec_g = (
        None
        if _g is None
        else pl.BlockSpec(
            (1, 1, BT, 128), index_map=lambda hv_idx, nt_idx: (hv_idx // num_v_tiles, nt_idx, 0, 0)
        )
    )
    spec_gamma = (
        None
        if g_gamma is None
        else pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)
    )
    spec_scale = pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)

    o = pl.pallas_call(
        functools.partial(_chunk_fwd_o_kernel, BT=BT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h, spec_g, spec_gamma, spec_scale],
            out_specs=pl.BlockSpec(
                (1, 1, BT, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((H_VT, total_NT, BT, BV), v.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _g, g_gamma, jnp.asarray(scale, dtype=jnp.float32).reshape(1))

    if num_v_tiles > 1:
        o = (
            o.reshape(H, num_v_tiles, total_NT, BT, BV)
            .transpose(0, 2, 3, 1, 4)
            .reshape(H, total_NT, BT, V)
        )

    o = o.reshape(H, B, NT, BT, V).transpose(1, 2, 3, 0, 4)
    return o.reshape(B, T, H, V)


def chunk_fwd_o(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunk forward output computation."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if scale is None:
        scale = K**-0.5

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size={C}"
    assert (cu_seqlens_cpu is None) or (
        cu_seqlens_cpu % chunk_size == 0
    ).all(), "All sequence lengths must be divisible by chunk_size"
    if cu_seqlens_cpu is not None or cu_seqlens_dev is not None:
        assert B == 1, f"Packed varlen chunk_fwd_o expects B=1, got B={B}"
    assert scale is not None

    return _chunk_fwd_o_pl(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        chunk_size=chunk_size,
    )


# =============================================================================
# Chunk forward varlen + simple_gla_fwd entry point
# (from tops/ops/simple_gla/chunk.py and tops/ops/simple_gla/__init__.py)
# =============================================================================


def chunk_simple_gla_fwd_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = cu_seqlens_dev.shape[0] - 1 if cu_seqlens_dev is not None else B

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (N, H, K, V))
    assert T % chunk_size == 0
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu is None."
    assert cu_seqlens_dev is not None, "cu_seqlens_dev is not None."
    assert (K % 128 == 0) and (V % 128 == 0)
    assert B == 1, "B must be 1."

    h, ht = chunk_fwd_h_kernel_varlen(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=h0,
        output_final_state=use_ht,
        states_in_fp32=False,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
    )
    return o, ht


class SimpleGLAKernelMode(enum.Enum):
    """Simple GLA kernel implementation mode."""

    CHUNK = "chunk"
    FUSED_CHUNK = "fused_chunk"


def simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK,
):
    if cu_seqlens_dev is not None:
        fn = chunk_simple_gla_fwd_varlen
    else:
        raise NotImplementedError(
            f"Non-varlen simple_gla_fwd (mode={mode}) is not vendored. "
            "Only the varlen path (cu_seqlens_dev != None) is supported."
        )
    return fn(
        q,
        k,
        v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        use_ht=use_ht,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
    )
