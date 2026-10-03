# Adapted from https://github.com/primatrix/pallas-kernel (rev 3c691ad3)
# Vendored to remove external dependency after the upstream repository went private.
#
# This file merges the following modules into a single file:
#   - tops/utils.py (cdiv, align_up, pad_to_multiple, prepare_lens, prepare_chunk_indices, assert_shape, assert_shape_or_none)
#   - tops/ops/utils.py (exp, exp2, get_interpret)
#   - tops/ops/common/cumsum.py (chunk_local_cumsum_vector via _chunk_cumsum_kernel)
#   - tops/ops/kda/chunk_intra_fwd.py (_solve_unit_lower_triangular, _kda_fwd_intra_kernel, kda_fwd_intra)
#   - tops/ops/common/chunk_delta_h.py (_prepare_chunk_offsets, _chunk_gated_delta_rule_fwd_kernel, chunk_gated_delta_rule_fwd_h)
#   - tops/ops/gla/chunk.py (_chunk_kda_fwd_o_gk_varlen_kernel renamed to _chunk_kda_fwd_o_gk_pl_kernel, chunk_kda_fwd_o_gk_varlen renamed to chunk_kda_fwd_o_gk)
#   - tops/ops/kda/gate.py (kda_gate_chunk_cumsum, pallas_kda_gate_cumsum)
#   - tops/ops/kda/chunk_fwd.py (_align_seqs, _unalign_output, chunk_kda_fwd)
"""KDA chunked forward pass for variable-length sequences (self-contained)."""

from __future__ import annotations

import functools
import math
import os
from functools import singledispatch

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

# ============================================================================
# Utilities
# ============================================================================


@singledispatch
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@cdiv.register
def _cdiv_jax(x: jax.Array, y: int):
    return (x + y - 1) // y


def align_up(x: int, align: int):
    return cdiv(x, align) * align


@singledispatch
def pad_to_multiple(x, multiple: int, axis: int, val):
    raise NotImplementedError(f"pad_to_multiple is not implemented for type {type(x)}")


@pad_to_multiple.register
def _pad_to_multiple_jax(x: jax.Array, multiple: int | list, axis: int | list, val):
    if isinstance(multiple, int):
        multiple = [multiple]
    if isinstance(axis, int):
        axis = [axis]
    assert len(multiple) == len(axis)
    shape = list(x.shape)
    pad_width = [(0, 0)] * len(shape)
    for idx in range(len(axis)):
        ax = axis[idx]
        mu = multiple[idx]
        remainder = shape[ax] % mu
        if remainder != 0:
            pad_width[ax] = (0, mu - remainder)
    return jnp.pad(x, pad_width, constant_values=val)


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
    lens = prepare_lens(cu_seqlens)
    n_chunks = cdiv(lens, chunk_size)
    num_seqs = len(lens)
    total_nt = max_T // chunk_size
    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32), n_chunks, total_repeat_length=total_nt
    )
    prefix_chunks = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_chunks)])
    seq_offsets = jnp.repeat(prefix_chunks[:-1], n_chunks, total_repeat_length=total_nt)
    block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
    return jnp.stack([seq_ids, block_ids], axis=1)


def assert_shape_or_none(x, expected_shape, name="tensor"):
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


def assert_shape(x, expected_shape, name="tensor"):
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


def exp2(x):
    return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# ============================================================================
# Chunk-local cumulative sum (varlen Pallas kernel only)
# ============================================================================

_VMEM_HW_LIMIT_BYTES = 30 * 1024 * 1024


def _chunk_cumsum_kernel(
    s_ref,
    o_ref,
    *,
    BT: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
):
    num_steps = int(math.log2(BT))
    s = s_ref[:, 0, :, :].astype(jnp.float32)

    if REVERSE:
        for d in range(num_steps):
            stride = 1 << d
            top = s[:, : BT - stride, :] + s[:, stride:, :]
            bot = s[:, BT - stride :, :]
            s = jnp.concatenate([top, bot], axis=1)
    else:
        for d in range(num_steps):
            stride = 1 << d
            top = s[:, :stride, :]
            bot = s[:, stride:, :] + s[:, :-stride, :]
            s = jnp.concatenate([top, bot], axis=1)

    if HAS_SCALE:
        s = s * scale

    o_ref[:, 0, dslice(0, BT), :] = s.astype(o_ref.dtype)


def chunk_local_cumsum_vector(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:
    assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be power of 2"
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"

    BT = chunk_size
    BS = 128
    BB = 8

    if head_first:
        B, H, T, S = g.shape
        g_flat = g.reshape(B * H, T, S)
    else:
        B, T, H, S = g.shape
        g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

    BH = B * H
    out_dtype = output_dtype or g.dtype
    HAS_SCALE = scale is not None
    scale_val = scale if scale is not None else 1.0

    interpret = get_interpret()

    pad_S = (BS - (S % BS)) % BS
    if pad_S > 0:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))
    S_padded = S + pad_S
    NS = S_padded // BS

    pad_BH = (BB - (BH % BB)) % BB
    if pad_BH > 0:
        g_flat = jnp.pad(g_flat, ((0, pad_BH), (0, 0), (0, 0)))
    BH_padded = BH + pad_BH

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NC_max = len(chunk_indices)

    g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))
    T_alloc = T + BT

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunk_indices_i32 = chunk_indices.astype(jnp.int32)
    N = cu_i32.shape[0] - 1
    chunks_per_seq = (jnp.diff(cu_i32) + BT - 1) // BT

    seq_id = chunk_indices_i32[:, 0]
    local_ci = chunk_indices_i32[:, 1]
    safe_seq_id = jnp.clip(seq_id, 0, N - 1)
    chunk_valid = (
        (seq_id == safe_seq_id) & (local_ci >= 0) & (local_ci < chunks_per_seq[safe_seq_id])
    )

    bos = cu_i32[safe_seq_id]
    eos = cu_i32[safe_seq_id + 1]
    chunk_starts = jnp.where(chunk_valid, bos + local_ci * BT, 0)
    positions = chunk_starts[:, None] + jnp.arange(BT, dtype=jnp.int32)[None, :]
    token_valid = chunk_valid[:, None] & (positions < eos[:, None]) & (positions < T)

    def _gather_chunk(start):
        return jax.lax.dynamic_slice(
            g_flat,
            (0, start, 0),
            (BH_padded, BT, S_padded),
        )

    g_chunks = jax.vmap(_gather_chunk)(chunk_starts).transpose(1, 0, 2, 3)
    g_chunks = jnp.where(token_valid[None, :, :, None], g_chunks, 0)

    elem_bytes = 4
    while BB > 1 and 4 * BB * BT * BS * elem_bytes > _VMEM_HW_LIMIT_BYTES:
        BB //= 2
    NBH = BH_padded // BB

    grid = (NS, NBH, NC_max)
    kernel = functools.partial(
        _chunk_cumsum_kernel,
        BT=BT,
        REVERSE=reverse,
        HAS_SCALE=HAS_SCALE,
        scale=scale_val,
    )

    def _index_map(i_s, i_bb, i_t):
        return (i_bb, i_t, 0, i_s)

    block_shape = (BB, 1, BT, BS)

    o_chunks = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block_shape=block_shape, index_map=_index_map)],
            out_specs=pl.BlockSpec(block_shape=block_shape, index_map=_index_map),
        ),
        out_shape=jax.ShapeDtypeStruct(g_chunks.shape, out_dtype),
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(g_chunks)

    o_chunks = jnp.where(token_valid[None, :, :, None], o_chunks, 0)
    sentinel = jnp.minimum(cu_i32[-1], T)
    scatter_positions = jnp.where(token_valid, positions, sentinel).reshape(-1)
    scatter_values = o_chunks.reshape(BH_padded, NC_max * BT, S_padded)

    o_flat = jnp.zeros((BH_padded, T_alloc, S_padded), dtype=o_chunks.dtype)
    o_flat = o_flat.at[:, scatter_positions, :].add(scatter_values)

    o_flat = o_flat[:BH, :T, :S]

    if head_first:
        return o_flat.reshape(B, H, T, S)
    else:
        return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# ============================================================================
# KDA intra-chunk forward (varlen only)
# ============================================================================


def _solve_unit_lower_triangular(A, b, block_size=16):
    N, D = b.shape
    BS = block_size
    assert BS > 0 and N % BS == 0
    num_blocks = N // BS
    A = A.astype(jnp.float32)
    b = b.astype(jnp.float32)

    blocks = jnp.split(b, num_blocks, axis=0)

    for i in range(num_blocks):
        start = i * BS
        end = (i + 1) * BS
        A_ii = A[start:end, start:end]
        x_block = blocks[i]

        rows = [x_block[r] for r in range(BS)]
        for j in range(BS):
            if j > 0:
                vec = A_ii[j, :j][None, :]
                mat = jnp.stack(rows[:j])
                correction = jax.lax.dot_general(
                    vec,
                    mat,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).squeeze(axis=0)
                rows[j] = rows[j] - correction

        x_block = jnp.stack(rows)
        blocks[i] = x_block

        if i < num_blocks - 1:
            rest_start = (i + 1) * BS
            x_rest = jnp.concatenate(blocks[i + 1 :], axis=0)
            A_rest = A[rest_start:, start:end]
            update = jax.lax.dot_general(
                A_rest,
                x_block,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            x_rest = x_rest - update
            remaining = num_blocks - 1 - i
            new_blocks = jnp.split(x_rest, remaining, axis=0)
            for idx, nb in enumerate(new_blocks):
                blocks[i + 1 + idx] = nb

    return jnp.concatenate(blocks, axis=0)


def _kda_fwd_intra_kernel(
    q_ref,
    k_ref,
    g_ref,
    beta_ref,
    v_ref,
    u_out_ref,
    w_out_ref,
    qg_out_ref,
    kg_out_ref,
    Aqk_out_ref,
    Akk_inv_out_ref,
    *,
    chunk_size,
    head_dim,
    value_dim,
    scale,
    disable_recompute,
    safe_gate,
    intra_block_size,
    COMPUTE_BLOCK_CHUNKS,
):
    dtype = q_ref.dtype
    BT = chunk_size
    causal_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    strict_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32), k=-1)
    identity = jnp.eye(BT, dtype=jnp.float32)

    # Adjacent logical chunks are independent in the intra stage.  Loading a
    # small group into one program amortizes the program/grid overhead while
    # preserving the exact per-chunk triangular solve and arithmetic order.
    for i_chunk in range(COMPUTE_BLOCK_CHUNKS):
        q = q_ref[0, 0, i_chunk]
        k = k_ref[0, 0, i_chunk]
        g = g_ref[0, 0, i_chunk]
        beta = beta_ref[0, 0, i_chunk]
        v = v_ref[0, 0, i_chunk]

        g_f32 = g.astype(jnp.float32)
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        beta_f32 = beta.astype(jnp.float32)

        # Build Aqk and L directly using exp2(g[i] - g[j]).
        # For causal (i >= j): g_cumsum[i] <= g_cumsum[j], so g[i]-g[j] <= 0,
        # giving exp2 in (0, 1].  This avoids the split-normalization overflow
        # that occurs with exp2(g-gn) when per-step gate changes exceed ~127.
        # g_diff[i, j, k] = g[i, k] - g[j, k];  shape [BT, BT, K]
        g_diff = g_f32[:, None, :] - g_f32[None, :, :]
        # Mask anti-causal entries to -126 before exp2 to prevent overflow;
        # they will be zeroed by causal_bt / strict_bt anyway.
        g_diff = jnp.where(causal_bt[:, :, None] > 0, g_diff, -126.0)
        decay = exp2(jnp.maximum(g_diff, -126.0))  # [BT, BT, K]

        # Aqk[i, j] = scale * sum_k q[i,k] * k[j,k] * decay[i,j,k]
        Aqk = scale * jnp.sum(q_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1)
        Aqk = (Aqk * causal_bt).astype(dtype)

        # L[i, j] = beta[i] * sum_k k[i,k] * k[j,k] * decay[i,j,k]   (i > j)
        L = jnp.sum(k_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1) * beta_f32 * strict_bt

        v_beta = v.astype(jnp.float32) * beta_f32
        k_eg_beta = k_f32 * exp2(g_f32) * beta_f32
        combined_b = jnp.concatenate([v_beta, k_eg_beta, identity], axis=-1)
        combined_x = _solve_unit_lower_triangular(L, combined_b, block_size=intra_block_size)

        u = combined_x[:, :value_dim]
        w = combined_x[:, value_dim : value_dim + head_dim]
        A_inv = combined_x[:, value_dim + head_dim :]

        g_last = g_f32[BT - 1 : BT, :]
        kg = k_f32 * exp2(g_last - g_f32)

        qg = q_f32 * exp2(g_f32) if disable_recompute else jnp.zeros_like(q_f32)

        u_out_ref[0, 0, i_chunk] = u.astype(u_out_ref.dtype)
        w_out_ref[0, 0, i_chunk] = w.astype(w_out_ref.dtype)
        qg_out_ref[0, 0, i_chunk] = qg.astype(qg_out_ref.dtype)
        kg_out_ref[0, 0, i_chunk] = kg.astype(kg_out_ref.dtype)
        Aqk_out_ref[0, 0, i_chunk] = Aqk.astype(Aqk_out_ref.dtype)
        Akk_inv_out_ref[0, 0, i_chunk] = A_inv.astype(Akk_inv_out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "intra_block_size",
        "scalar_intra_solve",
        "compute_block_chunks",
        "scale",
        "safe_gate",
        "disable_recompute",
    ],
)
def kda_fwd_intra(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    cu_seqlens,
    chunk_size=64,
    chunk_indices=None,
    safe_gate=True,
    disable_recompute=False,
    intra_block_size=16,
    scalar_intra_solve=False,
    compute_block_chunks=1,
):
    assert cu_seqlens is not None, "cu_seqlens must be provided for varlen"
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    solve_block_size = 1 if scalar_intra_solve else intra_block_size
    COMPUTE_BLOCK_CHUNKS = compute_block_chunks
    assert B == 1, f"varlen requires B=1 (packed layout), got B={B}"
    assert BT >= 16 and BT % 16 == 0
    assert solve_block_size > 0 and BT % solve_block_size == 0
    assert COMPUTE_BLOCK_CHUNKS >= 1, "compute_block_chunks must be positive"

    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(gk, (B, T, H, K), "gk")
    assert_shape(beta, (B, T, H), "beta")

    N = cu_seqlens.shape[0] - 1
    T_alloc = T + BT

    pad4d = lambda x: jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
    q_pad, k_pad, gk_pad, v_pad = pad4d(q), pad4d(k), pad4d(gk), pad4d(v)
    beta_pad = jnp.pad(beta.reshape(B, T, H, 1), ((0, 0), (0, BT), (0, 0), (0, 0)))

    cu_i32 = cu_seqlens.astype(jnp.int32)
    seq_lens = jnp.diff(cu_i32)
    chunks_per_seq = (seq_lens + BT - 1) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    total_chunks = cum_chunks[-1]

    NC_real = len(chunk_indices) if chunk_indices is not None else T // BT + N
    # Pad only the packed chunk axis.  Invalid tail chunks gather a harmless
    # duplicate tile and scatter it to the extra sentinel slot, so grouping
    # remains exact even when a varlen workload has an odd chunk count.
    NC_max = int(align_up(NC_real, COMPUTE_BLOCK_CHUNKS))
    flat_idx = jnp.arange(NC_max, dtype=jnp.int32)
    is_valid = flat_idx < total_chunks

    seq_id = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1)
    local_ci = flat_idx - cum_chunks[seq_id]
    bos = cu_i32[seq_id]
    # After _align_seqs, every sequence is BT-aligned, so all chunks are full.
    # No partial-chunk masking needed.
    chunk_starts = jnp.where(is_valid, bos + local_ci * BT, 0)

    def gather(x_pad, D):
        def extract(start):
            return jax.lax.dynamic_slice(x_pad, (0, start, 0, 0), (1, BT, H, D))[0]

        return jax.vmap(extract)(chunk_starts)

    q_c, k_c, gk_c, beta_c, v_c = (
        gather(q_pad, K),
        gather(k_pad, K),
        gather(gk_pad, K),
        gather(beta_pad, 1),
        gather(v_pad, V),
    )

    def _to_bhnd(x):
        return x.transpose(2, 0, 1, 3)[None]

    q_r, k_r, g_r, beta_r, v_r = (
        _to_bhnd(q_c),
        _to_bhnd(k_c),
        _to_bhnd(gk_c),
        _to_bhnd(beta_c),
        _to_bhnd(v_c),
    )

    grid = (B, H, NC_max // COMPUTE_BLOCK_CHUNKS)

    def _make_spec(last_dim):
        return pl.BlockSpec(
            index_map=lambda i, j, n: (i, j, n, 0, 0),
            block_shape=(1, 1, COMPUTE_BLOCK_CHUNKS, BT, last_dim),
        )

    u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r = pl.pallas_call(
        functools.partial(
            _kda_fwd_intra_kernel,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            disable_recompute=disable_recompute,
            safe_gate=safe_gate,
            intra_block_size=solve_block_size,
            COMPUTE_BLOCK_CHUNKS=COMPUTE_BLOCK_CHUNKS,
        ),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct((B, H, NC_max, BT, V), q.dtype),
            jax.ShapeDtypeStruct((B, H, NC_max, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, H, NC_max, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, H, NC_max, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, H, NC_max, BT, BT), q.dtype),
            jax.ShapeDtypeStruct((B, H, NC_max, BT, BT), q.dtype),
        ],
        in_specs=[_make_spec(K), _make_spec(K), _make_spec(K), _make_spec(1), _make_spec(V)],
        out_specs=[
            _make_spec(V),
            _make_spec(K),
            _make_spec(K),
            _make_spec(K),
            _make_spec(BT),
            _make_spec(BT),
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(q_r, k_r, g_r, beta_r, v_r)

    pos = chunk_starts[:, None] + jnp.arange(BT)[None, :]
    pos = jnp.where(is_valid[:, None], pos, T_alloc - 1)
    flat_pos = pos.reshape(-1)

    def _scatter(chunks_r, D):
        chunks = chunks_r[0].transpose(1, 2, 0, 3)
        flat_chunks = chunks.reshape(-1, H, D)
        out = jnp.zeros((T_alloc, H, D), dtype=chunks.dtype)
        out = out.at[flat_pos].add(flat_chunks)
        return out[:T][None]

    w_out, u_out, kg_out = _scatter(w_r, K), _scatter(u_r, V), _scatter(kg_r, K)
    Aqk_out, Akk_out = _scatter(Aqk_r, BT), _scatter(Akk_inv_r, BT)
    qg_out = _scatter(qg_r, K) if disable_recompute else None

    return w_out, u_out, qg_out, kg_out, Aqk_out, Akk_out


# ============================================================================
# Delta-rule inter-chunk state propagation (varlen only)
# ============================================================================


def _prepare_chunk_offsets(seqlens, chunk_size):
    return jnp.pad(
        cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32),
        (1, 0),
        constant_values=0,
    ).cumsum(-1)


def _chunk_gated_delta_rule_fwd_kernel(
    seqlens_ref,
    chunk_offsets_ref,
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
    NT,
    BT,
    STATE_BLOCK_CHUNKS,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
):
    idx_n = pl.program_id(0)
    idx_nb = pl.program_id(2)

    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    real_NT = (eos - bos) // BT

    K, V = k_ref.shape[-1], v_ref.shape[-1]

    @pl.when(idx_nb == 0)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    # The Pallas load/schedule tile may contain multiple logical KDA chunks.
    # Preserve the original recurrence by executing those chunk updates in
    # order while the state stays resident in the same VMEM scratch buffer.
    for i_chunk in range(STATE_BLOCK_CHUNKS):
        idx_nt = idx_nb * STATE_BLOCK_CHUNKS + i_chunk  # noqa: B023
        chunk_slice = dslice(i_chunk * BT, BT)

        @pl.when(idx_nt < real_NT)
        def _():
            h_ref[0, i_chunk, 0] = scratch_ref[...].astype(h_ref.dtype)  # noqa: B023

            b_k = k_ref[0, 0, chunk_slice, :]  # noqa: B023
            b_w = w_ref[0, 0, chunk_slice, :]  # noqa: B023
            b_v = jnp.dot(
                b_w.astype(jnp.float32),
                scratch_ref[...],
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            b_u = v_ref[0, 0, chunk_slice, :]  # noqa: B023
            b_v = b_u.astype(b_v.dtype) - b_v
            if SAVE_NEW_VALUE:
                v_new_ref[0, 0, chunk_slice, :] = b_v.astype(v_new_ref.dtype)  # noqa: B023

            if USE_G:
                b_g = g_ref[0, 0, chunk_slice, 0]  # noqa: B023
                b_g_last = g_ref[0, 0, i_chunk * BT + BT - 1, 0].astype(jnp.float32)  # noqa: B023
                if USE_EXP2:
                    b_v = b_v * exp2(b_g_last - b_g)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_v = b_v * exp(b_g_last - b_g)[:, None]
                    b_g_last = exp(b_g_last)
                scratch_ref[...] *= b_g_last
            if USE_GK:
                b_gk_last = gk_ref[0, 0, i_chunk * BT + BT - 1].astype(jnp.float32)  # noqa: B023
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

        @pl.when(idx_nt == real_NT - 1)
        def _():
            if STORE_FINAL_STATE:
                ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_gated_delta_rule_fwd_h(
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
    state_block_chunks=1,
    state_dim_alignment=128,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    STATE_BT = BT * state_block_chunks

    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    assert B == 1, f"varlen mode requires B==1, got B={B}"
    assert state_block_chunks >= 1, "state_block_chunks must be positive"
    assert (
        state_dim_alignment > 0 and (state_dim_alignment & (state_dim_alignment - 1)) == 0
    ), "state_dim_alignment must be a positive power of 2"
    assert T % STATE_BT == 0, f"T={T} must be divisible by state block size {STATE_BT}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(u, (B, T, H, V), "u")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # --- Varlen launcher ---
    k = k.astype(jnp.float32)
    w = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    K_PADSIZE = int(align_up(K, state_dim_alignment))
    V_ALIGNED = int(align_up(V, state_dim_alignment))

    assert chunk_indices is not None
    NT = len(chunk_indices)
    NB_max = T // STATE_BT
    chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
    assert initial_state is None or initial_state.shape == (N, H, K, V)

    T_alloc = T + STATE_BT

    k_pad = (
        jnp.pad(k, ((0, 0), (0, STATE_BT), (0, 0), (0, K_PADSIZE - K)))
        if K_PADSIZE > K
        else jnp.pad(k, ((0, 0), (0, STATE_BT), (0, 0), (0, 0)))
    )
    w_pad = (
        jnp.pad(w, ((0, 0), (0, STATE_BT), (0, 0), (0, K_PADSIZE - K)))
        if K_PADSIZE > K
        else jnp.pad(w, ((0, 0), (0, STATE_BT), (0, 0), (0, 0)))
    )
    k_t = jnp.transpose(k_pad, (0, 2, 1, 3))
    w_t = jnp.transpose(w_pad, (0, 2, 1, 3))

    v_pad = (
        jnp.pad(u_f32, ((0, 0), (0, STATE_BT), (0, 0), (0, V_ALIGNED - V)))
        if V_ALIGNED > V
        else jnp.pad(u_f32, ((0, 0), (0, STATE_BT), (0, 0), (0, 0)))
    )
    v_t = jnp.transpose(v_pad, (0, 2, 1, 3))

    if g is not None:
        g_fp32 = g.astype(jnp.float32).reshape(B, T, H, 1)
        g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
        g_fp32 = jnp.pad(g_fp32, ((0, 0), (0, STATE_BT), (0, 0), (0, 0)))
        g_t = jnp.transpose(g_fp32, (0, 2, 1, 3))
    else:
        g_t = None

    if gk is not None:
        gk_fp32 = gk.astype(jnp.float32)
        if K_PADSIZE > K:
            gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, STATE_BT), (0, 0), (0, 0)))
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
    v_new_spec = (
        jax.ShapeDtypeStruct([B, H, T_alloc, V_ALIGNED], jnp.float32) if save_new_value else None
    )
    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )

    def _t_index_map(n, h, nb, seqlens_ref, chunk_offsets_ref):
        bos = pl.multiple_of(seqlens_ref[n], STATE_BT)
        block_idx = jnp.minimum(bos // STATE_BT + nb, T // STATE_BT)
        return (0, h, block_idx, 0)

    def _h_index_map(n, h, nb, seqlens_ref, chunk_offsets_ref):
        bos = pl.multiple_of(seqlens_ref[n], STATE_BT)
        block_idx = jnp.minimum(bos // STATE_BT + nb, NT // state_block_chunks - 1)
        return (0, block_idx, h, 0, 0)

    k_blockspec = pl.BlockSpec([1, 1, STATE_BT, K_PADSIZE], index_map=_t_index_map)
    v_blockspec = pl.BlockSpec([1, 1, STATE_BT, V_ALIGNED], index_map=_t_index_map)
    w_blockspec = pl.BlockSpec([1, 1, STATE_BT, K_PADSIZE], index_map=_t_index_map)
    g_blockspec = (
        pl.BlockSpec([1, 1, STATE_BT, g_pad_size], index_map=_t_index_map)
        if g is not None
        else None
    )
    gk_blockspec = (
        pl.BlockSpec([1, 1, STATE_BT, K_PADSIZE], index_map=_t_index_map)
        if gk is not None
        else None
    )
    h0_blockspec = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda n, h, nt, *_: (n, h, 0, 0))
        if initial_state is not None
        else None
    )

    h_blockspec_out = pl.BlockSpec(
        [1, state_block_chunks, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map
    )
    v_new_blockspec_out = (
        pl.BlockSpec([1, 1, STATE_BT, V_ALIGNED], index_map=_t_index_map)
        if save_new_value
        else None
    )
    ht_blockspec_out = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda n, h, nt, *_: (n, h, 0, 0))
        if output_final_state
        else None
    )

    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    grid = (N, H, NB_max)
    interpret = get_interpret()

    h_out, v_new_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_gated_delta_rule_fwd_kernel,
            NT=NT,
            BT=BT,
            STATE_BLOCK_CHUNKS=state_block_chunks,
            USE_G=(g is not None),
            USE_GK=(gk is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            USE_EXP2=use_exp2,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
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
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        out_shape=[h_spec, v_new_spec, ht_spec],
        interpret=interpret,
    )(cu_seqlens, chunk_offsets, k_t, v_t, w_t, g_t, gk_t, h0)

    h_out = h_out[:, :, :, :K, :V]
    v_new_out = jnp.transpose(v_new_out[:, :, :T, :V], (0, 2, 1, 3)) if save_new_value else None
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None

    return h_out, v_new_out, ht_out


# ============================================================================
# kda forward O+GK (varlen only)
# ============================================================================


def _chunk_kda_fwd_o_gk_pl_kernel(
    q_ref,
    v_ref,
    g_ref,
    h_ref,
    A_ref,
    o_ref,
    *,
    BT,
    COMPUTE_BLOCK_CHUNKS,
    scale,
    USE_EXP2,
    ZERO_STATE_OUTPUT,
):
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]

    # Like the intra stage, output chunks have no cross-chunk dependence once
    # their boundary states are materialized.  Grouping them changes only the
    # Pallas load/program tile; each chunk retains its original arithmetic.
    for i_chunk in range(COMPUTE_BLOCK_CHUNKS):
        b_v = v_ref[0, i_chunk]
        b_A = A_ref[0, i_chunk]

        if ZERO_STATE_OUTPUT:
            # The only chunk entrance is the exact zero state, so its inter-
            # chunk contribution is identically zero.  Avoid loading q/g/h or
            # issuing the zero matmul; the retained intra-chunk term below is
            # unchanged.
            b_o = jnp.zeros_like(b_v, dtype=jnp.float32)
        else:
            b_q = q_ref[0, i_chunk]
            b_g = g_ref[0, i_chunk]
            b_h = h_ref[0, i_chunk]
            b_g_f32 = b_g.astype(jnp.float32)
            b_q_f32 = b_q.astype(jnp.float32)
            # Compute inter-chunk output: o = scale * q * exp2(g) @ h.
            # Use g[0] (first position, largest cumsum) as reference to avoid overflow/underflow:
            #   exp2(g[t]) = exp2(g[t] - g[0]) * exp2(g[0])
            # g[t] - g[0] <= 0 for all t (cumsum is monotonically decreasing), so exp2 is safe.
            # Factor exp2(g[0]) into h to preserve the matmul structure.
            _exp_fn = exp2 if USE_EXP2 else exp
            b_g_ref = b_g_f32[0:1, :]  # [1, K] — reference point
            b_qg = b_q_f32 * _exp_fn(jnp.maximum(b_g_f32 - b_g_ref, -126.0))
            # Scale h rows: h_scaled[k, v] = h[k, v] * exp2(g_ref[k])
            b_h_scaled = b_h.astype(jnp.float32) * _exp_fn(jnp.maximum(b_g_ref[0], -126.0))[:, None]
            b_o = jnp.dot(
                b_qg,
                b_h_scaled,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            b_o *= scale

        b_A_f32 = jnp.where(m_s, b_A, 0.0).astype(jnp.float32)
        b_o += jnp.dot(
            b_A_f32,
            b_v.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        o_ref[0, i_chunk] = b_o.astype(o_ref.dtype)


def chunk_kda_fwd_o_gk(
    q,
    v,
    g,
    A,
    h,
    scale,
    *,
    cu_seqlens,
    chunk_indices=None,
    chunk_size=64,
    compute_block_chunks=1,
    use_exp2=False,
    zero_state_output=False,
):
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    COMPUTE_BLOCK_CHUNKS = compute_block_chunks
    assert B == 1
    assert T % BT == 0
    assert COMPUTE_BLOCK_CHUNKS >= 1, "compute_block_chunks must be positive"

    N = cu_seqlens.shape[0] - 1
    T_alloc = T + BT

    assert zero_state_output or h is not None, "h is required unless its zero term is elided"
    pad4d = lambda x: jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
    v_pad, A_pad = pad4d(v), pad4d(A)
    q_pad = None if zero_state_output else pad4d(q)
    g_pad = None if zero_state_output else pad4d(g)

    cu_i32 = cu_seqlens.astype(jnp.int32)
    seq_lens = jnp.diff(cu_i32)
    chunks_per_seq = (seq_lens + BT - 1) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    total_chunks = cum_chunks[-1]

    NC_real = len(chunk_indices) if chunk_indices is not None else T // BT + N
    NC_max = int(align_up(NC_real, COMPUTE_BLOCK_CHUNKS))
    flat_idx = jnp.arange(NC_max, dtype=jnp.int32)
    is_valid = flat_idx < total_chunks

    seq_id = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1)
    local_ci = flat_idx - cum_chunks[seq_id]
    bos = cu_i32[seq_id]
    # After _align_seqs, every sequence is BT-aligned — no partial chunks.
    chunk_starts = jnp.where(is_valid, bos + local_ci * BT, 0)

    def gather(x_pad, D):
        def extract(start):
            return jax.lax.dynamic_slice(x_pad, (0, start, 0, 0), (1, BT, H, D))[0]

        return jax.vmap(extract)(chunk_starts)

    q_c = None if zero_state_output else gather(q_pad, K)
    g_c = None if zero_state_output else gather(g_pad, K)
    v_c, A_c = gather(v_pad, V), gather(A_pad, BT)

    _q = None if zero_state_output else q_c.transpose(2, 0, 1, 3)
    _v = v_c.transpose(2, 0, 1, 3)
    _g = None if zero_state_output else g_c.transpose(2, 0, 1, 3)
    _A = A_c.transpose(2, 0, 1, 3)

    if zero_state_output:
        _h = None
    else:
        NT_h = h.shape[1]
        _h = h[0].transpose(1, 0, 2, 3)
        if NC_max > NT_h:
            _h = jnp.pad(_h, ((0, 0), (0, NC_max - NT_h), (0, 0), (0, 0)))
        elif NC_max < NT_h:
            _h = _h[:, :NC_max]

    q_spec = (
        None
        if zero_state_output
        else pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    )
    g_spec = (
        None
        if zero_state_output
        else pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    )
    v_spec = pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = (
        None
        if zero_state_output
        else pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    )
    o_shape = jax.ShapeDtypeStruct([H, NC_max, BT, V], v.dtype)
    o_spec = pl.BlockSpec([1, COMPUTE_BLOCK_CHUNKS, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o_r = pl.pallas_call(
        functools.partial(
            _chunk_kda_fwd_o_gk_pl_kernel,
            BT=BT,
            COMPUTE_BLOCK_CHUNKS=COMPUTE_BLOCK_CHUNKS,
            scale=scale,
            USE_EXP2=use_exp2,
            ZERO_STATE_OUTPUT=zero_state_output,
        ),
        grid=(H, NC_max // COMPUTE_BLOCK_CHUNKS),
        out_shape=o_shape,
        in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=get_interpret(),
    )(_q, _v, _g, _h, _A)

    pos = chunk_starts[:, None] + jnp.arange(BT)[None, :]
    pos = jnp.where(is_valid[:, None], pos, T_alloc - 1)
    flat_pos = pos.reshape(-1)

    o_chunks = o_r.transpose(1, 2, 0, 3).reshape(-1, H, V)
    out = jnp.zeros((T_alloc, H, V), dtype=o_chunks.dtype)
    out = out.at[flat_pos].add(o_chunks)
    return out[:T][None]


# ============================================================================
# KDA gate cumsum helpers
# ============================================================================

_RCP_LN2 = 1.0 / math.log(2)


def kda_gate_chunk_cumsum(
    g,
    A_log,
    chunk_size,
    scale=None,
    dt_bias=None,
    cu_seqlens=None,
    output_dtype=jnp.float32,
    chunk_indices=None,
    lower_bound=None,
):
    B, T, H, K = g.shape
    assert_shape(g, (B, T, H, K), "g")
    assert A_log.shape == (H,), f"A_log shape {A_log.shape} != ({H},)"

    g_f32 = g.astype(jnp.float32)
    if dt_bias is not None:
        g_f32 = g_f32 + dt_bias.astype(jnp.float32).reshape(H, K)

    A = A_log.astype(jnp.float32)
    if lower_bound is None:
        g_act = -exp(A).reshape(1, 1, H, 1) * jax.nn.softplus(g_f32)
    else:
        g_act = lower_bound * jax.nn.sigmoid(exp(A).reshape(1, 1, H, 1) * g_f32)

    return chunk_local_cumsum_vector(
        g_act,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=output_dtype or jnp.float32,
        chunk_indices=chunk_indices,
    )


def pallas_kda_gate_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=_RCP_LN2,
    cu_seqlens=None,
    head_first=False,
    output_dtype=jnp.float32,
    chunk_indices=None,
):
    B, T, H, K = g.shape
    assert_shape(g, (B, T, H, K), "g")
    assert T % chunk_size == 0, f"T={T} must be divisible by chunk_size={chunk_size}"

    return chunk_local_cumsum_vector(
        g,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=False,
        output_dtype=jnp.float32,
    )


# ============================================================================
# Varlen alignment helpers
# ============================================================================


def _align_seqs(tensors_4d, tensors_3d, cu_seqlens, align):
    N = cu_seqlens.shape[0] - 1
    T_old = tensors_4d[0].shape[1]

    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_lens = ((seg_lens + align - 1) // align) * align
    padded_cu = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_lens)])
    T_new = ((T_old + N * (align - 1) + align - 1) // align) * align

    def _build_gather(i, gather_idx):
        old_start = cu_seqlens[i]
        new_start = padded_cu[i]
        sl = seg_lens[i]
        j = jnp.arange(T_new)
        in_seg = (j >= new_start) & (j < new_start + sl)
        src = old_start + (j - new_start)
        return jnp.where(in_seg, src, gather_idx)

    gather_idx = jnp.full(T_new, T_old, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)

    def repack_4d(t):
        return jnp.pad(t, ((0, 0), (0, T_new - T_old), (0, 0), (0, 0)))[:, gather_idx]

    def repack_3d(t):
        return jnp.pad(t, ((0, 0), (0, T_new - T_old), (0, 0)))[:, gather_idx]

    return (
        [repack_4d(t) for t in tensors_4d],
        [repack_3d(t) for t in tensors_3d],
        padded_cu,
        cu_seqlens,
    )


def _unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out):
    N = orig_cu_seqlens.shape[0] - 1
    orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

    def _build_gather(i, gather_idx):
        orig_start = orig_cu_seqlens[i]
        aligned_start = aligned_cu_seqlens[i]
        sl = orig_seg_lens[i]
        j = jnp.arange(T_out)
        in_seg = (j >= orig_start) & (j < orig_start + sl)
        src = aligned_start + (j - orig_start)
        return jnp.where(in_seg, src, gather_idx)

    gather_idx = jnp.zeros(T_out, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)
    return o[:, gather_idx]


# ============================================================================
# Main entry point
# ============================================================================


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "chunk_size",
        "intra_block_size",
        "scalar_intra_solve",
        "compute_block_chunks",
        "state_block_chunks",
        "state_dim_alignment",
        "single_chunk_state_elision",
        "zero_state_output_elision",
        "pipeline_impl",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "disable_recompute",
        "return_intermediate_states",
        "cp_context",
        "transpose_state_layout",
    ),
)
def chunk_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    cu_seqlens: jax.Array,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: None = None,
    transpose_state_layout: bool = False,
    intra_block_size: int = 16,
    state_block_chunks: int = 1,
    scalar_intra_solve: bool = False,
    compute_block_chunks: int = 1,
    state_dim_alignment: int = 128,
    single_chunk_state_elision: bool = False,
    zero_state_output_elision: bool = False,
    pipeline_impl: str = "incumbent",
):
    """KDA chunked forward pass for variable-length sequences (varlen).

    cu_seqlens must not be None. B must be 1 (packed layout).

    Four-stage pipeline:
      1. Gate activation + chunk-local cumsum
      2. Exact intra-chunk delta-rule triangular solve. ``scalar_intra_solve``
         selects one row per solve block; otherwise ``intra_block_size`` is used.
         ``compute_block_chunks`` groups independent logical chunks in this
         stage and the output stage under one physical Pallas program tile.
      3. Inter-chunk hidden state propagation via delta-rule recurrence
         using ``state_dim_alignment`` for its physical K/V state tile.
         ``single_chunk_state_elision`` skips this stage when its sole entrance
         state is known zero and its terminal state is not requested.
         ``zero_state_output_elision`` lets Stage 4 omit the corresponding
         zero inter-chunk term and its q/g/h loads.
      4. Output computation (inter-chunk state + intra-chunk attention)

    ``pipeline_impl`` selects which pipeline executes the recurrence.
    ``"incumbent"`` keeps the four-stage path documented above;
    ``"fused_resident"`` dispatches to the standalone
    ``resident_pipeline_kda_fwd`` handle, which runs stages 2-4 as one pass with
    the chunk state resident in VMEM (see kda/resident_pipeline.py).

    Returns:
        12-tuple: o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
    """
    # Imported here, not at module scope: resident_pipeline reuses this module's
    # solve/alignment helpers, so a top-level import would be circular.
    from sgl_jax.srt.kernels.kda.resident_pipeline import resident_pipeline_kda_fwd

    assert use_qk_l2norm_in_kernel is False
    assert cp_context is None
    assert not transpose_state_layout
    assert not return_intermediate_states
    assert not disable_recompute

    chunk_kda_pipeline_handle = (
        resident_pipeline_kda_fwd
        if not single_chunk_state_elision and pipeline_impl in ("incumbent", "fused_resident")
        else None
    )
    if chunk_kda_pipeline_handle is not None:
        return chunk_kda_pipeline_handle(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            chunk_size=chunk_size,
            intra_block_size=intra_block_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    STATE_BT = BT * state_block_chunks

    assert state_block_chunks >= 1, "state_block_chunks must be positive"
    assert compute_block_chunks >= 1, "compute_block_chunks must be positive"

    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert cu_seqlens is not None, "cu_seqlens must not be None for varlen path"
    assert B == 1, f"varlen requires B=1 (packed layout), got B={B}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")

    if single_chunk_state_elision:
        assert N == 1, "single-chunk state elision requires exactly one sequence"
        assert T == BT, (
            "single-chunk state elision requires chunk_size to equal the input length; "
            f"got T={T}, chunk_size={BT}"
        )
        assert initial_state is None, "single-chunk state elision requires initial_state=None"
        assert not output_final_state, "single-chunk state elision cannot return a final state"
    if zero_state_output_elision:
        assert (
            single_chunk_state_elision
        ), "zero-state output elision requires single_chunk_state_elision"

    # Varlen alignment
    _orig_cu_seqlens = cu_seqlens
    T_input = T
    # A grouped state block already spans this single packed sequence exactly
    # when its static extent is divisible by STATE_BT. Avoid materializing the
    # generic varlen helper's conservative trailing block in that case; the
    # default (one chunk per state block) retains the shipped alignment path.
    tight_single_chunk = single_chunk_state_elision and N == 1 and T == BT
    if not (tight_single_chunk or (state_block_chunks > 1 and N == 1 and T % STATE_BT == 0)):
        [q, k, v, g], [beta], cu_seqlens, _ = _align_seqs(
            [q, k, v, g],
            [beta],
            cu_seqlens,
            align=STATE_BT,
        )
    T = q.shape[1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T)

    assert T % BT == 0

    # Fix: _align_seqs pads g with 0, but softplus(0 + dt_bias) != 0 when
    # use_gate_in_kernel=True, producing non-zero gate activation at padding
    # positions.  This corrupts g_last (used for state propagation in Stage 3)
    # and kg (used for state update).  Set padding g to a large negative so
    # softplus(large_neg + dt_bias) ≈ 0, neutralising padding positions.
    if use_gate_in_kernel:
        orig_lens = _orig_cu_seqlens[1:] - _orig_cu_seqlens[:-1]
        aligned_starts = cu_seqlens[:-1]
        pos = jnp.arange(T)
        in_range = (pos[None, :] >= aligned_starts[:, None]) & (
            pos[None, :] < (aligned_starts + orig_lens)[:, None]
        )
        valid_mask = in_range.any(axis=0)  # [T]
        g = jnp.where(valid_mask[None, :, None, None], g, -1e4)

    # Step 1: Gate cumsum
    if use_gate_in_kernel:
        assert A_log is not None
        g_cumsum = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=BT,
            scale=_RCP_LN2,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        g_cumsum = pallas_kda_gate_cumsum(
            g=g,
            scale=_RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # Step 2: Intra-chunk solve
    w, u, qg, kg, Aqk, Akk = kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        scale=scale,
        safe_gate=safe_gate,
        chunk_size=BT,
        intra_block_size=intra_block_size,
        scalar_intra_solve=scalar_intra_solve,
        compute_block_chunks=compute_block_chunks,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # Step 3: Inter-chunk state propagation
    if single_chunk_state_elision:
        # Stage 3 stores the state at each chunk entrance for Stage 4.  With one
        # chunk and no initial state that entrance is exactly zero.  Its other
        # consumed output is v_new = u - w @ h, which therefore equals u.  The
        # state update after this chunk is terminal and unobserved when no final
        # state is requested, so the complete Stage-3 launch can be elided.
        h = None if zero_state_output_elision else jnp.zeros((B, 1, H, K, V), dtype=jnp.float32)
        v_new = u.astype(jnp.float32)
        final_state = None
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT,
            state_block_chunks=state_block_chunks,
            state_dim_alignment=state_dim_alignment,
            use_exp2=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # Step 4: Output computation
    o = chunk_kda_fwd_o_gk(
        q=q,
        v=v_new,
        g=g_cumsum,
        A=Aqk,
        h=h,
        scale=scale,
        chunk_size=BT,
        compute_block_chunks=compute_block_chunks,
        use_exp2=True,
        zero_state_output=zero_state_output_elision,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # Cast output back to input dtype (e.g. bfloat16)
    o = o.astype(q.dtype)

    # Unalign output
    o = _unalign_output(o, _orig_cu_seqlens, cu_seqlens, T_input)

    # Release intermediates
    w, u, qg, kg, v_new, h = None, None, None, None, None, None
    if use_gate_in_kernel:
        g_cumsum = None

    return o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
