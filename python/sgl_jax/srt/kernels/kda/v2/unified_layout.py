"""Layout transforms used by the fused KDA v2 kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def to_unified_layout(x: jax.Array) -> jax.Array:
    """Convert [B, T, H, D] to [B, H, T, D]."""
    return jnp.transpose(x, (0, 2, 1, 3))


def from_unified_layout(x: jax.Array) -> jax.Array:
    """Convert [B, H, T, D] to [B, T, H, D]."""
    return jnp.transpose(x, (0, 2, 1, 3))


def prepare_intra_layout(q, k, v, gk, beta, chunk_size, unified_layout, cu_seqlens):
    """Prepare K1 inputs and return the metadata needed to restore outputs."""
    BT = chunk_size
    if unified_layout:
        B, H, T_u, K = q.shape
        V = v.shape[-1]
        assert B == 1 and T_u % BT == 0
        NC = T_u // BT
        return (q, k, gk, beta, v), (B, H, K, V, NC, None)

    B, T, H, K = q.shape
    V = v.shape[-1]
    assert B == 1
    assert cu_seqlens is not None, "unified_layout=False needs cu_seqlens for gather"
    N = cu_seqlens.shape[0] - 1
    T_alloc = T + BT

    pad4d = lambda x: jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
    q_pad, k_pad, g_pad, v_pad = pad4d(q), pad4d(k), pad4d(gk), pad4d(v)
    beta_pad = jnp.pad(beta.reshape(B, T, H, 1), ((0, 0), (0, BT), (0, 0), (0, 0)))

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunks_per_seq = (jnp.diff(cu_i32) + BT - 1) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    total_chunks = cum_chunks[-1]
    NC = T // BT + N
    flat_idx = jnp.arange(NC, dtype=jnp.int32)
    is_valid = flat_idx < total_chunks
    seq_id = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1)
    local_ci = flat_idx - cum_chunks[seq_id]
    chunk_starts = jnp.where(is_valid, cu_i32[seq_id] + local_ci * BT, 0)

    def gather(x_pad, dim):
        def extract(start):
            return jax.lax.dynamic_slice(x_pad, (0, start, 0, 0), (1, BT, H, dim))[0]

        return jax.vmap(extract)(chunk_starts)

    def to_kernel_layout(x_chunks):
        return x_chunks.transpose(2, 0, 1, 3).reshape(1, H, NC * BT, x_chunks.shape[3])

    inputs = (
        to_kernel_layout(gather(q_pad, K)),
        to_kernel_layout(gather(k_pad, K)),
        to_kernel_layout(gather(g_pad, K)),
        to_kernel_layout(gather(beta_pad, 1)),
        to_kernel_layout(gather(v_pad, V)),
    )
    restore = (T, T_alloc, chunk_starts, is_valid)
    return inputs, (B, H, K, V, NC, restore)


def restore_intra_layout(outputs, *, chunk_size, layout_metadata):
    """Restore K1 outputs to [B, T, H, D] for the legacy addressing path."""
    B, H, K, V, NC, restore = layout_metadata
    if restore is None:
        return outputs

    BT = chunk_size
    T, T_alloc, chunk_starts, is_valid = restore
    w4, u4, qg4, kg4, Aqk4, Akk4, g_cum4 = outputs
    pos = chunk_starts[:, None] + jnp.arange(BT)[None, :]
    pos = jnp.where(is_valid[:, None], pos, T_alloc - 1)
    flat_pos = pos.reshape(-1)

    def scatter(x4, dim):
        chunks = x4.reshape(H, NC, BT, dim).transpose(1, 2, 0, 3).reshape(-1, H, dim)
        out = jnp.zeros((T_alloc, H, dim), dtype=x4.dtype)
        out = out.at[flat_pos].add(chunks)
        return out[:T][None]

    w_out = scatter(w4, K)
    u_out = scatter(u4, V)
    qg_out = scatter(qg4, K) if qg4 is not None else None
    kg_out = scatter(kg4, K)
    Aqk_out = scatter(Aqk4, BT)
    Akk_out = scatter(Akk4, BT)
    g_cum_out = scatter(g_cum4, K)
    return w_out, u_out, qg_out, kg_out, Aqk_out, Akk_out, g_cum_out
