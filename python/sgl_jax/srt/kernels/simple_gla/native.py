"""Native JAX reference implementation of GLA (Gated Linear Attention).

This module provides pure JAX implementations of GLA decode and prefill
operations. These implementations use jnp.einsum and jax.lax.scan without
Pallas kernels, matching the same dtype as the kernel under test.
"""

import jax
import jax.numpy as jnp


def naive_gla_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    h0: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Naive GLA decode using jnp.einsum.

    Args:
        q: Query tensor [B, 1, H, K]
        k: Key tensor [B, 1, H, K]
        v: Value tensor [B, 1, H, K]
        g_gamma: Gate decay per head [H], negative values (e.g., ALiBi slopes)
        h0: Initial state [B, H, K, K]
        scale: Optional output scaling factor

    Returns:
        output: [B, 1, H, K]
        h1: Updated state [B, H, K, K]
    """
    B, T, H, K = q.shape
    assert T == 1, f"Decode expects T=1, got {T}"

    q_t = q[:, 0].astype(jnp.float32)
    k_t = k[:, 0].astype(jnp.float32)
    v_t = v[:, 0].astype(jnp.float32)
    g_gamma = g_gamma.astype(jnp.float32)
    h0 = h0.astype(jnp.float32)

    decay = jnp.exp(g_gamma)[None, :, None, None]
    kv = jnp.einsum("bhk,bhv->bhkv", k_t, v_t)
    h1 = decay * h0 + kv
    o = jnp.einsum("bhk,bhkv->bhv", q_t, h1)

    if scale is not None:
        o = o * scale

    output = o[:, None, :, :]

    return output, h1


def naive_gla_prefill(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    h0: jax.Array,
    cu_seqlens: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Naive GLA prefill using per-request scan + jnp.einsum.

    Args:
        q: Query tensor [1, T_total, H, K] (varlen packed)
        k: Key tensor [1, T_total, H, K] (varlen packed)
        v: Value tensor [1, T_total, H, K] (varlen packed)
        g_gamma: Gate decay per head [H], negative values
        h0: Initial state per request [B, H, K, K]
        cu_seqlens: Cumulative sequence lengths [B+1], e.g., [0, 128, 384] for 2 requests
        scale: Optional output scaling factor

    Returns:
        output: [1, T_total, H, K]
        h_final: Final state per request [B, H, K, K]
    """
    assert q.shape[0] == 1, f"Prefill expects batch=1 (varlen), got {q.shape[0]}"

    q = q[0].astype(jnp.float32)
    k = k[0].astype(jnp.float32)
    v = v[0].astype(jnp.float32)
    g_gamma = g_gamma.astype(jnp.float32)
    h0 = h0.astype(jnp.float32)

    T = q.shape[0]
    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    reset_mask = token_idx == cu_seqlens[:-1][seq_ids]
    decay = jnp.exp(g_gamma)

    def scan_fn(carry, inputs):
        h_prev, final_states = carry
        seq_id, do_reset, q_t, k_t, v_t = inputs
        h = jnp.where(do_reset, h0[seq_id], h_prev)
        kv = jnp.einsum("hk,hv->hkv", k_t, v_t)
        h = decay[:, None, None] * h + kv
        o_t = jnp.einsum("hk,hkv->hv", q_t, h)
        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_t

    init_carry = (
        jnp.zeros_like(h0[0]),
        h0,
    )
    (_, h_final), output = jax.lax.scan(
        scan_fn,
        init_carry,
        (seq_ids, reset_mask, q, k, v),
    )

    if scale is not None:
        output = output * scale

    return output[None, :, :, :], h_final


__all__ = ["naive_gla_decode", "naive_gla_prefill"]
