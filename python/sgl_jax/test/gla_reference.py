"""Naive JAX reference implementation of GLA (Gated Linear Attention).

This module provides pure JAX implementations of GLA decode and prefill operations
for use as golden reference in tests. These implementations use jnp.einsum and
jax.lax.scan without Pallas kernels, matching the same dtype as the kernel under test.

Reference: RFC #1032 - Backend end-to-end validation uses "JAX naive jit implementation
(same dtype as kernel) using per-request scan + jnp.einsum, no Pallas."
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

    # Squeeze time dimension
    q_t = q[:, 0]  # [B, H, K]
    k_t = k[:, 0]  # [B, H, K]
    v_t = v[:, 0]  # [B, H, K]

    # Update state first: h1 = exp(g_gamma) * h0 + k^T @ v
    # g_gamma: [H] -> decay: [1, H, 1, 1]
    decay = jnp.exp(g_gamma)[None, :, None, None]
    # k: [B, H, K], v: [B, H, K] -> kv: [B, H, K, K]
    kv = jnp.einsum("bhk,bhv->bhkv", k_t, v_t)
    h1 = decay * h0 + kv

    # Compute output using updated state: o = q @ h1
    # q: [B, H, K], h1: [B, H, K, K] -> o: [B, H, K]
    o = jnp.einsum("bhk,bhkv->bhv", q_t, h1)

    if scale is not None:
        o = o * scale

    # Restore time dimension
    output = o[:, None, :, :]  # [B, 1, H, K]

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

    # Squeeze batch dimension
    q = q[0]  # [T_total, H, K]
    k = k[0]  # [T_total, H, K]
    v = v[0]  # [T_total, H, K]

    B = len(cu_seqlens) - 1
    decay = jnp.exp(g_gamma)  # [H]

    output_list = []
    h_final_list = []

    for i in range(B):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]

        q_seq = q[start:end]  # [T, H, K]
        k_seq = k[start:end]  # [T, H, K]
        v_seq = v[start:end]  # [T, H, K]
        h_init = h0[i]  # [H, K, K]

        # Scan over time steps
        def scan_fn(h_prev, inputs):
            q_t, k_t, v_t = inputs
            # q_t, k_t, v_t: [H, K]
            # h_prev: [H, K, K]

            # Update state first: h_next = decay * h + k^T @ v
            kv = jnp.einsum("hk,hv->hkv", k_t, v_t)  # [H, K, K]
            h_next = decay[:, None, None] * h_prev + kv  # [H, K, K]

            # Compute output using updated state: o = q @ h_next
            o = jnp.einsum("hk,hkv->hv", q_t, h_next)  # [H, K]

            return h_next, o

        h_final, o_seq = jax.lax.scan(
            scan_fn,
            h_init,
            (q_seq, k_seq, v_seq),
        )
        # o_seq: [T, H, K]
        # h_final: [H, K, K]

        if scale is not None:
            o_seq = o_seq * scale

        output_list.append(o_seq)
        h_final_list.append(h_final)

    # Concatenate outputs
    output = jnp.concatenate(output_list, axis=0)  # [T_total, H, K]
    output = output[None, :, :, :]  # [1, T_total, H, K]

    # Stack final states
    h_final = jnp.stack(h_final_list, axis=0)  # [B, H, K, K]

    return output, h_final


__all__ = ["naive_gla_decode", "naive_gla_prefill"]
