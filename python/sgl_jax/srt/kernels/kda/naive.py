import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


def naive_recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """
    Core recurrence (per timestep):
        S' = S_{t-1} * exp(g_t)                            decay
        residual = v_t - k_t^T @ S'                        prediction error
        S_t = S' + beta_t * k_t ⊗ residual                 delta update
        o_t = (q_t * scale)^T @ S_t                        output

    Dtype behavior (matching FLA):
      - All inputs cast to fp32 for computation
      - Hidden state S is fp32 accumulator
      - Output o computed in fp32, cast back to original dtype
      - Final state S stays in fp32
      - fp64 mode: all computation in fp64, no precision cast

    Args:
        q:               [B, T, H, K] — Queries
        k:               [B, T, H, K] — Keys
        v:               [B, T, H, V] — Values
        g:               [B, T, H, K] — Per-element gate in log-space (e.g., -exp(A)*softplus(g))
        beta:            [B, T, H]    — Learning rate / step size for delta rule
        scale:           Scalar query scale. Defaults to K ** -0.5.
        initial_state:   [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return the final hidden state.

    Returns:
        o:           [B, T, H, V] — Output (original input dtype)
        final_state: [B, H, K, V] in fp32 (or fp64), or None
    """
    orig_dtype, acc_dt = v.dtype, acc_dtype(q.dtype)

    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert (
        v.ndim == 4 and v.shape[:3] == q.shape[:3]
    ), f"v shape {v.shape} incompatible with q shape {q.shape}"
    assert g.ndim == 4 and g.shape == q.shape, f"g shape {g.shape} != q shape {q.shape}"
    assert beta.ndim == 3 and beta.shape == q.shape[:3], f"beta shape {beta.shape} != {q.shape[:3]}"

    B, T, H, K, V = *q.shape, v.shape[-1]

    if initial_state is not None:
        assert initial_state.shape == (
            B,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"

    if scale is None:
        scale = K**-0.5

    # [B, T, H, K] -> [B, H, T, K], cast to acc_dt
    q, k, v, g = (jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt) for x in (q, k, v, g))
    # q: [B, H, T, K]   k: [B, H, T, K]   v: [B, H, T, V]   g: [B, H, T, K]

    # [B, T, H] -> [B, H, T]
    beta = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt)  # [B, H, T]

    q = q * scale  # [B, H, T, K]

    S = jnp.zeros((B, H, K, V), dtype=acc_dt)  # [B, H, K, V] hidden state
    if initial_state is not None:
        S += initial_state.astype(acc_dt)  # [B, H, K, V]
    o = jnp.zeros((B, H, T, V), dtype=acc_dt)  # [B, H, T, V] output buffer

    for i in range(T):
        q_i = q[:, :, i]  # [B, H, K]
        k_i = k[:, :, i]  # [B, H, K]
        v_i = v[:, :, i]  # [B, H, V]
        g_i = g[:, :, i]  # [B, H, K]
        b_i = beta[:, :, i]  # [B, H]

        # 1. Decay the state
        # exp(g_i): [B, H, K] -> [B, H, K, 1] via broadcast
        S = S * jnp.exp(g_i)[..., None]  # [B, H, K, V]

        # 2. Delta rule update
        # k_i[..., None]: [B, H, K, 1],  k_i[..., None] * S: [B, H, K, V]
        v_predicted = (k_i[..., None] * S).sum(-2)  # [B, H, V]
        residual = v_i - v_predicted  # [B, H, V]

        # b_i[..., None] * k_i: [B, H, K],  einsum -> [B, H, K, V]
        S = S + jnp.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, residual)  # [B, H, K, V]

        # 3. Compute output: einsum [B,H,K] x [B,H,K,V] -> [B, H, V]
        o = o.at[:, :, i].set(
            jnp.einsum("bhk,bhkv->bhv", q_i, S), out_sharding=P(None, "tensor", None, None)
        )  # [B, H, V]

    final_state = S if output_final_state else None  # [B, H, K, V] or None
    # [B, H, T, V] -> [B, T, H, V], cast back to orig_dtype
    return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state
