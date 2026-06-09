"""Optimized Residual-Norm-Router (RNR) Pallas kernel for TPU.

Fuses the Attention Residual Add, the RMSNorm (post-attention layernorm),
and the MoE Router gating projection (GateLogit) into a single pipelined TPU kernel,
completely eliminating multiple intermediate HBM round-trips.
"""

from __future__ import annotations

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def apply_fused_rnr(
    attn_out: jax.Array,
    res: jax.Array,
    gamma: jax.Array,
    w_gate: jax.Array,
    b_seq: int = 128,
    hidden_size: int = 4096,
    num_experts: int = 256,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    seq_len, _ = attn_out.shape
    grid = (seq_len // b_seq,)

    # Output shapes and dtypes: [Y, Y_norm, logits]
    out_shapes = (
        jax.ShapeDtypeStruct((seq_len, hidden_size), attn_out.dtype),  # Y
        jax.ShapeDtypeStruct((seq_len, hidden_size), attn_out.dtype),  # Y_norm
        jax.ShapeDtypeStruct((seq_len, num_experts), attn_out.dtype),  # logits
    )

    in_specs = (
        pl.BlockSpec((b_seq, hidden_size), lambda seq_idx: (seq_idx, 0)),  # attn_out
        pl.BlockSpec((b_seq, hidden_size), lambda seq_idx: (seq_idx, 0)),  # res
        pl.BlockSpec((hidden_size,), lambda seq_idx: (0,)),               # gamma
        pl.BlockSpec((hidden_size, num_experts), lambda seq_idx: (0, 0)),  # w_gate
    )

    out_specs = (
        pl.BlockSpec((b_seq, hidden_size), lambda seq_idx: (seq_idx, 0)),  # Y
        pl.BlockSpec((b_seq, hidden_size), lambda seq_idx: (seq_idx, 0)),  # Y_norm
        pl.BlockSpec((b_seq, num_experts), lambda seq_idx: (seq_idx, 0)),  # logits
    )

    def rnr_kernel_fn(
        attn_out_ref,
        res_ref,
        gamma_ref,
        w_gate_ref,
        y_ref,
        y_norm_ref,
        logits_ref,
    ):
        attn_out = attn_out_ref[...]
        res = res_ref[...]
        gamma = gamma_ref[...]
        w_gate = w_gate_ref[...]

        # 1. Residual Add (attn_out + res)
        y = attn_out + res
        y_ref[...] = y

        # 2. RMSNorm (Y / sqrt(mean(Y^2) + eps) * gamma)
        y_f32 = y.astype(jnp.float32)
        var = jnp.mean(jnp.square(y_f32), axis=-1, keepdims=True)
        rsqrt = jax.lax.rsqrt(var + eps)
        gamma_f32 = gamma.astype(jnp.float32)
        y_norm = (y_f32 * rsqrt * gamma_f32).astype(attn_out.dtype)
        y_norm_ref[...] = y_norm

        # 3. Gating Projection + Sigmoid (Y_norm @ W_gate)
        logits = jnp.matmul(y_norm, w_gate, preferred_element_type=jnp.float32)
        logits = jax.nn.sigmoid(logits).astype(attn_out.dtype)
        logits_ref[...] = logits

    return pl.pallas_call(
        rnr_kernel_fn,
        out_shape=out_shapes,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(attn_out, res, gamma, w_gate)


def apply_fused_rnr_with_padding(
    attn_out: jax.Array,
    res: jax.Array,
    gamma: jax.Array,
    w_gate: jax.Array,
    b_seq: int = 128,
    hidden_size: int = 4096,
    num_experts: int = 256,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pads the input sequence length to be a multiple of b_seq if necessary."""
    seq_len, _ = attn_out.shape
    rem = seq_len % b_seq
    if rem == 0:
        return apply_fused_rnr(attn_out, res, gamma, w_gate, b_seq, hidden_size, num_experts, eps)

    pad_len = b_seq - rem
    attn_out_padded = jnp.pad(attn_out, ((0, pad_len), (0, 0)), mode="constant")
    res_padded = jnp.pad(res, ((0, pad_len), (0, 0)), mode="constant")
    
    y_padded, y_norm_padded, logits_padded = apply_fused_rnr(
        attn_out_padded, res_padded, gamma, w_gate, b_seq, hidden_size, num_experts, eps
    )
    
    return y_padded[:seq_len, :], y_norm_padded[:seq_len, :], logits_padded[:seq_len, :]
