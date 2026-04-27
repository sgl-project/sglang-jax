"""Gated RMS normalization for linear attention layers.

Computes ``RMSNorm(x) * sigmoid(gate)`` — used by KDA (Kimi Delta Attention)
as the output normalization before the final projection.

GPU reference: ``sglang/srt/layers/attention/fla/fused_norm_gate.py``
(``FusedRMSNormGated`` with ``activation="sigmoid"``).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype


class GatedRMSNorm(nnx.Module):
    """RMSNorm with a multiplicative sigmoid gate.

    Given input ``x`` and ``gate`` of the same shape, computes::

        output = (x / sqrt(mean(x^2) + eps)) * weight * sigmoid(gate)
    """

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-6,
        param_dtype: Dtype = jnp.float32,
    ):
        self.weight = nnx.Param(jnp.ones((num_features,), dtype=param_dtype))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.epsilon)
        x_norm = x_norm * self.weight[...].astype(jnp.float32)
        return (x_norm * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(orig_dtype)
