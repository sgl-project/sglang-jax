from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNorm(nnx.Module):
    """RMS normalization."""

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        kernel_axes: Optional[Sequence[str]] = None,
        params_dtype: Optional[jnp.dtype] = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.variance_epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.ones, kernel_axes)(
                rngs.params(), (hidden_size,), params_dtype
            )
        )

    def __call__(
        self, x: jax.Array, residual: Optional[jax.Array] = None
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Applies layer normalization on the input."""
        return rmsnorm_forward(x, residual, self.weight, self.variance_epsilon)


# @partial(jax.jit, static_argnames=["epsilon"])
def rmsnorm_forward(
    x, residual, weight, epsilon
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    orig_dtype = x.dtype
    x_f32 = jnp.asarray(x, jnp.float32)
    if residual is not None:
        x_f32 += jnp.asarray(residual, jnp.float32)
        residual = x_f32.astype(orig_dtype)
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    y = jnp.asarray(x_f32 * lax.rsqrt(mean2 + epsilon), jnp.float32)
    output = (y * jnp.asarray(weight, jnp.float32)).astype(orig_dtype)
    if residual is None:
        return output
    else:
        return output, residual


def dual_rmsnorm_forward(
    x: jax.Array,
    residual: jax.Array,
    weight1: jax.Array,
    weight2: jax.Array,
    epsilon: float,
) -> Tuple[jax.Array, jax.Array]:
    """Apply two RMSNorms with shared residual path, returning (y2, residual).

    Equivalent to fused_dual_residual_rmsnorm: first adds residual, applies
    norm with weight1 to produce y1 (discarded), then norm with weight2 to produce y2.
    """
    y1, residual_out = rmsnorm_forward(x, residual, weight1, epsilon)
    y2, _ = rmsnorm_forward(x, residual, weight2, epsilon)
    return y2, residual_out
