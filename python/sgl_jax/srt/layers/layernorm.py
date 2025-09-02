from functools import partial
from typing import Optional, Sequence, Tuple, Union
from venv import logger

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
        rngs: nnx.Rngs = None,
    ):
        self.variance_epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.ones, kernel_axes)(
                rngs.params(), (hidden_size,)
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
    logger.info("rmsnorm input 1 ")
    orig_dtype = x.dtype
    logger.info("rmsnorm input 2 ")

    x_f32 = jnp.asarray(x, jnp.float32)
    logger.info("rmsnorm input 3 ")

    if residual is not None:
        x_f32 += jnp.asarray(residual, jnp.float32)
        logger.info("rmsnorm input 4 ")

        residual = x_f32.astype(orig_dtype)
        logger.info("rmsnorm input 5 ")
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    logger.info("rmsnorm input 6 ")

    y = jnp.asarray(x_f32 * lax.rsqrt(mean2 + epsilon), jnp.float32)
    logger.info("rmsnorm input 7 ")
    weight = jnp.asarray(weight, jnp.float32)
    logger.info("rmsnorm input 8 ")
    out = y * weight
    logger.info("rmsnorm input 9 ")
    output = out
    logger.info("rmsnorm input 10 ")
    if residual is None:
        return output
    else:
        return output, residual
