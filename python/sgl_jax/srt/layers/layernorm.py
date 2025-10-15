from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh

from ..utils.weight_utils import lazy_init


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
            nnx.with_partitioning(lazy_init, kernel_axes)(
                rngs.params(), (hidden_size,), jnp.bfloat16
            )
        )

    def __call__(
        self, x: jax.Array, residual: Optional[jax.Array] = None
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Applies layer normalization on the input."""
        return rmsnorm_forward(x, residual, self.weight, self.variance_epsilon)

    def materialize(self, mesh: Mesh, rngs: nnx.Rngs):
        """Materialize and shard RMSNorm parameters in-place."""
        pspecs = nnx.get_partition_spec(nnx.state(self))

        real_weight_val = nnx.initializers.ones()(
            rngs.params(), (self.weight.value.shape[0],), self.weight.value.dtype
        )

        with mesh:
            sharded_weight = jax.lax.with_sharding_constraint(
                real_weight_val, pspecs["weight"]
            )
            nnx.update(self, {"weight": sharded_weight})


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
