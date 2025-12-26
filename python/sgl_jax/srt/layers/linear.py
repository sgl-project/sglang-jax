from collections.abc import Iterable, Sequence

import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def _normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class LinearBase(nnx.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        use_bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        partition_spec: Partition spec for the linear layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        mesh: jax.sharding.Mesh,
        use_bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        kernel_axes: Sequence[str | None] | None = None,
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes
        self.mesh = mesh
        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (input_size, output_size),
                dtype=params_dtype,
                out_sharding=P(*kernel_axes),
            ),
        )
        if use_bias:
            self.bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (output_size,),
                    dtype=params_dtype,
                    out_sharding=P(
                        kernel_axes[-1],
                    ),
                ),
            )
        else:
            self.bias = None

    def __call__(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass of the linear layer."""
        bias = self.bias if not self.skip_bias_add else None
        output_pspec = P(*([None] * (x.ndim - 1)), self.kernel_axes[-1])
        output_sharding = NamedSharding(self.mesh, output_pspec)
        output = lax.dot_general(
            x,
            self.weight.value,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=self.params_dtype,
            out_sharding=output_sharding,
        )
        if bias is not None:
            output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
