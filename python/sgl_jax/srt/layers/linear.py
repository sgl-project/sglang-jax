from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ..utils.weight_utils import lazy_init


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class LinearBase(nnx.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        partition_spec: Partition spec for the linear layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[jnp.dtype] = jnp.bfloat16,
        kernel_axes: Optional[Sequence[str]] = None,
        rngs: nnx.Rngs = None,
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.weight = nnx.Param(
            nnx.with_partitioning(lazy_init, self.kernel_axes)(
                rngs.params(), (self.input_size, self.output_size), self.params_dtype
            )
        )
        if self.use_bias:
            bias_axes = (self.kernel_axes[-1],) if self.kernel_axes else ()
            self.bias = nnx.Param(
                nnx.with_partitioning(lazy_init, bias_axes)(
                    rngs.params(), (self.output_size,), self.params_dtype
                )
            )
        else:
            self.bias = None

    def materialize(self, mesh: Mesh, rngs: nnx.Rngs):
        """
        Materializes and shards the model's parameters in-place.
        This method contains the full logic and is designed to be JIT-compiled.
        """
        # Create the real parameter values on a single device first
        real_weight_val = nnx.initializers.normal()(
            rngs.params(), (self.input_size, self.output_size), self.params_dtype
        )

        # Apply sharding constraints within the mesh context
        with mesh:
            weight_pspec = P(*self.kernel_axes) if self.kernel_axes else P()
            sharded_weight = jax.lax.with_sharding_constraint(
                real_weight_val, NamedSharding(mesh, weight_pspec)
            )
            updates = {"weight": sharded_weight}

            if self.use_bias:
                real_bias_val = nnx.initializers.zeros_init()(
                    rngs.params(), (self.output_size,), self.params_dtype
                )
                bias_axes = (self.kernel_axes[-1],) if self.kernel_axes else ()
                bias_pspec = P(*bias_axes) if bias_axes else P()
                sharded_bias = jax.lax.with_sharding_constraint(
                    real_bias_val, NamedSharding(mesh, bias_pspec)
                )
                updates["bias"] = sharded_bias

        # Update the model's state with the new sharded parameters
        nnx.update(self, updates)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        """Forward pass of the linear layer."""
        bias = self.bias if not self.skip_bias_add else None
        # Access the underlying JAX array using .value property
        output = jnp.dot(x, self.weight.value)
        if bias is not None:
            output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
