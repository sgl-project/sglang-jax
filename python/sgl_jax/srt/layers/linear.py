import logging
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
from flax import nnx
from jax import numpy as jnp

logger = logging.getLogger(__name__)


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
            nnx.with_partitioning(nnx.initializers.normal(), kernel_axes)(
                rngs.params(), (input_size, output_size), params_dtype
            )
        )
        if use_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(
                    nnx.initializers.zeros_init(), (kernel_axes[-1],)
                )(rngs.params(), (output_size,), params_dtype)
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        """Forward pass of the linear layer."""
        bias = self.bias if not self.skip_bias_add else None
        # Access the underlying JAX array using .value property
        logger.info("linear input 1 ")
        jax.debug.visualize_array_sharding(self.weight.value)

        # output = jnp.dot(
        #     jnp.arange(32 * 4096, dtype=jnp.bfloat16).reshape(32, 4096),
        #     self.weight.value,
        # )

        logger.info("linear input 2 ")

        # if bias is not None:
        #     output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None
        return jnp.arange(32 * 4096, dtype=jnp.bfloat16).reshape(32, 4096), output_bias
