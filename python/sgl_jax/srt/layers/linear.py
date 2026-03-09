# SPDX-License-Identifier: Apache-2.0
"""Linear layers."""

import math
from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


class LinearBase(nnx.Module):
    """Base class for all linear layers."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        mesh: jax.sharding.Mesh | None = None,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        kernel_axes: Sequence[str | None] | None = None,
        scope_name: str = "linear_base",
        use_weight_scale: bool = False,
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes
        self.mesh = mesh
        self.name = scope_name

        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (input_size, output_size),
                dtype=params_dtype,
                out_sharding=P(*kernel_axes),
            ),
        )
        if use_weight_scale:
            self.weight_scale = nnx.Param(jnp.ones((output_size,), dtype=jnp.float32))
        if use_bias:
            self.bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (output_size,),
                    dtype=params_dtype,
                    out_sharding=P(kernel_axes[1]),
                ),
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass."""
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        if self.mesh is not None and self.kernel_axes is not None:
            input_axis, output_axis = self.kernel_axes[0], self.kernel_axes[1]

            def _sharded_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
                y = jnp.dot(lhs, rhs)
                if input_axis is not None:
                    y = jax.lax.psum(y, input_axis)
                return y

            out = shard_map(
                _sharded_dot,
                mesh=self.mesh,
                in_specs=(P(None, input_axis), P(input_axis, output_axis)),
                out_specs=P(None, output_axis),
                check_vma=False,
            )(x_2d, self.weight.value)
        else:
            out = jnp.dot(x_2d, self.weight.value)

        if x.ndim > 2:
            out = out.reshape(x.shape[:-1] + (out.shape[-1],))

        if self.skip_bias_add:
            return out, (self.bias.value if self.bias is not None else None)
        if self.bias is not None:
            out = out + self.bias.value
        return out, None


class QuantizedLinear(nnx.Module):
    """Linear layer with pre-quantized weights."""

    def __init__(
        self,
        weight_q: jax.Array,
        weight_scale: jax.Array,
        bias: jax.Array | None,
        activation_dtype: jnp.dtype | None,
        mesh: jax.sharding.Mesh,
        kernel_axes: Sequence[str | None] | None = None,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        compute_dtype: jnp.dtype | None = None,
        weight_block_size: tuple[int, int] | None = None,
        scope_name: str = "quantized_linear",
    ):
        """Initialize the quantized linear layer with pre-quantized weights."""
        self.weight_q = nnx.Param(weight_q)
        self.weight_scale = nnx.Param(weight_scale)
        self.bias = nnx.Param(bias) if bias is not None else None
        self.activation_dtype = activation_dtype
        self.mesh = mesh
        self.kernel_axes = kernel_axes
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.compute_dtype = compute_dtype
        self.weight_block_size = weight_block_size
        self.name = scope_name

    @classmethod
    def from_linear(
        cls,
        linear: LinearBase,
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
        is_static_input: bool = False,
        weight_block_size: Sequence[int] | None = None,
    ) -> "QuantizedLinear":
        """Convert a LinearBase layer to a QuantizedLinear layer."""
        effective_weight_block_size = (
            tuple(weight_block_size) if weight_block_size is not None else None
        )

        if is_static_input:
            # Static checkpoint already stores pre-quantized weights and scales.
            weight = linear.weight.value

            if isinstance(weight, jax.ShapeDtypeStruct):
                in_features, out_features = map(int, weight.shape)
                kernel_axes = linear.kernel_axes or (None, None)
                wq_sharding = NamedSharding(linear.mesh, P(kernel_axes[1], kernel_axes[0]))
                weight_q = jax.ShapeDtypeStruct(
                    shape=(out_features, in_features),
                    dtype=weight_dtype,
                    sharding=wq_sharding,
                )

                if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                    block_n, block_k = int(effective_weight_block_size[0]), int(effective_weight_block_size[1])
                    out_blocks = (out_features + block_n - 1) // block_n
                    in_blocks = (in_features + block_k - 1) // block_k
                    scale_sharding = NamedSharding(linear.mesh, P(kernel_axes[1], kernel_axes[0]))
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(out_blocks, in_blocks), dtype=jnp.float32, sharding=scale_sharding
                    )
                else:
                    scale_sharding = NamedSharding(linear.mesh, P(kernel_axes[1]))
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(out_features,), dtype=jnp.float32, sharding=scale_sharding
                    )
                bias = linear.bias.value if linear.bias is not None else None
            else:
                weight_q = weight.T.astype(weight_dtype)
                if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                    block_n, block_k = int(effective_weight_block_size[0]), int(effective_weight_block_size[1])
                    out_blocks = (weight_q.shape[0] + block_n - 1) // block_n
                    in_blocks = (weight_q.shape[1] + block_k - 1) // block_k
                    weight_scale = jnp.ones((out_blocks, in_blocks), dtype=jnp.float32)
                else:
                    weight_scale = jnp.ones((weight_q.shape[0],), dtype=jnp.float32)
                bias = linear.bias.value if linear.bias is not None else None
        else:
            weight = linear.weight.value
            weight_t = weight.T

            if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype, tensor=weight_t, axis=(0, 1),
                    block_size=tuple(effective_weight_block_size), pad_tensor=True,
                )
            else:
                weight_q, weight_scale = quantize_tensor(dtype=weight_dtype, tensor=weight_t, axis=1)

            bias = linear.bias.value if linear.bias is not None else None

        return cls(
            weight_q=weight_q, weight_scale=weight_scale, bias=bias,
            activation_dtype=activation_dtype, mesh=linear.mesh,
            kernel_axes=linear.kernel_axes,
            skip_bias_add=linear.skip_bias_add or linear.bias is None,
            params_dtype=linear.params_dtype, weight_block_size=effective_weight_block_size,
            scope_name=f"quantized_{linear.name}",
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass with quantization."""
        quantize_activation = self.activation_dtype is not None
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        scale_val = self.weight_scale.value
        if scale_val.ndim == 2 and scale_val.shape[1] == 1 and scale_val.shape[0] == self.weight_q.value.shape[0]:
            scale_val = jnp.squeeze(scale_val, axis=1)

        input_axis, output_axis = self.kernel_axes[0], self.kernel_axes[1]
        w_scale_spec = P(output_axis) if scale_val.ndim == 1 else P(output_axis, input_axis)

        in_specs = (P(None, input_axis), P(output_axis, input_axis), w_scale_spec)
        out_specs = P(None, output_axis)

        # Handle block size inference
        effective_weight_block_size = self.weight_block_size
        if scale_val.ndim == 2 and self.weight_block_size is not None:
            global_out_size, global_in_size = self.weight_q.value.shape
            inferred_bs_out = math.ceil(global_out_size / scale_val.shape[0])
            inferred_bs_in = math.ceil(global_in_size / scale_val.shape[1])
            if (inferred_bs_out != self.weight_block_size[0] or inferred_bs_in != self.weight_block_size[1]):
                effective_weight_block_size = (inferred_bs_out, inferred_bs_in)

        output = shard_map(
            partial(
                xla_quantized_matmul_local,
                quantize_activation=quantize_activation,
                reduce_axis=input_axis,
                compute_dtype=self.compute_dtype,
                weight_block_size=effective_weight_block_size,
                activation_quant_dtype=self.activation_dtype,
            ),
            mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False,
        )(x_2d, self.weight_q.value, scale_val)

        if x.ndim > 2:
            output = output.reshape(x.shape[:-1] + (output.shape[-1],))

        if self.skip_bias_add:
            return output, (self.bias.value if self.bias is not None else None)
        if self.bias is not None:
            output = output + self.bias.value
        return output, None
