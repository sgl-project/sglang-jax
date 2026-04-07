from collections.abc import Sequence
from functools import partial

import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import should_use_blockwise_kernel
from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


class LinearBase(nnx.Module):
    """Base linear layer.

    Args:
        input_size: Input dimension of the linear layer.
        output_size: Output dimension of the linear layer.
        mesh: Device mesh for sharding.
        use_bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters and accumulation preference.
        kernel_axes: Partition spec for the weight tensor.
        scope_name: Name used for profiling scope.
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
        scope_name: str = "linear_base",
        force_dequant: bool = False,
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes
        self.mesh = mesh
        self.name = scope_name
        self.force_dequant = force_dequant

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
                    out_sharding=P(kernel_axes[-1]),
                ),
            )
        else:
            self.bias = None

    @named_scope
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass of the linear layer."""
        output_pspec = P(*([None] * (x.ndim - 1)), self.kernel_axes[-1])
        output_sharding = NamedSharding(self.mesh, output_pspec)
        out = lax.dot_general(
            x,
            self.weight.value,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=self.params_dtype,
            out_sharding=output_sharding,
        )
        if self.skip_bias_add:
            return out, self.bias
        if self.bias is not None:
            out = out + self.bias.value
        return out, None


class QuantizedLinear(nnx.Module):
    """Quantized linear layer using native quantized matmul.

    This layer stores pre-quantized weights and scales, and uses the native
    quantized matmul kernel for the forward pass. Weights are quantized once
    at initialization/conversion time, and activations are quantized at runtime.

    Args:
        weight_q: Quantized weight tensor with shape ``[output_size, input_size]``.
        weight_scale: Weight quantization scale. Per-channel uses
            ``[output_size]``; block-wise uses ``[in_blocks, 1, output_size]``.
        bias: Optional bias tensor with shape ``[output_size]``.
        activation_dtype: Dtype for activation quantization. ``None`` disables
            activation quantization.
        mesh: Device mesh for sharding.
        kernel_axes: Partition spec axes for the weight tensor.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Original parameter dtype used for output casting.
        compute_dtype: Optional dtype override for the quantized kernel.
        weight_block_size: Optional block size ``(block_n, block_k)`` for
            block-wise weight quantization.
        scope_name: Name used for profiling scope.
    """

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
        force_dequant: bool = False,
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
        self.force_dequant = force_dequant
        self._pre_dequanted = False

    def maybe_pre_dequant(self) -> bool:
        """Pre-dequant blockwise weights when the blockwise kernel can't handle them.

        Called once at model-loading time. Converts block-scaled int8 weights
        to bf16 so the forward pass uses a plain matmul instead of going
        through the quantized kernel fallback on every step.
        """
        if self._pre_dequanted or self.weight_scale is None:
            return False

        scale_val = self.weight_scale.value
        if scale_val.ndim != 3:
            return False

        w_q = self.weight_q.value
        out_dim, in_dim = w_q.shape

        # Compute the local out_dim that shard_map would see.
        output_axis = self.kernel_axes[1] if self.kernel_axes else None
        local_out = out_dim // self.mesh.shape[output_axis] if output_axis else out_dim
        block_size_out = int(self.weight_block_size[0]) if self.weight_block_size else local_out

        if not self.force_dequant and should_use_blockwise_kernel(out_dim=local_out, block_size_out=block_size_out):
            return False

        # Dequant: expand block scale and multiply into weights.
        in_blocks = scale_val.shape[0]
        block_size_in = in_dim // in_blocks
        scale_2d = scale_val[:, 0, :]  # [in_blocks, out_dim]
        scale_full = jnp.repeat(scale_2d, block_size_in, axis=0)[:in_dim]  # [in_dim, out_dim]
        scale_full = jnp.transpose(scale_full, (1, 0))  # [out_dim, in_dim]
        w_deq = (w_q.astype(jnp.float32) * scale_full.astype(jnp.float32)).astype(jnp.bfloat16)

        self.weight_q = nnx.Param(w_deq)
        self.weight_scale = None
        self.weight_block_size = None
        self._pre_dequanted = True
        return True

    @classmethod
    def from_linear(
        cls,
        linear: LinearBase,
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
        is_static_input: bool = False,
        weight_block_size: Sequence[int] | None = None,
    ) -> "QuantizedLinear":
        """Convert a LinearBase layer to a QuantizedLinear layer.

        Uses per-channel weight quantization by default, and block-wise
        weight quantization when ``weight_block_size`` is provided.

        Args:
            linear: The LinearBase layer to convert.
            weight_dtype: Target dtype for weight quantization.
            activation_dtype: Target dtype for activation quantization
                (``None`` = no activation quantization).
            is_static_input: If true, expect a static checkpoint with
                pre-quantized weights.
            weight_block_size: Optional ``(block_n, block_k)`` for
                block-wise weight quantization.

        Returns:
            A new QuantizedLinear layer with quantized weights.
        """
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

                if (
                    effective_weight_block_size is not None
                    and len(effective_weight_block_size) == 2
                ):
                    block_n, block_k = int(effective_weight_block_size[0]), int(
                        effective_weight_block_size[1]
                    )
                    in_blocks = (in_features + block_k - 1) // block_k
                    scale_sharding = NamedSharding(
                        linear.mesh, P(kernel_axes[0], None, kernel_axes[1])
                    )
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(in_blocks, 1, out_features),
                        dtype=jnp.float32,
                        sharding=scale_sharding,
                    )
                else:
                    scale_sharding = NamedSharding(linear.mesh, P(kernel_axes[1]))
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(out_features,), dtype=jnp.float32, sharding=scale_sharding
                    )
                bias = linear.bias.value if linear.bias is not None else None
            else:
                if weight.dtype != weight_dtype:
                    raise ValueError(
                        "QuantizedLinear.from_linear(..., is_static_input=True) requires "
                        "pre-quantized concrete weights or abstract shapes. "
                        f"Got weight.dtype={weight.dtype}, expected {weight_dtype}."
                    )
                weight_q = weight.T
                if (
                    effective_weight_block_size is not None
                    and len(effective_weight_block_size) == 2
                ):
                    block_n, block_k = int(effective_weight_block_size[0]), int(
                        effective_weight_block_size[1]
                    )
                    in_blocks = (weight_q.shape[1] + block_k - 1) // block_k
                    out_features = weight_q.shape[0]
                    weight_scale = jnp.ones((in_blocks, 1, out_features), dtype=jnp.float32)
                else:
                    weight_scale = jnp.ones((weight_q.shape[0],), dtype=jnp.float32)
                bias = linear.bias.value if linear.bias is not None else None
        else:
            # LinearBase weight shape: [input_size, output_size]
            # xla_quantized_matmul expects w_q: [output_size, input_size]
            # So we need to transpose the weight before quantizing.
            weight = linear.weight.value
            weight_t = weight.T  # [output_size, input_size]

            if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                # Block-wise quantization over [output_size, input_size].
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype,
                    tensor=weight_t,
                    axis=(0, 1),
                    block_size=tuple(effective_weight_block_size),
                    pad_tensor=True,
                )
            else:
                # Per-channel quantization along output dimension.
                # After transpose, output_size is axis 0 and input_size is axis 1.
                # We want per-output-channel, so reduce along axis 1.
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype, tensor=weight_t, axis=1
                )

            # Get bias if it exists.
            bias = linear.bias.value if linear.bias is not None else None

        return cls(
            weight_q=weight_q,
            weight_scale=weight_scale,
            bias=bias,
            activation_dtype=activation_dtype,
            mesh=linear.mesh,
            kernel_axes=linear.kernel_axes,
            skip_bias_add=linear.skip_bias_add,
            params_dtype=linear.params_dtype,
            weight_block_size=effective_weight_block_size,
            scope_name=f"quantized_{linear.name}",
            force_dequant=getattr(linear, 'force_dequant', False),
        )

    @named_scope
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass using quantized matmul.

        Args:
            x: Input tensor [..., input_size]

        Returns:
            Tuple of (output, bias) where output is [..., output_size]
            and bias is returned if skip_bias_add is True
        """
        # Fast path for pre-dequanted layers: pure bf16 matmul.
        if self._pre_dequanted:
            x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
            output_pspec = P(*([None] * (x_2d.ndim - 1)), self.kernel_axes[1])
            output_sharding = NamedSharding(self.mesh, output_pspec)
            output = lax.dot_general(
                x_2d,
                self.weight_q.value,
                (((1,), (1,)), ((), ())),
                preferred_element_type=self.params_dtype,
                out_sharding=output_sharding,
            )
            if x.ndim > 2:
                output = output.reshape(x.shape[:-1] + (output.shape[-1],))
            if self.skip_bias_add:
                return output, self.bias
            if self.bias is not None:
                output = output + self.bias.value
            return output, None

        # Determine if we should quantize activations.
        quantize_activation = self.activation_dtype is not None

        # Handle batched inputs by reshaping to 2D.
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        scale_val = self.weight_scale.value

        input_axis, output_axis = self.kernel_axes[0], self.kernel_axes[1]
        if scale_val.ndim == 1:
            w_scale_spec = P(output_axis)
        else:  # ndim == 3, block-quant
            w_scale_spec = P(input_axis, None, output_axis)
        in_specs = (P(None, input_axis), P(output_axis, input_axis), w_scale_spec)
        out_specs = P(None, output_axis)

        output = shard_map(
            partial(
                xla_quantized_matmul_local,
                quantize_activation=quantize_activation,
                reduce_axis=input_axis,
                compute_dtype=self.compute_dtype,
                weight_block_size=self.weight_block_size,
                activation_quant_dtype=self.activation_dtype,
            ),
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(x_2d, self.weight_q.value, scale_val)

        # Reshape back to original batch dimensions.
        if x.ndim > 2:
            output = output.reshape(x.shape[:-1] + (output.shape[-1],))

        # Handle bias.
        if self.skip_bias_add:
            return output, self.bias
        if self.bias is not None:
            output = output + self.bias.value
        return output, None
