import math
from collections.abc import Sequence
from functools import partial

import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


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
                    out_sharding=P(
                        kernel_axes[-1],
                    ),
                ),
            )
        else:
            self.bias = None

    @named_scope
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
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


class QuantizedLinear(nnx.Module):
    """Quantized linear layer using native quantized matmul.

    This layer stores pre-quantized weights and scales, and uses the native
    quantized matmul kernel for the forward pass. Weights are quantized once
    at initialization/conversion time, and activations are quantized at runtime.

    Args:
        weight_q: Quantized weight tensor [output_size, input_size]
        weight_scale: Weight quantization scale [output_size] for per-channel
        bias: Optional bias tensor [output_size]
        activation_dtype: Dtype for activation quantization (None = no activation quantization)
        mesh: Device mesh for sharding
        kernel_axes: Partition spec axes for the weight tensor
        skip_bias_add: If true, skip adding bias but instead return it
        params_dtype: Original data type of the parameters (for output casting)
        scope_name: Name for profiling scope
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
        activation_qdq_tile_size: int | None = None,
        activation_qdq_channelwise_axes: tuple[int, ...] | None = None,
        activation_qdq_calibration_method: str | None = None,
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
        self.activation_qdq_tile_size = activation_qdq_tile_size
        self.activation_qdq_channelwise_axes = activation_qdq_channelwise_axes
        self.activation_qdq_calibration_method = activation_qdq_calibration_method
        self.name = scope_name

        # Determine if we need tensor parallel reduction
        # kernel_axes[0] is input axis, kernel_axes[1] is output axis
        # For row-parallel (e.g., o_proj): kernel_axes = ("tensor", None)
        #   -> input is sharded, need psum over "tensor"
        # For column-parallel (e.g., q_proj): kernel_axes = (None, "tensor")
        #   -> input is replicated, no psum needed
        self.reduce_axis = kernel_axes[0]  # Axis to reduce over (or None)

        self._validate_blockwise_sharding_alignment()

    def _validate_blockwise_sharding_alignment(self):
        """Compatibility hook for block-wise QuantizedLinear.

        The blockwise path now accepts arbitrary TP/block alignment by treating
        `weight_scale` as a global (replicated) 2D block-scale matrix and
        reconstructing each local shard's elementwise scales inside the kernel
        using shard offsets. Keep this method as a no-op to avoid reintroducing
        stale alignment assumptions.
        """
        return

    @classmethod
    def from_linear(
        cls,
        linear: "LinearBase",
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
        is_static_input: bool = False,
        weight_block_size: Sequence[int] | None = None,
        activation_qdq_tile_size: int | None = None,
        activation_qdq_channelwise_axes: tuple[int, ...] | None = None,
        activation_qdq_calibration_method: str | None = None,
    ) -> "QuantizedLinear":
        """Convert a LinearBase layer to a QuantizedLinear layer.

        Uses per-channel weight quantization and dynamic per-token activation quantization.

        Args:
            linear: The LinearBase layer to convert
            weight_dtype: Target dtype for weight quantization (e.g., jnp.int8, jnp.float8_e4m3fn)
            activation_dtype: Target dtype for activation quantization (None = no activation quantization)

        Returns:
            A new QuantizedLinear layer with quantized weights
        """
        effective_weight_block_size = (
            tuple(weight_block_size) if weight_block_size is not None else None
        )

        if is_static_input:
            # Static checkpoint already stores pre-quantized (float8) weights and scales.
            weight = linear.weight.value  # [in_features, out_features]

            def _make_shape_struct(shape, dtype, sharding=None):
                try:
                    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)
                except TypeError:
                    # Older JAX may not support the sharding kwarg.
                    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)

            if isinstance(weight, jax.ShapeDtypeStruct):
                in_features, out_features = map(int, weight.shape)
                kernel_axes = linear.kernel_axes or (None, None)
                wq_sharding = NamedSharding(linear.mesh, P(kernel_axes[1], kernel_axes[0]))
                weight_q = _make_shape_struct(
                    (out_features, in_features),
                    weight_dtype,
                    sharding=wq_sharding,
                )

                if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                    block_n, block_k = int(effective_weight_block_size[0]), int(effective_weight_block_size[1])
                    out_blocks = (out_features + block_n - 1) // block_n
                    in_blocks = (in_features + block_k - 1) // block_k
                    scale_shape = (out_blocks, in_blocks)
                    scale_sharding = NamedSharding(linear.mesh, P(None, None))
                else:
                    scale_shape = (out_features,)
                    scale_sharding = NamedSharding(linear.mesh, P(kernel_axes[1]))
                weight_scale = _make_shape_struct(scale_shape, jnp.float32, sharding=scale_sharding)
                bias = linear.bias.value if linear.bias is not None else None
            else:
                weight_q = weight.T.astype(weight_dtype)

                if hasattr(linear, "weight_scale"):
                    weight_scale = linear.weight_scale.value
                    if (
                        effective_weight_block_size is not None
                        and len(effective_weight_block_size) == 2
                        and weight_scale.ndim != 2
                    ):
                        # The pre-wrap LinearBase placeholder is 1D. Rebuild a 2D blockwise
                        # placeholder so weight loading can assign checkpoint block scales.
                        block_n, block_k = int(effective_weight_block_size[0]), int(
                            effective_weight_block_size[1]
                        )
                        out_blocks = (weight_q.shape[0] + block_n - 1) // block_n
                        in_blocks = (weight_q.shape[1] + block_k - 1) // block_k
                        weight_scale = jnp.ones((out_blocks, in_blocks), dtype=jnp.float32)
                    elif weight_scale.ndim == 1 and weight_scale.shape[0] != weight_q.shape[0]:
                        # Broadcast/trim conservatively
                        weight_scale = jnp.reshape(weight_scale, (-1,))
                        weight_scale = jnp.broadcast_to(
                            weight_scale, (weight_q.shape[0],)
                        )[: weight_q.shape[0]]
                else:
                    if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                        block_n, block_k = int(effective_weight_block_size[0]), int(
                            effective_weight_block_size[1]
                        )
                        out_blocks = (weight_q.shape[0] + block_n - 1) // block_n
                        in_blocks = (weight_q.shape[1] + block_k - 1) // block_k
                        weight_scale = jnp.ones((out_blocks, in_blocks), dtype=jnp.float32)
                    else:
                        weight_scale = jnp.ones((weight_q.shape[0],), dtype=jnp.float32)
                bias = linear.bias.value if linear.bias is not None else None
        else:
            # LinearBase weight shape: [input_size, output_size]
            # xla_quantized_matmul expects w_q: [output_size, input_size]
            # So we need to transpose the weight before quantizing
            weight = linear.weight.value
            weight_t = weight.T  # [output_size, input_size]

            if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                # Static fp8 configs may request block-wise weight quantization.
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype,
                    tensor=weight_t,
                    axis=(0, 1),
                    block_size=tuple(effective_weight_block_size),
                    pad_tensor=True,
                )
            else:
                # Per-channel quantization along output dimension
                # After transpose, output_size is axis 0, input_size is axis 1
                # We want per-output-channel, so reduce along axis 1 (input features)
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype,
                    tensor=weight_t,
                    axis=1,
                )

            # Get bias if it exists
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
            activation_qdq_tile_size=activation_qdq_tile_size,
            activation_qdq_channelwise_axes=activation_qdq_channelwise_axes,
            activation_qdq_calibration_method=activation_qdq_calibration_method,
            scope_name=f"quantized_{linear.name}",
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
        # Whether activation quantization is enabled at all. The kernel decides
        # between fused per-token quantization and explicit QDQ (Qwix-style
        # tiled/channelwise rules) based on the extra activation_qdq_* fields.
        quantize_activation = self.activation_dtype is not None

        # Handle batched inputs by reshaping to 2D
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        scale_val = self.weight_scale.value
        if (
            scale_val.ndim == 2
            and scale_val.shape[1] == 1
            and scale_val.shape[0] == self.weight_q.value.shape[0]
        ):
            scale_val = jnp.squeeze(scale_val, axis=1)
        # Use shard_map for local computation with single all-reduce
        # kernel_axes[0] = input sharding axis (e.g., "tensor" for o_proj, None for q_proj)
        # kernel_axes[1] = output sharding axis (e.g., None for o_proj, "tensor" for q_proj)
        #
        # Weight w_q has shape [output_size, input_size]
        # After transpose from LinearBase, its sharding is P(kernel_axes[1], kernel_axes[0])
        # e.g., for o_proj with kernel_axes=("tensor", None): w_q has P(None, "tensor")
        input_axis = self.kernel_axes[0]
        output_axis = self.kernel_axes[1]

        # Input x sharding: for row-parallel, x is P(None, input_axis)
        # Weight w_q sharding: P(output_axis, input_axis)
        # Weight scale sharding:
        #   - Per-channel: P(output_axis)
        #   - Block-wise:  P(None, None) (replicated global block-scale matrix)
        # Output sharding: P(None, output_axis)
        w_scale_spec = P(output_axis) if scale_val.ndim == 1 else P(None, None)
        in_specs = (
            P(None, input_axis),  # x
            P(output_axis, input_axis),  # w_q
            w_scale_spec,  # w_scale
        )
        out_specs = P(None, output_axis)

        # Infer actual block sizes from global weight and scale shapes.
        # The config's weight_block_size may not match the actual scale layout
        # when k_proj/v_proj weights have been replicated for KV head padding
        effective_weight_block_size = self.weight_block_size
        if scale_val.ndim == 2 and self.weight_block_size is not None:
            global_out_size = self.weight_q.value.shape[0]
            global_in_size = self.weight_q.value.shape[1]
            inferred_bs_out = math.ceil(global_out_size / scale_val.shape[0])
            inferred_bs_in = math.ceil(global_in_size / scale_val.shape[1])
            if (inferred_bs_out != self.weight_block_size[0]
                    or inferred_bs_in != self.weight_block_size[1]):
                effective_weight_block_size = (inferred_bs_out, inferred_bs_in)

        output = shard_map(
            partial(
                xla_quantized_matmul_local,
                quantize_activation=quantize_activation,
                reduce_axis=input_axis,  # psum over input axis (e.g., "tensor" for o_proj)
                input_shard_axis=input_axis,
                output_shard_axis=output_axis,
                compute_dtype=self.compute_dtype,
                weight_block_size=effective_weight_block_size,
                activation_quant_dtype=self.activation_dtype,
                activation_qdq_tile_size=self.activation_qdq_tile_size,
                activation_qdq_channelwise_axes=self.activation_qdq_channelwise_axes,
                activation_qdq_calibration_method=self.activation_qdq_calibration_method,
            ),
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(x_2d, self.weight_q.value, scale_val)

        # Reshape back to original batch dimensions
        if x.ndim > 2:
            output = output.reshape(*orig_shape[:-1], output.shape[-1])

        # Handle bias
        bias = self.bias if not self.skip_bias_add else None
        if bias is not None:
            output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias
