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
from sgl_jax.srt.kernels.reduce_scatter_matmul import (
    bidirectional_reduce_scatter_matmul,
)
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

# ---------------------------------------------------------------------------
# Reduce-scatter matmul infrastructure (sequence parallel)
# ---------------------------------------------------------------------------
# Default tile sizes for the bidirectional RS-matmul kernel, keyed by device.
# TPU v7 (Ironwood): MXU supports up to 256x256.  Using bm=bn=256 for MXU
# utilization, bk=128 for compatibility with head_dim=128 multiples.
# TODO: Run tuning script to find optimal tile sizes per model shape.
_RS_MATMUL_TILE_CONFIGS: dict[str, tuple[int, int, int]] = {
    # (bm, bn, bk)
    "TPU v7": (256, 256, 128),
    "TPU v6e": (128, 128, 128),
    "TPU v5e": (128, 128, 128),
}
_RS_MATMUL_DEFAULT_TILES = (128, 128, 128)  # Conservative fallback


def _get_rs_matmul_tiles() -> tuple[int, int, int]:
    """Get tile sizes for the RS-matmul kernel based on the current device."""
    from sgl_jax.srt.utils.jax_utils import get_device_name

    try:
        device = get_device_name()
        return _RS_MATMUL_TILE_CONFIGS.get(device, _RS_MATMUL_DEFAULT_TILES)
    except Exception:
        return _RS_MATMUL_DEFAULT_TILES


def _fused_rs_matmul(
    x: jax.Array,
    weight: jax.Array,
    *,
    axis_name: str,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
) -> jax.Array:
    """Fused reduce-scatter matmul using bidirectional kernel (inside shard_map).

    Computes: reduce_scatter(x @ weight, scatter_dim=0) using the bidirectional
    ring algorithm that overlaps compute with communication.

    Args:
        x: Activation [M, K_local] (local shard)
        weight: Weight [K_local, H] (local shard)
        axis_name: Mesh axis for collective ops
        bm, bn, bk: Tile sizes for the Pallas kernel

    Returns:
        [M/num_devices, H] - scattered output
    """
    # The kernel computes x @ y.T, so transpose weight: y = weight.T → [H, K_local]
    y = weight.T
    return bidirectional_reduce_scatter_matmul(x, y, axis_name=axis_name, bm=bm, bn=bn, bk=bk)


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
        sequence_parallel: bool = False,
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes
        self.mesh = mesh
        self.name = scope_name
        self.sequence_parallel = sequence_parallel
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

    def _forward_reduce_scatter(self, x: jax.Array) -> jax.Array:
        """Forward with fused reduce-scatter instead of allreduce (sequence parallel).

        Instead of producing replicated [M, H] output (allreduce), produces
        scattered [M/TP, H] output (reduce-scatter on M dimension).
        """
        reduce_axis = self.kernel_axes[0]  # e.g. "tensor" for row-parallel
        bm, bn, bk = _get_rs_matmul_tiles()
        return shard_map(
            partial(_fused_rs_matmul, axis_name=reduce_axis, bm=bm, bn=bn, bk=bk),
            mesh=self.mesh,
            in_specs=(
                P(None, reduce_axis),  # x: full M, K sharded
                P(reduce_axis, None),  # weight: K sharded, full H
            ),
            out_specs=P(reduce_axis, None),  # M scattered, full H
            check_vma=False,
        )(x, self.weight.value)

    @named_scope
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass of the linear layer."""
        if self.sequence_parallel:
            return self._forward_reduce_scatter(x), None

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
        scope_name: str = "quantized_linear",
        sequence_parallel: bool = False,
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
        self.name = scope_name
        self.sequence_parallel = sequence_parallel

        # Determine if we need tensor parallel reduction
        # kernel_axes[0] is input axis, kernel_axes[1] is output axis
        # For row-parallel (e.g., o_proj): kernel_axes = ("tensor", None)
        #   -> input is sharded, need psum over "tensor"
        # For column-parallel (e.g., q_proj): kernel_axes = (None, "tensor")
        #   -> input is replicated, no psum needed
        self.reduce_axis = kernel_axes[0]  # Axis to reduce over (or None)

    @classmethod
    def from_linear(
        cls,
        linear: "LinearBase",
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
        is_static_input: bool = False,
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
        if is_static_input:
            w_shape = linear.weight.shape
            input_size, output_size = w_shape[0], w_shape[1]
            weight_q = jnp.zeros((output_size, input_size), dtype=weight_dtype)
            weight_scale = jnp.zeros((output_size,), dtype=jnp.float32)
            bias = linear.bias.value if linear.bias is not None else None
        else:
            # LinearBase weight shape: [input_size, output_size]
            # xla_quantized_matmul expects w_q: [output_size, input_size]
            # So we need to transpose the weight before quantizing
            weight = linear.weight.value
            weight_t = weight.T  # [output_size, input_size]

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
            scope_name=f"quantized_{linear.name}",
            sequence_parallel=linear.sequence_parallel,
        )

    def get_weight_bf16(self) -> jax.Array:
        """Dequantize weight to BF16, returning [input_size, output_size] layout.

        This matches the LinearBase weight layout so the result can be used
        interchangeably (e.g., for the reduce-scatter matmul kernel).
        """
        scale_val = self.weight_scale.value
        if scale_val.ndim == 2 and scale_val.shape[1] == 1:
            scale_val = jnp.squeeze(scale_val, axis=1)
        # weight_q is [output_size, input_size], dequantize and transpose to [input_size, output_size]
        return (self.weight_q.value.astype(jnp.bfloat16) * scale_val).T

    def _forward_reduce_scatter(self, x: jax.Array) -> jax.Array:
        """Forward with fused reduce-scatter instead of allreduce (sequence parallel).

        Passes FP8 weight_q directly to the RS-matmul kernel — the kernel casts
        tiles to f32 internally.  weight_q is already [H, K] which is the y layout
        the kernel expects, so no transpose is needed.  The per-channel weight
        scale is applied after the reduce-scatter (cheap element-wise on the
        already-reduced [M/TP, H] output).
        """
        reduce_axis = self.reduce_axis
        output_axis = self.kernel_axes[1]
        bm, bn, bk = _get_rs_matmul_tiles()

        # weight_q [H, K] with P(output_axis, reduce_axis) is already the y
        # layout for the kernel (computes x @ y.T), so call directly.
        output = shard_map(
            partial(
                bidirectional_reduce_scatter_matmul,
                axis_name=reduce_axis,
                bm=bm,
                bn=bn,
                bk=bk,
            ),
            mesh=self.mesh,
            in_specs=(
                P(None, reduce_axis),  # x: [M, K] full M, K sharded
                P(output_axis, reduce_axis),  # weight_q: [H, K] as-is
            ),
            out_specs=P(reduce_axis, None),  # [M/TP, H]
            check_vma=False,
        )(x, self.weight_q.value)

        # Apply per-channel weight scale after reduce-scatter
        scale_val = self.weight_scale.value
        if scale_val.ndim == 2 and scale_val.shape[1] == 1:
            scale_val = jnp.squeeze(scale_val, axis=1)
        return output * scale_val

    @named_scope
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass using quantized matmul.

        Args:
            x: Input tensor [..., input_size]

        Returns:
            Tuple of (output, bias) where output is [..., output_size]
            and bias is returned if skip_bias_add is True
        """
        if self.sequence_parallel:
            return self._forward_reduce_scatter(x), None

        # Determine if we should quantize activations
        quantize_activation = self.activation_dtype is not None

        # Handle batched inputs by reshaping to 2D
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        scale_val = self.weight_scale.value
        if scale_val.ndim == 2 and scale_val.shape[1] == 1:
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
        # Weight scale sharding: P(output_axis) - per output channel
        # Output sharding: P(None, output_axis)
        in_specs = (
            P(None, input_axis),  # x
            P(output_axis, input_axis),  # w_q
            P(output_axis),  # w_scale
        )
        out_specs = P(None, output_axis)

        output = shard_map(
            partial(
                xla_quantized_matmul_local,
                quantize_activation=quantize_activation,
                reduce_axis=input_axis,  # psum over input axis (e.g., "tensor" for o_proj)
                compute_dtype=self.compute_dtype,
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
