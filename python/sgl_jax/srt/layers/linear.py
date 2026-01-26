from collections.abc import Sequence

import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul
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
        self.name = scope_name

    @classmethod
    def from_linear(
        cls,
        linear: "LinearBase",
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
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
            scope_name=linear.name,
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
        # Determine if we should quantize activations
        quantize_activation = self.activation_dtype is not None

        # Handle batched inputs by reshaping to 2D
        orig_shape = x.shape
        if x.ndim > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # Call the native quantized matmul kernel
        # xla_quantized_matmul expects:
        #   x: [batch, n_input_features]
        #   w_q: [n_output_features, n_input_features]
        #   w_scale: [n_output_features]
        output = xla_quantized_matmul(
            x_2d,
            self.weight_q.value,
            self.weight_scale.value,
            quantize_activation=quantize_activation,
        )

        # Reshape back to original batch dimensions
        if x.ndim > 2:
            output = output.reshape(*orig_shape[:-1], output.shape[-1])

        # Handle bias
        bias = self.bias if not self.skip_bias_add else None
        if bias is not None:
            output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias
