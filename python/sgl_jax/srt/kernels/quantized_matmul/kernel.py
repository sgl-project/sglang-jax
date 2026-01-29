# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def xla_quantized_matmul_local(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool = True,
    reduce_axis: str | None = None,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map.

    All computation (quantize, matmul, dequantize) happens locally on each device.
    If reduce_axis is provided, uses psum to combine partial sums across devices.

    Args:
        x: Activation tensor [batch, n_input_features] (local slice)
        w_q: Quantized weight tensor [n_output_features, n_input_features] (local slice)
        w_scale: Weight quantization scale [n_output_features]
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.

    Returns:
        Output of the quantized matmul.
    """
    out_dtype = x.dtype

    if quantize_activation:
        # Local quantization - each device uses its local max
        x_q, x_scale = quantize_tensor_simple(x, w_q.dtype, dim=-1)

        # Local matmul
        out = lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=out_dtype,
        )

        # Local dequantization
        out = out * x_scale.astype(out_dtype) * jnp.expand_dims(w_scale, 0).astype(out_dtype)
    else:
        # Local matmul without activation quantization
        out = lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=out_dtype,
        )
        out = out * jnp.expand_dims(w_scale, 0).astype(out_dtype)

    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, reduce_axis)

    return out
