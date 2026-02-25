# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul utilities: XLA fallback and Pallas wrapper."""

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.kernel import quantized_matmul_kernel
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def xla_quantized_matmul_local(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool = True,
    reduce_axis: str | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map (pure XLA fallback).

    All computation (quantize, matmul, dequantize) happens locally on each device.
    If reduce_axis is provided, uses psum to combine partial sums across devices.

    Args:
        x: Activation tensor [batch, n_input_features] (local slice)
        w_q: Quantized weight tensor [n_output_features, n_input_features] (local slice)
        w_scale: Weight quantization scale [n_output_features]
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.
        compute_dtype: Dtype for intermediate computation. None uses x.dtype.

    Returns:
        Output of the quantized matmul.
    """
    out_dtype = x.dtype
    compute_dtype = out_dtype if compute_dtype is None else compute_dtype

    if quantize_activation:
        x_q, x_scale = quantize_tensor_simple(x, w_q.dtype, dim=-1)

        out = lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )

        out = out.astype(compute_dtype)
        out = (
            out * x_scale.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
        )
    else:
        out = lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        out = out.astype(compute_dtype)
        out = out * jnp.expand_dims(w_scale, 0).astype(compute_dtype)

    if reduce_axis is not None:
        out = lax.psum(out, reduce_axis)

    return out.astype(out_dtype)


def pallas_quantized_matmul_local(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool = True,
    reduce_axis: str | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map (Pallas kernel).

    Delegates the quantize + matmul + dequantize to the Pallas-based
    quantized_matmul_kernel, which handles tiling, padding, and VMEM
    management internally. If reduce_axis is provided, uses psum to
    combine partial sums across devices.

    Args:
        x: Activation tensor [batch, n_input_features] (local slice)
        w_q: Quantized weight tensor [n_output_features, n_input_features] (local slice)
        w_scale: Weight quantization scale [n_output_features]
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.
        compute_dtype: Unused (Pallas kernel manages its own accumulator dtype).

    Returns:
        Output of the quantized matmul.
    """
    out_dtype = x.dtype
    x_q_dtype = w_q.dtype if quantize_activation else None

    out = quantized_matmul_kernel(x, w_q, w_scale, x_q_dtype=x_q_dtype)

    if reduce_axis is not None:
        out = lax.psum(out, reduce_axis)

    return out.astype(out_dtype)
