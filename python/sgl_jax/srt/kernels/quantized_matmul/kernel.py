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
    compute_dtype: jnp.dtype | None = None,
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
    compute_dtype = out_dtype if compute_dtype is None else compute_dtype

    # Check for Block Quantization (scale varies along input dimension)
    # w_scale shape: [out_blocks, in_blocks] or similar, if ndim==2 and shape[1] > 1
    # Standard per-channel scale is [out_features] or [out_features, 1]
    is_block_quant = w_scale.ndim == 2 and w_scale.shape[1] > 1

    if is_block_quant:
        # === Block Quantization Path (Manual Dequantize -> BF16 Dot) ===
        out_dim, in_dim = w_q.shape
        out_blocks, in_blocks = w_scale.shape
        
        # Determine block sizes
        block_size_out = out_dim // out_blocks
        block_size_in = in_dim // in_blocks
        
        # Expand scale to match weight shape
        # scale: [out_blocks, in_blocks] -> [out_dim, in_dim]
        scale_expanded = jnp.repeat(w_scale, block_size_out, axis=0)
        scale_expanded = jnp.repeat(scale_expanded, block_size_in, axis=1)
        
        # Dequantize weight to compute_dtype
        w_dequant = w_q.astype(compute_dtype) * scale_expanded.astype(compute_dtype)
        
        # Perform standard dot product
        # x: [batch, in_dim], w_dequant: [out_dim, in_dim]
        # Contract last dim of x with last dim of w_dequant (dim 1)
        out = lax.dot_general(
            x.astype(compute_dtype),
            w_dequant,
            dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        # out: [batch, out_dim]
        
    else:
        # === Standard Per-Channel Quantization Path ===
        if quantize_activation:
            # Local quantization - each device uses its local max
            x_q, x_scale = quantize_tensor_simple(x, w_q.dtype, dim=-1)

            # Local matmul
            out = lax.dot_general(
                x_q,
                w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )

            # Local dequantization
            out = out.astype(compute_dtype)
            out = (
                out * x_scale.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
            )
        else:
            # Local matmul without activation quantization
            out = lax.dot_general(
                x,
                w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )
            out = out.astype(compute_dtype)
            out = out * jnp.expand_dims(w_scale, 0).astype(compute_dtype)

    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, reduce_axis)

    return out.astype(out_dtype)
