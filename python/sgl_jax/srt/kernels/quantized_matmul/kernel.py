# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import math

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
    weight_block_size: tuple[int, int] | None = None,
    activation_quant_dtype: jnp.dtype | None = None,
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
    Supports both per-channel and block-wise weight quantization.
    """
    out_dtype = x.dtype
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    act_quant_dtype = w_q.dtype if activation_quant_dtype is None else activation_quant_dtype

    # w_scale.ndim == 2 implies block-wise quantization
    is_block_quant = w_scale.ndim == 2

    if is_block_quant:
        # === Block Quantization Path ===
        out_dim, in_dim = w_q.shape
        out_blocks, in_blocks = w_scale.shape

        # Calculate local block size based on sharded shapes
        if weight_block_size is not None:
            block_size_out, block_size_in = int(weight_block_size[0]), int(weight_block_size[1])
        else:
            block_size_out = math.ceil(out_dim / out_blocks)
            block_size_in = math.ceil(in_dim / in_blocks)

        # Generate local indices for sharded scale lookup
        row_idx = jnp.arange(out_dim, dtype=jnp.int32) // jnp.int32(block_size_out)
        col_idx = jnp.arange(in_dim, dtype=jnp.int32) // jnp.int32(block_size_in)
        
        # Ensure indices are within local shard bounds
        row_idx = jnp.clip(row_idx, 0, out_blocks - 1)
        col_idx = jnp.clip(col_idx, 0, in_blocks - 1)
        
        scale_expanded = w_scale[row_idx[:, None], col_idx[None, :]]
        
        # Dequantize weight locally
        w_dequant = w_q.astype(compute_dtype) * scale_expanded.astype(compute_dtype)

        if quantize_activation:
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
            lhs = x_q.astype(compute_dtype)
        else:
            lhs = x.astype(compute_dtype)

        # Standard dot product with dequantized weight
        out = lax.dot_general(
            lhs, w_dequant,
            dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        if quantize_activation:
            out = out * x_scale.astype(compute_dtype)
        
    else:
        # === Standard Per-Channel Quantization Path ===
        if quantize_activation:
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
            out = lax.dot_general(
                x_q, w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )
            out = out.astype(compute_dtype) * x_scale.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
        else:
            out = lax.dot_general(
                x, w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )
            out = out.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)

    out = out.astype(out_dtype)
    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, axis_name=reduce_axis)

    return out
