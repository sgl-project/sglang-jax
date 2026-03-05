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
    input_shard_axis: str | None = None,
    output_shard_axis: str | None = None,
    compute_dtype: jnp.dtype | None = None,
    weight_block_size: tuple[int, int] | None = None,
    activation_quant_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map.

    Args:
        x: Input tensor [batch, in_features]
        w_q: Quantized weight tensor [out_features, in_features]
        w_scale: Weight scale tensor [out_features, 1] or [out_blocks, in_blocks]
        quantize_activation: Whether to quantize the activation
        reduce_axis: The axis to reduce over (for all-reduce)
        input_shard_axis: The mesh axis name for input sharding
        output_shard_axis: The mesh axis name for output sharding
        compute_dtype: The dtype to perform the computation in
        weight_block_size: Optional block size for block-wise weight quantization
        activation_quant_dtype: The dtype to use for activation quantization

    Returns:
        The result of the quantized matmul [batch, out_features]
    """
    out_dtype = x.dtype
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    act_quant_dtype = w_q.dtype if activation_quant_dtype is None else activation_quant_dtype

    # Block-wise scale uses 2D layout [out_blocks, in_blocks].
    # Per-channel [out_features, 1] is normalized to 1D before calling this kernel.
    is_block_quant = w_scale.ndim == 2

    if is_block_quant:
        # === Block Quantization Path (Manual Dequantize -> BF16 Dot) ===
        out_dim, in_dim = w_q.shape
        out_blocks, in_blocks = w_scale.shape

        if weight_block_size is not None:
            block_size_out, block_size_in = int(weight_block_size[0]), int(weight_block_size[1])
        else:
            block_size_out = math.ceil(out_dim / out_blocks)
            block_size_in = math.ceil(in_dim / in_blocks)

        out_axis_index = (
            lax.axis_index(output_shard_axis)
            if output_shard_axis is not None
            else jnp.array(0, dtype=jnp.int32)
        )
        in_axis_index = (
            lax.axis_index(input_shard_axis)
            if input_shard_axis is not None
            else jnp.array(0, dtype=jnp.int32)
        )
        out_global_offset = out_axis_index.astype(jnp.int32) * jnp.int32(out_dim)
        in_global_offset = in_axis_index.astype(jnp.int32) * jnp.int32(in_dim)

        row_idx = (jnp.arange(out_dim, dtype=jnp.int32) + out_global_offset) // jnp.int32(
            block_size_out
        )
        col_idx = (jnp.arange(in_dim, dtype=jnp.int32) + in_global_offset) // jnp.int32(
            block_size_in
        )
        row_idx = jnp.clip(row_idx, 0, out_blocks - 1)
        col_idx = jnp.clip(col_idx, 0, in_blocks - 1)
        scale_expanded = w_scale[row_idx[:, None], col_idx[None, :]]
        
        # Dequantize weight to compute_dtype
        w_dequant = w_q.astype(compute_dtype) * scale_expanded.astype(compute_dtype)

        if quantize_activation:
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
            lhs = x_q.astype(compute_dtype)
        else:
            lhs = x.astype(compute_dtype)

        # Perform standard dot product
        out = lax.dot_general(
            lhs,
            w_dequant,
            dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        if quantize_activation:
            out = out * x_scale.astype(compute_dtype)
        
    else:
        # === Standard Per-Channel Quantization Path ===
        if quantize_activation:
            # Local quantization - each device uses its local max
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)

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

    out = out.astype(out_dtype)
    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, axis_name=reduce_axis)

    return out
