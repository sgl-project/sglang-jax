# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import math

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.utils.quantization.quantization_utils import (
    dequantize_tensor,
    quantize_tensor,
    quantize_tensor_simple,
)


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
    activation_qdq_tile_size: int | None = None,
    activation_qdq_channelwise_axes: tuple[int, ...] | None = None,
    activation_qdq_calibration_method: str | None = None,
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
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    act_quant_dtype = w_q.dtype if activation_quant_dtype is None else activation_quant_dtype

    if quantize_activation and (
        activation_qdq_tile_size is not None or activation_qdq_channelwise_axes is not None
    ):
        if activation_qdq_calibration_method not in (None, "absmax", "maxabs"):
            raise NotImplementedError(
                "Only absmax/maxabs activation calibration is supported in QuantizedLinear"
            )

        if activation_qdq_channelwise_axes is None:
            # Qwix-style default for GEMM lhs: keep non-contraction axis (batch/token).
            channelwise_axes = (0,)
        else:
            channelwise_axes = tuple(sorted((a + x.ndim) % x.ndim for a in activation_qdq_channelwise_axes))

        reduce_axes = tuple(i for i in range(x.ndim) if i not in channelwise_axes)
        if not reduce_axes:
            # Degenerate per-element quantization doesn't buy us anything; skip explicit QDQ.
            reduce_axes = ()

        if reduce_axes:
            axis_arg: int | tuple[int, ...]
            axis_arg = reduce_axes[0] if len(reduce_axes) == 1 else reduce_axes
            orig_in_dim = x.shape[-1]
            if activation_qdq_tile_size is not None:
                if reduce_axes != (x.ndim - 1,):
                    raise NotImplementedError(
                        "Qwix-style tile_size is only supported on the contraction axis for Linear"
                    )
                x_q, x_scale = quantize_tensor(
                    dtype=act_quant_dtype,
                    tensor=x,
                    axis=axis_arg,
                    block_size=activation_qdq_tile_size,
                    pad_tensor=True,
                )
                x = dequantize_tensor(x_q, x_scale, axis=axis_arg, out_dtype=compute_dtype)
                x = x[..., :orig_in_dim]
            else:
                x_q, x_scale = quantize_tensor(
                    dtype=act_quant_dtype,
                    tensor=x,
                    axis=axis_arg,
                )
                x = dequantize_tensor(x_q, x_scale, axis=axis_arg, out_dtype=compute_dtype)
            quantize_activation = False

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
            # Fallback for older callers. This is unsafe under some sharding layouts;
            # QuantizedLinear now passes the original block sizes explicitly.
            block_size_out = math.ceil(out_dim / out_blocks)
            block_size_in = math.ceil(in_dim / in_blocks)

        # `w_scale` is treated as the global block-scale matrix [out_blocks, in_blocks].
        # For TP-sharded weights, local `w_q` may start/end in the middle of a block.
        # Build the exact local per-element scale by indexing global block ids using
        # the local shard's global feature offsets.
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
            # Keep activation quant/dequant semantics aligned with the non-block path.
            # We still dequantize weights manually (no fused blockwise kernel here).
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
            lhs = x_q.astype(compute_dtype)
        else:
            lhs = x.astype(compute_dtype)

        # Perform standard dot product
        # x: [batch, in_dim], w_dequant: [out_dim, in_dim]
        # Contract last dim of x with last dim of w_dequant (dim 1)
        out = lax.dot_general(
            lhs,
            w_dequant,
            dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        # out: [batch, out_dim]
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
        out = lax.psum(out, reduce_axis)

    return out
