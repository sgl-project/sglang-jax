# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
    get_blockwise_kernel,
    get_safe_blockwise_tuned_value,
    should_use_blockwise_kernel,
)
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
        w_scale: Weight quantization scale.  Per-channel: ``[n_output_features]``.
            Block-wise (pre-expanded): ``[in_blocks, 1, n_output_features]``.
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.
        weight_block_size: ``(block_n, block_k)`` for block-wise quantization.
        activation_quant_dtype: Dtype for activation quantization.

    Returns:
        Output of the quantized matmul.
    Supports both per-channel and block-wise weight quantization.
    """
    out_dtype = x.dtype
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    act_quant_dtype = w_q.dtype if activation_quant_dtype is None else activation_quant_dtype

    # w_scale.ndim == 3 implies pre-expanded block-wise quantization
    # (scale was expanded from [out_blocks, in_blocks] to [in_blocks, 1, n_out]
    #  at init time via expand_block_scale).
    is_block_quant = w_scale.ndim == 3

    if is_block_quant:
        # === Block Quantization Path ===
        out_dim, in_dim = w_q.shape
        in_blocks = w_scale.shape[0]
        block_size_in = in_dim // in_blocks

        if weight_block_size is not None:
            block_size_out = int(weight_block_size[0])
        else:
            block_size_out = block_size_in

        blockwise_kernel = get_blockwise_kernel()
        if blockwise_kernel is None:
            raise RuntimeError(
                "Block-wise quantized matmul requires the blockwise kernel, "
                "but it failed to load. Please check your installation."
            )
        if not should_use_blockwise_kernel(
            out_dim=int(out_dim),
            block_size_out=int(block_size_out),
        ):
            raise RuntimeError(
                f"Block-wise kernel does not support out_dim={out_dim} with "
                f"block_size_out={block_size_out} (known to cause NaNs)."
            )

        # w_scale is already in kernel-ready layout [in_blocks, 1, n_out].
        x_q_dtype = act_quant_dtype if quantize_activation else x.dtype
        tuned_value = get_safe_blockwise_tuned_value(
            n_batch=int(x.shape[0]),
            n_out=int(out_dim),
            n_in=int(in_dim),
            x_q_dtype=x_q_dtype,
            w_q_dtype=w_q.dtype,
            block_size_in=block_size_in,
        )
        out = blockwise_kernel(
            x=x,
            w_q=w_q,
            w_scale=w_scale,
            block_size=block_size_in,
            x_q_dtype=x_q_dtype,
            tuned_value=tuned_value,
        )

    else:
        # === Standard Per-Channel Quantization Path ===
        if quantize_activation:
            x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
            out = lax.dot_general(
                x_q,
                w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )
            out = (
                out.astype(compute_dtype)
                * x_scale.astype(compute_dtype)
                * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
            )
        else:
            out = lax.dot_general(
                x,
                w_q,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=compute_dtype,
            )
            out = out.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)

    out = out.astype(out_dtype)
    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, axis_name=reduce_axis)

    return out
