# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import math

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
    convert_block_scale_to_kernel_layout,
    get_blockwise_kernel,
    get_safe_blockwise_tuned_value,
    should_use_blockwise_kernel,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def _get_effective_block_sizes(
    w_q: jax.Array,
    w_scale: jax.Array,
    weight_block_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Infer effective ``(block_n, block_k)`` from weights and scales.

    When a rule provides ``weight_block_size`` we trust that explicitly.
    Otherwise, this recovers the block sizes from the compact block-scale shape
    carried by an offline checkpoint.
    """
    out_dim, in_dim = w_q.shape
    out_blocks, in_blocks = w_scale.shape

    if weight_block_size is not None:
        block_size_out, block_size_in = int(weight_block_size[0]), int(weight_block_size[1])
    else:
        if out_blocks <= 0 or in_blocks <= 0:
            raise ValueError(
                f"Invalid w_scale shape: {w_scale.shape}. Both dimensions must be positive."
            )
        block_size_out = math.ceil(out_dim / out_blocks)
        block_size_in = math.ceil(in_dim / in_blocks)

    if block_size_out <= 0 or block_size_in <= 0:
        raise ValueError(
            f"Invalid block sizes: block_size_out={block_size_out}, block_size_in={block_size_in}."
        )
    return block_size_out, block_size_in


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
        block_size_out, block_size_in = _get_effective_block_sizes(
            w_q=w_q,
            w_scale=w_scale,
            weight_block_size=weight_block_size,
        )

        blockwise_kernel = get_blockwise_kernel()
        if jax.default_backend() != "tpu":
            raise RuntimeError(
                "Block-wise quantized matmul requires TPU backend, "
                f"but got {jax.default_backend()!r}."
            )
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

        w_scale_kernel = convert_block_scale_to_kernel_layout(
            w_scale=w_scale,
            out_dim=out_dim,
            in_dim=in_dim,
            block_size_out=block_size_out,
            block_size_in=block_size_in,
        )
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
            w_scale=w_scale_kernel,
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
