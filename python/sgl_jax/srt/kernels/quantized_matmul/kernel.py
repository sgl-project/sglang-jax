# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import logging
import math

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.blockwise_3rd_utils import (
    convert_block_scale_to_3rd_layout,
    get_blockwise_3rd_kernel,
    get_perchannel_3rd_kernel,
    get_safe_blockwise_tuned_value,
    should_use_3rd_party_blockwise_kernel,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple

logger = logging.getLogger(__name__)


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
        block_size_out, block_size_in = int(weight_block_size[0]), int(
            weight_block_size[1]
        )
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


def _expand_block_scales_to_weight_shape(
    w_scale: jax.Array,
    out_dim: int,
    in_dim: int,
    block_size_out: int,
    block_size_in: int,
) -> jax.Array:
    """Expand compact block scales to full ``[n_out, n_in]`` layout.

    This is only used by the local dequantized fallback path. The third-party
    TPU kernel consumes compact scales directly after a dedicated layout
    conversion.
    """
    out_blocks, in_blocks = w_scale.shape

    row_idx = jnp.arange(out_dim, dtype=jnp.int32) // jnp.int32(block_size_out)
    col_idx = jnp.arange(in_dim, dtype=jnp.int32) // jnp.int32(block_size_in)
    row_idx = jnp.clip(row_idx, 0, out_blocks - 1)
    col_idx = jnp.clip(col_idx, 0, in_blocks - 1)
    return w_scale[row_idx[:, None], col_idx[None, :]]


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
    act_quant_dtype = (
        w_q.dtype if activation_quant_dtype is None else activation_quant_dtype
    )

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

        # Prefer third-party blockwise kernel on TPU. Keep the local dequantized
        # path as fallback for non-TPU (e.g., CPU/GPU) or unavailable environments.
        # Fallback to pure JAX dequantization happens if:
        # 1. We are not on TPU (`jax.default_backend() != "tpu"`).
        # 2. The 3rd party kernel failed to load (`blockwise_3rd_kernel is None`).
        # 3. Narrow-N shapes known to cause NaNs in 3rd party kernel are detected (`should_use_3rd_party_blockwise_kernel` returns False).
        out = None
        blockwise_3rd_kernel = get_blockwise_3rd_kernel()
        if (
            jax.default_backend() == "tpu"
            and blockwise_3rd_kernel is not None
            and should_use_3rd_party_blockwise_kernel(
                out_dim=int(out_dim),
                block_size_out=int(block_size_out),
            )
        ):
            try:
                w_scale_3rd = convert_block_scale_to_3rd_layout(
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
                out = blockwise_3rd_kernel(
                    x=x,
                    w_q=w_q,
                    w_scale=w_scale_3rd,
                    block_size=block_size_in,
                    x_q_dtype=x_q_dtype,
                    tuned_value=tuned_value,
                )
            except Exception:
                logger.warning(
                    "Falling back from third-party blockwise kernel to local dequant path.",
                    exc_info=True,
                )
                out = None

        if out is None:
            scale_expanded = _expand_block_scales_to_weight_shape(
                w_scale=w_scale,
                out_dim=out_dim,
                in_dim=in_dim,
                block_size_out=block_size_out,
                block_size_in=block_size_in,
            )
            w_dequant = w_q.astype(compute_dtype) * scale_expanded.astype(compute_dtype)

            if quantize_activation:
                x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
                lhs = x_q.astype(compute_dtype)
            else:
                lhs = x.astype(compute_dtype)

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
        # Prefer third-party per-channel kernel on TPU. It accumulates in int32
        # across the full K dimension for better precision, then applies the 1D
        # scale at the end. Falls back to pure JAX if unavailable.
        out = None
        perchannel_3rd_kernel = get_perchannel_3rd_kernel()
        if jax.default_backend() == "tpu" and perchannel_3rd_kernel is not None:
            try:
                x_q_dtype = act_quant_dtype if quantize_activation else x.dtype
                out = perchannel_3rd_kernel(
                    x=x,
                    w_q=w_q,
                    w_scale=w_scale,
                    x_q_dtype=x_q_dtype,
                )
            except Exception:
                logger.warning(
                    "Falling back from third-party per-channel kernel to JAX path.",
                    exc_info=True,
                )
                out = None

        if out is None:
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
                out = out.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(
                    compute_dtype
                )

    out = out.astype(out_dtype)
    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, axis_name=reduce_axis)

    return out
