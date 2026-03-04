# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import functools
import importlib
import logging
import math
import re

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple

logger = logging.getLogger(__name__)


_BLOCKWISE_3RD_KERNEL = None
_TRIED_LOADING_BLOCKWISE_3RD_KERNEL = False
_BLOCKWISE_3RD_TUNED_VALUE_CLS = None
_BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES = None
_BLOCKWISE_3RD_TUNED_BLOCK_SIZES = None
_TRIED_LOADING_BLOCKWISE_3RD_TUNING = False


def _get_blockwise_3rd_kernel():
    """Lazily load the third-party blockwise kernel implementation."""
    global _BLOCKWISE_3RD_KERNEL, _TRIED_LOADING_BLOCKWISE_3RD_KERNEL

    if _TRIED_LOADING_BLOCKWISE_3RD_KERNEL:
        return _BLOCKWISE_3RD_KERNEL
    _TRIED_LOADING_BLOCKWISE_3RD_KERNEL = True

    try:
        package = __package__ or "sgl_jax.srt.kernels.quantized_matmul"
        module = importlib.import_module(f"{package}.3rd_quantized_matmul")
        _BLOCKWISE_3RD_KERNEL = getattr(module, "quantized_matmul", None)
    except Exception:
        logger.debug(
            "Failed to import third-party blockwise quantized matmul kernel.", exc_info=True
        )
        _BLOCKWISE_3RD_KERNEL = None
    return _BLOCKWISE_3RD_KERNEL


def _get_blockwise_3rd_tuning_api():
    """Lazily load third-party tuned-size helpers for blockwise kernel."""
    global _BLOCKWISE_3RD_TUNED_VALUE_CLS
    global _BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES
    global _BLOCKWISE_3RD_TUNED_BLOCK_SIZES
    global _TRIED_LOADING_BLOCKWISE_3RD_TUNING

    if _TRIED_LOADING_BLOCKWISE_3RD_TUNING:
        return (
            _BLOCKWISE_3RD_TUNED_VALUE_CLS,
            _BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES,
            _BLOCKWISE_3RD_TUNED_BLOCK_SIZES,
        )
    _TRIED_LOADING_BLOCKWISE_3RD_TUNING = True

    try:
        package = __package__ or "sgl_jax.srt.kernels.quantized_matmul"
        module = importlib.import_module(f"{package}.3rd_quantized_matmul.tuned_block_sizes")
        _BLOCKWISE_3RD_TUNED_VALUE_CLS = getattr(module, "TunedValue", None)
        _BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES = getattr(module, "get_tuned_block_sizes", None)
        _BLOCKWISE_3RD_TUNED_BLOCK_SIZES = getattr(module, "TUNED_BLOCK_SIZES", None)
    except Exception:
        logger.debug("Failed to import third-party blockwise tuning metadata.", exc_info=True)
        _BLOCKWISE_3RD_TUNED_VALUE_CLS = None
        _BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES = None
        _BLOCKWISE_3RD_TUNED_BLOCK_SIZES = None

    return (
        _BLOCKWISE_3RD_TUNED_VALUE_CLS,
        _BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES,
        _BLOCKWISE_3RD_TUNED_BLOCK_SIZES,
    )


def _next_multiple(x: int, m: int) -> int:
    """Round ``x`` up to the next multiple of ``m``."""
    if m <= 0:
        return x
    return ((x + m - 1) // m) * m


def _floor_multiple(x: int, m: int) -> int:
    """Round ``x`` down to a positive multiple of ``m``."""
    if m <= 0:
        return x
    return max(m, (x // m) * m)


def _nearest_power_of_two_multiple(x: int, base: int, upper_bound: int) -> int:
    """Snap ``x`` to a nearby power-of-two multiple of ``base``.

    The imported TPU blockwise kernel is more reliable with tile sizes that are
    aligned to the compute tile width. This helper keeps the candidate near the
    requested value while respecting the local matrix bound.
    """
    if base <= 0:
        return x

    x = max(base, x)
    units = max(1, x // base)
    lower_units = 1 << (units.bit_length() - 1)
    upper_units = lower_units if lower_units == units else lower_units << 1

    def _candidate(units_value: int) -> int:
        return units_value * base

    lower = _candidate(lower_units)
    upper = _candidate(upper_units)
    candidates = [value for value in (lower, upper) if value <= upper_bound]
    if not candidates:
        candidates = [lower]

    return min(candidates, key=lambda value: (abs(value - x), -value))


@functools.lru_cache(maxsize=1)
def _get_current_tpu_version() -> int:
    """Return the current TPU major version, or ``-1`` when unavailable."""
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        return -1
    match = re.match(r"^TPU[^\d]*(\d+)", kind)
    if match is None:
        return -1
    return int(match.group(1))


def _iter_blockwise_tuned_candidates(
    tuned_block_sizes: dict | None,
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    tpu_version: int,
):
    """Return compatible tuned-size candidates ordered by closeness.

    We first filter by TPU version and weight dtype, then rank surviving
    entries by activation dtype compatibility and distance from the requested
    ``(n_batch, n_out, n_in)`` shape.
    """
    if not tuned_block_sizes:
        return []

    x_q_dtype_name = jnp.dtype(x_q_dtype).name
    w_q_dtype_name = jnp.dtype(w_q_dtype).name
    compatible_x_dtype_names = [x_q_dtype_name]
    if jnp.issubdtype(w_q_dtype, jnp.integer) and x_q_dtype_name != "int8":
        compatible_x_dtype_names.append("int8")

    candidates = []
    for key, value in tuned_block_sizes.items():
        if getattr(key, "tpu_version", tpu_version) != tpu_version:
            continue
        if key.w_q_dtype != w_q_dtype_name:
            continue
        if key.x_q_dtype not in compatible_x_dtype_names:
            continue

        score = (
            compatible_x_dtype_names.index(key.x_q_dtype),
            key.n_in != n_in,
            abs(key.n_in - n_in),
            key.n_batch != n_batch,
            abs(key.n_batch - n_batch),
            key.n_out != n_out,
            abs(key.n_out - n_out),
        )
        candidates.append((score, value))

    candidates.sort(key=lambda item: item[0])
    return [value for _, value in candidates]


def _get_safe_blockwise_tuned_value(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    block_size_in: int,
):
    """Build a safe tuned value for third-party blockwise kernel on TPU."""
    tuned_value_cls, get_tuned_block_sizes, tuned_block_sizes = _get_blockwise_3rd_tuning_api()
    if tuned_value_cls is None:
        return None

    tuned = None
    compatible_candidates = _iter_blockwise_tuned_candidates(
        tuned_block_sizes=tuned_block_sizes,
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q_dtype,
        tpu_version=_get_current_tpu_version(),
    )
    if compatible_candidates:
        tuned = compatible_candidates[0]
    elif get_tuned_block_sizes is not None:
        try:
            tuned = get_tuned_block_sizes(
                n_batch=n_batch,
                n_out=n_out,
                n_in=n_in,
                x_q_dtype=jnp.dtype(x_q_dtype).name,
                w_q_dtype=jnp.dtype(w_q_dtype).name,
            )
        except Exception:
            logger.debug(
                "Failed to query tuned block sizes from third-party kernel.", exc_info=True
            )
            tuned = None
    if tuned is None:
        # Last-resort seed. Final sizes are still clamped to the current local
        # shape below, so this does not force a fixed launch shape.
        tuned = tuned_value_cls(128, 128, 128, 1)

    n_lane_multiplier = max(1, int(tuned.n_lane_multiplier))
    compute_tile_n = 256 * n_lane_multiplier

    batch_block_size = max(1, min(int(tuned.batch_block_size), int(n_batch)))
    out_block_size = _next_multiple(max(int(tuned.out_block_size), compute_tile_n), compute_tile_n)
    out_block_size = min(out_block_size, _floor_multiple(int(n_out), compute_tile_n))
    out_block_size = _nearest_power_of_two_multiple(
        out_block_size,
        compute_tile_n,
        _floor_multiple(int(n_out), compute_tile_n),
    )
    in_block_size = max(int(tuned.in_block_size), int(block_size_in))
    in_block_size = _next_multiple(in_block_size, int(block_size_in))
    in_block_size = min(in_block_size, _floor_multiple(int(n_in), int(block_size_in)))

    return tuned_value_cls(batch_block_size, out_block_size, in_block_size, n_lane_multiplier)


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
        block_size_out = math.ceil(out_dim / out_blocks)
        block_size_in = math.ceil(in_dim / in_blocks)

    if block_size_out <= 0 or block_size_in <= 0:
        raise ValueError(
            f"Invalid block sizes: block_size_out={block_size_out}, block_size_in={block_size_in}."
        )
    return block_size_out, block_size_in


def _should_use_3rd_party_blockwise_kernel(
    *,
    out_dim: int,
    block_size_out: int,
) -> bool:
    """Guard known-bad narrow-N TPU blockwise cases.

    When a tensor-parallel column shard collapses to a single output block
    (for example local N=128 with block_size_out=128), the third-party TPU
    blockwise kernel can produce NaNs on Qwen3-MoE k/v projections. The local
    dequantized fallback remains numerically stable for the same inputs.
    """
    return out_dim > block_size_out


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


def _convert_block_scale_to_3rd_layout(
    w_scale: jax.Array,
    out_dim: int,
    in_dim: int,
    block_size_out: int,
    block_size_in: int,
) -> jax.Array:
    """Convert our block-scale layout to the imported TPU kernel layout.

    The layer/checkpoint-facing format is ``[out_blocks, in_blocks]``.
    The third-party kernel expects one scale per input block and output
    channel: ``[in_blocks, 1, n_out]``. We therefore replicate each output
    block's scale across the channels inside that block before transposing.
    """
    needed_out_blocks = math.ceil(out_dim / block_size_out)
    needed_in_blocks = math.ceil(in_dim / block_size_in)

    if w_scale.shape[0] < needed_out_blocks or w_scale.shape[1] < needed_in_blocks:
        raise ValueError(
            "Block scale shape is smaller than required by weight shape: "
            f"w_scale.shape={w_scale.shape}, needed=({needed_out_blocks}, {needed_in_blocks})."
        )

    # Third-party kernel expects per-output-channel scales for each input block.
    # Replicate each output-block scale value across channels in that output block.
    scale_2d = w_scale[:needed_out_blocks, :needed_in_blocks]
    scale_per_out = jnp.repeat(scale_2d, repeats=block_size_out, axis=0)[:out_dim, :]
    return jnp.transpose(scale_per_out, (1, 0))[:, None, :]


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
    compute_dtype = out_dtype if compute_dtype is None else compute_dtype
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

        # Prefer third-party blockwise kernel on TPU. Keep the local dequantized
        # path as fallback for non-TPU / unavailable environments.
        out = None
        blockwise_3rd_kernel = _get_blockwise_3rd_kernel()
        if (
            jax.default_backend() == "tpu"
            and blockwise_3rd_kernel is not None
            and _should_use_3rd_party_blockwise_kernel(
                out_dim=int(out_dim),
                block_size_out=int(block_size_out),
            )
        ):
            try:
                w_scale_3rd = _convert_block_scale_to_3rd_layout(
                    w_scale=w_scale,
                    out_dim=out_dim,
                    in_dim=in_dim,
                    block_size_out=block_size_out,
                    block_size_in=block_size_in,
                )
                x_q_dtype = act_quant_dtype if quantize_activation else x.dtype
                tuned_value = _get_safe_blockwise_tuned_value(
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
                logger.debug(
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

    # Sum partial results across devices (single all-reduce)
    if reduce_axis is not None:
        out = lax.psum(out, axis_name=reduce_axis)

    return out.astype(out_dtype)
