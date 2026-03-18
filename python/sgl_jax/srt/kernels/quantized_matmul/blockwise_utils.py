# SPDX-License-Identifier: Apache-2.0
"""Utilities for the TPU blockwise quantized matmul kernel.

This module provides lazy-loading and safe parameter resolution for the
TPU blockwise quantized matmul kernel.  The overall call flow is:

    kernel.py (xla_quantized_matmul_local)
        |
        |-- get_blockwise_kernel()                   # lazy-load the kernel function
        |-- should_use_blockwise_kernel()            # narrow-N shape guard
        |-- convert_block_scale_to_kernel_layout()   # scale format conversion
        |-- get_safe_blockwise_tuned_value()          # resolve TPU tile sizes
        |       |
        |       |-- _get_blockwise_tuning_api()          # lazy-load tuning tables
        |       |-- _iter_blockwise_tuned_candidates()   # find best match in table
        |       +-- clamp / snap sizes to local matrix   # ensure launch safety
        |
        +-- blockwise_kernel(...)                    # invoke the kernel

The tuned value resolution follows a 3-tier fallback strategy:
  1. Look up a compatible entry from a pre-computed tuning table (best).
  2. Query ``get_tuned_block_sizes()`` API at runtime.
  3. Fall back to a conservative default seed ``(128, 128, 128, 1)``.

After obtaining a seed, the final tile sizes are clamped and snapped to
the actual local matrix dimensions so that the kernel launch is always valid.
"""

import functools
import importlib
import logging
import math
import re

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded blockwise kernel and tuning state.
#
# These globals are populated on first access via get_blockwise_kernel()
# and _get_blockwise_tuning_api().  The ``_TRIED_LOADING_*`` flags ensure
# we only attempt the import once per process, even if it fails.
# ---------------------------------------------------------------------------

# The loaded kernel callable, or None if unavailable.
_BLOCKWISE_KERNEL = None
_TRIED_LOADING_BLOCKWISE_KERNEL = False

# Tuning metadata: the namedtuple class, the runtime query function, and
# the static lookup table from the tuned_block_sizes module.
_BLOCKWISE_TUNED_VALUE_CLS = None  # e.g. TunedValue namedtuple class
_BLOCKWISE_GET_TUNED_BLOCK_SIZES = None  # e.g. get_tuned_block_sizes()
_BLOCKWISE_TUNED_BLOCK_SIZES = None  # e.g. TUNED_BLOCK_SIZES dict
_TRIED_LOADING_BLOCKWISE_TUNING = False


def get_blockwise_kernel():
    """Lazily load the blockwise kernel implementation."""
    global _BLOCKWISE_KERNEL, _TRIED_LOADING_BLOCKWISE_KERNEL

    if _TRIED_LOADING_BLOCKWISE_KERNEL:
        return _BLOCKWISE_KERNEL
    _TRIED_LOADING_BLOCKWISE_KERNEL = True

    try:
        package = __package__ or "sgl_jax.srt.kernels.quantized_matmul"
        module = importlib.import_module(f"{package}.quantized_matmul_kernels")
        _BLOCKWISE_KERNEL = getattr(module, "quantized_matmul", None)
    except Exception:
        logger.debug("Failed to import blockwise quantized matmul kernel.", exc_info=True)
        _BLOCKWISE_KERNEL = None
    return _BLOCKWISE_KERNEL


def _get_blockwise_tuning_api():
    """Lazily load tuned-size helpers for the blockwise kernel."""
    global _BLOCKWISE_TUNED_VALUE_CLS
    global _BLOCKWISE_GET_TUNED_BLOCK_SIZES
    global _BLOCKWISE_TUNED_BLOCK_SIZES
    global _TRIED_LOADING_BLOCKWISE_TUNING

    if _TRIED_LOADING_BLOCKWISE_TUNING:
        return (
            _BLOCKWISE_TUNED_VALUE_CLS,
            _BLOCKWISE_GET_TUNED_BLOCK_SIZES,
            _BLOCKWISE_TUNED_BLOCK_SIZES,
        )
    _TRIED_LOADING_BLOCKWISE_TUNING = True

    try:
        package = __package__ or "sgl_jax.srt.kernels.quantized_matmul"
        module = importlib.import_module(f"{package}.quantized_matmul_kernels.tuned_block_sizes")
        _BLOCKWISE_TUNED_VALUE_CLS = getattr(module, "TunedValue", None)
        _BLOCKWISE_GET_TUNED_BLOCK_SIZES = getattr(module, "get_tuned_block_sizes", None)
        _BLOCKWISE_TUNED_BLOCK_SIZES = getattr(module, "TUNED_BLOCK_SIZES", None)
    except Exception:
        logger.debug("Failed to import blockwise tuning metadata.", exc_info=True)
        _BLOCKWISE_TUNED_VALUE_CLS = None
        _BLOCKWISE_GET_TUNED_BLOCK_SIZES = None
        _BLOCKWISE_TUNED_BLOCK_SIZES = None

    return (
        _BLOCKWISE_TUNED_VALUE_CLS,
        _BLOCKWISE_GET_TUNED_BLOCK_SIZES,
        _BLOCKWISE_TUNED_BLOCK_SIZES,
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

    The TPU blockwise kernel is more reliable with tile sizes that are
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
        # Hard filter: must match TPU version and weight dtype exactly.
        if getattr(key, "tpu_version", tpu_version) != tpu_version:
            continue
        if key.w_q_dtype != w_q_dtype_name:
            continue
        if key.x_q_dtype not in compatible_x_dtype_names:
            continue

        # Soft ranking: tuple comparison gives lexicographic priority to
        # (1) exact activation dtype match, then (2) closest n_in, then
        # (3) closest n_batch, then (4) closest n_out.
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


def should_use_blockwise_kernel(
    *,
    out_dim: int,
    block_size_out: int,
) -> bool:
    """Guard known-bad narrow-N TPU blockwise cases.

    When a tensor-parallel column shard collapses to a single output block
    (for example local N=128 with block_size_out=128), the TPU blockwise
    kernel can produce NaNs on Qwen3-MoE k/v projections.
    """
    return out_dim > block_size_out


def expand_block_scale(
    scale_2d: jax.Array,
    n_out: int,
    block_size_out: int,
    channel_to_block: jax.Array | None = None,
) -> jax.Array:
    """Expand a 2D block scale to the 3D kernel-ready layout.

    This should be called **once at init / weight-loading time**, not on
    every inference step.

    Args:
        scale_2d: Compact block scale ``[out_blocks, in_blocks]``.
        n_out: Total number of output channels.
        block_size_out: Uniform block size along the output dimension.
        channel_to_block: Optional ``[n_out]`` int array that maps each
            output channel to its block index.  When ``None`` (the default),
            a uniform mapping ``channel // block_size_out`` is used.

            .. note::

                For non-uniform block quant (e.g. per-head boundaries),
                pass an explicit ``channel_to_block`` index array.

    Returns:
        Kernel-ready scale ``[in_blocks, 1, n_out]``.
    """
    if channel_to_block is not None:
        # Non-uniform block mapping (e.g., per-head block quant).
        scale_per_channel = scale_2d[channel_to_block]  # [n_out, in_blocks]
    else:
        # Standard uniform block quant: repeat each block's scale to its
        # constituent channels, then truncate to the actual output size.
        scale_per_channel = jnp.repeat(scale_2d, repeats=block_size_out, axis=0)[:n_out]

    # Transpose to [in_blocks, n_out] and insert the singleton dim expected
    # by the blockwise kernel: [in_blocks, 1, n_out].
    return jnp.transpose(scale_per_channel, (1, 0))[:, None, :]


def convert_block_scale_to_kernel_layout(
    w_scale: jax.Array,
    out_dim: int,
    in_dim: int,
    block_size_out: int,
    block_size_in: int,
) -> jax.Array:
    """Convert our block-scale layout to the TPU kernel layout.

    .. deprecated::
        Use :func:`expand_block_scale` at init time instead.  This function
        is kept only for internal / test compatibility.
    """
    needed_out_blocks = math.ceil(out_dim / block_size_out)
    needed_in_blocks = math.ceil(in_dim / block_size_in)

    if w_scale.shape[0] < needed_out_blocks or w_scale.shape[1] < needed_in_blocks:
        raise ValueError(
            "Block scale shape is smaller than required by weight shape: "
            f"w_scale.shape={w_scale.shape}, needed=({needed_out_blocks}, {needed_in_blocks})."
        )

    scale_2d = w_scale[:needed_out_blocks, :needed_in_blocks]
    return expand_block_scale(scale_2d, out_dim, block_size_out)


def get_safe_blockwise_tuned_value(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    block_size_in: int,
):
    """Build a safe tuned value for the blockwise kernel on TPU.

    Returns a ``TunedValue(batch_block_size, out_block_size, in_block_size,
    n_lane_multiplier)`` whose tile sizes are guaranteed to be valid for the
    current local matrix shape ``(n_batch, n_out, n_in)``.

    Resolution strategy (3-tier fallback):
      1. Search the pre-computed tuning table for a compatible entry.
      2. Query ``get_tuned_block_sizes()`` API at runtime.
      3. Use a conservative default seed ``(128, 128, 128, 1)``.

    After obtaining a seed, every dimension is clamped and aligned to the
    local matrix so the kernel launch never exceeds the actual tensor bounds.
    """
    # --- Tier 0: load tuning metadata (lazy, once per process) ---
    tuned_value_cls, get_tuned_block_sizes, tuned_block_sizes = _get_blockwise_tuning_api()
    if tuned_value_cls is None:
        return None

    tuned = None
    # --- Tier 1: look up from pre-computed tuning table ---
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
    # --- Tier 2: query the tuning API at runtime ---
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
            logger.debug("Failed to query tuned block sizes for blockwise kernel.", exc_info=True)
            tuned = None
    if tuned is None:
        # --- Tier 3: conservative default seed ---
        # Final sizes are still clamped to the current local shape below,
        # so this does not force a fixed launch shape.
        tuned = tuned_value_cls(128, 128, 128, 1)

    # ------------------------------------------------------------------
    # Clamp and align tile sizes to the actual local matrix dimensions.
    #
    # compute_tile_n = 256 * n_lane_multiplier is the TPU MXU output tile
    # width; out_block_size must be a power-of-two multiple of this to
    # avoid hardware under-utilisation and potential NaN issues.
    # ------------------------------------------------------------------
    n_lane_multiplier = max(1, int(tuned.n_lane_multiplier))
    compute_tile_n = 256 * n_lane_multiplier

    # batch: simply cap to actual batch size.
    batch_block_size = max(1, min(int(tuned.batch_block_size), int(n_batch)))

    # out (N): round up to compute_tile_n, cap to matrix N, then snap to
    # nearest power-of-two multiple for TPU alignment.
    out_block_size = _next_multiple(max(int(tuned.out_block_size), compute_tile_n), compute_tile_n)
    out_block_size = min(out_block_size, _floor_multiple(int(n_out), compute_tile_n))
    out_block_size = _nearest_power_of_two_multiple(
        out_block_size,
        compute_tile_n,
        _floor_multiple(int(n_out), compute_tile_n),
    )
    # in (K): must be a multiple of the quantization block_size_in so that
    # each tile boundary aligns with a scale-block boundary.
    in_block_size = max(int(tuned.in_block_size), int(block_size_in))
    in_block_size = _next_multiple(in_block_size, int(block_size_in))
    in_block_size = min(in_block_size, _floor_multiple(int(n_in), int(block_size_in)))

    return tuned_value_cls(batch_block_size, out_block_size, in_block_size, n_lane_multiplier)
