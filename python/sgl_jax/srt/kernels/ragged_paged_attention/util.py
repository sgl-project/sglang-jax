"""Utility functions for ragged paged attention."""

import jax
import jax.numpy as jnp
from jax._src import dtypes

from sgl_jax.srt.utils.jax_utils import get_device_kind


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = get_device_kind()
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]

    # Extract version number after "TPU"
    # Supports formats: "TPU v5", "TPU v6", "TPU7x", "TPU6e", etc.
    if kind.startswith("TPU v"):
        # Format: "TPU v7x" or "TPU v5"
        version_str = kind[len("TPU v") :]
    elif kind.startswith("TPU"):
        # Format: "TPU7x" or "TPU5"
        version_str = kind[len("TPU") :]
    else:
        raise ValueError(f"Unexpected TPU device kind format: {kind}")

    # Extract the numeric part (first consecutive digits)
    version_digits = ""
    for char in version_str:
        if char.isdigit():
            version_digits += char
        else:
            break

    if not version_digits:
        raise ValueError(f"Could not extract TPU version from: {kind}")

    return int(version_digits)


def get_dtype_bitwidth(dtype):
    return dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype)


def next_power_of_2(x):
    if x <= 0:
        return 1
    x = x - 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def quantize_to_kv_dtype(
    x: jax.Array,
    dtype: jnp.dtype,
    scale: jax.Array | float | None = None,
) -> jax.Array:
    """Quantize activation tensor to the target KV cache data type.

    If dtype is an FP8 type (e.g. float8_e4m3fn or float8_e5m2), values are scaled
    (divided by scale if provided and != 1.0) and clipped to the representable range in float32
    before downcasting, avoiding NaN overflow on out-of-range values.
    Otherwise, casts directly to dtype (or no-op when already dtype and unscaled).
    """
    is_unscaled = scale is None or scale == 1.0
    if x.dtype == dtype and is_unscaled:
        return x

    target_dtype = jnp.dtype(dtype)
    if target_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2):
        finfo = jnp.finfo(target_dtype)
        x_f32 = x.astype(jnp.float32)
        if not is_unscaled:
            x_f32 = x_f32 / scale
        return jnp.clip(x_f32, float(finfo.min), float(finfo.max)).astype(target_dtype)

    if not is_unscaled:
        return (x / scale).astype(target_dtype)
    return x.astype(target_dtype)
