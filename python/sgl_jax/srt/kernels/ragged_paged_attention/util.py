"""Utility functions for ragged paged attention."""

import jax
from jax._src import dtypes


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
    kind = jax.devices()[0].device_kind
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


def get_device_name(num_devices: int | None = None):
    name = " ".join(jax.devices()[0].device_kind.split()[:2])
    if num_devices is not None:
        name += f"-{num_devices}"
    return name
