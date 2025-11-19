"""Utility functions for ragged paged attention."""

import jax
from jax._src import dtypes


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.bit_width(dtype)
    return 32 // bits


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


def get_device_name(num_devices: int | None = None):
    name = " ".join(jax.devices()[0].device_kind.split()[:2])
    if num_devices is not None:
        name += f"-{num_devices}"
    return name
