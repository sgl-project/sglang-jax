import re

import jax
import jax.numpy as jnp


def is_tpu() -> bool:
    return "TPU" in jax.devices()[0].device_kind


def tpu_kind() -> str:
    """Query identification string for the currently attached TPU."""
    return jax.devices()[0].device_kind


# Ex: VPU v5; TPU v5 lite; TPU7x
_TPU_KIND_PATTERN = re.compile(r"TPU(?: v)?(\d+)")


def tpu_generation() -> int:
    """Generation number of the currently attached TPU."""
    my_tpu_kind = tpu_kind()
    if version := _TPU_KIND_PATTERN.match(my_tpu_kind):
        return int(version[1])
    raise NotImplementedError(
        f"Only TPU devices are supported: Invalid device_kind: '{my_tpu_kind}'"
    )


def supports_bfloat16_matmul() -> bool:
    """Does the currently attached CPU support bfloat16 inputs?"""
    return not is_tpu() or tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
    if dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
    """A type to which both input should be adapted to before dot product."""
    # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
    # input precision, we need to convert bf16 argument to fp32 beforehand.
    if supports_bfloat16_matmul() and lhs.dtype == jnp.bfloat16 and rhs.dtype == jnp.bfloat16:
        return jnp.bfloat16
    else:
        return jnp.float32
