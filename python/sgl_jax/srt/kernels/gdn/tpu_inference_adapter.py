"""Startup validation and dispatch placeholder for TPU-Inference GDN v3."""

from __future__ import annotations

import os
from collections.abc import Iterable

import jax.numpy as jnp


class GDNPrefillCapabilityError(RuntimeError):
    """An explicitly requested GDN prefill implementation is unsupported."""


def pallas_interpret_enabled() -> bool:
    """Whether Pallas interpret mode explicitly permits local correctness runs."""
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _mesh_devices(mesh) -> tuple[object, ...]:
    devices = getattr(mesh, "devices", None)
    if devices is None:
        return ()
    flat = getattr(devices, "flat", None)
    if flat is not None:
        return tuple(flat)
    if isinstance(devices, Iterable):
        return tuple(devices)
    return (devices,)


def validate_tpu_inference_v3_capability(
    *,
    mesh,
    dtype,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> None:
    """Validate only construction inputs; never probe the global JAX backend."""
    devices = _mesh_devices(mesh)
    mesh_is_tpu = bool(devices) and all(
        getattr(device, "platform", None) == "tpu" for device in devices
    )
    if not mesh_is_tpu and not pallas_interpret_enabled():
        raise GDNPrefillCapabilityError(
            "tpu_inference_v3 requires a TPU mesh unless PALLAS_INTERPRET is enabled."
        )
    if dtype is None:
        raise GDNPrefillCapabilityError("tpu_inference_v3 requires a BF16 activation dtype.")
    if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        raise GDNPrefillCapabilityError("tpu_inference_v3 requires BF16 activation dtype.")
    if min(num_k_heads, num_v_heads, head_k_dim, head_v_dim) <= 0:
        raise GDNPrefillCapabilityError(
            "tpu_inference_v3 requires positive head counts and dimensions."
        )
    if num_v_heads % num_k_heads != 0:
        raise GDNPrefillCapabilityError(
            "tpu_inference_v3 requires num_v_heads to be divisible by num_k_heads."
        )
    if conv_kernel_size < 2:
        raise GDNPrefillCapabilityError(
            "tpu_inference_v3 requires conv_kernel_size >= 2 for its state shape."
        )


def tpu_inference_v3_prefill(*args, **kwargs):
    """Reserved vendor prefill adapter; state-pool adaptation is a later task."""
    del args, kwargs
    raise NotImplementedError(
        "tpu_inference_v3 prefill dispatch requires the later state-pool adapter task."
    )
