"""int4 per-channel routed-expert weights: quantization math and checkpoint conventions.

Checkpoint layout produced by ``sgl_jax.tools.convert_moe_experts_int4`` for
each routed expert projection ``<prefix>.{gate,up,down}_proj``:

* ``<prefix>.weight``       int8 ``[out, in]`` holding the int4 codes ``[-7, 7]``
                            (one code per byte; safetensors has no 4-bit dtype)
* ``<prefix>.weight_scale`` float32 ``[out, 1]`` per-output-channel scale

so that ``dequant = weight.astype(f32) * weight_scale``. The loader stacks the
experts to ``[E, in, out]``, casts the codes to ``jnp.int4`` (halving HBM) and
promotes the scale to the GMM layout ``[E, 1, 1, out]``; ``gmm_v2`` then runs
the int4 weights with fp8 activations (W4A8) on chips without an int8 MXU.

The rounding matches ``quantize_tensor(jnp.int4, ..., axis=1)`` exactly:
symmetric ``absmax / 7`` scale, round-half-to-even, clip to ``[-7, 7]``.
"""

from __future__ import annotations

import math
import re

import jax.numpy as jnp
import numpy as np

INT4_MAX = 7
INT4_STORAGE_DTYPE = np.int8
INT4_SCALE_SUFFIX = ".weight_scale"
FP8_BLOCK_SCALE_SUFFIX = ".weight_scale_inv"

ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj))\.weight$"
)


def routed_expert_scale_suffix(quant_config) -> str:
    """HF sidecar suffix for routed-expert scales under this quantization config."""
    dtype = getattr(quant_config, "moe_weight_dtype", None)
    if dtype is not None and jnp.dtype(dtype) == jnp.dtype(jnp.int4):
        return INT4_SCALE_SUFFIX
    return FP8_BLOCK_SCALE_SUFFIX


def match_routed_expert_weight(key: str) -> tuple[str, int] | None:
    """Returns ``(tensor prefix, layer index)`` for a routed expert weight key, else None."""
    m = ROUTED_EXPERT_WEIGHT_RE.match(key)
    if m is None:
        return None
    return m.group("prefix"), int(m.group("layer"))


def dequantize_fp8_block(
    weight: np.ndarray, scale_inv: np.ndarray, block_size: tuple[int, int] = (128, 128)
) -> np.ndarray:
    """FP8 ``[out, in]`` weight with ``[out_blocks, in_blocks]`` block scales -> f32 ``[out, in]``."""
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(f"expected 2-D weight/scale, got {weight.shape} / {scale_inv.shape}")
    n, k = weight.shape
    bn, bk = block_size
    expected = (math.ceil(n / bn), math.ceil(k / bk))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"scale_inv shape {scale_inv.shape} does not match weight {weight.shape} "
            f"with block size {block_size} (expected {expected})"
        )
    scale = np.repeat(scale_inv.astype(np.float32), bn, axis=0)[:n]
    scale = np.repeat(scale, bk, axis=1)[:, :k]
    return weight.astype(np.float32) * scale


def quantize_int4_per_channel(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f32 ``[out, in]`` -> (int8 codes ``[out, in]`` in ``[-7, 7]``, f32 scale ``[out, 1]``).

    Mirrors ``quantize_tensor(jnp.int4, w, axis=1)``: an all-zero row keeps a
    zero scale and zero codes.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {w.shape}")
    abs_max = np.max(np.abs(w), axis=1, keepdims=True)
    scale = abs_max / INT4_MAX
    scale_safe = np.where(scale == 0, 1.0, scale).astype(np.float32)
    codes = np.clip(np.rint(w / scale_safe), -INT4_MAX, INT4_MAX).astype(INT4_STORAGE_DTYPE)
    return codes, scale.astype(np.float32)


def dequantize_int4_per_channel(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of ``quantize_int4_per_channel`` (f32 ``[out, in]``)."""
    return codes.astype(np.float32) * np.asarray(scale, dtype=np.float32).reshape(-1, 1)


def convert_fp8_block_expert_to_int4(
    weight: np.ndarray, scale_inv: np.ndarray, block_size: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray]:
    """FP8 block-scaled expert projection -> (int4 codes as int8, per-channel f32 scale)."""
    return quantize_int4_per_channel(dequantize_fp8_block(weight, scale_inv, block_size))
