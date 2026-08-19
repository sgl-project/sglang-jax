"""MXFP4 (``mxfp4-pack-quantized``) dequantization for sglang-jax.

sglang-jax's quantization support is int8-only; ``float4_e2m1fn`` appears in its tuned-block-size
tables but nothing consumes it. Kimi-K3 ships as::

    "format": "mxfp4-pack-quantized", "num_bits": 4, "group_size": 32,
    "scale_dtype": "torch.uint8"   # e8m0, bitcast to u8
    "strategy": "group", "symmetric": true, "type": "float"

so K3 cannot be loaded at all without this. Ported from tpu-inference's
``layers/common/quantization`` (the JAX-native path), which is pure JAX and carries no
tpu-inference dependencies.

Layout, which is the part that is easy to get wrong:

* **weights** are e2m1 (fp4) *pairs packed into uint8* — two values per byte, so the packed array's
  last dim is ``K/2`` and unpacking doubles it.
* **scales** are e8m0 bitcast to uint8 — a raw 8-bit exponent, NOT a float. Converting them means
  exponent arithmetic (``ldexp``), not a cast.
* one scale covers ``group_size`` (32) contiguous weight elements along the quantized axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# K3's checkpoint group size. The kernel-side re-blocking target is separate and larger; see the
# moe-gmm-fp4 kernel family for why (group_size < mxu_column_size forces an in-VMEM widen).
MXFP4_GROUP_SIZE = 32


def u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    """Unpack an e2m1 (fp4) tensor that was packed two-per-byte into uint8.

    ``bitcast_convert_type`` to a narrower dtype appends a trailing axis of size 2 (the two fp4
    values inside each byte); flattening it into the last dim restores the logical shape, so the
    result's final dimension is twice the input's.
    """
    assert u8_packed_e2m1.dtype == jnp.uint8, u8_packed_e2m1.dtype
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1,))


def e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    """Convert an e8m0 scale (bitcast to uint8) into fp32.

    e8m0 is a bare 8-bit exponent with no sign and no mantissa, so the value is ``2**(u8 - bias)``.
    Casting the uint8 to float would be wrong by many orders of magnitude -- this must be exponent
    arithmetic.
    """
    assert u8.dtype == jnp.uint8, u8.dtype
    minexp = jnp.finfo(jnp.float8_e8m0fnu).minexp
    exponents = u8.astype(jnp.int32) + minexp
    return jnp.ldexp(jnp.ones_like(u8, dtype=jnp.float32), exponents)


def dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | tuple | None = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
    block_size: tuple[int, ...] | None = None,
) -> jax.Array:
    """Multiply a group-quantized tensor by its per-group scale.

    ``axis`` names the quantized axis/axes; ``block_size`` gives the group size per axis and is
    used to pad a trailing partial block so the reshape divides evenly.
    """
    if axis is None:
        axis = list(range(tensor_q.ndim))
    if isinstance(axis, int):
        axis = [axis]
    # Normalize negative indices. Without this, axis=-1 never matches the positional loop
    # counter below, no grouping happens, and the scale reshape fails with a confusing
    # "cannot reshape (4,4) into [4,128]" -- or worse, silently broadcasts if shapes align.
    axis = [a % tensor_q.ndim for a in axis]

    orig_shape = tensor_q.shape
    if block_size is not None:
        pad_width = [[0, 0] for _ in range(tensor_q.ndim)]
        for ax, bs in zip(axis, block_size):
            pad_width[ax][1] = scale.shape[ax] * bs - tensor_q.shape[ax]
        if any(w[1] for w in pad_width):
            tensor_q = jnp.pad(tensor_q, pad_width)

    # split each quantized axis into (n_groups, group_size) so the scale broadcasts over the group
    new_shape, scale_shape = [], []
    for i, dim in enumerate(tensor_q.shape):
        if i in axis:
            n = scale.shape[i]
            new_shape += [n, dim // n]
            scale_shape += [n, 1]
        else:
            new_shape.append(dim)
            scale_shape.append(dim)

    out = tensor_q.reshape(new_shape).astype(jnp.float32) * scale.reshape(scale_shape).astype(
        jnp.float32
    )
    out = out.reshape(tensor_q.shape)
    if block_size is not None and out.shape != orig_shape:
        out = jax.lax.slice(out, [0] * out.ndim, orig_shape)
    return out.astype(out_dtype)


def dequantize_tensor_from_mxfp4_packed(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | tuple = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Full MXFP4 dequantization: unpack fp4 pairs, decode e8m0 scales, apply.

    ``tensor_q`` is uint8 with the quantized axis at half length; ``scale`` is uint8 e8m0 with one
    entry per ``MXFP4_GROUP_SIZE`` weights.
    """
    return dequantize_tensor(
        u8_unpack_e2m1(tensor_q),
        e8m0_to_fp32(scale),
        axis,
        out_dtype,
    )


def is_mxfp4_packed_config(quantization_config: dict | None) -> bool:
    """Recognize K3's ``mxfp4-pack-quantized`` compressed-tensors block.

    The format string lives under ``config_groups.<group>.format``, not at the top level, which is
    why a naive ``quantization_config["format"]`` lookup misses it.
    """
    if not quantization_config:
        return False
    for group in (quantization_config.get("config_groups") or {}).values():
        if str(group.get("format", "")).startswith("mxfp4"):
            return True
    return str(quantization_config.get("format", "")).startswith("mxfp4")
