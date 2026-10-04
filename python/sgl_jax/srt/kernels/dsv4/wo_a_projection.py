# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused inverse-RoPE + ``wo_a`` output projection for DSv4 attention (Pallas TPU).

Ported from tpu-inference ``kernels/experimental/deepseek_v4/o_projection.py``
(``wo_a_projection``). One kernel replaces, per layer: the interleaved inverse RoPE
on the trailing ``rope_head_dim`` features of every head (three constant matmuls
plus glue in XLA), the ``[T, H, D] -> [T, G, H/G*D]`` regroup, and the grouped
``tgd,gdr->tgr`` contraction. Each grid step handles one output group of eight
heads, so the group's ``8 * head_dim`` reduction is one MXU pass.

Semantics (per token ``t``, group ``g``)::

    x_roped = rope^-1(x[t, g*8:(g+1)*8, :])            # trailing rotary lanes only
    out[t, g*R:(g+1)*R] = x_roped.reshape(8*D) @ wo_a[:, g*R:(g+1)*R] (* scale)

``wo_a`` may be bf16 (no scale) or fp8 with a per-output-column scale; with fp8
weights the activations are quantized to fp8 per row inside the kernel.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128
SUBLANE = 8
FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _largest_divisor(x: int, cap: int) -> int:
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _lane_gather(operand, indices):
    """``out[b..., i] = operand[b..., indices[i]]``; constant indices -> lane shuffle."""
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        start_index_map=(rank - 1,),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1,) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _rotate_gptj(x):
    """``[-x1, x0, -x3, x2, ...]`` along the lanes: partner lane ``i ^ 1``, sign on even."""
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    return _lane_gather(x, lane ^ 1) * sign


def widen_cos_sin(cos, sin, *, rope_head_dim: int, inverse: bool):
    """``[T, rope_head_dim//2]`` cos/sin -> ``[T, 2*LANE]`` f32 ``[cos_lanes | sin_lanes]``.

    The kernel rotates the last ``LANE`` lanes of every head; the roped channels are
    the tail of that block and NoPE lanes get the identity (cos 1, sin 0). Channels
    ``2k`` and ``2k+1`` share frequency ``k`` (interleaved pairs), matching
    `dsv4.compressor.interleaved_rope`. ``inverse`` negates sin.
    """
    if rope_head_dim % 2 or rope_head_dim > LANE:
        raise ValueError(f"rope_head_dim must be even and <= {LANE}, got {rope_head_dim}")
    cos = jnp.asarray(cos, jnp.float32)
    sin = jnp.asarray(sin, jnp.float32)
    channel = jnp.arange(LANE) - (LANE - rope_head_dim)
    is_rot = channel >= 0
    freq = jnp.maximum(channel, 0) // 2
    cos_l = jnp.where(is_rot[None, :], cos[:, freq], 1.0)
    sin_l = jnp.where(is_rot[None, :], sin[:, freq], 0.0)
    if inverse:
        sin_l = -sin_l
    return jnp.concatenate([cos_l, sin_l], axis=-1)


def _rope_heads(x, cos_sin, *, head_dim):
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    tail = x[:, :, lo:].astype(jnp.float32)
    roped = tail * cos[:, None, :] + _rotate_gptj(tail) * sin[:, None, :]
    return jnp.concatenate([x[:, :, :lo], roped.astype(x.dtype)], axis=-1)


def _kernel(
    x_ref,  # (tile_t, 8, head_dim) bf16
    w_ref,  # (D, tile_r) bf16 | fp8
    scale_ref,  # (1, tile_r) f32
    cos_sin_ref,  # (tile_t, 2*LANE) f32
    out_ref,  # (tile_t, tile_r)
    *,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
    head_dim: int,
    interpret: bool,
):
    rhs = w_ref[...]
    if interpret:  # XLA:CPU cannot run bf16 x bf16 -> f32 dots; parity tests only
        rhs = rhs.astype(jnp.float32)
    scale = scale_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = out_ref.dtype
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x = _rope_heads(x_ref[rows], cos_sin[rows], head_dim=head_dim)
        x = x.reshape(sub_t, -1)
        inv = None
        if quantize_activations:
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(jnp.bfloat16)
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
        else:
            lhs = x.astype(jnp.float32) if interpret else x
        partial = jax.lax.dot_general(
            lhs, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        if quantize_activations:
            partial = partial * (1.0 / inv.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)


def wo_a_projection(
    x,  # [T, G*8, head_dim] bf16
    wo_a,  # [8*head_dim, G*R] bf16 or fp8
    cos_sin,  # [T, 2*LANE] f32 (see widen_cos_sin)
    wo_a_scale=None,  # [G*R] f32 for fp8 weights
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    out_dtype=jnp.bfloat16,
    interpret: bool = False,
):
    """``einsum("tgd,dgr->tgr", rope^-1(x).view(T,G,8*D), wo_a.view(8*D,G,R)) [* scale]``."""
    x = jnp.asarray(x)
    wo_a = jnp.asarray(wo_a)
    if x.ndim != 3:
        raise ValueError(f"x must be [T, heads, head_dim], got {x.shape}")
    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    if reduction != SUBLANE * head_dim or num_heads % SUBLANE:
        raise ValueError("wo_a_projection needs 8 heads per group: wo_a [8*head_dim, G*R]")
    if head_dim % LANE:
        raise ValueError("head_dim must be a multiple of 128")
    num_groups = num_heads // SUBLANE
    if out_features % num_groups:
        raise ValueError("wo_a columns must split evenly into groups")
    lora_rank = out_features // num_groups
    if cos_sin.shape != (num_tokens, 2 * LANE):
        raise ValueError(f"cos_sin must be [T, {2 * LANE}], got {cos_sin.shape}")
    quantize = wo_a.dtype == jnp.float8_e4m3fn
    if quantize and wo_a_scale is None:
        raise ValueError("fp8 wo_a needs wo_a_scale")
    scale = (
        jnp.ones((out_features,), jnp.float32)
        if wo_a_scale is None
        else jnp.asarray(wo_a_scale, jnp.float32)
    )
    x = x.astype(jnp.bfloat16)

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = _largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        sub_t = _largest_divisor(tile_t, cap=128)
    if lora_rank % tile_r or num_tokens % tile_t or tile_t % sub_t:
        raise ValueError("tile sizes must divide their extents")
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize,
            head_dim=head_dim,
            interpret=interpret,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), out_dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            pl.BlockSpec((tile_t, SUBLANE, head_dim), lambda g, t, r: (t, g, 0)),
            pl.BlockSpec((reduction, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            pl.BlockSpec((1, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec((tile_t, tile_r), lambda g, t, r: (t, g * num_r_tiles + r)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
    )(x, wo_a, scale.reshape(1, out_features), cos_sin)
