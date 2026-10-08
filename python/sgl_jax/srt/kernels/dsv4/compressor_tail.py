"""Fused compressor record tail for DSv4 CSA/indexer layers (Pallas TPU).

`layers/attention/dsv4/compressor.compress_chunk` gathers each record's window rows
into ``combined`` ``[N, W, 2*width]`` and then runs, as separate XLA ops per layer:
field selection (older half of the window reads field 0, newer half field 1), the
per-feature softmax pool over the window, RMSNorm, and the interleaved RoPE on the
trailing rotary channels. This kernel does that tail in one pass per record block;
with 21 CSA layers and 21 indexer compressors that is a few hundred XLA ops per
decode step folded into 42 kernel calls. Values match the XLA path to f32 rounding.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.dsv4.wo_a_projection import LANE, _rotate_gptj, widen_cos_sin

# ``DSV4_TAIL_INKERNEL_MASK=1|2``: apply the window validity mask inside the tail
# kernel (1) and additionally skip the row pad with a partial last block (2).
_INKERNEL_MASK = int(os.environ.get("DSV4_TAIL_INKERNEL_MASK", "2"))


def _kernel(comb_ref, nw_ref, cs_ref, out_ref, *, ratio, coff, head_dim, width, eps):
    comb = comb_ref[...]  # [tn, W, 2*width] f32; invalid rows' scores are already -inf
    _tail_body(
        comb,
        None,
        nw_ref,
        cs_ref,
        out_ref,
        ratio=ratio,
        coff=coff,
        head_dim=head_dim,
        width=width,
        eps=eps,
    )


def _kernel_masked(
    comb_ref, bias_ref, nw_ref, cs_ref, out_ref, *, ratio, coff, head_dim, width, eps
):
    # ``bias_ref`` [tn, W, LANE] f32: 0 for in-sequence window rows, -inf otherwise
    # (lane-expanded so it tiles across the score features without a lane broadcast).
    _tail_body(
        comb_ref[...].astype(jnp.float32),
        bias_ref[...],
        nw_ref,
        cs_ref,
        out_ref,
        ratio=ratio,
        coff=coff,
        head_dim=head_dim,
        width=width,
        eps=eps,
    )


def _tail_body(comb, bias, nw_ref, cs_ref, out_ref, *, ratio, coff, head_dim, width, eps):
    tn, window = comb.shape[0], comb.shape[1]
    if coff == 2:
        # Full-shape iota: Mosaic cannot broadcast a (1, W, 1) predicate across lanes.
        newer = jax.lax.broadcasted_iota(jnp.int32, (tn, window, head_dim), 1) >= ratio
        kv = jnp.where(newer, comb[..., head_dim:width], comb[..., :head_dim])
        score = jnp.where(newer, comb[..., width + head_dim :], comb[..., width : width + head_dim])
    else:
        kv = comb[..., :width]
        score = comb[..., width:]
    if bias is not None:
        score = score + jnp.concatenate([bias] * (head_dim // LANE), axis=-1)
    peak = jnp.max(score, axis=1, keepdims=True)
    weights = jnp.exp(score - peak)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    pooled = jnp.sum(weights * kv, axis=1)  # [tn, D]
    variance = jnp.mean(jnp.square(pooled), axis=-1, keepdims=True)
    normed = pooled * jax.lax.rsqrt(variance + eps) * nw_ref[...]
    cos_sin = cs_ref[...]
    lo = head_dim - LANE
    tail = normed[:, lo:]
    roped = tail * cos_sin[:, :LANE] + _rotate_gptj(tail) * cos_sin[:, LANE:]
    out_ref[...] = jnp.concatenate([normed[:, :lo], roped], axis=-1) if lo else roped


def compressor_tail_pallas(
    combined,
    valid_mask,
    norm_weight,
    cos,
    sin,
    *,
    ratio: int,
    coff: int,
    head_dim: int,
    width: int,
    rope_head_dim: int,
    norm_eps: float,
    block_n: int = 16,
    interpret: bool | None = None,
):
    """``[N, W, 2*width]`` gathered window rows -> ``[N, head_dim]`` f32 records."""
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    combined = jnp.asarray(combined, jnp.float32)
    n, window, two_width = combined.shape
    if two_width != 2 * width or width != coff * head_dim or head_dim % LANE:
        raise ValueError(f"bad compressor geometry: {combined.shape}, width {width}, D {head_dim}")
    tn = min(block_n, -(-n // 8) * 8)
    n_pad = -(-n // tn) * tn
    # Mask the score fields in XLA (one fused pass over the gathered rows): the
    # kernel then needs no separate validity operand. Padded records are all-zero
    # and stay finite; their output is dropped.
    cos_sin = widen_cos_sin(cos, sin, rope_head_dim=rope_head_dim, inverse=False)
    nw = jnp.asarray(norm_weight, jnp.float32).reshape(1, head_dim)
    if _INKERNEL_MASK:
        # Mask inside the kernel: no XLA pass over the [N, W, 2*width] f32 window
        # rows (the where and the row pad were ~6.5 ms of an 8K prefill step on
        # v7x). Level 2 also skips the row pad and lets the last block run
        # partially past ``n`` (rows past n are never stored; padded records'
        # outputs are dropped by the caller anyway).
        valid = jnp.asarray(valid_mask, bool)
        bias = jnp.broadcast_to(
            jnp.where(valid, 0.0, -jnp.inf).astype(jnp.float32)[:, :, None], (n, window, LANE)
        )
        rows_in = n if _INKERNEL_MASK >= 2 else n_pad
        if rows_in != n:
            combined = jnp.pad(combined, ((0, n_pad - n), (0, 0), (0, 0)))
            bias = jnp.pad(bias, ((0, n_pad - n), (0, 0), (0, 0)))
            cos_sin = jnp.pad(cos_sin, ((0, n_pad - n), (0, 0)))
        out = pl.pallas_call(
            functools.partial(
                _kernel_masked,
                ratio=ratio,
                coff=coff,
                head_dim=head_dim,
                width=width,
                eps=float(norm_eps),
            ),
            grid=(n_pad // tn,),
            in_specs=[
                pl.BlockSpec((tn, window, two_width), lambda i: (i, 0, 0)),
                pl.BlockSpec((tn, window, LANE), lambda i: (i, 0, 0)),
                pl.BlockSpec((1, head_dim), lambda i: (0, 0)),
                pl.BlockSpec((tn, 2 * LANE), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tn, head_dim), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((rows_in, head_dim), jnp.float32),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",), vmem_limit_bytes=32 * 1024 * 1024
            ),
            interpret=interpret,
        )(combined, bias, nw, cos_sin)
        return out[:n]
    valid = jnp.asarray(valid_mask, bool)[:, :, None]
    lane = jnp.arange(two_width)[None, None, :]
    is_score = lane >= width
    combined = jnp.where(is_score & ~valid, -jnp.inf, combined)
    comb = jnp.pad(combined, ((0, n_pad - n), (0, 0), (0, 0)))
    cos_sin = jnp.pad(cos_sin, ((0, n_pad - n), (0, 0)))
    out = pl.pallas_call(
        functools.partial(
            _kernel, ratio=ratio, coff=coff, head_dim=head_dim, width=width, eps=float(norm_eps)
        ),
        grid=(n_pad // tn,),
        in_specs=[
            pl.BlockSpec((tn, window, two_width), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, head_dim), lambda i: (0, 0)),
            pl.BlockSpec((tn, 2 * LANE), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tn, head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n_pad, head_dim), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=32 * 1024 * 1024
        ),
        interpret=interpret,
    )(comb, nw, cos_sin)
    return out[:n]
