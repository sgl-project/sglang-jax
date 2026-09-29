"""SwiGLU activation for EPMoE restricted to the rows that hold local-expert tokens.

The expert-parallel gather buffer is statically sized ``[T * top_k, D]`` although
only the rows of the device's own experts (``[start, end)``) are materialised and
consumed by the grouped matmuls. XLA's element-wise activation still streams the
whole buffer: at an 8K prefill chunk that is 3 x 200 MB per layer (0.21 ms on v7x,
9 ms per step) for ~1/8 useful rows. This kernel clamps the input block index into
the valid block range, so the pipeline never fetches the other blocks (Pallas skips
a DMA when the block index does not change), computes the valid blocks, and writes
zeros elsewhere so no downstream reader can see uninitialised rows.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _act(gate, up, limit):
    # Same expression as layers/activation.silu_and_mul_with_clamp, same dtype.
    return jax.nn.silu(jnp.minimum(gate, limit)) * jnp.clip(up, -limit, limit)


def _kernel(bounds_ref, g_ref, u_ref, o_ref, *, limit):
    i = pl.program_id(0)
    valid = jnp.logical_and(i >= bounds_ref[0], i <= bounds_ref[1])

    @pl.when(valid)
    def _compute():
        o_ref[...] = _act(g_ref[...], u_ref[...], limit).astype(o_ref.dtype)

    @pl.when(jnp.logical_not(valid))
    def _zero():
        o_ref[...] = jnp.zeros_like(o_ref)


def silu_mul_rows(
    gate, up, start, end, *, limit: float, block_rows: int = 512, interpret: bool = False
):
    """``silu(min(gate, limit)) * clip(up, -limit, limit)`` on rows ``[start, end)``.

    ``gate``/``up`` are ``[rows, n]``; rows outside the blocks that cover
    ``[start, end)`` come back as zeros without being read. Rows inside those
    blocks but outside ``[start, end)`` are computed from whatever the buffer
    holds, exactly as the whole-buffer form did.
    """
    if gate.shape != up.shape or gate.ndim != 2:
        raise ValueError(f"gate/up must be matching 2-D arrays, got {gate.shape} vs {up.shape}")
    rows, n = gate.shape
    block_rows = min(block_rows, rows)
    if block_rows % 8:
        raise ValueError("block_rows must be a multiple of 8")
    pad = (-rows) % block_rows
    if pad:
        gate = jnp.pad(gate, ((0, pad), (0, 0)))
        up = jnp.pad(up, ((0, pad), (0, 0)))
    nb = (rows + pad) // block_rows
    start = jnp.asarray(start, jnp.int32)
    end = jnp.asarray(end, jnp.int32)
    first = jnp.clip(start // block_rows, 0, nb - 1)
    last = jnp.clip((end + block_rows - 1) // block_rows - 1, first - 1, nb - 1)
    bounds = jnp.stack([first, last]).astype(jnp.int32)

    def in_map(i, b_ref):
        return (jnp.clip(i, b_ref[0], jnp.maximum(b_ref[1], b_ref[0])), 0)

    out = pl.pallas_call(
        functools.partial(_kernel, limit=float(limit)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(nb,),
            in_specs=[
                pl.BlockSpec((block_rows, n), in_map),
                pl.BlockSpec((block_rows, n), in_map),
            ],
            out_specs=pl.BlockSpec((block_rows, n), lambda i, b_ref: (i, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((rows + pad, n), gate.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name=f"dsv4-moe-act-rows-b{block_rows}-n{n}",
    )(bounds, gate, up)
    return out[:rows] if pad else out
