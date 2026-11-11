"""Fixed-tile matmul and RMSNorm for batch-invariant numerics (issue #1795).

XLA picks its matmul and reduction tiling from the operand shape, so the same token row can get
a different f32/bf16 accumulation order depending on the padded batch size and on the row's
position mod the tile height. These kernels fix the tiling: rows are processed in blocks of
``BLOCK_M`` and every row's reduction runs over the same K blocks in the same order, so a row's
result depends only on that row's values.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_M = 128
_BLOCK_CANDIDATES = (512, 384, 256, 128)


def _interpret() -> bool:
    return jax.default_backend() != "tpu"


def _pick_block(dim: int) -> int | None:
    for b in _BLOCK_CANDIDATES:
        if dim % b == 0:
            return b
    return None


def _pad_rows(x, multiple):
    rem = x.shape[0] % multiple
    if rem == 0:
        return x
    return jnp.pad(x, ((0, multiple - rem),) + ((0, 0),) * (x.ndim - 1))


def _matmul_kernel(x_ref, w_ref, o_ref, acc_ref, *, nk):
    # nk is passed statically instead of read from pl.num_programs(2). On TPU, two calls with the
    # same block shapes but a different K-grid length shared one compiled body that kept the
    # first call's K-grid length, so the output was written at the wrong K step or never.
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def fixed_tile_matmul(x: jax.Array, w: jax.Array, out_dtype=jnp.float32) -> jax.Array:
    """x: [M, K] @ w: [K, N] -> [M, N] with a row-invariant, M-invariant accumulation order.

    Runs on one device's local shards (call inside shard_map for sharded operands).
    """
    m, kdim = x.shape
    kw, n = w.shape
    assert kdim == kw, (x.shape, w.shape)
    bk = _pick_block(kdim)
    if bk is None:  # zero-pad K; adding exact zeros does not change the f32 sum
        pad = (-kdim) % 128
        x = jnp.pad(x, ((0, 0), (0, pad)))
        w = jnp.pad(w, ((0, pad), (0, 0)))
        kdim += pad
        bk = _pick_block(kdim)
    bn = _pick_block(n) or n
    xp = _pad_rows(x, BLOCK_M)
    mp = xp.shape[0]
    out = pl.pallas_call(
        functools.partial(_matmul_kernel, nk=kdim // bk),
        out_shape=jax.ShapeDtypeStruct((mp, n), out_dtype),
        grid=(mp // BLOCK_M, n // bn, kdim // bk),
        in_specs=[
            pl.BlockSpec((BLOCK_M, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((BLOCK_M, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=_interpret(),
    )(xp, w)
    return out[:m]


def _rmsnorm_kernel(x_ref, w_ref, o_ref, *, eps, add_unit_offset):
    x = x_ref[...].astype(jnp.float32)
    var = jnp.sum(x * x, axis=-1, keepdims=True) / x.shape[-1]
    y = x * jax.lax.rsqrt(var + eps)
    w = w_ref[...].astype(jnp.float32)
    if add_unit_offset:
        w = 1.0 + w
    o_ref[...] = (y * w).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "add_unit_offset", "out_dtype"))
def fixed_tile_rmsnorm(x, weight, eps, add_unit_offset=False, out_dtype=None):
    """RMSNorm over the last axis of x (any rank), one fixed reduction per row.

    weight: [H] or None (no scale). Runs on local shards; the last axis must be whole.
    """
    out_dtype = out_dtype or x.dtype
    shape = x.shape
    h = shape[-1]
    x2 = _pad_rows(x.reshape(-1, h), BLOCK_M)
    if weight is None:
        weight, add_unit_offset = jnp.ones((h,), jnp.float32), False
    w2 = weight.reshape(1, h)
    out = pl.pallas_call(
        functools.partial(_rmsnorm_kernel, eps=eps, add_unit_offset=add_unit_offset),
        out_shape=jax.ShapeDtypeStruct(x2.shape, out_dtype),
        grid=(x2.shape[0] // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, h), lambda i: (i, 0)),
            pl.BlockSpec((1, h), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, h), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=_interpret(),
    )(x2, w2)
    n_rows = 1
    for d in shape[:-1]:
        n_rows *= d
    return out[:n_rows].reshape(shape)
