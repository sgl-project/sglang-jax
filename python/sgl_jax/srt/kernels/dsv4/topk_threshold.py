"""Per-row top-k *threshold* for the CSA indexer (Pallas TPU).

The prefill attention kernel consumes a membership mask, not an index list, so the
indexer only has to answer "is this entry among the row's k best?". That is a
comparison against the row's k-th largest score. This module finds that value by a
32-step bisection over the f32 bit pattern (mapped to a monotone integer key): each
step compares the whole ``[rows, E]`` block against a candidate and counts, so the
cost is linear in ``E`` and the block stays in VMEM for all steps, unlike
``jax.lax.approx_max_k(recall_target=1.0)`` which sorts every row (superlinear in
``E``: 1.5 ms/layer at E=2048, 4.2 ms at E=4096 on v7x for 8192 rows).

Semantics: ``threshold[t]`` is the k-th largest key of row ``t`` (ties included, so
``key >= threshold`` may hold for more than k entries when scores tie exactly); a
row with fewer than k finite scores gets a threshold at or below its smallest
entry (the caller masks non-finite scores out). ``score_key`` and the threshold are
uint32 in the same monotone order as the f32 scores.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SIGN = 0x80000000  # Python int: kernels must not capture jax constants


def score_key(scores):
    """f32 -> uint32 with the same total order (NaN-free inputs; -inf maps low)."""
    bits = jax.lax.bitcast_convert_type(jnp.asarray(scores, jnp.float32), jnp.uint32)
    sign = jnp.uint32(_SIGN)
    negative = (bits & sign) != 0
    return jnp.where(negative, ~bits, bits | sign)


def _to_signed(key_u32):
    # Flip the sign bit so unsigned order becomes signed int32 order (Mosaic
    # compares int32 natively).
    return jax.lax.bitcast_convert_type(key_u32 ^ jnp.uint32(_SIGN), jnp.int32)


def _from_signed(key_i32):
    return jax.lax.bitcast_convert_type(key_i32, jnp.uint32) ^ jnp.uint32(_SIGN)


def _bisect(skey, k):
    """``skey`` [rows, E] int32 (signed view of the key) -> [rows, 1] int32 threshold."""
    # Search the unsigned key bit by bit; keep the signed view for comparisons.
    # The running threshold is derived from the data block (not a fresh zeros
    # constant) so every intermediate carries the block's sharding under explicit
    # meshes; jnp.where rejects mixed shardings there.
    thr_u = jax.lax.bitcast_convert_type(skey[:, :1] & jnp.int32(0), jnp.uint32)
    for b in range(31, -1, -1):
        cand_u = thr_u | jnp.uint32(1 << b)
        cand_s = _to_signed(cand_u)
        count = jnp.sum((skey >= cand_s).astype(jnp.int32), axis=1, keepdims=True)
        thr_u = jnp.where(count >= k, cand_u, thr_u)
    return _to_signed(thr_u)


def _kernel(skey_ref, out_ref, *, k):
    out_ref[...] = _bisect(skey_ref[...], k)


def _default_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "0") == "1" or jax.default_backend() == "cpu"


def topk_threshold(scores, k: int, *, block_rows: int | None = None, interpret: bool | None = None):
    """``[T, E]`` f32 scores -> ``[T]`` uint32 threshold keys (see module docstring)."""
    if interpret is None:
        interpret = _default_interpret()
    scores = jnp.asarray(scores, jnp.float32)
    T, E = scores.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if E % 128:
        scores = jnp.pad(scores, ((0, 0), (0, -E % 128)), constant_values=-jnp.inf)
        E = scores.shape[1]
    if block_rows is None:
        # Keep one block near 4 MB of int32 so it sits in VMEM with room to spare.
        block_rows = max(8, min(256, (4 << 20) // (E * 4) // 8 * 8))
    tr = min(block_rows, -(-T // 8) * 8)
    Tp = -(-T // tr) * tr
    skey = _to_signed(score_key(scores))
    if Tp != T:
        skey = jnp.pad(skey, ((0, Tp - T), (0, 0)), constant_values=jnp.iinfo(jnp.int32).min)
    out = pl.pallas_call(
        functools.partial(_kernel, k=int(k)),
        grid=(Tp // tr,),
        in_specs=[pl.BlockSpec((tr, E), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tr, 1), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Tp, 1), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
        interpret=interpret,
        name=f"dsv4-topk-threshold-k{k}-e{E}",
    )(skey)
    return _from_signed(out[:T, 0])


def topk_membership_mask(scores, k: int, *, interpret: bool | None = None):
    """``[T, E]`` bool: score is finite and >= the row's k-th largest score."""
    scores = jnp.asarray(scores, jnp.float32)
    thr = topk_threshold(scores, k, interpret=interpret)
    return (score_key(scores) >= thr[:, None]) & (scores > -jnp.inf)
