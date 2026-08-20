"""Independent numpy reference for the mHC ops (no JAX import).

Written from the semantics DeepSeek-V4-Flash publishes, not from the kernels, so
it shares no lowering, reduction order or framework with the code under test.

* ``sinkhorn_gates``  <- inference/kernel.py:hc_split_sinkhorn_kernel
* ``pre`` / ``post``  <- inference/model.py:Block.hc_pre / hc_post
* ``head_collapse``   <- inference/model.py:ParallelHead.hc_head
"""

from __future__ import annotations

import ml_dtypes
import numpy as np


def mix_hc_width(hc_mult: int) -> int:
    """Width of the gate projection: hc for pre, hc for post, hc*hc for comb."""
    return (2 + hc_mult) * hc_mult


def bf16(x):
    """Round through bf16 so both sides start from identical values."""
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _softmax(x, axis):
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def sinkhorn_gates(mixes, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps):
    """Split the projection into the three per-token gates.

    Load-bearing: ``pre`` adds eps after the sigmoid, ``post`` does not and
    carries a factor of two; the schedule is a row softmax, one column pass,
    then ``sinkhorn_iters - 1`` pairs; every division is by ``sum + eps``.
    """
    hc = hc_mult
    mixes = np.asarray(mixes, np.float64)
    scale = np.asarray(hc_scale, np.float64)
    base = np.asarray(hc_base, np.float64)

    pre = 1.0 / (1.0 + np.exp(-(mixes[..., :hc] * scale[0] + base[:hc]))) + eps
    post = 2.0 / (
        1.0 + np.exp(-(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]))
    )

    comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc)
    comb = comb * scale[2] + base[2 * hc :].reshape(hc, hc)
    comb = _softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _projection(x_streams, hc_fn, *, norm_eps):
    """RMS-scaled gate projection over the flattened multi-stream state."""
    x = np.asarray(x_streams, np.float64)
    flat = x.reshape(*x.shape[:-2], -1)
    rsqrt = 1.0 / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + norm_eps)
    return flat @ np.asarray(hc_fn, np.float64).T * rsqrt


def pre(
    x_streams, hc_fn, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, norm_eps, hc_eps
):
    """Collapse hc streams to one and emit the gates the post step will need."""
    x = np.asarray(x_streams, np.float64)
    mixes = _projection(x, hc_fn, norm_eps=norm_eps)
    pre_gate, post_gate, comb = sinkhorn_gates(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    y = (pre_gate[..., None] * x).sum(axis=-2)
    return y, post_gate, comb


def post(x, residual_streams, post_gate, comb):
    """Expand one stream to hc: out[j] = post[j]*x + Sum_i comb[i,j]*res[i]."""
    x = np.asarray(x, np.float64)
    residual = np.asarray(residual_streams, np.float64)
    mixed = np.einsum("...ij,...ih->...jh", np.asarray(comb, np.float64), residual)
    return np.asarray(post_gate, np.float64)[..., None] * x[..., None, :] + mixed


def head_collapse(x_streams, hc_fn, hc_scale, hc_base, *, norm_eps, hc_eps):
    """Final hc -> 1 collapse, sigmoid gates only, no Sinkhorn.

    Unlike ``pre``, the RMS scale hits the activation *before* the projection
    with a bf16 rounding in between; that boundary is part of the op.
    """
    x = np.asarray(x_streams, np.float64)
    flat = x.reshape(*x.shape[:-2], -1)
    rsqrt = 1.0 / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + norm_eps)
    normalized = bf16(flat * rsqrt).astype(np.float64)
    mixes = normalized @ np.asarray(hc_fn, np.float64).T
    gate = (
        1.0 / (1.0 + np.exp(-(mixes * np.asarray(hc_scale, np.float64)[0] + hc_base)))
        + hc_eps
    )
    return (gate[..., None] * x).sum(axis=-2)


__all__ = [
    "bf16",
    "head_collapse",
    "mix_hc_width",
    "post",
    "pre",
    "sinkhorn_gates",
]
