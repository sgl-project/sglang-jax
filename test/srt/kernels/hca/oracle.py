"""Independent fp32 dense-math reference for HCA (numpy only, no JAX).

Implements the documented semantics the kernels are supposed to realize: bf16
projection with fp32 accumulation, per-feature gating softmax over each
128-token group, RMSNorm, interleaved RoPE on the trailing 64 features, the
sliding-window union with compressed records visible from ``(p+1)//128``, and an
attention sink that appears in the denominator only.

Written densely and without JAX on purpose: it shares no code, no reduction
order, and no lowering with the Pallas kernels, so agreement is evidence about
the semantics rather than about a shared implementation. fp32 dense math cannot
match bf16 tiled math bit for bit, so callers compare with a tolerance.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

RATIO, WINDOW, HEAD_DIM = 128, 128, 512


def bf16(x):
    """Round through bf16 so both sides start from identical values."""
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def records(hidden_req, weights, upto_groups: int):
    """Compressed records for complete groups ``[0, upto_groups)`` of one request."""
    tokens = upto_groups * RATIO
    x = bf16(hidden_req[:tokens])
    kv = x @ bf16(weights["wkv"]).T
    score = x @ bf16(weights["wgate"]).T + weights["ape"][np.arange(tokens) % RATIO]
    out = []
    for group in range(upto_groups):
        span = slice(group * RATIO, (group + 1) * RATIO)
        weight = np.exp(score[span] - score[span].max(axis=0, keepdims=True))
        weight /= weight.sum(axis=0, keepdims=True)
        pooled = (weight * kv[span]).sum(axis=0)
        normed = pooled / np.sqrt(np.mean(pooled**2) + 1e-6) * weights["norm"]
        cos, sin = weights["cos"][group * RATIO], weights["sin"][group * RATIO]
        rotated = normed.copy()
        even, odd = normed[448::2].copy(), normed[449::2].copy()
        rotated[448::2] = even * cos - odd * sin
        rotated[449::2] = even * sin + odd * cos
        out.append(bf16(rotated))
    return np.stack(out) if out else np.zeros((0, HEAD_DIM), np.float32)


def attention_token(q_token, kv_req, recs, position: int, weights, softmax_scale):
    """Reference output for the query at absolute ``position``."""
    window = bf16(kv_req[max(0, position - WINDOW + 1) : position + 1])
    groups = (position + 1) // RATIO
    keys = np.concatenate([window, recs[:groups]], axis=0) if groups else window
    scores = (bf16(q_token) @ keys.T) * softmax_scale
    sink = weights["sink"][:, None]
    shift = np.maximum(scores.max(axis=1, keepdims=True), sink)
    probs = np.exp(scores - shift)
    denominator = probs.sum(axis=1, keepdims=True) + np.exp(sink - shift)
    return (probs @ keys) / denominator


def request_outputs(stream, request: int, positions, weights, softmax_scale):
    """Stack of reference outputs for one request's query positions."""
    kv_req = np.asarray(stream["kv"][request], np.float32)
    recs = records(
        np.asarray(stream["hidden"][request], np.float32),
        weights,
        (int(positions[-1]) + 1) // RATIO,
    )
    return np.stack(
        [
            attention_token(
                np.asarray(stream["q"][request][p], np.float32),
                kv_req,
                recs,
                int(p),
                weights,
                softmax_scale,
            )
            for p in positions
        ]
    )


def compare(actual, expected) -> dict:
    """Metrics for one comparison, accumulated in float64.

    fp32 accumulation over tens of millions of elements loses roughly 1e-4 of
    cosine on its own, which would otherwise look like a kernel defect.
    """
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    difference = np.abs(actual - expected)
    flat_a, flat_e = actual.ravel(), expected.ravel()
    cosine = float(
        (flat_a @ flat_e) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_e) + 1e-30)
    )
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "cosine": cosine,
        "nan": int(np.isnan(actual).sum()),
        "inf": int(np.isinf(actual).sum()),
    }


__all__ = [
    "HEAD_DIM",
    "RATIO",
    "WINDOW",
    "bf16",
    "compare",
    "records",
    "request_outputs",
]
