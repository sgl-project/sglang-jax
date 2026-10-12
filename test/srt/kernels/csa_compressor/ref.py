"""Independent sequential NumPy oracle: BF16 operands, FP32 math, BF16 records."""

import ml_dtypes
import numpy as np


def compressor(
    x,
    weight,
    apes,
    norms,
    cos,
    sin,
    states,
    caches,
    positions,
    cu_lens,
    slots,
    locations,
    *,
    norm_eps=1e-6,
):
    states = [np.array(s, copy=True) for s in states]
    caches = [np.array(c, copy=True) for c in caches]
    projection = np.asarray(x, np.float32) @ np.asarray(weight, np.float32)
    for request, slot in enumerate(slots):
        if not 0 <= slot < states[0].shape[0] - 1:
            continue
        for token in range(cu_lens[request], cu_lens[request + 1]):
            pos = int(positions[token])
            if not 0 <= pos < len(cos):
                continue
            offset = 0
            for state, cache, ape, norm in zip(states, caches, apes, norms, strict=True):
                dim = len(norm)
                new = projection[token, offset : offset + 4 * dim].copy()
                offset += 4 * dim
                new[2 * dim :] += ape[pos % 4]
                state[slot, pos % 8] = new
                loc = locations[token]
                if (pos + 1) % 4 or not cache.shape[1] <= loc < np.prod(cache.shape[:2]):
                    continue
                values, scores = [], []
                for row, absolute in enumerate(range(pos - 7, pos + 1)):
                    field = 0 if row < 4 else dim
                    s = state[slot, absolute % 8]
                    values.append(
                        s[field : field + dim] if absolute >= 0 else np.zeros(dim, np.float32)
                    )
                    scores.append(
                        s[2 * dim + field : 3 * dim + field]
                        if absolute >= 0
                        else np.full(dim, -np.inf, np.float32)
                    )
                values, scores = np.asarray(values), np.asarray(scores)
                probabilities = np.exp(scores - scores.max(axis=0))
                probabilities /= probabilities.sum(axis=0, dtype=np.float32)
                pooled = np.sum(values * probabilities, axis=0, dtype=np.float32)
                pooled *= np.float32(1) / np.sqrt(
                    np.mean(pooled * pooled, dtype=np.float32) + np.float32(norm_eps)
                )
                pooled *= norm
                real, imag = pooled[-64::2].copy(), pooled[-63::2].copy()
                c, s = cos[pos - 3], sin[pos - 3]
                pooled[-64::2] = real * c - imag * s
                pooled[-63::2] = real * s + imag * c
                cache[loc // cache.shape[1], loc % cache.shape[1]] = pooled.astype(
                    ml_dtypes.bfloat16
                )
    return (*states, *caches)
