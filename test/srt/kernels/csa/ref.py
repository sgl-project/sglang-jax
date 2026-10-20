"""Independent FP32 index scores and causal record selection, with BF16 inputs."""

import numpy as np


def select_records(q, weights, cache, pages, cu, lengths, k, *, candidates=None):
    result = np.full((len(q), k), -1, np.int32)
    page_rows = cache.shape[1]
    for request in range(len(lengths)):
        prefix = int(lengths[request] - (cu[request + 1] - cu[request]))
        for token in range(cu[request], cu[request + 1]):
            visible = (prefix + token - cu[request] + 1) // 4
            if visible <= 0:
                continue
            entries = np.arange(visible)
            physical = pages[request, entries // page_rows]
            keys = cache[physical, entries % page_rows].astype(np.float32)
            scores = np.maximum(np.asarray(q[token], np.float32) @ keys.T, 0)
            scores = np.sum(scores * np.asarray(weights[token], np.float32)[:, None], axis=0)
            order = np.argsort(-scores, kind="stable")[:k]
            if candidates is not None:
                chosen = np.asarray(candidates[token])
                assert np.all(chosen >= -1)
                chosen = chosen[chosen >= 0]
                assert len(chosen) == len(order) and len(np.unique(chosen)) == len(order)
                assert np.all(chosen < visible)
                # Equal-score ties need not share NumPy's stable index ordering.
                np.testing.assert_array_equal(np.sort(scores[chosen]), np.sort(scores[order]))
                order = chosen
            result[token, : len(order)] = order
    return result
