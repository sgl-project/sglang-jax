"""Real-TPU tests for runtime live-context scoring and selection.

NumPy computes scores independently; exact set membership is checked away
from FP32 rounding ties. One executable crosses multiple context buckets.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.streamindex_live import streamindex_topk_live

if jax.default_backend() != "tpu":
    pytest.skip("Live indexer requires real TPU DMA", allow_module_level=True)

PAGE, HEADS, DIM, K = 128, 32, 128, 2048


def make_inputs(lengths, capacity=65, active=None, seed=7):
    batch = len(lengths)
    active = batch if active is None else active
    rng = np.random.default_rng(seed)
    cache = rng.normal(0, 0.2, (batch * capacity + 1, PAGE, DIM)).astype(ml_dtypes.bfloat16)
    q = rng.normal(0, 0.2, (batch, HEADS, DIM)).astype(ml_dtypes.bfloat16)
    weights = rng.normal(0, 0.2, (batch, HEADS)).astype(ml_dtypes.bfloat16)
    page_counts = (np.asarray(lengths, np.int32) + PAGE - 1) // PAGE
    offsets = np.r_[0, np.cumsum(page_counts, dtype=np.int32)]
    physical = rng.permutation(np.arange(1, batch * capacity + 1, dtype=np.int32))
    pages = np.zeros(batch * capacity, np.int32)
    pages[: offsets[-1]] = physical[: offsets[-1]]
    return (
        q,
        weights,
        cache,
        np.asarray(lengths, np.int32),
        pages,
        np.minimum(np.arange(batch + 1, dtype=np.int32), active),
        offsets.astype(np.int32) * PAGE,
        np.asarray([active] * 3, np.int32),
    )


def verify(host, result):
    q, weights, cache, lengths, pages, _, offsets, dist = host
    result = np.asarray(result)
    assert result.shape == (len(lengths), K)
    assert result.dtype == np.int32
    for row, length in enumerate(lengths):
        count = min(int(length), K) if row < dist[0] else 0
        selected = result[row, :count]
        assert np.all(result[row, count:] == -1)
        assert np.all((selected >= 0) & (selected < length))
        assert len(np.unique(selected)) == count
        if not count:
            continue
        start = offsets[row] // PAGE
        n_pages = (int(length) + PAGE - 1) // PAGE
        keys = cache[pages[start : start + n_pages]].reshape(-1, DIM)[:length].astype(np.float32)
        dot = np.asarray(q[row], np.float32) @ keys.T
        scores = (np.maximum(dot, 0) * np.asarray(weights[row], np.float32)[:, None]).sum(0)
        cutoff = np.partition(scores, len(scores) - count)[len(scores) - count]
        tolerance = 2e-5 * max(1.0, float(np.abs(scores).max()))
        assert scores[selected].min() >= cutoff - tolerance
        # A padded score must never displace a negative but valid score.
        assert np.all(np.diff(scores[selected]) <= tolerance)


def test_bucket_crossings_and_packed_page_offsets():
    def run(*args):
        return streamindex_topk_live(*args, k=K, pages_per_seq=65)

    lengths_cases = [
        [2047, 2048, 2049, 0],
        [4095, 4096, 4097, 129],
        [8191, 8192, 8193, 8320],  # last, non-power-of-two capacity bucket
        [1, 650, 0, 8320],  # returning to smaller buckets uses the same executable
    ]
    first = make_inputs(lengths_cases[0])
    compiled = jax.jit(run).lower(*map(jnp.asarray, first)).compile()
    for lengths in lengths_cases:
        host = make_inputs(lengths)
        inputs = tuple(map(jnp.asarray, host))
        actual = compiled(*inputs)
        verify(host, actual)
        np.testing.assert_array_equal(inputs[2], host[2])


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_ties_negative_scores_and_inactive_rows(weight):
    host = list(make_inputs([128, 1, 128], capacity=1, active=2))
    host[0].fill(1)
    host[1].fill(weight)
    host[2].fill(1)
    actual = streamindex_topk_live(*map(jnp.asarray, host), k=K, pages_per_seq=1)
    verify(host, actual)
