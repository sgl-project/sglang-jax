"""CPU-side tests for live indexer row grouping and output restoration."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.dsa import streamindex_live


def test_bucket_groups_restore_rows_and_packed_page_offsets(monkeypatch):
    page_size, head_dim, heads, topk, capacity = 128, 128, 32, 8, 65
    lengths = np.asarray([100, 200, 300, 2049, 2050, 4097, 5000, 8193], np.int32)
    batch = len(lengths)
    page_counts = (lengths + page_size - 1) // page_size
    offsets = np.r_[0, np.cumsum(page_counts, dtype=np.int32)]

    # Encode request and page number in each packed page id. The fake scorer
    # returns each row's first page id, checking the repacked offsets and that
    # the grouped results are scattered back to their original rows.
    page_indices = np.full(batch * capacity, -1, np.int32)
    for row, count in enumerate(page_counts):
        start = offsets[row]
        page_indices[start : start + count] = row * 100 + np.arange(count) + 1

    seen_batch_sizes = []

    def fake_streamindex_topk(
        q, weights, cache, seq_lens, pages, cu_q, distribution, *, k, **kwargs
    ):
        del weights, cache, cu_q, distribution, kwargs
        group_size = seq_lens.shape[0]
        seen_batch_sizes.append(group_size)
        pages_per_row = pages.shape[0] // group_size
        first_pages = pages.reshape(group_size, pages_per_row)[:, 0]
        return jnp.broadcast_to(first_pages[:, None], (group_size, k))

    monkeypatch.setattr(streamindex_live, "streamindex_topk", fake_streamindex_topk)

    q = np.zeros((batch, heads, head_dim), ml_dtypes.bfloat16)
    weights = np.zeros((batch, heads), ml_dtypes.bfloat16)
    cache = np.zeros((1, page_size, head_dim), ml_dtypes.bfloat16)
    cu_q_lens = np.arange(batch + 1, dtype=np.int32)
    cu_kv_lens = offsets * page_size
    distribution = np.asarray([batch, batch, batch], np.int32)

    inputs = map(
        jnp.asarray,
        (q, weights, cache, lengths, page_indices, cu_q_lens, cu_kv_lens, distribution),
    )
    result = streamindex_live.streamindex_topk_live(
        *inputs,
        k=topk,
        pages_per_seq=capacity,
    )

    expected = np.repeat((np.arange(batch, dtype=np.int32) * 100 + 1)[:, None], topk, axis=1)
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert seen_batch_sizes and set(seen_batch_sizes) == {4}
