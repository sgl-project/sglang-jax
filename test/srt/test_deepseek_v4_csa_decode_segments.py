"""Page-run DMA segments for the request-local CSA decode scorer.

The host segmentation (``page_run_segments``) and the device per-page split
(``trivial_segments``) must drive the multi-row kernel to the same scores, and both
must match a NumPy reference. Runs on CPU in
interpret mode.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels import csa_decode
from sgl_jax.srt.kernels.csa_decode import (
    decode_segments,
    page_run_segments,
    paged_csa_decode_scores,
    scorer_pages_per_block,
    trivial_segments,
)

H, D, CPS = 4, 128, 32
NEG = float(jnp.finfo(jnp.float32).min)


def _expand(segments, counts, pages_per_block):
    """Segments -> sorted (logical page, physical page) pairs they cover, per row."""
    out = []
    for t in range(segments.shape[0]):
        covered = []
        for b in range(counts.shape[1]):
            for i in range(int(counts[t, b])):
                physical, offset, log2 = decode_segments(int(segments[t, b * pages_per_block + i]))
                for j in range(1 << log2):
                    covered.append((b * pages_per_block + offset + j, physical + j))
        out.append(sorted(covered))
    return out


@pytest.mark.parametrize("pages_per_block", [32, 8])
def test_page_run_segments_cover_exactly_the_valid_pages(pages_per_block):
    rng = np.random.default_rng(0)
    table = 4 * pages_per_block
    blocks = 4
    counts = np.array([0, 1, pages_per_block, table, min(70, table - 1), table - 5])
    rows = len(counts)
    pages = np.zeros((rows, table), np.int64)
    for t in range(rows):
        n = int(counts[t])
        if t % 2:
            pages[t, :n] = 1000 + t * table + np.arange(n)  # one run per allocation
        else:
            pages[t, :n] = rng.permutation(1 << 18)[:n]  # fully fragmented
    pages[2, 5:8] = [3, 4, 5]  # a short run inside a fragmented row
    segments, seg_counts = page_run_segments(pages, counts, pages_per_block)
    assert segments.shape == (rows, table) and seg_counts.shape == (rows, blocks)
    for t, covered in enumerate(_expand(segments, seg_counts, pages_per_block)):
        expected = sorted((k, int(pages[t, k])) for k in range(int(counts[t])))
        assert covered == expected, t
    # one allocation run per block is one DMA per block
    assert seg_counts[3].tolist() == [1] * blocks
    # a fragmented block needs (pages - merged) DMAs: 3 pages merged into a 2 + 1 split
    assert seg_counts[2].tolist() == [pages_per_block - 3 + 2] + [0] * (blocks - 1)
    assert seg_counts[0].tolist() == [0] * blocks
    assert seg_counts[1].tolist() == [1] + [0] * (blocks - 1)


def test_page_run_segments_rejects_bad_tables():
    with pytest.raises(ValueError):
        page_run_segments(np.zeros((2, 65), np.int32), np.array([1, 1]), 64)
    with pytest.raises(ValueError):
        page_run_segments(np.full((1, 64), csa_decode.MAX_PHYSICAL_PAGE), np.array([64]), 64)


def _reference(q, weights, cache, lengths, pages):
    q = np.asarray(q.astype(jnp.float32))
    weights = np.asarray(weights, np.float32)
    cache = np.asarray(cache.astype(jnp.float32))
    pages = np.asarray(pages)
    tokens, table = pages.shape
    capacity = table * CPS
    out = np.full((tokens, capacity), NEG, np.float32)
    for t in range(tokens):
        n = int(lengths[t])
        if n == 0:
            continue
        slots = pages[t, : (n + CPS - 1) // CPS][:, None] * CPS + np.arange(CPS)[None, :]
        keys = cache[slots.reshape(-1)][:n]  # [n, D]
        sims = q[t] @ keys.T  # [H, n]
        out[t, :n] = (np.maximum(sims, 0) * weights[t][:, None]).sum(0)
    return out


def _case(seed, tokens, table, lengths):
    rng = np.random.default_rng(seed)
    total_pages = 1 + tokens * table
    pages = np.zeros((tokens, table), np.int32)
    for t in range(tokens):
        if t % 3 == 0:
            pages[t] = rng.permutation(np.arange(1, total_pages))[:table]
        else:
            pages[t] = 1 + t * table + np.arange(table)
    return dict(
        q=jnp.asarray(rng.standard_normal((tokens, H, D)), jnp.bfloat16),
        weights=jnp.asarray(rng.random((tokens, H)), jnp.float32),
        cache=jnp.asarray(rng.standard_normal((total_pages * CPS, D)), jnp.bfloat16),
        lengths=jnp.asarray(lengths, jnp.int32),
        pages=jnp.asarray(pages),
    )


@pytest.mark.parametrize(
    "tokens,table,lengths,rows_per_step",
    [
        (5, 64, [2048, 1000, 0, 33, 2047], 8),  # one 2048 tile; rows padded to 8
        (9, 128, [4096, 4095, 2049, 2048, 1, 0, 3000, 2100, 4096], 4),  # two tiles, mixed
        (3, 4, [128, 100, 0], 2),  # capacity below the max tile: one 128-row tile
        (4, 80, [2560, 2304, 2049, 513], 4),  # fine bucket 2560: five 512-row tiles
    ],
)
def test_segment_kernel_matches_reference(tokens, table, lengths, rows_per_step):
    case = _case(1, tokens, table, lengths)
    capacity = table * CPS
    pages_per_block = scorer_pages_per_block(capacity, CPS)
    page_counts = (np.asarray(lengths) + CPS - 1) // CPS
    host_segments, host_counts = page_run_segments(
        np.asarray(case["pages"]), page_counts, pages_per_block
    )
    dev_segments, dev_counts = trivial_segments(
        case["pages"], jnp.asarray(page_counts), pages_per_block
    )
    assert _expand(np.asarray(dev_segments), np.asarray(dev_counts), pages_per_block) == _expand(
        host_segments, host_counts, pages_per_block
    )
    ref = _reference(case["q"], case["weights"], case["cache"], lengths, case["pages"])
    args = (case["q"], case["weights"], case["cache"], case["lengths"], case["pages"])
    kw = dict(page_size=CPS, interpret=True)
    got_host = np.asarray(
        paged_csa_decode_scores(
            *args,
            segments=host_segments,
            segment_counts=host_counts,
            rows_per_step=rows_per_step,
            **kw,
        )
    )
    got_dev = np.asarray(paged_csa_decode_scores(*args, rows_per_step=rows_per_step, **kw))
    assert got_host.shape == (tokens, capacity)
    np.testing.assert_array_equal(got_host, got_dev)
    valid = ref > NEG
    np.testing.assert_array_equal(got_host > NEG, valid)
    np.testing.assert_allclose(got_host[valid], ref[valid], rtol=2e-2, atol=2e-2)


def test_three_d_cache_and_flat_cache_agree():
    case = _case(2, 4, 8, [256, 200, 0, 1])
    args = (case["q"], case["weights"])
    tail = (case["lengths"], case["pages"])
    flat = paged_csa_decode_scores(*args, case["cache"], *tail, page_size=CPS, interpret=True)
    cubed = paged_csa_decode_scores(*args, case["cache"].reshape(-1, CPS, D), *tail, interpret=True)
    np.testing.assert_array_equal(np.asarray(flat), np.asarray(cubed))
    with pytest.raises(ValueError):
        paged_csa_decode_scores(*args, case["cache"], *tail, interpret=True)
