"""Page deduplication must preserve page order, truncation and invalid slots.

Input rows follow the indexer contract: descending score, -1 padding.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages


def _expected(tokens, page_size, pages_per_seq, budget):
    """Pages ranked by their first (best-scored) token; kept pages ascending."""
    result = np.full((len(tokens), budget), -1, dtype=np.int32)
    for row, ids in enumerate(tokens):
        valid = ids[(ids >= 0) & (ids // page_size < pages_per_seq)]
        pages, first = np.unique(valid // page_size, return_index=True)
        pages = np.sort(pages[np.argsort(first)][:budget])
        result[row, : len(pages)] = pages
    return result


def _lowest_ids_expected(tokens, page_size, pages_per_seq, budget):
    """Previous behaviour: the lowest page ids win when truncated."""
    result = np.full((len(tokens), budget), -1, dtype=np.int32)
    for row, ids in enumerate(tokens):
        valid = ids[(ids >= 0) & (ids // page_size < pages_per_seq)]
        pages = np.unique(valid // page_size)[:budget]
        result[row, : len(pages)] = pages
    return result


@pytest.mark.parametrize(
    "batch,selected,page_size,pages_per_seq,budget",
    [
        (1, 16, 128, 8, 4),
        (4, 32, 7, 17, 8),
        (2, 8, 128, 3, 16),
        (4, 2048, 128, 7635, 512),
        (2, 0, 128, 4, 8),
        (2, 8, 128, 4, 0),
    ],
)
@pytest.mark.parametrize("pattern", ["random", "invalid", "duplicates"])
def test_page_selection(batch, selected, page_size, pages_per_seq, budget, pattern):
    rng = np.random.default_rng(42)
    if pattern == "invalid":
        tokens = np.full((batch, selected), -1, dtype=np.int32)
    elif pattern == "duplicates":
        tokens = rng.integers(
            0, min(pages_per_seq, 3) * page_size, (batch, selected), dtype=np.int32
        )
    else:
        tokens = rng.integers(
            -page_size, (pages_per_seq + 2) * page_size, (batch, selected), dtype=np.int32
        )
    actual = compute_topk_pages(
        jnp.asarray(tokens),
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        k_pages_max=budget,
    )
    np.testing.assert_array_equal(actual, _expected(tokens, page_size, pages_per_seq, budget))


@pytest.mark.parametrize("pattern", ["random", "duplicates", "padded"])
def test_within_budget_matches_lowest_ids_behaviour(pattern):
    rng = np.random.default_rng(7)
    page_size, pages_per_seq, budget = 128, 782, 512
    if pattern == "duplicates":
        tokens = rng.integers(0, 3 * page_size, (4, 2048), dtype=np.int32)
    else:
        # 400 distinct pages, 2048 tokens: never more than the budget.
        pages = rng.choice(pages_per_seq, 400, replace=False)
        tokens = (pages[rng.integers(0, 400, (4, 2048))] * page_size).astype(np.int32)
        tokens += rng.integers(0, page_size, tokens.shape, dtype=np.int32)
    if pattern == "padded":
        tokens[:, 1500:] = -1
        tokens[3] = -1
    actual = compute_topk_pages(
        jnp.asarray(tokens), page_size=page_size, pages_per_seq=pages_per_seq, k_pages_max=budget
    )
    np.testing.assert_array_equal(
        actual, _lowest_ids_expected(tokens, page_size, pages_per_seq, budget)
    )


def test_overflow_keeps_best_scored_pages():
    # 100k context = 782 pages; 2048 tokens spread over them, the most recent
    # pages scored highest (indexer order: descending score).
    page_size, pages_per_seq, budget = 128, 782, 512
    rng = np.random.default_rng(0)
    tokens = rng.choice(100_000, 2048, replace=False).astype(np.int32)
    tokens = np.sort(tokens)[::-1][None, :].copy()
    touched = np.unique(tokens // page_size)
    assert len(touched) > budget
    actual = np.asarray(
        compute_topk_pages(
            jnp.asarray(tokens),
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            k_pages_max=budget,
        )
    )
    np.testing.assert_array_equal(actual, touched[-budget:][None, :])
    assert (actual >= 0).sum() == budget
    np.testing.assert_array_equal(actual, _expected(tokens, page_size, pages_per_seq, budget))


def test_overflow_ranks_pages_by_best_token_not_page_id():
    # Scores interleave old and new pages; the budget cuts by rank, so the kept
    # set is neither the lowest nor the highest ids. -1 padding sits at the tail.
    page_size, pages_per_seq, budget = 128, 1024, 4
    order = [900, 3, 700, 5, 3, 900, 100, 800, 1]
    tokens = np.array([[p * page_size + 1 for p in order] + [-1, -1]], dtype=np.int32)
    actual = compute_topk_pages(
        jnp.asarray(tokens), page_size=page_size, pages_per_seq=pages_per_seq, k_pages_max=budget
    )
    np.testing.assert_array_equal(actual, [[3, 5, 700, 900]])


def test_padding_rows_and_out_of_range_do_not_take_budget():
    page_size, pages_per_seq, budget = 128, 8, 3
    tokens = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [9 * page_size, -1, 7 * page_size, 2 * page_size, 5 * page_size, 0],
        ],
        dtype=np.int32,
    )
    actual = compute_topk_pages(
        jnp.asarray(tokens), page_size=page_size, pages_per_seq=pages_per_seq, k_pages_max=budget
    )
    np.testing.assert_array_equal(actual, [[-1, -1, -1], [2, 5, 7]])


def test_page_boundaries_and_out_of_range_tokens():
    tokens = np.array([[127, 128, 129, 255, 256, 511, 512, -1, -128]], dtype=np.int32)
    actual = compute_topk_pages(jnp.asarray(tokens), page_size=128, pages_per_seq=4, k_pages_max=6)
    np.testing.assert_array_equal(actual, [[0, 1, 2, 3, -1, -1]])
