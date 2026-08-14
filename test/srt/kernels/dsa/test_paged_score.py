"""Correctness tests for the paged-cache DSA decode scorer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.indexer import compute_scores_and_select_topk_indices
from sgl_jax.srt.kernels.dsa.paged_score import (
    paged_decode_scores_pallas,
    paged_extend_score_and_map_block_pallas,
    paged_extend_score_block_pallas,
)


def _reference_scores(
    q_idx,
    idx_weights,
    cache,
    seq_lens,
    page_indices,
    cu_kv_lens,
    *,
    pages_per_seq,
    active_num_seqs,
    first_dot_bf16=False,
):
    max_kv = pages_per_seq * cache.shape[1]
    rows = []
    for seq_id in range(q_idx.shape[0]):
        if seq_id >= active_num_seqs:
            rows.append(jnp.full((max_kv,), -jnp.inf, jnp.float32))
            continue
        page_start = int(cu_kv_lens[seq_id]) // cache.shape[1]
        seq_pages = page_indices[page_start : page_start + pages_per_seq]
        keys = cache[seq_pages].reshape(max_kv, cache.shape[-1])
        query = q_idx[seq_id]
        if first_dot_bf16:
            query = query.astype(jnp.bfloat16)
            keys = keys.astype(jnp.bfloat16)
        similarities = jnp.einsum(
            "hd,kd->hk",
            query,
            keys,
            preferred_element_type=jnp.float32,
        )
        scores = jnp.einsum(
            "h,hk->k",
            idx_weights[seq_id].astype(jnp.float32),
            jax.nn.relu(similarities),
        )
        rows.append(
            jnp.where(
                jnp.arange(max_kv) < seq_lens[seq_id],
                scores,
                -jnp.inf,
            )
        )
    return jnp.stack(rows)


def _reference_extend_score_block(
    q_idx,
    idx_weights,
    cache,
    valid_lengths,
    page_indices,
    *,
    first_dot_bf16=False,
):
    keys = cache[page_indices].reshape(-1, cache.shape[-1])
    query = q_idx
    if first_dot_bf16:
        query = query.astype(jnp.bfloat16)
        keys = keys.astype(jnp.bfloat16)
    similarities = jnp.einsum(
        "bhd,kd->bhk",
        query,
        keys,
        preferred_element_type=jnp.float32,
    )
    scores = jnp.einsum(
        "bh,bhk->bk",
        idx_weights.astype(jnp.float32),
        jax.nn.relu(similarities),
    )
    return jnp.where(
        jnp.arange(keys.shape[0])[None, :] < valid_lengths[:, None],
        scores,
        -jnp.inf,
    )


@pytest.mark.parametrize("q_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("first_dot_bf16", [False, True])
@pytest.mark.parametrize(
    ("coalesce_page_dma", "contiguous_pages"),
    [(False, False), (True, False), (True, True)],
)
def test_paged_extend_score_block_pallas_interpret_matches_reference(
    q_dtype,
    first_dot_bf16,
    coalesce_page_dma,
    contiguous_pages,
):
    num_queries, num_heads, head_dim = 3, 3, 128
    page_size, pages_per_seq, block_k = 64, 4, 128
    keys = jax.random.split(jax.random.key(31), 3)
    q_idx = jax.random.normal(
        keys[0],
        (num_queries, num_heads, head_dim),
        dtype=jnp.float32,
    ).astype(q_dtype)
    idx_weights = jax.random.normal(
        keys[1],
        (num_queries, num_heads),
        dtype=jnp.float32,
    ).astype(q_dtype)
    cache = jax.random.normal(
        keys[2],
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    page_indices = jnp.asarray(
        jnp.arange(pages_per_seq) if contiguous_pages else [3, 1, 0, 2],
        jnp.int32,
    )
    valid_lengths = jnp.asarray([0, 197, 256], jnp.int32)

    actual = paged_extend_score_block_pallas(
        q_idx,
        idx_weights,
        cache,
        valid_lengths,
        page_indices,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        coalesce_page_dma=coalesce_page_dma,
        interpret=True,
    )
    expected = _reference_extend_score_block(
        q_idx,
        idx_weights,
        cache,
        valid_lengths,
        page_indices,
        first_dot_bf16=first_dot_bf16,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    ("coalesce_page_dma", "contiguous_pages"),
    [(False, False), (True, False), (True, True)],
)
def test_paged_extend_score_and_map_block_pallas_interpret_matches_reference(
    coalesce_page_dma,
    contiguous_pages,
):
    num_queries, num_heads, head_dim = 3, 3, 128
    page_size, pages_per_seq, block_k, topk = 64, 4, 128, 128
    keys = jax.random.split(jax.random.key(34), 3)
    q_idx = jax.random.normal(
        keys[0],
        (num_queries, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        keys[1],
        (num_queries, num_heads),
        dtype=jnp.float32,
    )
    cache = jax.random.normal(
        keys[2],
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    page_indices = jnp.asarray(
        jnp.arange(pages_per_seq) if contiguous_pages else [3, 1, 0, 2],
        jnp.int32,
    )
    score_valid_lengths = jnp.asarray([0, 197, 256], jnp.int32)
    mapping_valid_lengths = jnp.asarray([0, 129, 256], jnp.int32)
    logical_topk = jnp.tile(jnp.arange(topk, dtype=jnp.int32), (num_queries, 1))
    logical_topk = logical_topk.at[0, 0].set(-1)
    logical_topk = logical_topk.at[1, -1].set(255)

    actual_scores, actual_slots = paged_extend_score_and_map_block_pallas(
        q_idx,
        idx_weights,
        cache,
        score_valid_lengths,
        page_indices,
        logical_topk,
        mapping_valid_lengths,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        coalesce_page_dma=coalesce_page_dma,
        interpret=True,
    )
    expected_scores = _reference_extend_score_block(
        q_idx,
        idx_weights,
        cache,
        score_valid_lengths,
        page_indices,
    )
    safe_logical = jnp.maximum(logical_topk, 0)
    expected_slots = (
        page_indices[safe_logical // page_size] * page_size + safe_logical % page_size
    )
    expected_slots = jnp.where(
        (logical_topk >= 0) & (safe_logical < mapping_valid_lengths[:, None]),
        expected_slots,
        -1,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_scores), np.asarray(expected_scores)
    )
    np.testing.assert_array_equal(np.asarray(actual_slots), np.asarray(expected_slots))


def test_paged_extend_score_and_map_block_falls_back_for_many_page_runs():
    num_queries, num_heads, head_dim = 2, 2, 128
    page_size, pages_per_seq, block_k, topk = 64, 16, 128, 128
    q_idx = jax.random.normal(
        jax.random.key(37),
        (num_queries, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        jax.random.key(38),
        (num_queries, num_heads),
        dtype=jnp.float32,
    )
    cache = jax.random.normal(
        jax.random.key(39),
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    page_indices = jnp.asarray(
        [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
        jnp.int32,
    )
    score_valid_lengths = jnp.asarray([900, 1024], jnp.int32)
    mapping_valid_lengths = jnp.asarray([700, 1024], jnp.int32)
    logical_topk = jnp.tile(jnp.arange(topk, dtype=jnp.int32), (num_queries, 1))

    actual_scores, actual_slots = paged_extend_score_and_map_block_pallas(
        q_idx,
        idx_weights,
        cache,
        score_valid_lengths,
        page_indices,
        logical_topk,
        mapping_valid_lengths,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        interpret=True,
    )
    expected_scores = _reference_extend_score_block(
        q_idx,
        idx_weights,
        cache,
        score_valid_lengths,
        page_indices,
    )
    expected_slots = (
        page_indices[logical_topk // page_size] * page_size + logical_topk % page_size
    )
    expected_slots = jnp.where(
        logical_topk < mapping_valid_lengths[:, None],
        expected_slots,
        -1,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_scores), np.asarray(expected_scores)
    )
    np.testing.assert_array_equal(np.asarray(actual_slots), np.asarray(expected_slots))


def test_paged_decode_is_single_query_extend_semantics():
    num_heads, head_dim = 3, 128
    page_size, pages_per_seq = 64, 4
    q_idx = jax.random.normal(
        jax.random.key(35),
        (1, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        jax.random.key(36),
        (1, num_heads),
        dtype=jnp.float32,
    )
    cache = jax.random.normal(
        jax.random.key(37),
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    seq_len = jnp.asarray([211], jnp.int32)
    page_indices = jnp.asarray([2, 0, 3, 1], jnp.int32)

    decode_scores = paged_decode_scores_pallas(
        q_idx,
        idx_weights,
        cache,
        seq_len,
        page_indices,
        jnp.asarray([0, pages_per_seq * page_size], jnp.int32),
        jnp.asarray([0, 1, 1], jnp.int32),
        pages_per_seq=pages_per_seq,
        block_k=128,
        interpret=True,
    )
    extend_score_block = paged_extend_score_block_pallas(
        q_idx,
        idx_weights,
        cache,
        seq_len,
        page_indices,
        pages_per_seq=pages_per_seq,
        block_k=128,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(decode_scores),
        np.asarray(extend_score_block),
    )


@pytest.mark.parametrize("q_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("active_num_seqs", [0, 1, 2])
@pytest.mark.parametrize("first_dot_bf16", [False, True])
@pytest.mark.parametrize("persistent_two_seq", [False, True])
@pytest.mark.parametrize(
    ("coalesce_page_dma", "contiguous_pages"),
    [(False, False), (True, False), (True, True)],
)
def test_paged_decode_scores_pallas_interpret_matches_reference(
    q_dtype,
    active_num_seqs,
    first_dot_bf16,
    persistent_two_seq,
    coalesce_page_dma,
    contiguous_pages,
):
    num_seqs, num_heads, head_dim = 2, 3, 128
    page_size, pages_per_seq, block_k = 64, 4, 128
    keys = jax.random.split(jax.random.key(41), 3)
    q_idx = jax.random.normal(
        keys[0],
        (num_seqs, num_heads, head_dim),
        dtype=jnp.float32,
    ).astype(q_dtype)
    idx_weights = jax.random.normal(
        keys[1],
        (num_seqs, num_heads),
        dtype=jnp.float32,
    ).astype(q_dtype)
    cache = jax.random.normal(
        keys[2],
        (num_seqs * pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    seq_lens = jnp.asarray([197, 255], jnp.int32)
    page_indices = jnp.asarray(
        jnp.arange(8) if contiguous_pages else [3, 1, 0, 2, 7, 5, 4, 6],
        jnp.int32,
    )
    cu_kv_lens = jnp.asarray([0, 3 * page_size, 7 * page_size], jnp.int32)
    distribution = jnp.asarray(
        [0, active_num_seqs, active_num_seqs],
        jnp.int32,
    )

    actual = paged_decode_scores_pallas(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        distribution,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        coalesce_page_dma=coalesce_page_dma,
        interpret=True,
    )
    expected = _reference_scores(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        pages_per_seq=pages_per_seq,
        active_num_seqs=active_num_seqs,
        first_dot_bf16=first_dot_bf16,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("num_seqs", [4, 8, 16])
@pytest.mark.parametrize("coalesce_page_dma", [False, True])
def test_paged_decode_grouped_two_seq_interpret_matches_reference(
    num_seqs,
    coalesce_page_dma,
):
    num_heads, head_dim = 3, 128
    page_size, pages_per_seq, block_k = 64, 4, 128
    keys = jax.random.split(jax.random.key(45 + num_seqs), 3)
    q_idx = jax.random.normal(
        keys[0],
        (num_seqs, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        keys[1],
        (num_seqs, num_heads),
        dtype=jnp.float32,
    )
    total_pages = num_seqs * pages_per_seq
    cache = jax.random.normal(
        keys[2],
        (total_pages, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    seq_lens = jnp.arange(num_seqs, dtype=jnp.int32) % 57 + 192
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    if not coalesce_page_dma:
        page_indices = page_indices.reshape(num_seqs, pages_per_seq)[:, ::-1].reshape(
            -1
        )
    cu_kv_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * pages_per_seq * page_size
    distribution = jnp.asarray([0, num_seqs, num_seqs], jnp.int32)

    actual = paged_decode_scores_pallas(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        distribution,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        persistent_two_seq=True,
        coalesce_page_dma=coalesce_page_dma,
        interpret=True,
    )
    expected = _reference_scores(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        pages_per_seq=pages_per_seq,
        active_num_seqs=num_seqs,
        first_dot_bf16=False,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("first_dot_bf16", [False, True])
@pytest.mark.parametrize("persistent_two_seq", [False, True])
@pytest.mark.parametrize("coalesce_page_dma", [False, True])
@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="Pallas TPU kernel requires TPU"
)
def test_paged_decode_scores_pallas_tpu_matches_reference(
    first_dot_bf16,
    persistent_two_seq,
    coalesce_page_dma,
):
    num_seqs, num_heads, head_dim = 2, 32, 128
    page_size, pages_per_seq = 64, 4
    q_idx = jax.random.normal(
        jax.random.key(51),
        (num_seqs, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        jax.random.key(52),
        (num_seqs, num_heads),
        dtype=jnp.bfloat16,
    )
    cache = jax.random.normal(
        jax.random.key(53),
        (num_seqs * pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    seq_lens = jnp.asarray([223, 255], jnp.int32)
    page_indices = jnp.arange(8, dtype=jnp.int32)
    cu_kv_lens = jnp.asarray([0, 3 * page_size, 7 * page_size], jnp.int32)
    distribution = jnp.asarray([0, num_seqs, num_seqs], jnp.int32)

    actual = paged_decode_scores_pallas(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        distribution,
        pages_per_seq=pages_per_seq,
        block_k=128,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        coalesce_page_dma=coalesce_page_dma,
    )
    expected = _reference_scores(
        q_idx,
        idx_weights,
        cache,
        seq_lens,
        page_indices,
        cu_kv_lens,
        pages_per_seq=pages_per_seq,
        active_num_seqs=num_seqs,
        first_dot_bf16=first_dot_bf16,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("first_dot_bf16", [False, True])
@pytest.mark.parametrize("coalesce_page_dma", [False, True])
@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="Pallas TPU kernel requires TPU"
)
def test_paged_extend_score_block_pallas_tpu_matches_reference(
    first_dot_bf16,
    coalesce_page_dma,
):
    num_queries, num_heads, head_dim = 4, 32, 128
    page_size, pages_per_seq = 64, 4
    q_idx = jax.random.normal(
        jax.random.key(61),
        (num_queries, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        jax.random.key(62),
        (num_queries, num_heads),
        dtype=jnp.bfloat16,
    )
    cache = jax.random.normal(
        jax.random.key(63),
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    valid_lengths = jnp.asarray([0, 127, 223, 256], jnp.int32)
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    actual = paged_extend_score_block_pallas(
        q_idx,
        idx_weights,
        cache,
        valid_lengths,
        page_indices,
        pages_per_seq=pages_per_seq,
        block_k=128,
        first_dot_bf16=first_dot_bf16,
        coalesce_page_dma=coalesce_page_dma,
    )
    expected = _reference_extend_score_block(
        q_idx,
        idx_weights,
        cache,
        valid_lengths,
        page_indices,
        first_dot_bf16=first_dot_bf16,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="Pallas TPU kernel requires TPU"
)
def test_paged_extend_score_block_pallas_tpu_production_shape_smoke():
    """Compile the v7x Extend schedule at Bq=32 and the 135168 KV bucket."""

    num_queries, num_heads, head_dim = 32, 32, 128
    page_size, pages_per_seq = 64, 2112
    max_kv = page_size * pages_per_seq
    q_idx = jnp.ones((num_queries, num_heads, head_dim), jnp.float32)
    idx_weights = jnp.ones((num_queries, num_heads), jnp.bfloat16)
    cache = jnp.ones((pages_per_seq, page_size, head_dim), jnp.bfloat16)
    valid_lengths = jnp.full((num_queries,), max_kv, jnp.int32)
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    actual = paged_extend_score_block_pallas(
        q_idx,
        idx_weights,
        cache,
        valid_lengths,
        page_indices,
        pages_per_seq=pages_per_seq,
        coalesce_page_dma=True,
    )
    actual = np.asarray(actual)

    assert actual.shape == (num_queries, max_kv)
    np.testing.assert_array_equal(actual, np.float32(num_heads * head_dim))


@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="Pallas TPU kernel requires TPU"
)
def test_paged_extend_score_topk_pipeline_tpu_matches_reference():
    """Exercise two Pallas score blocks inside the Extend score/Top-K loop."""

    num_queries, num_heads, head_dim = 64, 32, 128
    page_size, pages_per_seq, k = 64, 4, 8
    max_kv = page_size * pages_per_seq
    q_idx = jax.random.normal(
        jax.random.key(71),
        (num_queries, num_heads, head_dim),
        dtype=jnp.float32,
    )
    idx_weights = jax.random.normal(
        jax.random.key(72),
        (num_queries, num_heads),
        dtype=jnp.bfloat16,
    )
    cache = jax.random.normal(
        jax.random.key(73),
        (pages_per_seq, page_size, head_dim),
        dtype=jnp.bfloat16,
    )
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    actual = compute_scores_and_select_topk_indices(
        q_idx,
        idx_weights,
        cache,
        jnp.asarray([max_kv], jnp.int32),
        page_indices,
        jnp.asarray([0, num_queries], jnp.int32),
        jnp.asarray([0, max_kv], jnp.int32),
        jnp.asarray([0, 1, 1], jnp.int32),
        k=k,
        pages_per_seq=pages_per_seq,
        topk_impl="exact_lax",
        score_query_block_size=32,
    )

    scores = _reference_extend_score_block(
        q_idx,
        idx_weights,
        cache,
        jnp.arange(max_kv - num_queries + 1, max_kv + 1, dtype=jnp.int32),
        page_indices,
    )
    expected = jax.lax.top_k(scores, k)[1]
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
