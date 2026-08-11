"""Correctness tests for the paged-cache DSA decode scorer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.paged_score import paged_decode_scores_pallas


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


@pytest.mark.parametrize("first_dot_bf16", [False, True])
@pytest.mark.parametrize("persistent_two_seq", [False, True])
@pytest.mark.parametrize("coalesce_page_dma", [False, True])
@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="Pallas TPU kernel requires TPU")
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
