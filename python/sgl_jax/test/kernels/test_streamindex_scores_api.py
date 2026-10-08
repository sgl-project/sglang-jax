"""Trace the real StreamIndex API and exit stage on CPU.

Only the TPU-only score-producing Pallas call is substituted; these checks
do not establish the numerical correctness of the TPU scorer.
"""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.indexer import csa_indexer_topk_kernel

kernel = importlib.import_module("sgl_jax.srt.kernels.dsa.streamindex_topk")


@pytest.fixture
def raw_scores(monkeypatch):
    scores = np.full((3, 128), -np.inf, np.float32)
    scores[0, [2, 5]] = [9, 4]
    scores[2, [70, 127]] = [7, 3]

    def scored_call(*args, out_shape, **kwargs):
        assert out_shape.shape[0] == 3 and out_shape.shape[-1] == 128
        width = out_shape.shape[1] * 128
        padded = jnp.pad(jnp.asarray(scores), ((0, 0), (0, width - 128)), constant_values=-jnp.inf)
        return lambda *inputs: padded.reshape(out_shape.shape)

    kernel.streamindex_topk.clear_cache()
    monkeypatch.setattr(kernel.pl, "pallas_call", scored_call)
    yield scores
    kernel.streamindex_topk.clear_cache()


def _call(**kwargs):
    return kernel.streamindex_topk(
        q=jnp.zeros((3, 2, 128), jnp.bfloat16),
        indexer_weights=jnp.ones((3, 2), jnp.float32),
        cache_kv=jnp.zeros((1, 64, 2, 128), jnp.bfloat16),
        seq_lens=jnp.asarray([16], jnp.int32),
        page_indices=jnp.asarray([0], jnp.int32),
        cu_q_lens=jnp.asarray([0, 3], jnp.int32),
        distribution=jnp.asarray([0, 0, 1], jnp.int32),
        k=4,
        compression_ratio=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=8,
        topk_backend="xla",
        **kwargs,
    )


def test_scores_mode_preserves_masked_scores_and_skips_selection(raw_scores, monkeypatch):
    def unexpected_selection(*args, **kwargs):
        raise AssertionError("score mode must skip top-K selection")

    monkeypatch.setattr(kernel, "select_topk_indices", unexpected_selection)
    got = _call(return_scores=True)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), raw_scores)


@pytest.mark.parametrize("kwargs", [{}, {"return_scores": False}])
def test_default_and_explicit_index_modes_keep_invalid_padding(raw_scores, kwargs):
    got = _call(**kwargs)
    assert got.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(got), [[2, 5, -1, -1], [-1, -1, -1, -1], [70, 127, -1, -1]]
    )


def test_csa_caller_reaches_real_scores_api(raw_scores):
    scores, offsets = csa_indexer_topk_kernel(
        jnp.zeros((3, 2, 128), jnp.bfloat16),
        jnp.ones((3, 2), jnp.float32),
        jnp.zeros((128, 128), jnp.bfloat16),
        compressed_rows=jnp.arange(4, dtype=jnp.int32),
        seq_lens=jnp.asarray([16], jnp.int32),
        q_lens=jnp.asarray([3], jnp.int32),
        cu_q_lens=jnp.asarray([0, 3], jnp.int32),
        query_request_ids=jnp.zeros(3, jnp.int32),
        valid_token_mask=jnp.ones(3, bool),
        k=4,
        ratio=4,
        compressed_page_size=32,
        return_scores=True,
    )
    np.testing.assert_array_equal(np.asarray(scores[:, :128]), raw_scores)
    assert np.isneginf(np.asarray(scores[:, 128:])).all()
    np.testing.assert_array_equal(np.asarray(offsets), [0])
