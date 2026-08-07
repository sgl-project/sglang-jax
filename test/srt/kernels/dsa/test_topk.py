"""Dispatch and ABI tests for DSA Indexer top-k selection."""

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.topk import select_indexer_topk


def test_exact_lax_matches_jax_top_k():
    scores = jax.random.normal(jax.random.key(0), (3, 37), dtype=jnp.float32)

    expected_values, expected_indices = jax.lax.top_k(scores, 7)
    actual_values, actual_indices = select_indexer_topk(
        scores,
        k=7,
        implementation="exact_lax",
    )

    np.testing.assert_array_equal(np.asarray(actual_values), np.asarray(expected_values))
    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))


def test_radix_dispatch_pads_and_preserves_unordered_exact_set(monkeypatch):
    scores = jax.random.normal(jax.random.key(1), (2, 130), dtype=jnp.float32)
    called = False

    def fake_radix_topk(keys, *, k, **kwargs):
        nonlocal called
        called = True
        assert keys.shape == (2, 256)
        assert kwargs == {
            "use_approx_top_k": False,
            "num_seq_windows": 1,
            "digit_width": 8,
            "num_digits": 4,
            "use_tc_tiling_on_sc": False,
        }
        values, indices = jax.lax.top_k(keys, k)
        # The real radix kernel returns an exact, potentially unordered set.
        return values[:, ::-1], indices[:, ::-1]

    monkeypatch.setitem(
        sys.modules,
        "sgl_jax.srt.kernels.radix_topk",
        SimpleNamespace(radix_topk_pallas=fake_radix_topk),
    )

    expected_values, expected_indices = jax.lax.top_k(scores, 5)
    actual_values, actual_indices = select_indexer_topk(
        scores,
        k=5,
        implementation="radix",
    )

    assert called
    np.testing.assert_array_equal(np.asarray(actual_values), np.asarray(expected_values)[:, ::-1])
    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices)[:, ::-1])


def test_radix_dispatch_uses_score_size_topk_tuned_config(monkeypatch):
    from sgl_jax.srt.kernels.dsa import topk as dsa_topk
    from sgl_jax.srt.kernels.radix_topk.tuned_configs import RadixTopKConfig

    scores = jax.random.normal(jax.random.key(2), (1, 513), dtype=jnp.float32)
    config = RadixTopKConfig(
        num_seq_windows=1,
        digit_width=8,
        num_digits=4,
        use_tc_tiling_on_sc=True,
    )

    def fake_lookup(score_size, topk):
        assert (score_size, topk) == (513, 9)
        return config

    def fake_radix_topk(keys, *, k, **kwargs):
        assert keys.shape == (1, 768)
        assert kwargs == {
            "use_approx_top_k": False,
            "num_seq_windows": 1,
            "digit_width": 8,
            "num_digits": 4,
            "use_tc_tiling_on_sc": True,
        }
        return jax.lax.top_k(keys, k)

    monkeypatch.setattr(dsa_topk, "get_tuned_radix_topk_config", fake_lookup)
    monkeypatch.setitem(
        sys.modules,
        "sgl_jax.srt.kernels.radix_topk",
        SimpleNamespace(radix_topk_pallas=fake_radix_topk),
    )

    expected_values, expected_indices = jax.lax.top_k(scores, 9)
    actual_values, actual_indices = select_indexer_topk(
        scores,
        k=9,
        implementation="radix",
    )

    np.testing.assert_array_equal(np.asarray(actual_values), np.asarray(expected_values))
    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))


@pytest.mark.parametrize(
    ("scores", "error", "match"),
    [
        (jnp.zeros((2, 3, 4), jnp.float32), ValueError, "rank 2"),
        (jnp.zeros((2, 4), jnp.bfloat16), TypeError, "float32"),
    ],
)
def test_rejects_invalid_score_abi(scores, error, match):
    with pytest.raises(error, match=match):
        select_indexer_topk(scores, k=2, implementation="exact_lax")


def test_rejects_unknown_implementation():
    with pytest.raises(ValueError, match="unknown DSA top-k implementation"):
        select_indexer_topk(
            jnp.zeros((2, 8), jnp.float32),
            k=2,
            implementation="unknown",
        )
