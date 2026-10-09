"""V4's short-vector permutation matches the stable sorting reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers import moe


@pytest.mark.parametrize("n,experts,seed", [(512, 256, 0), (896, 256, 1), (8, 4, 2), (64, 1, 3)])
def test_small_permutations_match_argsort(n, experts, seed):
    rng = np.random.default_rng(seed)
    keys = jnp.asarray(rng.integers(0, experts, size=n), jnp.int32)
    forward = np.asarray(moe._stable_argsort_small(keys))
    np.testing.assert_array_equal(forward, np.asarray(jnp.argsort(keys, stable=True)))
    inverse = np.asarray(moe._inverse_permutation_small(jnp.asarray(forward)))
    np.testing.assert_array_equal(inverse, np.asarray(jnp.argsort(jnp.asarray(forward))))
    np.testing.assert_array_equal(inverse[forward], np.arange(n))


def test_large_vectors_fall_back_to_sort(monkeypatch):
    monkeypatch.setattr(moe, "_RANK_SORT_MAX_ENTRIES", 16)
    keys = jnp.asarray(np.random.default_rng(4).integers(0, 8, size=40), jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(moe._stable_argsort_small(keys)), np.asarray(jnp.argsort(keys, stable=True))
    )
