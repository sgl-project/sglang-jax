"""The argsort-based un-permute index map equals the scatter-based one bit for bit."""

import unittest

import jax.numpy as jnp
import numpy as np


def _scatter_inverse(sorted_selected_experts, n):
    return (
        jnp.zeros(n, dtype=jnp.int32)
        .at[sorted_selected_experts]
        .set(jnp.arange(n, dtype=jnp.int32))
    )


class InvSortUnpermuteTest(unittest.TestCase):
    def test_matches_scatter_for_random_routing(self):
        rng = np.random.default_rng(0)
        for tokens, top_k, experts in [(4, 8, 36), (256, 8, 36), (4096, 8, 36)]:
            top_k_indices = rng.integers(0, experts, size=(tokens, top_k)).astype(np.int32)
            flat = jnp.ravel(jnp.asarray(top_k_indices))
            sorted_sel = jnp.argsort(flat, stable=True)
            ref = _scatter_inverse(sorted_sel, tokens * top_k)
            got = jnp.argsort(sorted_sel)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
            self.assertEqual(got.dtype, jnp.int32)
            # it is an inverse permutation
            np.testing.assert_array_equal(
                np.asarray(sorted_sel)[np.asarray(got)], np.arange(tokens * top_k)
            )


if __name__ == "__main__":
    unittest.main()
