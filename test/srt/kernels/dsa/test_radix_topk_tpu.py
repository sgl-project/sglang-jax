"""TPU equivalence test for the SparseCore radix top-k kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.radix_topk import radix_topk_pallas


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="SparseCore requires TPU")
@pytest.mark.parametrize("width", [4096, 8192])
def test_radix_topk_matches_lax_membership(width):
    scores = jax.random.normal(jax.random.key(width), (2, width), dtype=jnp.float32)
    k = 128

    expected_values, _ = jax.lax.top_k(scores, k)
    actual_values, actual_indices = radix_topk_pallas(scores, k=k)
    actual_values = jnp.take_along_axis(scores, actual_indices, axis=-1)

    np.testing.assert_array_equal(
        np.sort(np.asarray(actual_values), axis=-1),
        np.sort(np.asarray(expected_values), axis=-1),
    )
