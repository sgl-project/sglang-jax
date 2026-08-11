"""TPU equivalence test for the SparseCore radix top-k kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.radix_topk import radix_topk_pallas


@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="SparseCore requires TPU"
)
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


@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="SparseCore requires TPU"
)
@pytest.mark.parametrize(
    ("batch_size", "width", "k"),
    [
        (4, 8192, 128),
        (14, 135168, 2048),
    ],
)
def test_radix_topk_pipelined_batch_rows_match_lax_membership(batch_size, width, k):
    scores = jax.random.normal(
        jax.random.key(batch_size + width + k),
        (batch_size, width),
        dtype=jnp.float32,
    )

    expected_values, _ = jax.lax.top_k(scores, k)
    actual_values, actual_indices = radix_topk_pallas(
        scores,
        k=k,
        use_tc_tiling_on_sc=width == 135168,
    )
    indices_only = radix_topk_pallas(
        scores,
        k=k,
        use_tc_tiling_on_sc=width == 135168,
        indices_only=True,
    )
    gathered_values = jnp.take_along_axis(scores, actual_indices, axis=-1)
    indices_only_values = jnp.take_along_axis(scores, indices_only, axis=-1)

    np.testing.assert_array_equal(
        np.sort(np.asarray(gathered_values), axis=-1),
        np.sort(np.asarray(expected_values), axis=-1),
    )
    np.testing.assert_array_equal(
        np.sort(np.asarray(actual_values), axis=-1),
        np.sort(np.asarray(expected_values), axis=-1),
    )
    np.testing.assert_array_equal(
        np.sort(np.asarray(indices_only_values), axis=-1),
        np.sort(np.asarray(expected_values), axis=-1),
    )


@pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="SparseCore requires TPU"
)
@pytest.mark.parametrize("batch_size", [2, 4])
def test_radix_topk_overlapped_histogram_clear_drains_on_early_exit(batch_size):
    """The async histogram clear must finish when the first digit finds all K."""

    width, k = 8192, 128
    scores = jnp.full((batch_size, width), -1000.0, dtype=jnp.float32)
    scores = scores.at[:, :k].set(1000.0)

    actual_values, actual_indices = radix_topk_pallas(scores, k=k)

    np.testing.assert_array_equal(
        np.asarray(actual_values), np.full((batch_size, k), 1000.0)
    )
    np.testing.assert_array_equal(
        np.sort(np.asarray(actual_indices), axis=-1),
        np.broadcast_to(np.arange(k), (batch_size, k)),
    )
