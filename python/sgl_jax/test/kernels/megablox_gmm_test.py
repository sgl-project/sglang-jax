import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm


def _reference_gmm(lhs, rhs, group_sizes, start_group=0):
    m = lhs.shape[0]
    n = rhs.shape[1]
    out = np.zeros((m, n), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(np.asarray(group_sizes, dtype=np.int32))])
    for local_group, rhs_group in enumerate(range(start_group, start_group + rhs.shape[0])):
        start = offsets[rhs_group]
        end = offsets[rhs_group + 1]
        if end > start:
            out[start:end] = np.asarray(lhs[start:end], dtype=np.float32) @ np.asarray(
                rhs[local_group], dtype=np.float32
            ).T
    return out


def test_gmm_handles_irregular_m():
    lhs = jnp.arange(13 * 16, dtype=jnp.float32).reshape(13, 16) / 17.0
    rhs = jnp.arange(4 * 8 * 16, dtype=jnp.float32).reshape(4, 8, 16) / 23.0
    group_sizes = jnp.array([3, 4, 2, 4], dtype=jnp.int32)

    actual = gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=jnp.float32,
        tiling=(8, 8, 8),
        interpret=True,
    )
    expected = _reference_gmm(lhs, rhs, group_sizes)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_gmm_handles_irregular_m_with_group_offset():
    lhs = jnp.arange(13 * 16, dtype=jnp.float32).reshape(13, 16) / 19.0
    rhs_full = jnp.arange(4 * 8 * 16, dtype=jnp.float32).reshape(4, 8, 16) / 29.0
    rhs = rhs_full[2:]
    group_sizes = jnp.array([3, 4, 2, 4], dtype=jnp.int32)

    actual = gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=jnp.float32,
        tiling=(8, 8, 8),
        group_offset=jnp.array(2, dtype=jnp.int32),
        interpret=True,
    )
    expected = _reference_gmm(lhs, rhs, group_sizes, start_group=2)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
