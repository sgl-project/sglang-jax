"""CPU: TSA zigzag (block-strided) token assignment helpers.

The contiguous a2a split after ``tsa_zigzag_permute`` must hand device k exactly the
rows ``tsa_zigzag_token_index(k)`` of the original token order, and
``tsa_zigzag_unpermute`` must invert the rank-major concatenation.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from sgl_jax.srt.layers.attention.dsa_sparse_backend import (  # noqa: E402
    tsa_zigzag_permute,
    tsa_zigzag_token_index,
    tsa_zigzag_unpermute,
)


@pytest.mark.parametrize("T, tp, qb", [(1024, 8, 16), (4096, 8, 16), (512, 4, 16), (128, 8, 16)])
def test_zigzag_permute_split_matches_token_index(T, tp, qb):
    x = np.arange(T * 3 * 2, dtype=np.int32).reshape(T, 3, 2)
    xp = np.asarray(tsa_zigzag_permute(x, tp, qb))
    t_local = T // tp
    seen = []
    for k in range(tp):
        idx = np.asarray(tsa_zigzag_token_index(t_local, tp, qb, k))
        assert idx.min() >= 0 and idx.max() < T
        np.testing.assert_array_equal(xp[k * t_local : (k + 1) * t_local], x[idx])
        # device k owns blocks k, k+tp, ...; each block is qb consecutive tokens
        blocks = idx.reshape(-1, qb) // qb
        assert (blocks == blocks[:, :1]).all()
        np.testing.assert_array_equal(blocks[:, 0], np.arange(k, T // qb, tp))
        seen.append(idx)
    assert sorted(np.concatenate(seen).tolist()) == list(range(T))
    # rank-major concatenation of device slices -> unpermute restores token order
    np.testing.assert_array_equal(np.asarray(tsa_zigzag_unpermute(xp, tp, qb)), x)


def test_zigzag_rejects_non_divisible_T():
    with pytest.raises((ValueError, TypeError)):
        tsa_zigzag_permute(np.zeros((192, 2), np.float32), 8, 16)
