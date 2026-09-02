# SPDX-License-Identifier: Apache-2.0
"""EPMoE._unpermute takes 2-D [tokens, top_k] routing weights and rejects anything else."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.layers.moe import EPMoE

HIDDEN, EXPERTS, TOP_K, TOKENS = 8, 4, 2, 6


def _epmoe():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("data", "tensor"))
    import sgl_jax.srt.layers.moe as moe_module

    with mock.patch.object(moe_module, "P", lambda *a: None):
        return EPMoE(
            hidden_size=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=HIDDEN,
        )


def test_unpermute():
    moe = _epmoe()
    k_ids, k_w, k_x = jax.random.split(jax.random.PRNGKey(0), 3)
    ids = jax.random.randint(k_ids, (TOKENS, TOP_K), 0, EXPERTS, dtype=jnp.int32)
    weights = jax.random.uniform(k_w, (TOKENS, TOP_K)).astype(jnp.bfloat16)
    # _permute sorts the flattened (token, slot) pairs by expert id.
    order = jnp.argsort(jnp.ravel(ids), stable=True)
    gmm_out = jax.random.normal(k_x, (TOKENS * TOP_K, HIDDEN), dtype=jnp.bfloat16)

    out = moe._unpermute(gmm_out, order, weights)

    inverse = np.zeros(TOKENS * TOP_K, dtype=int)
    inverse[np.asarray(order)] = np.arange(TOKENS * TOP_K)
    rows = np.asarray(gmm_out, np.float32)[inverse].reshape(TOKENS, TOP_K, HIDDEN)
    expected = np.einsum("tkh,tk->th", rows, np.asarray(weights, np.float32))
    np.testing.assert_allclose(np.asarray(out, np.float32), expected, rtol=2e-2, atol=2e-2)

    # GMM pads its output to a row alignment; the extra rows must be dropped.
    padded = jnp.concatenate([gmm_out, jnp.ones((32, HIDDEN), gmm_out.dtype)])
    np.testing.assert_array_equal(
        np.asarray(moe._unpermute(padded, order, weights)), np.asarray(out)
    )

    # sglang-jax is token-flat, so anything but [tokens, top_k] is a caller bug --
    # including (tokens * top_k, 1), which reshapes cleanly but means something else.
    for bad_shape in ((1, TOKENS, TOP_K), (TOKENS * TOP_K, 1)):
        with pytest.raises(ValueError, match="expects 2-D routing weights"):
            moe._unpermute(gmm_out, order, jnp.reshape(weights, bad_shape))


if __name__ == "__main__":
    test_unpermute()
    print("PASSED")
