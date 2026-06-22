"""Training-safe grouped-topk Pallas kernel tests.

The Pallas kernel in this variant returns only top-k ids. The differentiable weights are gathered
outside the kernel so reverse-mode autodiff follows JAX's gather rule instead of transposing Pallas.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.grouped_topk.topk_v1_training import (
    grouped_topk_ids_pallas,
    grouped_topk_pallas_training,
)
from sgl_jax.test.kernels.grouped_topk_test import ref_biased_grouped_topk


def _logits(bs, e, seed):
    return jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(seed), (bs, e), dtype=jnp.float32))


def test_grouped_topk_training_eq_ref():
    bs, e, n_group, topk_group, topk = 256, 256, 8, 4, 8
    logits = _logits(bs, e, seed=17)
    bias = jax.random.normal(jax.random.PRNGKey(19), (e,), dtype=jnp.float32) * 0.1

    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=n_group, topk_group=topk_group, topk=topk
    )
    w_pal, ids_pal = grouped_topk_pallas_training(
        logits,
        bias,
        num_expert_group=n_group,
        topk_group=topk_group,
        topk=topk,
        block_tokens=256,
        interpret=True,
    )

    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_ref), rtol=0, atol=1e-6)


def test_grouped_topk_ids_kernel_returns_ids_only():
    logits = _logits(32, 64, seed=23)
    bias = jnp.zeros((64,), dtype=jnp.float32)

    ids = grouped_topk_ids_pallas(
        logits,
        bias,
        num_expert_group=8,
        topk_group=4,
        topk=8,
        block_tokens=32,
        interpret=True,
    )

    assert ids.shape == (32, 8)
    assert ids.dtype == jnp.int32


def test_grouped_topk_training_weights_are_differentiable():
    bs, e, n_group, topk_group, topk = 32, 64, 8, 4, 8
    logits = _logits(bs, e, seed=29)
    bias = jax.random.normal(jax.random.PRNGKey(31), (e,), dtype=jnp.float32) * 0.1

    _, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=n_group, topk_group=topk_group, topk=topk
    )
    expected_grad = (
        jnp.zeros_like(logits)
        .at[jnp.arange(bs)[:, None], ids_ref]
        .add(jnp.ones((bs, topk), dtype=jnp.float32))
    )

    def loss_fn(x):
        weights, _ = grouped_topk_pallas_training(
            x,
            bias,
            num_expert_group=n_group,
            topk_group=topk_group,
            topk=topk,
            block_tokens=32,
            interpret=True,
        )
        return jnp.sum(weights)

    grad = jax.grad(loss_fn)(logits)
    np.testing.assert_allclose(np.array(grad), np.array(expected_grad), rtol=0, atol=1e-6)
