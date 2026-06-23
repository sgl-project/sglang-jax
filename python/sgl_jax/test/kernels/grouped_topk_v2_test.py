"""Equivalence tests for the stable lowest-index tie-break grouped-topk V2 (inference) kernel.

V2 must match `jax.lax.top_k` (the reference) id-for-id, INCLUDING the order among experts/groups
with equal post-bias scores. Runs in CPU interpret mode. (On CPU `jnp.argmax` already breaks ties
toward the lowest index, so V1 also passes these; the V1-vs-V2 divergence only shows on TPU — see
the on-device comparison. These tests pin V2's contract against the reference regardless of backend.)

Run:  PALLAS_INTERPRET=1 python -m pytest python/sgl_jax/test/kernels/grouped_topk_v2_test.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.grouped_topk.grouped_topk_v2 import grouped_topk_pallas_v2
from sgl_jax.test.kernels.grouped_topk_test import ref_biased_grouped_topk


def _logits(bs, e, seed):
    return jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(seed), (bs, e), dtype=jnp.float32))


CONFIGS = [
    (256, 8, 4, 8, "A_E256"),
    (512, 8, 4, 8, "B_E512"),
    (128, 4, 2, 6, "small_E128"),
]
BATCHES = [256, 512, 1024]


@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
@pytest.mark.parametrize("bs", BATCHES)
def test_v2_eq_ref(E, G, Gtop, k, name, bs):
    logits = _logits(bs, E, seed=2)
    bias = jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_v2, ids_v2 = grouped_topk_pallas_v2(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=256, interpret=True
    )
    np.testing.assert_array_equal(
        np.array(ids_v2), np.array(ids_ref), err_msg=f"{name} bs={bs}: ids"
    )
    np.testing.assert_allclose(
        np.array(w_v2), np.array(w_ref), rtol=0, atol=1e-6, err_msg=f"{name} bs={bs}: weights"
    )


def test_v2_matches_ref_on_flat_ties():
    # All scores equal -> reference returns the lowest indices in order; V2 must match exactly.
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = jnp.full((bs, E), 0.5, dtype=jnp.float32)
    bias = jnp.zeros((E,), dtype=jnp.float32)
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_v2, ids_v2 = grouped_topk_pallas_v2(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=bs, interpret=True
    )
    np.testing.assert_array_equal(np.array(ids_v2), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_v2), np.array(w_ref), rtol=0, atol=1e-6)


def test_v2_matches_ref_on_partial_ties():
    # Force a within-group tie (experts 3 and 5 share a score) with distinct pre-bias weights, so a
    # wrong tie-break would swap their topk positions / gathered weights. V2 must match the reference.
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = _logits(bs, E, seed=11)
    logits = logits.at[:, 5].set(logits[:, 3])
    bias = jax.random.normal(jax.random.PRNGKey(5), (E,), dtype=jnp.float32) * 0.1
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_v2, ids_v2 = grouped_topk_pallas_v2(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=bs, interpret=True
    )
    np.testing.assert_array_equal(np.array(ids_v2), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_v2), np.array(w_ref), rtol=0, atol=1e-6)
