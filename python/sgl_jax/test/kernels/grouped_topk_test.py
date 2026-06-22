"""Equivalence test: Pallas argmax-selection grouped-topk == gate.py `_biased_grouped_topk`.

Runs the Pallas kernel in CPU interpret mode (no TPU needed) and checks it is id-for-id identical
to the reference sort-based routing (verbatim copy of `gate.py:TopK._biased_grouped_topk`,
L159-193), with matching weights.

Run:  PALLAS_INTERPRET=1 python -m pytest python/sgl_jax/test/kernels/grouped_topk_test.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.grouped_topk.grouped_topk import grouped_topk_pallas


def ref_biased_grouped_topk(router_logits, correction_bias, *, num_expert_group, topk_group, topk):
    """Verbatim `gate.py:TopK._biased_grouped_topk` (the sort-based reference)."""
    router_logits = router_logits.astype(jnp.float32)
    num_token = router_logits.shape[0]
    scores_for_choice = router_logits.reshape(num_token, -1) + jnp.expand_dims(correction_bias, 0)
    scores_grouped = scores_for_choice.reshape(num_token, num_expert_group, -1)
    group_scores = jnp.sum(jax.lax.top_k(scores_grouped, k=2)[0], axis=-1)
    group_idx = jax.lax.top_k(group_scores, k=topk_group)[1]
    group_mask = jnp.clip(jax.nn.one_hot(group_idx, num_expert_group).sum(axis=1), 0, 1)
    experts_per_group = router_logits.shape[-1] // num_expert_group
    score_mask = jnp.expand_dims(group_mask, axis=-1)
    score_mask = jnp.broadcast_to(score_mask, (num_token, num_expert_group, experts_per_group))
    score_mask = score_mask.reshape(num_token, -1)
    tmp_scores = jnp.where(score_mask, scores_for_choice, float("-inf"))
    topk_ids = jax.lax.top_k(tmp_scores, k=topk)[1]
    topk_weights = jnp.take_along_axis(router_logits, topk_ids, axis=1)
    return topk_weights, topk_ids


def _logits(bs, e, seed):
    # sigmoid'd gate output, like the real router_logits fed to TopK
    return jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(seed), (bs, e), dtype=jnp.float32))


CONFIGS = [
    # (E, G, Gtop, k, name)
    (256, 8, 4, 8, "A_E256_G8_Gtop4_k8"),  # sgl-jax DeepSeek-V3 / Ling
    (512, 8, 4, 8, "B_E512_G8_Gtop4_k8"),  # MaxText
    (128, 4, 2, 6, "small_E128_G4_Gtop2_k6"),
]
BATCHES = [256, 512, 1024]


@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
@pytest.mark.parametrize("bs", BATCHES)
def test_pallas_eq_ref(E, G, Gtop, k, name, bs):
    logits = _logits(bs, E, seed=2)
    bias = jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1

    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits,
        bias,
        num_expert_group=G,
        topk_group=Gtop,
        topk=k,
        block_tokens=256,
        interpret=True,
    )

    # exact expert-id match (descending-by-score order, ties by lowest index — both agree)
    np.testing.assert_array_equal(
        np.array(ids_pal),
        np.array(ids_ref),
        err_msg=f"{name} bs={bs}: top-k ids differ",
    )
    # weights are the pre-bias logits at those ids → exact up to fp
    np.testing.assert_allclose(
        np.array(w_pal),
        np.array(w_ref),
        rtol=0,
        atol=1e-6,
        err_msg=f"{name} bs={bs}: weights differ",
    )


def test_against_real_topk_module():
    """If the sgl_jax model stack imports cleanly, also check vs the real TopK module."""
    real_topk = pytest.importorskip("sgl_jax.srt.layers.gate", reason="gate.py import").TopK
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = _logits(bs, E, seed=7)
    bias = jax.random.normal(jax.random.PRNGKey(3), (E,), dtype=jnp.float32) * 0.1
    mod = real_topk(topk=k, renormalize=False, num_expert_group=G, topk_group=Gtop)
    w_real, ids_real = mod._biased_grouped_topk(logits, bias)
    w_pal, ids_pal = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, interpret=True
    )
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_real))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_real), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "E,G,Gtop,k,name",
    [
        (256, 8, 4, 8, "n_pad>0_k8"),  # padded_topk=128, n_pad=120 (filler-column path)
        (256, 4, 4, 128, "n_pad==0_k128"),  # padded_topk=128, n_pad=0 (no filler)
    ],
)
def test_topk_pad_boundary(E, G, Gtop, k, name):
    """Exercise both the n_pad>0 (topk<128) and n_pad==0 (topk==128) output-padding paths,
    and that the returned shape is sliced back to exactly (bs, topk)."""
    bs = 512
    logits = _logits(bs, E, seed=11)
    bias = jax.random.normal(jax.random.PRNGKey(5), (E,), dtype=jnp.float32) * 0.1
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=256, interpret=True
    )
    assert ids_pal.shape == (bs, k), f"{name}: shape {ids_pal.shape} != {(bs, k)}"
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref), err_msg=f"{name}: ids")
    np.testing.assert_allclose(
        np.array(w_pal), np.array(w_ref), rtol=0, atol=1e-6, err_msg=f"{name}: weights"
    )
