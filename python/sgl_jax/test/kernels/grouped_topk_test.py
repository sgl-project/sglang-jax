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

from sgl_jax.srt.kernels.grouped_topk.v1.kernel import (
    SAFE_AUTO_BT,
    _largest_safe_divisor,
    grouped_topk_pallas,
)


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


def ref_biased_grouped_topk_bf16(
    router_logits_bf16, correction_bias_bf16, *, num_expert_group, topk_group, topk
):
    """Faithful reference for the ``packed=True`` bf16 path.

    Mirrors the kernel exactly: group selection runs on the f32 post-bias scores, but the FINAL
    select is done on bf16-rounded scores (the kernel rounds ``masked`` to bf16 before packing the
    id into the freed low-16 mantissa bits). Weights are the PRE-bias (bf16) logits at the winners.
    """
    logits = router_logits_bf16.astype(jnp.float32)  # lossless upcast of bf16
    bias = correction_bias_bf16.astype(jnp.float32)
    num_token, E = logits.shape
    scores = logits + jnp.expand_dims(bias, 0)
    scores_grouped = scores.reshape(num_token, num_expert_group, -1)
    group_scores = jnp.sum(jax.lax.top_k(scores_grouped, k=2)[0], axis=-1)
    group_idx = jax.lax.top_k(group_scores, k=topk_group)[1]
    group_mask = jnp.clip(jax.nn.one_hot(group_idx, num_expert_group).sum(axis=1), 0, 1)
    S = E // num_expert_group
    score_mask = jnp.broadcast_to(
        jnp.expand_dims(group_mask, -1), (num_token, num_expert_group, S)
    ).reshape(num_token, -1)
    masked = jnp.where(score_mask, scores, float("-inf"))
    masked_bf16 = masked.astype(jnp.bfloat16).astype(jnp.float32)  # kernel rounds here
    topk_ids = jax.lax.top_k(masked_bf16, k=topk)[1]
    topk_weights = jnp.take_along_axis(logits, topk_ids, axis=1)
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
    "bs,expected",
    [
        (5000, None),  # 5000=2^3·5^4 has no 128-multiple divisor
        (4000, None),  # 4000=2^5·5^3 has no 128-multiple divisor
        (2048, 2048),  # 2048=128*16, a safe power-of-2 tile
        (1024, 1024),  # 1024=128*8
        (1536, 1536),  # 1536=128*12 (<=2048), a 128-aligned divisor
        (5003, None),  # prime -> no 128-aligned divisor, caller falls back to whole batch
        (100, None),  # 100=2^2·5^2 has no 128-multiple divisor
    ],
)
def test_largest_safe_divisor(bs, expected):
    """The auto fallback must pick a VMEM-safe, 128-aligned divisor of bs (tokens are in the lane
    dim, so the block must be a 128-multiple), or None."""
    d = _largest_safe_divisor(bs)
    assert d == expected, f"bs={bs}: got {d}, want {expected}"
    if d is not None:
        assert bs % d == 0 and d % 128 == 0 and d <= SAFE_AUTO_BT


def test_auto_block_tokens_nondivisible():
    """`block_tokens='auto'` on an odd bucket (divisible by neither tuned BT nor 512) must still
    produce a correct result — and not silently tile the whole batch (the Codex P2)."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 1000  # 1000 % 512 != 0; on CPU the tuned lookup misses
    logits = _logits(bs, E, seed=13)
    bias = jax.random.normal(jax.random.PRNGKey(9), (E,), dtype=jnp.float32) * 0.1
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, interpret=True
    )  # block_tokens defaults to "auto"
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_ref), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "E,G,Gtop,k,name",
    [
        (256, 8, 4, 8, "k8"),  # typical routing k
        (256, 4, 4, 128, "k128"),  # large k (topk == experts/2)
    ],
)
def test_topk_pad_boundary(E, G, Gtop, k, name):
    """Output is [BS, topk] (topk in the sublane dim, no 128 padding); check small and large k both
    return the exact (bs, topk) shape and correct values."""
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


def test_matches_ref_on_flat_ties():
    """All scores equal -> reference returns the lowest indices in order; the kernel must match
    (the stable lowest-index tie-break, vs the hardware argmax which would reorder on TPU)."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = jnp.full((bs, E), 0.5, dtype=jnp.float32)
    bias = jnp.zeros((E,), dtype=jnp.float32)
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=bs, interpret=True
    )
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_ref), rtol=0, atol=1e-6)


# --- bf16 packed-key final-select path (packed=True; caller enables it when logits are bf16) -------
#
# IMPORTANT: the packed path builds an int32 order key with bitcast_convert_type + bf16 rounding +
# bit masking. Pallas *interpret* mode does NOT emulate these bit-tricks faithfully — it diverges
# from the real Mosaic-lowered kernel (verified on v7x: interpret mismatches lax.top_k by ~0.3% of
# rows, the real kernel is bit-exact to lax.top_k). So these tests must run the REAL kernel
# (interpret=False), which requires a TPU. They are skipped off-TPU rather than giving a false pass.
# (The f32 path has no such bit-tricks and IS faithful under interpret, so those tests run anywhere.)

_ON_TPU = jax.default_backend() == "tpu"
_requires_tpu = pytest.mark.skipif(
    not _ON_TPU,
    reason="packed bf16 path uses bitcast/bf16-key ops that Pallas interpret mode does not emulate; "
    "validate against the real Mosaic kernel on TPU (interpret=False)",
)


@_requires_tpu
@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
@pytest.mark.parametrize("bs", BATCHES)
def test_packed_eq_ref_bf16(E, G, Gtop, k, name, bs):
    """The packed bf16 final-select must be id-for-id identical to the bf16 reference (group select
    in f32, final select on bf16-rounded scores) with matching pre-bias weights."""
    logits = _logits(bs, E, seed=2).astype(jnp.bfloat16)
    bias = (jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1).astype(
        jnp.bfloat16
    )
    w_ref, ids_ref = ref_biased_grouped_topk_bf16(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits,
        bias,
        num_expert_group=G,
        topk_group=Gtop,
        topk=k,
        block_tokens=256,
        interpret=False,
        packed=True,
    )
    np.testing.assert_array_equal(
        np.array(ids_pal), np.array(ids_ref), err_msg=f"{name} bs={bs}: packed ids differ"
    )
    np.testing.assert_allclose(
        np.array(w_pal), np.array(w_ref), rtol=0, atol=0, err_msg=f"{name} bs={bs}: packed weights"
    )


@_requires_tpu
def test_packed_eq_unpacked_when_scores_exact_bf16():
    """With bias=0 the final-select scores are exactly bf16 (the packed path's bf16 rounding is a
    no-op), so packed and the plain f32 path must produce identical ids and weights."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = _logits(bs, E, seed=4).astype(jnp.bfloat16)
    bias = jnp.zeros((E,), dtype=jnp.bfloat16)
    w_u, ids_u = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=256, interpret=False
    )
    w_p, ids_p = grouped_topk_pallas(
        logits,
        bias,
        num_expert_group=G,
        topk_group=Gtop,
        topk=k,
        block_tokens=256,
        interpret=False,
        packed=True,
    )
    np.testing.assert_array_equal(np.array(ids_u), np.array(ids_p))
    np.testing.assert_array_equal(np.array(w_u), np.array(w_p))


@_requires_tpu
def test_packed_flat_ties_lowest_index():
    """All scores equal -> the packed key must decode the lowest expert id per pick (experts 0..k-1),
    exercising the (score DESC, index ASC) tie-break through the pack/unpack round-trip."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = jnp.full((bs, E), 0.5, dtype=jnp.bfloat16)
    bias = jnp.zeros((E,), dtype=jnp.bfloat16)
    _, ids_pal = grouped_topk_pallas(
        logits,
        bias,
        num_expert_group=G,
        topk_group=Gtop,
        topk=k,
        block_tokens=bs,
        interpret=False,
        packed=True,
    )
    expect = np.tile(np.arange(k), (bs, 1))
    np.testing.assert_array_equal(np.array(ids_pal), expect)


@_requires_tpu
def test_packed_partial_ties():
    """Force a within-group tie (experts 3 and 5 share a score) in bf16; the packed path must still
    match the bf16 reference (a wrong tie-break would swap positions / gathered weights)."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = _logits(bs, E, seed=11)
    logits = logits.at[:, 5].set(logits[:, 3]).astype(jnp.bfloat16)
    bias = (jax.random.normal(jax.random.PRNGKey(5), (E,), dtype=jnp.float32) * 0.1).astype(
        jnp.bfloat16
    )
    w_ref, ids_ref = ref_biased_grouped_topk_bf16(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits,
        bias,
        num_expert_group=G,
        topk_group=Gtop,
        topk=k,
        block_tokens=bs,
        interpret=False,
        packed=True,
    )
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_ref), rtol=0, atol=0)


def test_matches_ref_on_partial_ties():
    """Force a within-group tie (experts 3 and 5 share a score) with distinct pre-bias weights, so a
    wrong tie-break would swap their topk positions / gathered weights. Must match the reference."""
    E, G, Gtop, k, bs = 256, 8, 4, 8, 512
    logits = _logits(bs, E, seed=11)
    logits = logits.at[:, 5].set(logits[:, 3])
    bias = jax.random.normal(jax.random.PRNGKey(5), (E,), dtype=jnp.float32) * 0.1
    w_ref, ids_ref = ref_biased_grouped_topk(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
    )
    w_pal, ids_pal = grouped_topk_pallas(
        logits, bias, num_expert_group=G, topk_group=Gtop, topk=k, block_tokens=bs, interpret=True
    )
    np.testing.assert_array_equal(np.array(ids_pal), np.array(ids_ref))
    np.testing.assert_allclose(np.array(w_pal), np.array(w_ref), rtol=0, atol=1e-6)
