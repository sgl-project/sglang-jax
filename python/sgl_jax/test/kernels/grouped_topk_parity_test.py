"""Bit-exact parity test: live grouped-topk kernel == frozen main kernel == sort-based reference.

PR #1758 adds an all-groups fast path to `_grouped_topk_kernel` (when every group is retained the
correction bias is added in the native [BT, E] layout before the transpose, and the group
selection is skipped) and claims bit-exact parity with main. This test pins that claim instead of
asserting it: the live kernel is compared against `grouped_topk_main_ref.py` (a verbatim copy of
main's kernel at c136c733) and against the `lax.top_k` reference, over the (E, G, Gtop, k) grid
used in production plus adversarial inputs -- exact ties (flat, pairwise, whole-group, dyadic
grids, post-bias-only ties), +/-inf scores and biases, NaN, signed zeros, subnormals, bf16-exact
values -- for both select modes (f32 and packed bf16-key) and multi-block grids. It also checks
that configurations outside the routing contract (e.g. `topk_group > num_expert_group`, for which
main silently produced input-dependent ids) are rejected at the entry point.

Checks per case:
  PARITY  ids identical AND weights bit-identical (uint32 view, so -0.0/+0.0 and NaN payloads
          count) between the live kernel and the frozen main kernel.
  REF     ids identical to the sort-based reference and weights equal bit-for-bit modulo signed
          zero (the one-hot masked-sum gather canonicalises -0.0 to +0.0 in main and live alike).

On TPU the real Mosaic kernel runs; elsewhere Pallas interpret mode is used, with two caveats
mirrored from grouped_topk_test.py:
  * packed=True vs REF is TPU-only (interpret does not emulate the bf16-key bit tricks faithfully).
    packed PARITY (main vs live) is still asserted everywhere: both sides share the emulation.
  * NaN inputs are TPU-only. XLA:CPU's reduce-max is not an IEEE `maximum`: under interpret the
    result depends on the reduction order XLA:CPU picks for each program, so main and live (which
    lower to different CPU programs) can disagree on a few ids when exactly one score is NaN
    while each is self-consistent. On TPU both emit the same `vector.multi_reduction<maximumf>`
    over the same [E, BT] value.

Run:  python -m pytest python/sgl_jax/test/kernels/grouped_topk_parity_test.py -q
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas
from sgl_jax.test.kernels.grouped_topk_main_ref import (
    grouped_topk_pallas as grouped_topk_pallas_main,
)
from sgl_jax.test.kernels.grouped_topk_test import (
    ref_biased_grouped_topk,
    ref_biased_grouped_topk_bf16,
)

_ON_TPU = jax.default_backend() == "tpu"
_INTERPRET = not _ON_TPU
_requires_tpu = pytest.mark.skipif(not _ON_TPU, reason="needs the real Mosaic kernel (TPU)")

CONFIGS = [
    # (E, G, Gtop, k, name)
    (256, 8, 4, 8, "dsv3_E256_G8_Gtop4_k8"),  # DeepSeek-V3 / Ling (group-drop path)
    (512, 8, 4, 8, "maxtext_E512_G8_Gtop4_k8"),  # group-drop path
    (128, 4, 2, 6, "small_E128_G4_Gtop2_k6"),  # group-drop path
    (384, 8, 3, 6, "oddE_E384_G8_Gtop3_k6"),  # non-pow2 E, odd Gtop
    (256, 8, 4, 32, "bigk_E256_G8_Gtop4_k32"),  # large k
    (896, 1, 1, 16, "all_E896_G1_Gtop1_k16"),  # all-groups path (fused bias add in #1758)
    (128, 4, 4, 8, "all_E128_G4_Gtop4_k8"),  # all-groups path
    (256, 8, 8, 8, "all_E256_G8_Gtop8_k8"),  # all-groups path, DS-V3 shape
]
BS, BT = 512, 128  # 4 grid steps; blocking must not change results (see test_blocking_invariance)

# name -> (is_nan_pattern, is_degenerate). Degenerate = fewer than k finite candidates per token
# (all -inf / 95% -inf): main and live must still agree, but the sort-based reference legitimately
# differs there (the kernel re-picks the lowest -inf index), so REF is not asserted.
PATTERNS = {
    "sigmoid/rand_bias": (False, False),
    "sigmoid/zero_bias": (False, False),
    "normal*3/rand_bias": (False, False),
    "sigmoid/big_bias*5": (False, False),
    "flat_ties": (False, False),
    "partial_tie_e3=e5": (False, False),
    "group_tie_g0=g1": (False, False),
    "dyadic_1/64_ties": (False, False),
    "dyadic_1/8_ties": (False, False),
    "postbias_tie_e0=e1": (False, False),
    "bf16_exact": (False, False),
    "neg_inf_30pct": (False, False),
    "pos_inf_some": (False, False),
    "bias_neg_inf_some": (False, False),
    "signed_zeros": (False, False),
    "subnormal": (False, False),
    "neg_logits": (False, False),
    "neg_inf_95pct": (False, True),
    "all_neg_inf": (False, True),
    "nan_sparse": (True, True),
    "nan_bias": (True, True),
    "nan_rows": (True, True),
}


def _inputs(pattern, bs, E, G, seed=0):
    """f32 (logits[bs, E], bias[E]) for a named adversarial pattern."""
    S = E // G
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    sig = jax.nn.sigmoid(jax.random.normal(k1, (bs, E), jnp.float32))
    bias_r = jax.random.normal(k2, (E,), jnp.float32) * 0.1
    zero_b = jnp.zeros((E,), jnp.float32)
    m30 = jax.random.bernoulli(k4, 0.3, (bs, E))
    if pattern == "sigmoid/rand_bias":
        return sig, bias_r
    if pattern == "sigmoid/zero_bias":
        return sig, zero_b
    if pattern == "normal*3/rand_bias":
        return jax.random.normal(k3, (bs, E), jnp.float32) * 3, bias_r
    if pattern == "sigmoid/big_bias*5":
        return sig, jax.random.normal(k2, (E,), jnp.float32) * 5
    if pattern == "flat_ties":
        return jnp.full((bs, E), 0.5, jnp.float32), zero_b
    if pattern == "partial_tie_e3=e5":
        return sig.at[:, 5].set(sig[:, 3]), bias_r
    if pattern == "group_tie_g0=g1":
        if G < 2:  # degenerate to a within-group tie block
            return sig.at[:, 1:3].set(sig[:, 0:1]), bias_r
        return sig.at[:, S : 2 * S].set(sig[:, :S]), bias_r.at[S : 2 * S].set(bias_r[:S])
    if pattern == "dyadic_1/64_ties":
        return jnp.round(sig * 64) / 64, jnp.round(bias_r * 64) / 64
    if pattern == "dyadic_1/8_ties":
        return jnp.round(sig * 8) / 8, jnp.round(bias_r * 8) / 8
    if pattern == "postbias_tie_e0=e1":
        # distinct pre-bias logits whose post-bias sums tie exactly (dyadic grid, exact sums)
        dl, db = jnp.round(sig * 64) / 64, jnp.round(bias_r * 64) / 64
        return dl.at[:, 1].set(dl[:, 0] + (db[0] - db[1])), db
    if pattern == "bf16_exact":
        return (
            sig.astype(jnp.bfloat16).astype(jnp.float32),
            bias_r.astype(jnp.bfloat16).astype(jnp.float32),
        )
    if pattern == "neg_inf_30pct":
        return jnp.where(m30, -jnp.inf, sig), bias_r
    if pattern == "pos_inf_some":
        return jnp.where(m30, jnp.inf, sig), bias_r
    if pattern == "bias_neg_inf_some":
        return sig, jnp.where(m30[0], -jnp.inf, bias_r)
    if pattern == "signed_zeros":
        return jnp.where(m30, -0.0, 0.0).astype(jnp.float32), zero_b
    if pattern == "subnormal":
        return sig * 1e-40, bias_r * 1e-40
    if pattern == "neg_logits":
        return -sig, -bias_r
    if pattern == "neg_inf_95pct":
        m95 = jax.random.bernoulli(k3, 0.95, (bs, E))
        return jnp.where(m95, -jnp.inf, sig), bias_r
    if pattern == "all_neg_inf":
        return jnp.full((bs, E), -jnp.inf, jnp.float32), bias_r
    if pattern == "nan_sparse":
        return sig.at[::7, 11].set(jnp.nan), bias_r
    if pattern == "nan_bias":
        return sig, bias_r.at[3].set(jnp.nan)
    if pattern == "nan_rows":
        return sig.at[::5, :].set(jnp.nan), bias_r
    raise KeyError(pattern)


@functools.cache
def _jitted(which, E, G, Gtop, k, bt, packed):
    fn = {"live": grouped_topk_pallas, "main": grouped_topk_pallas_main}[which]
    return jax.jit(
        functools.partial(
            fn,
            num_expert_group=G,
            topk_group=Gtop,
            topk=k,
            block_tokens=bt,
            interpret=_INTERPRET,
            packed=packed,
        )
    )


def _run(which, logits, bias, *, E, G, Gtop, k, bt=BT, packed=False):
    w, ids = _jitted(which, E, G, Gtop, k, bt, packed)(logits, bias)
    return np.asarray(w), np.asarray(ids)


def _bits(w):
    return np.asarray(w, dtype=np.float32).view(np.uint32)


def _assert_parity(w_live, ids_live, w_main, ids_main, msg):
    np.testing.assert_array_equal(ids_live, ids_main, err_msg=f"{msg}: ids differ from main")
    n_bits = int(np.sum(_bits(w_live) != _bits(w_main)))
    assert n_bits == 0, f"{msg}: {n_bits} weight words differ bitwise from main"


def _assert_ref(w_live, ids_live, w_ref, ids_ref, msg):
    np.testing.assert_array_equal(ids_live, ids_ref, err_msg=f"{msg}: ids differ from reference")
    # +0.0 folds -0.0 into +0.0 on both sides; everything else must be bit-exact.
    n_bits = int(np.sum(_bits(w_live + 0.0) != _bits(np.asarray(w_ref, np.float32) + 0.0)))
    assert n_bits == 0, f"{msg}: {n_bits} weight words differ from reference"


def _cast(logits, bias, packed):
    if packed:  # the caller enables packed only for bf16 router logits
        return logits.astype(jnp.bfloat16), bias.astype(jnp.bfloat16)
    return logits, bias


# ---------------------------------------------------------------------------------------------
# PARITY: live kernel vs frozen main kernel (bit-exact), every config x pattern x select mode
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
def test_parity_vs_main(E, G, Gtop, k, name, pattern, packed):
    is_nan, _ = PATTERNS[pattern]
    if is_nan and not _ON_TPU:
        pytest.skip("NaN parity is only defined on TPU (XLA:CPU reduce-max is order-dependent)")
    logits, bias = _cast(*_inputs(pattern, BS, E, G), packed)
    w_live, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    w_main, ids_main = _run("main", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    _assert_parity(w_live, ids_live, w_main, ids_main, f"{name}/{pattern}/packed={packed}")


# ---------------------------------------------------------------------------------------------
# REF: live kernel vs sort-based lax.top_k reference (ids exact; weights exact modulo -0.0)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("pattern", [p for p, (nan, deg) in PATTERNS.items() if not deg])
@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
def test_matches_reference(E, G, Gtop, k, name, pattern, packed):
    if packed and not _ON_TPU:
        pytest.skip("packed bf16-key path vs reference needs the real Mosaic kernel (TPU)")
    logits, bias = _cast(*_inputs(pattern, BS, E, G), packed)
    ref = ref_biased_grouped_topk_bf16 if packed else ref_biased_grouped_topk
    w_ref, ids_ref = ref(logits, bias, num_expert_group=G, topk_group=Gtop, topk=k)
    w_live, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    _assert_ref(w_live, ids_live, np.asarray(w_ref), np.asarray(ids_ref), f"{name}/{pattern}")


# ---------------------------------------------------------------------------------------------
# Exact-tie order: every pick must be the lowest expert id achieving the max, in both modes.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("E,G,Gtop,k,name", CONFIGS)
def test_flat_ties_pick_lowest_indices(E, G, Gtop, k, name, packed):
    logits, bias = _cast(jnp.full((BS, E), 0.5, jnp.float32), jnp.zeros((E,), jnp.float32), packed)
    w_live, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    w_main, ids_main = _run("main", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    np.testing.assert_array_equal(ids_live, np.tile(np.arange(k), (BS, 1)), err_msg=name)
    np.testing.assert_array_equal(w_live, np.full((BS, k), 0.5, np.float32), err_msg=name)
    _assert_parity(w_live, ids_live, w_main, ids_main, f"{name}/flat_ties/packed={packed}")


# ---------------------------------------------------------------------------------------------
# Blocking invariance: a 128-token grid and a whole-batch block must agree bit-for-bit (and with
# main), for both the group-drop and the all-groups code paths.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("pattern", ["sigmoid/rand_bias", "dyadic_1/64_ties", "neg_inf_30pct"])
@pytest.mark.parametrize("E,G,Gtop,k,name", [CONFIGS[0], CONFIGS[7]])
def test_blocking_invariance(E, G, Gtop, k, name, pattern, packed):
    logits, bias = _cast(*_inputs(pattern, BS, E, G, seed=3), packed)
    outs = {
        (which, bt): _run(which, logits, bias, E=E, G=G, Gtop=Gtop, k=k, bt=bt, packed=packed)
        for which in ("live", "main")
        for bt in (128, BS)
    }
    w0, i0 = outs[("live", 128)]
    for key, (w, ids) in outs.items():
        _assert_parity(w, ids, w0, i0, f"{name}/{pattern}/{key}")


# ---------------------------------------------------------------------------------------------
# Routing-level agreement on a large synthetic DS-V3 gate batch (sigmoid gate output + learned
# correction bias magnitudes): the expert-id agreement rate between live and main must be 100%.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
def test_expert_id_agreement_rate_dsv3(packed):
    E, G, Gtop, k, bs = 256, 8, 4, 8, 4096
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    logits = jax.nn.sigmoid(jax.random.normal(k1, (bs, E), jnp.float32) * 2.0)
    bias = jax.random.normal(k2, (E,), jnp.float32) * 0.05
    logits, bias = _cast(logits, bias, packed)
    _, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, bt=512, packed=packed)
    _, ids_main = _run("main", logits, bias, E=E, G=G, Gtop=Gtop, k=k, bt=512, packed=packed)
    agreement = float(np.mean(ids_live == ids_main))
    tokens_identical = float(np.mean(np.all(ids_live == ids_main, axis=1)))
    print(
        f"\n[dsv3 packed={packed}] expert-id agreement {agreement:.6f}, tokens identical {tokens_identical:.6f}"
    )
    assert agreement == 1.0 and tokens_identical == 1.0


# ---------------------------------------------------------------------------------------------
# NaN / -inf contract on hardware: NaN scores never produce an out-of-range expert id when any
# finite candidate exists, and live == main bit-for-bit (the CPU interpreter cannot pin this).
# ---------------------------------------------------------------------------------------------
@_requires_tpu
@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("pattern", ["nan_sparse", "nan_bias", "nan_rows", "all_neg_inf"])
@pytest.mark.parametrize("E,G,Gtop,k,name", [CONFIGS[0], CONFIGS[7]])
def test_nan_and_inf_parity_on_tpu(E, G, Gtop, k, name, pattern, packed):
    logits, bias = _cast(*_inputs(pattern, BS, E, G), packed)
    w_live, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    w_main, ids_main = _run("main", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    _assert_parity(w_live, ids_live, w_main, ids_main, f"{name}/{pattern}/packed={packed}")
    assert ids_live.min() >= 0 and ids_live.max() <= E, f"{name}/{pattern}: id out of [0, E]"


# ---------------------------------------------------------------------------------------------
# Contract: configurations the routing does not define are rejected at the entry point, before
# anything is traced. main computed something for these (e.g. `topk_group > num_expert_group`
# looped `topk_group` times over `num_expert_group` groups and produced input-dependent ids).
# ---------------------------------------------------------------------------------------------
INVALID_CONFIGS = [
    # (E, G, Gtop, k, packed, error-message fragment)
    (128, 4, 5, 8, False, "topk_group"),  # more groups selected than exist
    (128, 4, 5, 8, True, "topk_group"),
    (128, 4, 0, 8, False, "topk_group"),
    (128, 0, 1, 8, False, "num_expert_group"),
    (130, 4, 2, 8, False, "divisible"),  # E % G != 0
    (128, 4, 2, 0, False, "topk"),
    (128, 4, 1, 33, False, "topk"),  # more picks than the one retained group holds (32)
    (128, 4, 4, 129, True, "topk"),  # more picks than experts
]

# Boundaries of the contract that must still be accepted (and stay bit-exact with main / REF).
BOUNDARY_CONFIGS = [
    (128, 4, 1, 32, "one_group_all_its_experts"),  # topk == topk_group * S
    (128, 4, 4, 128, "all_groups_all_experts"),  # topk == E
    (64, 1, 1, 64, "single_group_all_experts"),
]


@pytest.mark.parametrize("E,G,Gtop,k,packed,match", INVALID_CONFIGS)
def test_rejects_out_of_contract_config(E, G, Gtop, k, packed, match):
    logits, bias = _cast(jnp.zeros((BS, E), jnp.float32), jnp.zeros((E,), jnp.float32), packed)
    with pytest.raises(ValueError, match=match):
        grouped_topk_pallas(
            logits,
            bias,
            num_expert_group=G,
            topk_group=Gtop,
            topk=k,
            interpret=_INTERPRET,
            packed=packed,
        )


def test_rejects_mismatched_shapes():
    E, G, Gtop, k, _ = CONFIGS[0]
    kw = dict(num_expert_group=G, topk_group=Gtop, topk=k, interpret=_INTERPRET)
    with pytest.raises(ValueError, match="correction_bias"):
        grouped_topk_pallas(jnp.zeros((BS, E)), jnp.zeros((E + 1,)), **kw)
    with pytest.raises(ValueError, match="router_logits"):
        grouped_topk_pallas(jnp.zeros((2, BS, E)), jnp.zeros((E,)), **kw)


def test_rejects_packed_with_more_than_65536_experts():
    E = (1 << 16) + 128
    with pytest.raises(ValueError, match="packed"):
        grouped_topk_pallas(
            jnp.zeros((128, E), jnp.bfloat16),
            jnp.zeros((E,), jnp.bfloat16),
            num_expert_group=1,
            topk_group=1,
            topk=8,
            interpret=_INTERPRET,
            packed=True,
        )


@pytest.mark.parametrize("packed", [False, True], ids=["f32", "packed"])
@pytest.mark.parametrize("E,G,Gtop,k,name", BOUNDARY_CONFIGS)
def test_contract_boundaries_accepted_and_exact(E, G, Gtop, k, name, packed):
    logits, bias = _cast(*_inputs("dyadic_1/64_ties", BS, E, G), packed)
    w_live, ids_live = _run("live", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    w_main, ids_main = _run("main", logits, bias, E=E, G=G, Gtop=Gtop, k=k, packed=packed)
    _assert_parity(w_live, ids_live, w_main, ids_main, f"{name}/packed={packed}")
    if packed and not _ON_TPU:
        return  # packed vs REF needs the real Mosaic kernel (see module docstring)
    ref = ref_biased_grouped_topk_bf16 if packed else ref_biased_grouped_topk
    w_ref, ids_ref = ref(logits, bias, num_expert_group=G, topk_group=Gtop, topk=k)
    _assert_ref(w_live, ids_live, np.asarray(w_ref), np.asarray(ids_ref), name)
