"""M1.2 -- the mHC layer.

The reference is `test/srt/kernels/mhc/ref.py`, the independent float64 NumPy oracle
that landed with the kernels in #341. It was written from the published semantics
rather than from the kernels, so agreement is evidence about the semantics.

Checked on CPU against that oracle: the three gates, the pre/post pair, the head
collapse, and the sequencing a decoder layer performs.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config, mhc_param_shapes
from sgl_jax.srt.layers.deepseek_v4_mhc import (
    DeepseekV4MHC,
    collapse_head_reference,
    expand_streams,
    mhc_gates_reference,
    post_reference,
    pre_reference,
    resolve_backend,
)

# The oracle lives under test/srt, which is not a package on the path.
_ORACLE_DIR = Path(__file__).resolve().parents[4] / "test" / "srt" / "kernels" / "mhc"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))
ref = pytest.importorskip("ref", reason="mHC NumPy oracle not importable")

HC = 4
D = 16
T = 6
EPS = 1e-6
ITERS = 3


@pytest.fixture
def highest_matmul_precision():
    # TPU defaults to reduced-precision float32 dots. The strict NumPy oracle
    # comparisons need float32 dot precision on every backend.
    with jax.default_matmul_precision("highest"):
        yield


def _cfg(**kw):
    values = dict(hc_mult=HC, hidden_size=D, hc_sinkhorn_iters=ITERS, hc_eps=EPS)
    values.update(kw)
    return DeepseekV4Config(**values)


def _params(seed=0, *, head=False):
    rng = np.random.default_rng(seed)
    shapes = mhc_param_shapes(_cfg())
    if head:
        return (
            (rng.normal(size=shapes["head_fn"]) * 0.2).astype(np.float32),
            (rng.normal(size=shapes["head_base"]) * 0.2).astype(np.float32),
            (1.0 + 0.1 * rng.normal(size=shapes["head_scale"])).astype(np.float32),
        )
    return (
        (rng.normal(size=shapes["fn"]) * 0.2).astype(np.float32),
        (rng.normal(size=shapes["base"]) * 0.2).astype(np.float32),
        (1.0 + 0.1 * rng.normal(size=shapes["scale"])).astype(np.float32),
    )


def _streams(seed=0, tokens=T):
    return np.random.default_rng(seed).normal(size=(tokens, HC, D)).astype(np.float32)


def _apply_sublayer(mhc, streams, params, sublayer):
    collapsed, post_gate, comb = mhc.pre(streams, *params)
    return mhc.post(sublayer(collapsed), streams, post_gate, comb)


# --------------------------------------------------------------------------
# shapes and stream handling
# --------------------------------------------------------------------------


def test_expand_streams_replicates_the_embedding():
    hidden = np.arange(T * D, dtype=np.float32).reshape(T, D)
    out = np.asarray(expand_streams(hidden, HC))
    assert out.shape == (T, HC, D)
    for h in range(HC):
        np.testing.assert_array_equal(out[:, h], hidden)


def test_expand_streams_rejects_a_flat_input():
    with pytest.raises(ValueError, match="hidden must be"):
        expand_streams(np.zeros((D,), np.float32)[0], HC)


def test_param_shapes_come_from_the_config():
    shapes = mhc_param_shapes(_cfg())
    assert shapes["fn"] == (24, HC * D)  # (2+4)*4 = 24
    assert shapes["base"] == (24,)
    assert shapes["scale"] == (3,)
    assert shapes["head_fn"] == (HC, HC * D)


# --------------------------------------------------------------------------
# the gates, against the oracle
# --------------------------------------------------------------------------


def test_gates_match_the_oracle():
    rng = np.random.default_rng(3)
    mixes = rng.normal(size=(T, 24)).astype(np.float32)
    fn, base, scale = _params(4)
    got = [
        np.asarray(a)
        for a in mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)
    ]
    want = ref.sinkhorn_gates(mixes, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)


def test_pre_adds_eps_after_the_sigmoid_and_post_does_not():
    """The asymmetry the oracle calls load-bearing: pre gets +eps, post gets x2."""
    mixes = np.full((1, 24), -60.0, np.float32)  # sigmoid ~ 0
    base = np.zeros((24,), np.float32)
    scale = np.ones((3,), np.float32)
    pre_gate, post_gate, _ = mhc_gates_reference(
        mixes, scale, base, hc_mult=HC, sinkhorn_iters=1, eps=0.25
    )
    # pre floors at eps; post has no eps at all.
    np.testing.assert_allclose(np.asarray(pre_gate), 0.25, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(post_gate), 0.0, rtol=1e-5, atol=1e-6)
    # And post's ceiling is 2, not 1.
    _, big, _ = mhc_gates_reference(
        np.full((1, 24), 60.0, np.float32), scale, base, hc_mult=HC, sinkhorn_iters=1, eps=0.0
    )
    np.testing.assert_allclose(np.asarray(big), 2.0, rtol=1e-5, atol=1e-5)


def test_sinkhorn_iteration_count_changes_the_result():
    """Guards the schedule: row softmax, one column pass, then iters-1 pairs."""
    rng = np.random.default_rng(5)
    mixes = rng.normal(size=(T, 24)).astype(np.float32)
    fn, base, scale = _params(6)
    one = np.asarray(
        mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=1, eps=EPS)[2]
    )
    many = np.asarray(
        mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=8, eps=EPS)[2]
    )
    assert not np.allclose(one, many, rtol=1e-4)
    # The schedule always *ends* on a column pass, so columns are normalised at
    # every iteration count; rows only approach 1 as the pairs accumulate. Asserting
    # the rows at one iteration would be asserting the wrong side of the schedule.
    for arr in (one, many):
        np.testing.assert_allclose(arr.sum(axis=-2), 1.0, atol=2e-5)
    row_error_one = np.abs(one.sum(axis=-1) - 1.0).max()
    row_error_many = np.abs(many.sum(axis=-1) - 1.0).max()
    assert row_error_many < row_error_one


# --------------------------------------------------------------------------
# pre / post / head, against the oracle
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("highest_matmul_precision")
def test_pre_matches_the_oracle():
    x = _streams(7)
    fn, base, scale = _params(8)
    y, post_gate, comb = pre_reference(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    np.testing.assert_allclose(np.asarray(y), wy, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(post_gate), wpost, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(comb), wcomb, rtol=2e-5, atol=2e-5)


@pytest.mark.usefixtures("highest_matmul_precision")
def test_post_matches_the_oracle():
    x = _streams(9)
    fn, base, scale = _params(10)
    y, post_gate, comb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    sublayer_out = np.random.default_rng(11).normal(size=(T, D)).astype(np.float32)
    got = np.asarray(post_reference(sublayer_out, x, post_gate, comb))
    want = ref.post(sublayer_out, x, post_gate, comb)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert got.shape == (T, HC, D)


@pytest.mark.usefixtures("highest_matmul_precision")
def test_head_collapse_matches_the_oracle():
    x = _streams(12)
    fn, base, scale = _params(13, head=True)
    got = np.asarray(collapse_head_reference(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS))
    want = ref.head_collapse(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert got.shape == (T, D)


def test_head_collapse_is_not_the_same_normalisation_as_pre():
    """`pre` scales the projection; `head_collapse` scales the activation before it,
    with a bf16 rounding in between. Same-looking code, different numbers -- so a
    test has to distinguish them or the asymmetry will get 'cleaned up'."""
    x = _streams(14) * 40.0  # large enough that the bf16 rounding is visible
    fn, base, scale = _params(15, head=True)
    collapsed = np.asarray(collapse_head_reference(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS))
    # Recompute with pre's ordering (scale after the projection, no rounding).
    flat = x.reshape(T, -1).astype(np.float64)
    rsq = 1.0 / np.sqrt((flat**2).mean(-1, keepdims=True) + EPS)
    wrong_mixes = (flat @ fn.astype(np.float64).T) * rsq
    wrong_gate = 1.0 / (1.0 + np.exp(-(wrong_mixes * scale[0] + base))) + EPS
    wrong = (wrong_gate[..., None] * x.astype(np.float64)).sum(-2)
    assert not np.allclose(collapsed, wrong, rtol=1e-3)


# --------------------------------------------------------------------------
# the layer object
# --------------------------------------------------------------------------


def test_backend_resolution():
    assert resolve_backend("reference") == "reference"
    assert resolve_backend("pallas") == "pallas"
    assert resolve_backend("auto") == ("pallas" if jax.default_backend() == "tpu" else "reference")
    with pytest.raises(ValueError, match="unknown mHC backend"):
        resolve_backend("cuda")


@pytest.mark.usefixtures("highest_matmul_precision")
def test_layer_pre_post_round_trip_matches_the_oracle():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(16)
    fn, base, scale = _params(17)
    collapsed, post_gate, comb = mhc.pre(x, fn, base, scale)
    out = np.asarray(mhc.post(np.asarray(collapsed) * 0.5, x, post_gate, comb))

    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    want = ref.post(wy * 0.5, x, wpost, wcomb)
    np.testing.assert_allclose(out, want, rtol=2e-5, atol=2e-5)


@pytest.mark.usefixtures("highest_matmul_precision")
def test_pre_post_is_the_sequential_semantics():
    """The unfused form: residual = X, collapse, run F, expand against the residual."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(18)
    fn, base, scale = _params(19)
    calls = []

    def sublayer(u):
        calls.append(np.asarray(u).shape)
        return np.asarray(u) * 2.0 + 1.0

    out = np.asarray(_apply_sublayer(mhc, x, (fn, base, scale), sublayer))
    assert calls == [(T, D)]  # the sublayer sees the collapsed stream
    assert out.shape == (T, HC, D)

    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    np.testing.assert_allclose(out, ref.post(wy * 2.0 + 1.0, x, wpost, wcomb), rtol=2e-5, atol=2e-5)


def test_the_residual_is_the_input_not_the_collapsed_stream():
    """`post` must mix against the original hc streams. Passing the collapsed value
    would typecheck and produce the right shape."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(20)
    fn, base, scale = _params(21)
    correct = np.asarray(_apply_sublayer(mhc, x, (fn, base, scale), lambda u: np.asarray(u)))
    collapsed, post_gate, comb = mhc.pre(x, fn, base, scale)
    wrong = np.asarray(
        mhc.post(
            collapsed, np.repeat(np.asarray(collapsed)[:, None, :], HC, axis=1), post_gate, comb
        )
    )
    assert not np.allclose(correct, wrong, rtol=1e-3)


def test_two_sublayers_use_independent_gate_parameters():
    """Attention and FFN each get their own pre/post set; sharing them silently
    couples the two halves of a layer."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(22)
    attn = _params(23)
    ffn = _params(24)
    after_attn = _apply_sublayer(mhc, x, attn, lambda u: np.asarray(u))
    a = np.asarray(_apply_sublayer(mhc, after_attn, ffn, lambda u: np.asarray(u)))
    b = np.asarray(_apply_sublayer(mhc, after_attn, attn, lambda u: np.asarray(u)))
    assert not np.allclose(a, b, rtol=1e-3)


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


def test_gate_parameters_must_stay_float32():
    """They are Sinkhorn coefficients, not projections, so they do not follow the
    activation dtype."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    fn, base, scale = _params(25)
    with pytest.raises(ValueError, match="must stay float32"):
        mhc.check_params(fn.astype(np.float16), base, scale)


def test_gate_parameter_shapes_are_checked():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    fn, base, scale = _params(26)
    with pytest.raises(ValueError, match="mHC fn fn must be"):
        mhc.check_params(fn[:, :-1], base, scale)
    with pytest.raises(ValueError, match="mHC fn base must be"):
        mhc.check_params(fn, base[:-1], scale)
    with pytest.raises(ValueError, match="mHC fn scale must be"):
        mhc.check_params(fn, base, scale[:-1])


def test_head_parameter_shapes_are_checked_separately():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    head = _params(27, head=True)
    mhc.check_params(*head, kind="head_fn")
    layer_fn, base, scale = _params(28)
    with pytest.raises(ValueError, match="mHC head_fn fn must be"):
        mhc.check_params(layer_fn, base, scale, kind="head_fn")


def test_shipped_config_geometry():
    """hc_mult=4, hidden=4096 -> mix_hc=24 and hc_dim=16384."""
    shapes = mhc_param_shapes(DeepseekV4Config())
    assert shapes["fn"] == (24, 16384)
    assert shapes["head_fn"] == (4, 16384)
    assert DeepseekV4Config().hc_sinkhorn_iters == 20


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("tokens", [1, 7])
def test_model_facing_dtype_and_irregular_token_count(dtype, tokens):
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    streams = jnp.asarray(_streams(30, tokens), dtype)
    fn, base, scale = _params(31)
    collapsed, post_gate, comb = mhc.pre(streams, fn, base, scale)
    assert collapsed.shape == (tokens, D)
    assert post_gate.shape == (tokens, HC)
    assert comb.shape == (tokens, HC, HC)
    assert post_gate.dtype == comb.dtype == jnp.float32
    # M casts at this boundary before normalization and again after post.
    output = jnp.asarray(collapsed, dtype)
    next_streams = jnp.asarray(mhc.post(output, streams, post_gate, comb), dtype)
    assert next_streams.shape == (tokens, HC, D)
    assert next_streams.dtype == dtype
    head = mhc.collapse_head(next_streams, *_params(32, head=True))
    assert head.shape == (tokens, D)
    assert head.dtype == dtype


def test_stacked_attention_and_ffn_follow_sequential_reference():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    streams = jnp.asarray(_streams(33, 5), jnp.bfloat16)
    attn, ffn = _params(34), _params(35)

    def apply(x, params, factor):
        collapsed, post_gate, comb = mhc.pre(x, *params)
        y = (collapsed.astype(jnp.bfloat16) * factor).astype(jnp.bfloat16)
        return mhc.post(y, x, post_gate, comb).astype(jnp.bfloat16)

    def oracle(x, params, factor):
        fn, base, scale = params
        hidden, post_gate, comb = ref.pre(
            x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
        )
        y = jnp.asarray(hidden, jnp.bfloat16)
        y = np.asarray((y * factor).astype(jnp.bfloat16), np.float32)
        return np.asarray(jnp.asarray(ref.post(y, x, post_gate, comb), jnp.bfloat16), np.float32)

    got, want = streams, np.asarray(streams, np.float32)
    for _ in range(2):
        got = apply(apply(got, attn, 0.5), ffn, -0.25)
        want = oracle(oracle(want, attn, 0.5), ffn, -0.25)
    np.testing.assert_allclose(np.asarray(got, np.float32), want, rtol=3e-2, atol=3e-2)


def test_padding_and_multiple_leading_dimensions_are_token_local():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    streams = jnp.asarray(_streams(36, 6)).reshape(2, 3, HC, D)
    fn, base, scale = _params(37)
    full = mhc.pre(streams, fn, base, scale)
    assert full[0].shape == (2, 3, D)
    padded = streams.at[1, 2].set(jnp.nan)
    changed = mhc.pre(padded, fn, base, scale)
    for before, after in zip(full, changed):
        np.testing.assert_allclose(
            np.asarray(before).reshape(6, -1)[:5], np.asarray(after).reshape(6, -1)[:5]
        )


def test_invalid_config_and_operation_boundaries_are_rejected():
    for overrides in (
        {"hc_mult": 0},
        {"hc_sinkhorn_iters": 0},
        {"hc_eps": 0},
        {"hc_eps": float("nan")},
        {"rms_norm_eps": 0},
    ):
        with pytest.raises(ValueError):
            DeepseekV4MHC(_cfg(**overrides), backend="reference")
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    fn, base, scale = _params(38)
    with pytest.raises(ValueError, match="streams must end"):
        mhc.pre(jnp.zeros((2, HC, D - 1)), fn, base, scale)
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        mhc.pre(jnp.zeros((2, HC, D), jnp.int32), fn, base, scale)
    streams = jnp.zeros((2, HC, D), jnp.bfloat16)
    y, post_gate, comb = mhc.pre(streams, fn, base, scale)
    with pytest.raises(ValueError, match="post output must be"):
        mhc.post(y[:1], streams, post_gate, comb)
    with pytest.raises(ValueError, match="must stay float32"):
        mhc.post(y, streams, post_gate.astype(jnp.bfloat16), comb)


def test_row_sharding_preserves_valid_results():
    if len(jax.devices("cpu")) < 2:
        pytest.skip("needs at least two CPU devices")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("tokens",))
    sharding = NamedSharding(mesh, P("tokens", None, None))
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    streams = jnp.asarray(_streams(39, 8), jnp.bfloat16)
    padded = streams.at[-1].set(jnp.nan)
    distributed = jax.device_put(padded, sharding)
    fn, base, scale = _params(40)
    got = mhc.pre(distributed, fn, base, scale)
    want = mhc.pre(streams, fn, base, scale)
    for array in got:
        assert isinstance(array.sharding, NamedSharding)
        assert array.sharding.spec[0] == "tokens"
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a, np.float32)[:-1], np.asarray(b, np.float32)[:-1])
    got_post = mhc.post(got[0].astype(jnp.bfloat16), distributed, got[1], got[2])
    want_post = mhc.post(want[0].astype(jnp.bfloat16), streams, want[1], want[2])
    np.testing.assert_allclose(
        np.asarray(got_post, np.float32)[:-1], np.asarray(want_post, np.float32)[:-1]
    )
    head_params = _params(41, head=True)
    got_head = mhc.collapse_head(got_post.astype(jnp.bfloat16), *head_params)
    want_head = mhc.collapse_head(want_post.astype(jnp.bfloat16), *head_params)
    np.testing.assert_allclose(
        np.asarray(got_head, np.float32)[:-1], np.asarray(want_head, np.float32)[:-1]
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="mHC Pallas kernels need TPU")
def test_model_facing_tpu_kernel_matches_reference():
    config = DeepseekV4Config(hidden_size=4096)
    mhc = DeepseekV4MHC(config, backend="pallas")
    reference = DeepseekV4MHC(config, backend="reference")
    rng = np.random.default_rng(42)
    streams = jnp.asarray(rng.normal(size=(8, 4, 4096)) * 0.1, jnp.bfloat16)
    shapes = mhc_param_shapes(config)
    params = (
        jnp.asarray(rng.normal(size=shapes["fn"]) * 0.01, jnp.float32),
        jnp.asarray(rng.normal(size=shapes["base"]) * 0.01, jnp.float32),
        jnp.asarray([0.8, 1.1, 0.9], jnp.float32),
    )
    got = mhc.pre(streams, *params)
    want = reference.pre(streams, *params)
    for a, b in zip(got, want):
        np.testing.assert_allclose(
            np.asarray(a, np.float32), np.asarray(b, np.float32), rtol=2e-2, atol=1e-2
        )
    output = got[0].astype(jnp.bfloat16)
    got_post = mhc.post(output, streams, got[1], got[2])
    want_post = reference.post(output, streams, want[1], want[2])
    np.testing.assert_allclose(
        np.asarray(got_post, np.float32), np.asarray(want_post, np.float32), rtol=2e-2, atol=1e-2
    )
    head_params = (
        jnp.asarray(rng.normal(size=shapes["head_fn"]) * 0.01, jnp.float32),
        jnp.asarray(rng.normal(size=shapes["head_base"]) * 0.01, jnp.float32),
        jnp.asarray([0.8], jnp.float32),
    )
    got_head = mhc.collapse_head(streams, *head_params)
    want_head = reference.collapse_head(streams, *head_params)
    np.testing.assert_allclose(
        np.asarray(got_head, np.float32), np.asarray(want_head, np.float32), rtol=2e-2, atol=1e-2
    )
