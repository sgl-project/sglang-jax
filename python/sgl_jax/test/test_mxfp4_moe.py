"""MXFP4 -> EPMoE weight assembly."""
import jax.numpy as jnp, numpy as np, pytest
from sgl_jax.srt.layers.quantization.mxfp4_moe import (
    dequant_expert_weight, stack_experts, build_epmoe_weights, EXPERT_PROJ_TO_EPMOE)

E2M1 = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]


def _mk(out_dim, in_dim, seed=0, exp=127):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 16, size=(out_dim, in_dim)).astype(np.uint8)
    vals = np.array(E2M1, np.float32)[idx]
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    scale = np.full((out_dim, in_dim // 32), exp, np.uint8)
    return packed, scale, vals * (2.0 ** (exp - 127))


def test_dequant_transposes_to_epmoe_layout():
    """Checkpoint stores [out, in]; EPMoE wants [in, out]."""
    packed, scale, want = _mk(8, 64)
    got = dequant_expert_weight(jnp.asarray(packed), jnp.asarray(scale), jnp.float32)
    assert got.shape == (64, 8), got.shape
    np.testing.assert_allclose(np.asarray(got), want.T, rtol=0, atol=0)


def test_dequant_happens_before_transpose():
    """The quantized axis is the stored tensor's LAST axis. Scaling after transposing would
    apply the per-32 group scale along the wrong dim; a non-uniform scale exposes it."""
    rng = np.random.default_rng(3)
    out_dim, in_dim = 4, 64
    idx = rng.integers(1, 8, size=(out_dim, in_dim)).astype(np.uint8)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    exps = np.array([[127, 130]] * out_dim, np.uint8)          # differs per group
    got = np.asarray(dequant_expert_weight(jnp.asarray(packed), jnp.asarray(exps), jnp.float32))
    vals = np.array(E2M1, np.float32)[idx]
    want = np.concatenate([vals[:, :32] * 1.0, vals[:, 32:] * 8.0], axis=1).T
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_scale_group_count_is_validated():
    packed, _, _ = _mk(4, 64)
    bad = np.full((4, 3), 127, np.uint8)                        # should be 2 groups
    with pytest.raises(ValueError, match="scale groups"):
        dequant_expert_weight(jnp.asarray(packed), jnp.asarray(bad), jnp.float32)


def test_missing_expert_raises_rather_than_stacking_short():
    d = {0: jnp.zeros((4, 4)), 2: jnp.zeros((4, 4))}
    with pytest.raises(KeyError, match="missing 2 experts"):
        stack_experts(d, 4)


def test_packed_without_scale_raises():
    t = {"language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed":
         jnp.asarray(_mk(4, 64)[0])}
    with pytest.raises(KeyError, match="weight_scale missing|missing -- dequant"):
        build_epmoe_weights(t, 0, 1)


def test_projection_mapping_is_gate_up_down():
    """w1->gate(wi_0), w3->up(wi_1), w2->down(wo). Swapping w2/w3 is the classic error."""
    assert EXPERT_PROJ_TO_EPMOE == {"w1": "wi_0", "w3": "wi_1", "w2": "wo"}


def test_builds_stacked_weights_for_all_three_projections():
    n_exp, hid, inter = 3, 64, 32
    t = {}
    for e in range(n_exp):
        for proj, (o, i) in (("w1", (inter, hid)), ("w3", (inter, hid)), ("w2", (hid, inter))):
            p, s, _ = _mk(o, i, seed=e)
            b = f"language_model.model.layers.0.block_sparse_moe.experts.{e}.{proj}"
            t[f"{b}.weight_packed"] = jnp.asarray(p); t[f"{b}.weight_scale"] = jnp.asarray(s)
    out = build_epmoe_weights(t, 0, n_exp, out_dtype=jnp.float32)
    assert set(out) == {"wi_0", "wi_1", "wo"}
    assert out["wi_0"].shape == (n_exp, hid, inter)
    assert out["wo"].shape == (n_exp, inter, hid)
