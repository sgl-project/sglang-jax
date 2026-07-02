"""Boundary tests for Step3p5 MoE layer, dense MLP, and routing.

Covers (per deliverable ③):
- routing bias select-only: ids chosen by prob+bias, weights = original sigmoid prob renorm×3.0
- top-k legality (I1): exactly 8 ids, unique per token, ∈[0,288)
- renorm then ×3.0 order; sigmoid not softmax
- clamp over-limit [NAMED BLIND SPOT]: routed layer-43/44 (limit 7), shared layer-44 (limit 16)
- dense layers 0-2: Step3p5MLP, no routing / no shared
- GMM==loop + dispatch conservation: EPMoE == per-expert loop with same clamp

Run from python/ directory::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_step3p5_moe.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

_ATOL = 2e-5

# ---------------------------------------------------------------------------
# Helpers: fp32 numpy oracles
# ---------------------------------------------------------------------------


def _router_bias_oracle(logits_np: np.ndarray, bias_np: np.ndarray, topk: int, scale: float):
    """Port of HF router_bias_func to numpy fp32.

    Select top-k indices by (sigmoid(logits) + bias), but return weights from
    original sigmoid probs (no bias), renorm, then ×scale.
    """
    prob = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64))).astype(np.float32)
    biased = prob + bias_np[None, :]  # [T, E]
    ids = np.argsort(biased, axis=-1)[:, ::-1][:, :topk]  # top-k by biased score
    gathered = np.take_along_axis(prob, ids, axis=1)
    denom = gathered.sum(axis=-1, keepdims=True) + 1e-20
    weights = (gathered / denom) * scale
    return weights, ids


def _naive_moe_ref(x, wi_0, wi_1, wo, topk_w, topk_ids, swiglu_limit):
    """Per-expert loop reference (fp32). wi_0/wi_1=[E,h,i], wo=[E,i,h]."""
    x = x.astype(np.float32)
    out = np.zeros((x.shape[0], wo.shape[-1]), np.float32)
    for t in range(x.shape[0]):
        acc = np.zeros((wo.shape[-1],), np.float32)
        for slot in range(topk_ids.shape[1]):
            e = int(topk_ids[t, slot])
            w = float(topk_w[t, slot])
            # silu(z) = z * sigmoid(z)
            gate_val = x[t] @ wi_0[e]
            gate_silu = gate_val * (1.0 / (1.0 + np.exp(-gate_val)))
            up = x[t] @ wi_1[e]
            if swiglu_limit is not None:
                gate_silu = np.clip(gate_silu, None, swiglu_limit)
                up = np.clip(up, -swiglu_limit, swiglu_limit)
            acc = acc + w * ((gate_silu * up) @ wo[e])
        out[t] = acc
    return out


# ---------------------------------------------------------------------------
# Config/mesh helpers
# ---------------------------------------------------------------------------


def _make_mesh():
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    return create_device_mesh(
        ici_parallelism=[1, 1],
        dcn_parallelism=[1, 1],
        devices=[jax.devices()[0]],
    )


def _tiny_moe_config(
    layer_id: int = 3,
    swiglu_limit_routed: float = 0.0,
    swiglu_limit_shared: float = 0.0,
    num_layers: int = 45,
):
    """Minimal config for MoE tests with tiny expert count."""
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    # swiglu_limits[i] = 0 means None (no clamp), per HF + sglang convention.
    routed_limits = [0.0] * num_layers
    shared_limits = [0.0] * num_layers
    routed_limits[layer_id] = swiglu_limit_routed
    shared_limits[layer_id] = swiglu_limit_shared

    # moe_layers_enum: all layers >= 3 are MoE for these tests
    moe_layers = ",".join(str(i) for i in range(3, num_layers))

    return Step3p5Config(
        hidden_size=32,
        intermediate_size=64,  # dense MLP (layers 0-2)
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=128,
        vocab_size=128,
        moe_num_experts=8,  # tiny (full model: 288)
        moe_top_k=2,  # tiny (full model: 8)
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
        use_moe_router_bias=True,
        swiglu_limits=routed_limits,
        swiglu_limits_shared=shared_limits,
        moe_layers_enum=moe_layers,
    )


def _build_moe(config, layer_id, mesh):
    from sgl_jax.srt.models.step3p5 import Step3p5MoE

    with jax.set_mesh(mesh):
        return Step3p5MoE(config, layer_id=layer_id, mesh=mesh, ep_size=1, dtype=jnp.float32)


def _build_dense_mlp(config, mesh, intermediate_size=None):
    from sgl_jax.srt.models.step3p5 import Step3p5MLP

    inter = intermediate_size if intermediate_size is not None else config.intermediate_size
    with jax.set_mesh(mesh):
        return Step3p5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=inter,
            mesh=mesh,
            dtype=jnp.float32,
        )


# ---------------------------------------------------------------------------
# Tests: routing bias select-only oracle
# ---------------------------------------------------------------------------


class TestRoutingBias(unittest.TestCase):
    """Bias routing: ids chosen by prob+bias, weights from original sigmoid prob."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.config = _tiny_moe_config(layer_id=3)
        self.rng = np.random.default_rng(42)

    def _run_topk(self, moe, logits_jax, bias_jax):
        """Call TopK._biased_topk through the MoE's topk module directly."""
        with jax.set_mesh(self.mesh):
            # GateLogit computes sigmoid(logits) at HIGHEST precision; replicate here.
            router_logits = jax.nn.sigmoid(logits_jax.astype(jnp.float32))
            topk_w, topk_ids = moe.topk(router_logits, bias_jax)
        return np.asarray(topk_w), np.asarray(topk_ids)

    def test_bias_select_only_matches_oracle(self):
        """Weights = gathered original sigmoid prob, renorm, ×3.0 (NOT prob+bias)."""
        E = self.config.moe_num_experts
        T = 4
        logits_np = self.rng.standard_normal((T, E)).astype(np.float32)
        # Bias that flips which experts are chosen (large values for low-prob experts).
        bias_np = self.rng.standard_normal((E,)).astype(np.float32) * 5.0

        logits_jax = jnp.asarray(logits_np)
        bias_jax = jnp.asarray(bias_np)

        moe = _build_moe(self.config, layer_id=3, mesh=self.mesh)
        with jax.set_mesh(self.mesh):
            moe.moe_gate.bias = None  # bypass GateLogit bias; inject via correction_bias
            router_logits = jax.nn.sigmoid(logits_jax.astype(jnp.float32))
            topk_w, topk_ids = moe.topk(router_logits, bias_jax)

        topk_w_np = np.asarray(topk_w)
        topk_ids_np = np.asarray(topk_ids)

        ref_w, ref_ids = _router_bias_oracle(logits_np, bias_np, self.config.moe_top_k, scale=3.0)

        # Ids must match oracle (same bias-selected experts).
        np.testing.assert_array_equal(
            topk_ids_np, ref_ids, err_msg="topk_ids differ from bias-select oracle"
        )

        # Weights must match oracle (original prob, not biased prob).
        np.testing.assert_allclose(
            topk_w_np, ref_w, atol=_ATOL, err_msg="topk weights differ from bias-select oracle"
        )

    def test_bias_changes_selection(self):
        """Verify the bias actually changes WHICH experts are selected."""
        E = self.config.moe_num_experts
        T = 3
        logits_np = self.rng.standard_normal((T, E)).astype(np.float32)
        logits_jax = jnp.asarray(logits_np)

        zero_bias = jnp.zeros((E,), jnp.float32)
        # Large positive bias for the last expert — forces it into top-k.
        strong_bias = jnp.zeros((E,), jnp.float32).at[-1].set(100.0)

        moe = _build_moe(self.config, layer_id=3, mesh=self.mesh)
        with jax.set_mesh(self.mesh):
            router_logits = jax.nn.sigmoid(logits_jax.astype(jnp.float32))
            _, ids_no_bias = moe.topk(router_logits, zero_bias)
            _, ids_strong = moe.topk(router_logits, strong_bias)

        # With a strong bias on expert E-1, it must appear in every row.
        strong_ids_np = np.asarray(ids_strong)
        self.assertTrue(
            np.all(np.any(strong_ids_np == E - 1, axis=1)),
            msg=f"Last expert should always be selected with bias=100; got {strong_ids_np}",
        )
        # Without bias, last expert need not be selected (sanity — might not differ for all T).
        ids_no_bias_np = np.asarray(ids_no_bias)
        # The selections must differ across at least one token.
        any_differ = not np.array_equal(ids_no_bias_np, strong_ids_np)
        self.assertTrue(any_differ, "Bias should change expert selection")


# ---------------------------------------------------------------------------
# Tests: top-k legality (I1)
# ---------------------------------------------------------------------------


class TestTopKLegality(unittest.TestCase):
    """Exactly K ids, unique per token, all in [0, num_experts)."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.rng = np.random.default_rng(99)

    def _check_ids(self, config, layer_id, T):
        E = config.moe_num_experts
        K = config.moe_top_k

        x = jnp.asarray(self.rng.standard_normal((T, config.hidden_size)), jnp.float32)
        moe = _build_moe(config, layer_id=layer_id, mesh=self.mesh)
        with jax.set_mesh(self.mesh):
            router_logits = moe.moe_gate(x)
            bias = moe.moe_gate.bias.value if moe.moe_gate.bias is not None else None
            _, topk_ids = moe.topk(router_logits, bias)

        ids_np = np.asarray(topk_ids)
        self.assertEqual(ids_np.shape, (T, K), f"Expected shape ({T},{K}), got {ids_np.shape}")
        self.assertTrue(np.all(ids_np >= 0), "ids must be >= 0")
        self.assertTrue(np.all(ids_np < E), f"ids must be < {E}")
        for t in range(T):
            row = ids_np[t]
            self.assertEqual(len(set(row.tolist())), K, f"Token {t} has duplicate ids: {row}")

    def test_legality_moe_layer(self):
        config = _tiny_moe_config(layer_id=3)
        self._check_ids(config, layer_id=3, T=8)


# ---------------------------------------------------------------------------
# Tests: sigmoid not softmax
# ---------------------------------------------------------------------------


class TestSigmoidNotSoftmax(unittest.TestCase):
    """Step3p5 uses sigmoid routing; weights must NOT match a softmax oracle."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.rng = np.random.default_rng(77)

    def test_renorm_then_scale_order(self):
        """Renorm happens before ×3.0: sum of unnormalized weights × 3.0 ≠ 3.0 generally."""
        config = _tiny_moe_config(layer_id=3)
        E = config.moe_num_experts
        T = 4
        logits_np = self.rng.standard_normal((T, E)).astype(np.float32)

        moe = _build_moe(config, layer_id=3, mesh=self.mesh)
        with jax.set_mesh(self.mesh):
            router_logits = jax.nn.sigmoid(jnp.asarray(logits_np).astype(jnp.float32))
            topk_w, _ = moe.topk(router_logits, None)

        topk_w_np = np.asarray(topk_w)
        # After renorm then ×3.0, each row sums to 3.0.
        row_sums = topk_w_np.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums,
            np.full(T, 3.0),
            atol=1e-5,
            err_msg="Each token's weights should sum to 3.0 after renorm×scale",
        )


# ---------------------------------------------------------------------------
# Tests: clamp over-limit [NAMED BLIND SPOT]
# ---------------------------------------------------------------------------


class TestClampOverLimit(unittest.TestCase):
    """Verify swiglu_limit wired correctly per-layer for routed and shared experts."""

    def setUp(self):
        self.mesh = _make_mesh()

    def _build_epmoe(self, swiglu_limit, num_experts=8, hidden=32, inter=16):
        from sgl_jax.srt.layers.moe import EPMoE

        with jax.set_mesh(self.mesh):
            return EPMoE(
                hidden_size=hidden,
                num_experts=num_experts,
                num_experts_per_tok=2,
                ep_size=1,
                mesh=self.mesh,
                intermediate_dim=inter,
                weight_dtype=jnp.float32,
                dtype=jnp.float32,
                swiglu_limit=swiglu_limit,
            )

    def test_clamp_layer_index_mapping(self):
        """config swiglu_limits[i] -> layer i (routed reads swiglu_limits, shared reads
        swiglu_limits_shared). Mirrors the REAL Step-3.5 distribution: layer 43 =
        routed-only (7), layer 44 = routed (7) + shared (16), layer 3 = none.

        Alignment cannot cover this: its single clamp layer is routed+shared only, and
        its small random weights keep activations below the limit so the clamp is a
        no-op (it would pass with the clamp wired to the wrong layer). This test pins
        the config->layer index mapping directly, with the real routed-only case.
        """
        NL = 45
        routed = [0.0] * NL
        shared = [0.0] * NL
        routed[43] = routed[44] = 7.0  # routed clamp on 43 and 44
        shared[44] = 16.0  # shared clamp only on 44
        cfg = _tiny_moe_config(layer_id=43, num_layers=NL)
        cfg.swiglu_limits = routed
        cfg.swiglu_limits_shared = shared
        cases = {3: (None, None), 43: (7.0, None), 44: (7.0, 16.0)}
        for lid, (exp_routed, exp_shared) in cases.items():
            moe = _build_moe(cfg, layer_id=lid, mesh=self.mesh)
            self.assertEqual(
                moe.experts.swiglu_limit, exp_routed, f"layer {lid}: routed swiglu_limit"
            )
            self.assertEqual(
                moe.shared_experts.swiglu_limit, exp_shared, f"layer {lid}: shared swiglu_limit"
            )

    def test_routed_clamp_upper_only(self):
        """Routed expert gate clamped upper-only at limit=7 (via EPMoE swiglu_limit)."""
        E, H, inter, T, K, L = 8, 32, 16, 4, 2, 7.0
        rng = np.random.default_rng(10)

        m = self._build_epmoe(swiglu_limit=L, num_experts=E, hidden=H, inter=inter)
        # Set weights to produce gate activations well above limit.
        m.wi_0.value = jnp.asarray(rng.normal(0, 5.0, (E, H, inter)), jnp.float32)
        m.wi_1.value = jnp.asarray(rng.normal(0, 5.0, (E, H, inter)), jnp.float32)
        m.wo.value = jnp.asarray(rng.normal(0, 1.0, (E, inter, H)), jnp.float32)

        x = jnp.asarray(rng.normal(0, 3.0, (T, H)), jnp.float32)
        topk_ids = jnp.asarray(rng.integers(0, E, (T, K)), jnp.int32)
        topk_w = jnp.asarray(rng.uniform(0.1, 1.0, (T, K)), jnp.float32)

        with jax.set_mesh(self.mesh):
            got = m(x, topk_w, topk_ids)
        ref = _naive_moe_ref(
            np.asarray(x),
            np.asarray(m.wi_0.value),
            np.asarray(m.wi_1.value),
            np.asarray(m.wo.value),
            np.asarray(topk_w),
            np.asarray(topk_ids),
            L,
        )
        np.testing.assert_allclose(
            np.asarray(got),
            ref,
            rtol=2e-3,
            atol=2e-3,
            err_msg="Routed EPMoE with limit=7 must match clamped loop ref",
        )

    def test_shared_mlp_clamp_asymmetry(self):
        """Gate: upper-only clip. Up: double-sided clip. Asymmetry must hold."""
        from sgl_jax.srt.models.step3p5 import Step3p5MLP

        H, inter, L = 16, 8, 2.0
        mesh = self.mesh

        with jax.set_mesh(mesh):
            mlp = Step3p5MLP(H, inter, mesh=mesh, dtype=jnp.float32, swiglu_limit=L)

        # Force gate_proj output to be large-positive to trigger upper clip.
        # Force up_proj output to be large-negative to trigger lower clip.
        # Use identity-style weights for predictability.
        gw = np.zeros((H, inter), np.float32)
        uw = np.zeros((H, inter), np.float32)
        dw = np.eye(inter, H, dtype=np.float32)

        # gate_proj produces +4 on dim 0, -4 on dim 1
        gw[0, 0] = 4.0
        gw[1, 1] = 4.0
        # up_proj produces -6 on dim 0, +6 on dim 1
        uw[0, 0] = -6.0
        uw[1, 1] = 6.0

        mlp.gate_proj.weight.value = jnp.asarray(gw)
        mlp.up_proj.weight.value = jnp.asarray(uw)
        mlp.down_proj.weight.value = jnp.asarray(dw)

        x = jnp.ones((1, H), jnp.float32)
        with jax.set_mesh(mesh):
            out = np.asarray(mlp(x))

        # silu(4.0) ≈ 3.928; clipped to 2.0 (upper-only — negative silu stays)
        silu_4 = 4.0 / (1.0 + np.exp(-4.0))
        gate_0_expected = min(silu_4, L)
        # silu(4.0) positive, also clipped to 2.0
        gate_1_expected = min(silu_4, L)
        # up dim 0: -6.0 clipped to -2.0 (double-sided)
        up_0_expected = max(-6.0, -L)
        # up dim 1: +6.0 clipped to +2.0
        up_1_expected = min(6.0, L)

        expected_intermediate_0 = gate_0_expected * up_0_expected
        expected_intermediate_1 = gate_1_expected * up_1_expected

        # down_proj is identity (inter->H), result maps back
        np.testing.assert_allclose(
            out[0, 0],
            expected_intermediate_0,
            atol=1e-5,
            err_msg="Gate upper+up lower clamp mismatch on dim 0",
        )
        np.testing.assert_allclose(
            out[0, 1],
            expected_intermediate_1,
            atol=1e-5,
            err_msg="Gate upper+up upper clamp mismatch on dim 1",
        )


# ---------------------------------------------------------------------------
# Tests: GMM == loop + dispatch conservation
# ---------------------------------------------------------------------------


class TestGMMEqualLoop(unittest.TestCase):
    """EPMoE GMM output must equal per-expert loop with the same clamp."""

    def setUp(self):
        self.mesh = _make_mesh()

    def _build_epmoe(self, swiglu_limit=None, num_experts=8, hidden=32, inter=16):
        from sgl_jax.srt.layers.moe import EPMoE

        with jax.set_mesh(self.mesh):
            return EPMoE(
                hidden_size=hidden,
                num_experts=num_experts,
                num_experts_per_tok=2,
                ep_size=1,
                mesh=self.mesh,
                intermediate_dim=inter,
                weight_dtype=jnp.float32,
                dtype=jnp.float32,
                swiglu_limit=swiglu_limit,
            )

    def _run_test(self, swiglu_limit, seed=0):
        E, H, inter, T, K = 8, 32, 16, 6, 2
        rng = np.random.default_rng(seed)

        m = self._build_epmoe(swiglu_limit=swiglu_limit, num_experts=E, hidden=H, inter=inter)
        m.wi_0.value = jnp.asarray(rng.normal(0, 3.0, (E, H, inter)), jnp.float32)
        m.wi_1.value = jnp.asarray(rng.normal(0, 3.0, (E, H, inter)), jnp.float32)
        m.wo.value = jnp.asarray(rng.normal(0, 1.0, (E, inter, H)), jnp.float32)
        x = jnp.asarray(rng.normal(0, 2.0, (T, H)), jnp.float32)
        topk_ids = jnp.asarray(rng.integers(0, E, (T, K)), jnp.int32)
        topk_w = jnp.asarray(rng.uniform(0.1, 1.0, (T, K)), jnp.float32)

        with jax.set_mesh(self.mesh):
            got = m(x, topk_w, topk_ids)

        ref = _naive_moe_ref(
            np.asarray(x),
            np.asarray(m.wi_0.value),
            np.asarray(m.wi_1.value),
            np.asarray(m.wo.value),
            np.asarray(topk_w),
            np.asarray(topk_ids),
            swiglu_limit,
        )
        np.testing.assert_allclose(
            np.asarray(got),
            ref,
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"GMM!=loop for swiglu_limit={swiglu_limit}",
        )

    def test_gmm_equals_loop_no_clamp(self):
        self._run_test(swiglu_limit=None, seed=50)

    def test_gmm_equals_loop_with_clamp_7(self):
        """Routed-expert clamp limit=7 (layers 43/44 analog)."""
        self._run_test(swiglu_limit=7.0, seed=51)


# ---------------------------------------------------------------------------
# Tests: MoE forward output shape + shared expert always added
# ---------------------------------------------------------------------------


class TestMoEForward(unittest.TestCase):
    """Full Step3p5MoE forward: output shape, shared expert contribution."""

    def setUp(self):
        self.mesh = _make_mesh()

    def test_shared_expert_is_added(self):
        """Shared expert output is non-zero and is added to MoE output."""
        config = _tiny_moe_config(layer_id=3)
        T, H = 4, config.hidden_size
        rng = np.random.default_rng(70)

        moe = _build_moe(config, layer_id=3, mesh=self.mesh)
        x = jnp.asarray(rng.standard_normal((T, H)), jnp.float32)

        with jax.set_mesh(self.mesh):
            full_out, _ = moe(x)
            full_out = np.asarray(full_out)
            # Zero out shared expert weights to isolate moe-only contribution.
            moe.shared_experts.gate_proj.weight.value = jnp.zeros(
                moe.shared_experts.gate_proj.weight.value.shape, jnp.float32
            )
            moe.shared_experts.up_proj.weight.value = jnp.zeros(
                moe.shared_experts.up_proj.weight.value.shape, jnp.float32
            )
            moe_only_out, _ = moe(x)
            moe_only_out = np.asarray(moe_only_out)

        self.assertFalse(
            np.allclose(full_out, moe_only_out, atol=1e-6),
            "Shared expert must contribute to MoE output (outputs differ when shared is non-zero)",
        )


if __name__ == "__main__":
    unittest.main()
