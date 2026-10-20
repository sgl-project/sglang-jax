"""V4 routing semantics and real TPU EPMoE numerics (no mocked kernels).

Reference: epic/dsv4 commit 5f2e7ea12, with the NumPy equations below.
TPU cases require actual TPU execution and are skipped on CPU. The small
CPU routing cases use BF16 activations and four logical CPU devices.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp
from sgl_jax.srt.layers.deepseek_v4_moe import DeepseekV4EPMoE, DeepseekV4MoE
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.moe import EPMoE


def mesh_for(data=1, tensor=1):
    if len(jax.devices()) < data * tensor:
        pytest.skip("requires more devices")
    return Mesh(
        np.array(jax.devices()[: data * tensor]).reshape(data, tensor),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )


def config(**kwargs):
    values = dict(
        hidden_size=8,
        num_hidden_layers=43,
        n_routed_experts=4,
        num_experts_per_tok=2,
        vocab_size=16,
        num_hash_layers=3,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        moe_intermediate_size=8,
        swiglu_limit=2.0,
        n_shared_experts=1,
        ep_size=1,
        moe_dp_size=1,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def assign(param, value):
    param.value = jax.device_put(np.asarray(value, dtype=param.value.dtype), param.value.sharding)


def expert_layer(layer_cls=EPMoE, **kwargs):
    mesh = mesh_for()
    with jax.set_mesh(mesh):
        return layer_cls(
            hidden_size=8,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            **kwargs,
        )


@pytest.mark.parametrize("activation", ["silu", "gelu", "swigluoai"])
def test_shared_activation_hooks_preserve_existing_models(activation):
    layer = expert_layer(activation=activation)
    gate = jnp.asarray([[-9.0, -1.0, 1.0, 9.0]], jnp.float32)
    up = jnp.asarray([[-9.0, 0.0, 2.0, 9.0]], jnp.float32)
    if activation == "silu":
        expected = jax.nn.silu(gate) * up
    elif activation == "gelu":
        expected = jax.nn.gelu(gate) * up
    else:
        g = np.minimum(np.asarray(gate), 7.0)
        u = np.clip(np.asarray(up), -7.0, 7.0)
        expected = (u + 1.0) * g / (1.0 + np.exp(-1.702 * g))
    actual = jax.jit(lambda g, u: layer._apply_activation_with_context(g, u, local_range=(0, 1)))(
        gate, up
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert layer.swiglu_alpha == 1.702
    assert layer.swiglu_limit == 7.0


def test_context_adapter_preserves_two_input_subclass_override():
    class TwoInputMoE(EPMoE):
        def _apply_activation(self, gate, up):
            return gate + 2 * up

    layer = expert_layer(TwoInputMoE)
    gate, up = jnp.ones((2, 8)), jnp.full((2, 8), 3.0)
    actual = jax.jit(lambda g, u: layer._apply_activation_with_context(g, u, local_range=(0, 1)))(
        gate, up
    )
    np.testing.assert_array_equal(actual, np.full((2, 8), 7.0))


@pytest.mark.parametrize("v4", [False, True])
def test_expert_compute_dispatches_all_three_gmms_through_hook(v4):
    calls = []

    class DenseHookMoE(DeepseekV4EPMoE if v4 else EPMoE):
        def _call_gmm(self, **kwargs):
            # One expert: a real dense backend exercises the shared pipeline on CPU.
            calls.append(kwargs)
            return kwargs["lhs"] @ kwargs["rhs"][0]

    layer = expert_layer(DenseHookMoE, **({"swiglu_limit": 2.0} if v4 else {}))
    x = jnp.asarray(np.arange(16).reshape(2, 8) / 4, jnp.float32)
    gate_weight = jnp.eye(8)[None, :, :]
    up_weight = (2 * jnp.eye(8))[None, :, :]
    down_weight = (3 * jnp.eye(8))[None, :, :]
    result = layer._gmm_compute(
        x,
        jnp.arange(2, dtype=jnp.int32),
        jnp.asarray([2], jnp.int32),
        gate_weight,
        up_weight,
        down_weight,
        jnp.asarray(0, jnp.int32),
    )
    h = np.asarray(x, dtype=jnp.bfloat16).astype(np.float32)
    g = np.minimum(h, 2.0) if v4 else h
    u = np.clip(2 * h, -2.0, 2.0) if v4 else 2 * h
    expected = g / (1 + np.exp(-g)) * u * 3
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert len(calls) == 3
    assert [call["zero_initialize"] for call in calls] == [False, False, True]
    assert all(call["rhs_scale"] is None for call in calls)


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_v4_expert_clamp_validation_is_owned_by_subclass(limit):
    with pytest.raises(ValueError, match="finite positive limit"):
        expert_layer(DeepseekV4EPMoE, swiglu_limit=limit)


@pytest.mark.parametrize("limit,rows_enabled", [(None, True), (2.0, False), (2.0, True)])
def test_v4_context_fallback_preserves_activation(monkeypatch, limit, rows_enabled):
    from sgl_jax.srt.layers import deepseek_v4_moe as v4_module

    monkeypatch.setattr(v4_module, "_ACT_ROWS", rows_enabled)
    layer = expert_layer(DeepseekV4EPMoE, swiglu_limit=limit)
    gate = jnp.asarray([[-5.0, -1.0, 1.0, 5.0]])
    up = jnp.asarray([[-5.0, 5.0, -5.0, 5.0]])
    bounds = (0, 1) if limit is None or not rows_enabled else None
    actual = layer._apply_activation_with_context(gate, up, local_range=bounds)
    g = np.asarray(gate) if limit is None else np.minimum(np.asarray(gate), limit)
    u = np.asarray(up) if limit is None else np.clip(np.asarray(up), -limit, limit)
    np.testing.assert_allclose(actual, g / (1 + np.exp(-g)) * u, rtol=1e-6)
    assert layer.swiglu_limit == limit


@pytest.mark.parametrize("start,end", [(512, 768), (512, 512), (0, 1024)])
def test_v4_context_runs_real_row_kernel_against_numpy(monkeypatch, start, end):
    from sgl_jax.srt.layers import deepseek_v4_moe as v4_module

    monkeypatch.setattr(v4_module, "_ACT_ROWS", True)
    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    layer = expert_layer(DeepseekV4EPMoE, swiglu_limit=2.0)
    rng = np.random.default_rng(19)
    gate, up = [rng.normal(0, 3, (1024, 128)).astype(np.float32) for _ in range(2)]
    actual = jax.jit(
        lambda g, u, s, e: layer._apply_activation_with_context(g, u, local_range=(s, e))
    )(jnp.asarray(gate), jnp.asarray(up), jnp.asarray(start), jnp.asarray(end))
    expected = np.zeros_like(gate)
    # The kernel computes complete blocks intersecting [start, end).
    if end > start:
        lo, hi = start // 512 * 512, (end + 511) // 512 * 512
        g = np.minimum(gate[lo:hi], 2.0)
        expected[lo:hi] = g / (1 + np.exp(-g)) * np.clip(up[lo:hi], -2.0, 2.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_v4_expert_permutation_defaults_match_epic(monkeypatch):
    mesh = mesh_for()
    monkeypatch.delenv("SGL_JAX_MOE_SC_PERMUTE", raising=False)
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(config(), mesh, 0, backend="reference")
    assert layer.experts.use_sc_permute
    assert layer.experts.sort_free_permute
    assert layer.experts.inverse_by_sort

    monkeypatch.setenv("SGL_JAX_MOE_SC_PERMUTE", "false")
    with jax.set_mesh(mesh):
        disabled = DeepseekV4MoE(config(), mesh, 0, backend="reference")
    assert not disabled.experts.use_sc_permute


def test_v4_sort_free_permute_and_combine_match_stable_sort():
    mesh = mesh_for()
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(config(), mesh, 0, backend="reference")
        x = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
        ids = jnp.asarray([[2, 1], [0, 2], [1, 0], [2, 1]], jnp.int32)
        _, _, perm, _ = layer.experts._permute(x, ids)
        flat_ids = jnp.ravel(ids)
        np.testing.assert_array_equal(perm, jnp.argsort(flat_ids, stable=True))

        routed = jnp.arange(8, dtype=jnp.float32)[:, None] * jnp.ones((1, 8))
        weights = jnp.asarray([[0.25, 0.75]] * 4, jnp.float32)
        actual = layer.experts._unpermute(routed[perm], perm, weights)
        expected = (routed.reshape(4, 2, 8) * weights[:, :, None]).sum(axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_sqrtsoftplus_extremes_and_fp32():
    mesh = mesh_for()
    with jax.set_mesh(mesh):
        gate = GateLogit(5, 5, score_func="sqrtsoftplus")
        assign(gate.kernel, np.eye(5))
        logits = np.array([[-80, -5, 0, 5, 1000]], np.float32)
        actual = gate(jnp.array(logits, dtype=jnp.bfloat16))
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, np.sqrt(np.logaddexp(0, logits)), rtol=1e-6)


def test_hash_gathers_before_normalizing_without_bias():
    scores = jnp.array([[0.2, 3, 0.7, 2], [2, 0.1, 0.5, 4]])
    ids = jnp.array([[2, 0], [1, 3]], dtype=jnp.int32)
    route = TopK(2, True, routed_scaling_factor=1.5)
    weights, actual_ids = jax.jit(lambda x: route(x, selected_experts=ids))(scores)
    selected = np.take_along_axis(np.asarray(scores), np.asarray(ids), axis=-1)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_allclose(weights, selected / selected.sum(-1, keepdims=True) * 1.5)
    with pytest.raises(ValueError, match="correction bias"):
        route(scores, jnp.zeros(4), selected_experts=ids)
    with pytest.raises(ValueError, match="integers"):
        route(scores, selected_experts=ids.astype(jnp.float32))


def test_single_token_hash_routing_on_explicit_mesh():
    mesh = mesh_for()
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(config(), mesh, 0, backend="reference")
        layer.load_hash_table(np.tile([[3, 1], [0, 2]], (8, 1)))
        sharding = NamedSharding(mesh, P("data", None))
        weights, ids = layer.route(
            jnp.ones((1, 8), jnp.bfloat16),
            jnp.asarray([0], jnp.int32),
            routing_sharding=sharding,
        )
    np.testing.assert_array_equal(np.asarray(ids), [[3, 1]])
    np.testing.assert_allclose(np.asarray(weights).sum(axis=-1), 1.5)


def test_learned_bias_changes_choice_only():
    weights, ids = TopK(2, True, routed_scaling_factor=1.5)(
        jnp.array([[1.0, 2.0, 3.0, 4.0]]), jnp.array([10.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(ids, [[0, 3]])
    np.testing.assert_allclose(weights, [[0.3, 1.2]])


def test_clamp_is_before_silu_and_gate_has_no_lower_bound():
    gate = jnp.array([-5.0, -1.0, 1.0, 5.0])
    up = jnp.array([-5.0, 5.0, -5.0, 5.0])
    actual = silu_and_mul_with_clamp(gate, up, 2.0)
    g = np.minimum(np.asarray(gate), 2.0)
    expected = g / (1 + np.exp(-g)) * np.clip(np.asarray(up), -2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert not np.allclose(actual, jnp.minimum(jax.nn.silu(gate), 2) * jnp.clip(up, -2, 2))


@pytest.mark.parametrize("data,tensor", [(1, 1), (2, 2)])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_hash_table_roundtrip_hidden_dependence_and_padding(data, tensor, sequence_parallel):
    mesh = mesh_for(data, tensor)
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(config(), mesh, 0)
        table = np.tile([[3, 1], [0, 2]], (8, 1)).astype(np.int64)
        layer.load_hash_table(table)
        assert layer.gate.tid2eid.value.dtype == jnp.int32
        assert layer.gate.tid2eid.value.sharding.spec == P(None, None)
        assign(layer.gate.kernel, np.arange(32).reshape(8, 4) / 32)
        token_axis = ("data", "tensor") if sequence_parallel else "data"
        sharding = NamedSharding(mesh, P(token_axis, None))
        x = jax.device_put(np.ones((4, 8), np.float32).astype(jnp.bfloat16), sharding)
        input_ids = jax.device_put(
            np.array([7, 6, -1, 100], np.int32), NamedSharding(mesh, P(token_axis))
        )
        run = jax.jit(lambda h, t: layer.route(h, t, routing_sharding=sharding))
        weights, ids = run(x, input_ids)
        other, other_ids = run(-x, input_ids)
        weights, other = np.asarray(weights), np.asarray(other)
        np.testing.assert_array_equal(ids, [[0, 2], [3, 1], [3, 1], [3, 1]])
        np.testing.assert_array_equal(ids, other_ids)
        assert not np.allclose(weights[:2], other[:2])
        np.testing.assert_array_equal(weights[2:], 0)
        np.testing.assert_allclose(weights[:2].sum(-1), 1.5)
        changed_ids = jax.device_put(np.array([6, 7, -1, 100], np.int32), input_ids.sharding)
        changed_weights, changed = run(x, changed_ids)
        np.testing.assert_array_equal(np.asarray(changed)[:2], [[3, 1], [0, 2]])
        np.testing.assert_allclose(np.asarray(changed_weights)[:2], weights[:2][::-1])
        masked, _ = layer.route(x, input_ids, token_valid_mask=jnp.array([False] * 4))
        np.testing.assert_array_equal(masked, 0)
        with pytest.raises(ValueError, match="requires input_ids"):
            layer.route(x)
        with pytest.raises(ValueError, match="integer expert"):
            layer.load_hash_table(table.astype(np.float32))
        with pytest.raises(ValueError, match="logical expert range"):
            layer.load_hash_table(np.full((16, 2), 4))
        learned = DeepseekV4MoE(config(), mesh, 3)
        assert learned.gate.bias.value.dtype == jnp.float32
        assert not hasattr(learned.gate, "tid2eid")
        learned.route(x)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="real TPU GMM required")
@pytest.mark.parametrize(
    "ep,data,tensor,replicated,sequence_parallel",
    [
        (1, 1, 1, False, False),
        (2, 1, 2, False, False),
        (1, 1, 2, False, False),
        (1, 2, 2, True, False),
        (1, 1, 2, False, True),
    ],
)
@pytest.mark.parametrize("hash_layer", [True, False])
@pytest.mark.parametrize("fp8", [False, True])
def test_tpu_full_block_against_numpy(
    ep, data, tensor, replicated, sequence_parallel, hash_layer, fp8
):
    if fp8 and replicated:
        pytest.skip("replicated EPMoE only supports unquantized weights")
    mesh = mesh_for(data, tensor)
    cfg = config(
        hidden_size=256,
        moe_intermediate_size=256,
        ep_size=ep,
        moe_dp_size=data if replicated else 1,
        quantization_config=(
            QuantizationConfig(moe_weight_dtype=jnp.float8_e4m3fn, weight_block_size=(128, 128))
            if fp8
            else None
        ),
    )
    rng = np.random.default_rng(15)
    # Deliberately activate both clamp boundaries; use independent expert weights.
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(cfg, mesh, 0 if hash_layer else 3)
        for param in (layer.experts.wi_0, layer.experts.wi_1, layer.experts.wo):
            assign(
                param,
                rng.normal(0, 0.12 if param is not layer.experts.wo else 0.03, param.value.shape),
            )
        if fp8:
            layer.experts.quantize_weights(is_static=True)
            for name in ("wi_0", "wi_1", "wo"):
                param = getattr(layer.experts, name)
                scale = 0.5 if name != "wo" else 0.125
                param.value = jax.device_put(
                    (np.asarray(param.value, dtype=np.float32) / scale).astype(jnp.float8_e4m3fn),
                    param.value.sharding,
                )
                scale_param = getattr(layer.experts, name + "_scale")
                assign(scale_param, np.full(scale_param.value.shape, scale))
        for name in ("gate_proj", "up_proj", "down_proj"):
            p = getattr(layer.shared_experts, name).weight
            assign(p, rng.normal(0, 0.12 if name != "down_proj" else 0.03, p.value.shape))
        assign(layer.gate.kernel, rng.normal(0, 0.03, layer.gate.kernel.value.shape))
        if hash_layer:
            layer.load_hash_table(np.tile([[3, 0], [1, 2]], (8, 1)))
        else:
            assign(layer.gate.bias, [0.2, -0.3, 0.5, 0.0])
        token_axis = ("data", "tensor") if sequence_parallel else "data"
        shard = NamedSharding(mesh, P(token_axis, None))
        x = jax.device_put(rng.normal(0, 2, (16, 256)).astype(jnp.bfloat16), shard)
        tids = jax.device_put(np.arange(16, dtype=np.int32), NamedSharding(mesh, P(token_axis)))
        mask = jnp.arange(16) < 13
        out, ids = jax.jit(
            lambda h, t, m: layer(
                h, t, token_valid_mask=m, out_sharding=shard, return_expert_ids=True
            )
        )(x, tids, mask)
        out.block_until_ready()
        h = np.asarray(x, dtype=np.float32)
        scores = np.sqrt(np.logaddexp(0, h @ np.asarray(layer.gate.kernel.value)))
        if hash_layer:
            selected = np.asarray(layer.gate.tid2eid.value)[np.arange(16)]
        else:
            selected = np.argsort(-(scores + np.asarray(layer.gate.bias.value)), axis=-1)[:, :2]
        weights = np.take_along_axis(scores, selected, axis=-1)
        weights = weights / weights.sum(-1, keepdims=True) * 1.5

        def mlp(a, b, c):
            g = np.minimum(h @ a, cfg.swiglu_limit)
            u = np.clip(h @ b, -cfg.swiglu_limit, cfg.swiglu_limit)
            return (g / (1 + np.exp(-g)) * u) @ c

        def expert_weight(name):
            w = np.asarray(getattr(layer.experts, name).value, dtype=np.float32)
            if fp8:
                scale = np.asarray(getattr(layer.experts, name + "_scale").value)
                w = w * np.repeat(scale[:, :, 0, :], 128, axis=1)
            return w

        a, b, c = [expert_weight(name) for name in ("wi_0", "wi_1", "wo")]
        expert_out = np.stack([mlp(a[e], b[e], c[e]) for e in range(4)], axis=1)
        expected = (
            np.take_along_axis(expert_out, selected[:, :, None], axis=1) * weights[:, :, None]
        ).sum(1)
        expected += mlp(
            *[
                np.asarray(getattr(layer.shared_experts, n).weight.value, dtype=np.float32)
                for n in ("gate_proj", "up_proj", "down_proj")
            ]
        )
        expected[13:] = 0
        np.testing.assert_array_equal(np.asarray(ids)[:13], selected[:13])
        # BF16 intermediate GMM and shared linear rounding versus FP32 dense oracle.
        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float32), expected, atol=0.12, rtol=0.035
        )
        np.testing.assert_array_equal(np.asarray(out)[13:], 0)


def test_preselected_ids_gather_logical_weights_before_eplb(monkeypatch):
    from sgl_jax.srt.layers import gate as gate_module

    def translate(ids, metadata, layer_id):
        assert metadata == "dispatch" and layer_id == 2
        return ids + 4

    monkeypatch.setattr(gate_module, "topk_ids_logical_to_physical", translate)
    weights, ids, logical_ids = TopK(2, True, layer_id=2)(
        jnp.array([[1.0, 2.0, 3.0, 4.0]]),
        dispatch_info="dispatch",
        selected_experts=jnp.array([[3, 0]], dtype=jnp.int32),
        return_logical_ids=True,
    )
    np.testing.assert_array_equal(ids, [[7, 4]])
    np.testing.assert_array_equal(logical_ids, [[3, 0]])
    np.testing.assert_allclose(weights, [[0.8, 0.2]])


@pytest.mark.parametrize("hash_layer", [True, False])
def test_model_facing_ids_remain_logical_with_dispatch_mapping(monkeypatch, hash_layer):
    from sgl_jax.srt.layers import gate as gate_module

    mesh = mesh_for()
    monkeypatch.setattr(
        gate_module,
        "topk_ids_logical_to_physical",
        lambda ids, metadata, layer_id: (ids + 1) % 4,
    )
    with jax.set_mesh(mesh):
        layer = DeepseekV4MoE(config(), mesh, 0 if hash_layer else 3, backend="reference")
        if hash_layer:
            layer.load_hash_table(np.tile([[3, 1], [0, 2]], (8, 1)))
        hidden = jnp.ones((2, 8), jnp.bfloat16)
        token_ids = jnp.asarray([0, 1], jnp.int32)
        _, expected_logical_ids = layer.route(hidden, token_ids)
        output, logical_ids = layer(
            hidden, token_ids, dispatch_info="dispatch", return_expert_ids=True
        )
        output_only = layer(hidden, token_ids, dispatch_info="dispatch")
    np.testing.assert_array_equal(logical_ids, expected_logical_ids)
    np.testing.assert_array_equal(output, output_only)
