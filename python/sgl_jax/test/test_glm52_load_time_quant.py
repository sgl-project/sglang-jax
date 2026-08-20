from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.fused_moe import FusedEPMoERS, FusedEPMoEV2
from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.glm5_moe import (
    Glm5ForCausalLM,
    _use_fused_rs_for_forward_mode,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader
from sgl_jax.srt.kernels.fused_moe.fused_rs import (
    fused_moe_rs,
    gmm_fused_rs_nodedup,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    _build_packed_index_tile_table,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs.fused_moe_rs import (
    _compute_rs_routing,
)
def _cpu_mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


class _TinyQuantizedModel(nnx.Module):
    def __init__(self, mesh):
        self.proj = QuantizedLinear(
            weight_q=jnp.zeros((3, 4), dtype=jnp.float8_e4m3fn),
            weight_scale=jnp.zeros((3,), dtype=jnp.float32),
            bias=None,
            activation_dtype=jnp.float8_e4m3fn,
            mesh=mesh,
            kernel_axes=(None, None),
        )


class _MoEQuantConfig:
    quantize_on_load = True
    weight_block_size = None

    @staticmethod
    def get_moe_weight_dtype():
        return jnp.float8_e4m3fn

    @staticmethod
    def get_moe_activation_dtype():
        return jnp.float8_e4m3fn


class _TinyMoEWeights(nnx.Module):
    def __init__(self):
        self.w1 = nnx.Param(jnp.zeros((2, 3, 4), dtype=jnp.float8_e4m3fn))
        self.w1_scale = nnx.Param(jnp.zeros((2, 1, 1, 4), dtype=jnp.float32))
        self.w1_shared = nnx.Param(jnp.zeros((3, 4), dtype=jnp.float8_e4m3fn))
        self.w1_shared_scale = nnx.Param(jnp.zeros((1, 1, 4), dtype=jnp.float32))


class _TinyMoEModel(nnx.Module):
    def __init__(self):
        self.mlp = _TinyMoEWeights()


def test_weight_loader_quantizes_linear_and_assigns_scale_immediately():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        model = _TinyQuantizedModel(mesh)
    params = nnx.state(model)
    loader = object.__new__(WeightLoader)
    loader.model_config = SimpleNamespace(
        quantization_config=SimpleNamespace(quantize_on_load=True)
    )

    weight = jnp.asarray(
        [[-4.0, -2.0, 1.0, 3.0], [1.0, 2.0, 4.0, 8.0], [-3.0, 0.0, 2.0, 1.0]],
        dtype=jnp.bfloat16,
    )
    assert loader._assign_load_time_quantized_weight(params, "proj.weight_q", weight)

    weight_q = params["proj"]["weight_q"].value
    scale = params["proj"]["weight_scale"].value
    assert weight_q.dtype == jnp.float8_e4m3fn
    assert scale.shape == (3,)
    reconstructed = weight_q.astype(jnp.float32) * scale[:, None]
    np.testing.assert_allclose(
        np.asarray(reconstructed), np.asarray(weight, dtype=np.float32), rtol=0.05
    )


def test_weight_loader_quantizes_routed_and_shared_moe_per_channel():
    params = nnx.state(_TinyMoEModel())
    loader = object.__new__(WeightLoader)
    loader.model_config = SimpleNamespace(quantization_config=_MoEQuantConfig())

    routed = jnp.arange(24, dtype=jnp.bfloat16).reshape(2, 3, 4) - 12
    assert loader._assign_load_time_quantized_weight(params, "mlp.w1", routed)
    routed_reconstructed = (
        params["mlp"]["w1"].value.astype(jnp.float32)
        * params["mlp"]["w1_scale"].value[:, 0, 0, :][:, None, :]
    )
    np.testing.assert_allclose(
        np.asarray(routed_reconstructed), np.asarray(routed, dtype=np.float32), rtol=0.05
    )

    shared = jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4) - 6
    assert loader._assign_load_time_quantized_weight(params, "mlp.w1_shared", shared)
    shared_reconstructed = (
        params["mlp"]["w1_shared"].value.astype(jnp.float32)
        * params["mlp"]["w1_shared_scale"].value[0, 0, :][None, :]
    )
    np.testing.assert_allclose(
        np.asarray(shared_reconstructed), np.asarray(shared, dtype=np.float32), rtol=0.05
    )


def test_fused_moe_v2_dynamic_per_channel_scale_layout():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        layer = FusedEPMoEV2(
            hidden_size=4,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=8,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            quantization_config=_MoEQuantConfig(),
        )
        layer.quantize_weights(is_static=False)

    assert layer.w1.value.dtype == jnp.float8_e4m3fn
    assert layer.w2.value.dtype == jnp.float8_e4m3fn
    assert layer.w3.value.dtype == jnp.float8_e4m3fn
    assert layer.w1_scale.value.shape == (1, 1, 1, 8)
    assert layer.w2_scale.value.shape == (1, 1, 1, 4)
    assert layer.w3_scale.value.shape == (1, 1, 1, 8)
    assert layer.w1_shared_scale.value.shape == (1, 1, 8)
    assert layer.w2_shared_scale.value.shape == (1, 1, 4)
    assert layer.w3_shared_scale.value.shape == (1, 1, 8)


def test_fused_moe_v2_static_per_channel_scale_layout():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        layer = FusedEPMoEV2(
            hidden_size=4,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=8,
            weight_dtype=jnp.float8_e4m3fn,
            dtype=jnp.bfloat16,
            quantization_config=_MoEQuantConfig(),
        )
        layer.quantize_weights(is_static=True)

    assert layer.w1_shared_scale.value.shape == (1, 1, 8)
    assert layer.w2_shared_scale.value.shape == (1, 1, 4)
    assert layer.w3_shared_scale.value.shape == (1, 1, 8)


def test_glm52_load_time_mapping_targets_quantized_linears_without_scale_sidecars():
    model = object.__new__(Glm5ForCausalLM)
    mappings = model._create_moe_layer_mappings(
        layer_idx=0,
        target_idx=0,
        is_mlp_layer=True,
        is_static_quant=False,
        is_load_time_quant=True,
        has_indexer=True,
    )

    q_a = mappings["model.layers.0.self_attn.q_a_proj.weight"]
    o_proj = mappings["model.layers.0.self_attn.o_proj.weight"]
    dense = mappings["model.layers.0.mlp.gate_proj.weight"]
    indexer_gate = mappings["model.layers.0.self_attn.indexer.weights_proj.weight"]
    assert q_a.target_path.endswith("q_a_proj.weight_q")
    assert o_proj.target_path.endswith("o_proj.weight_q")
    assert dense.target_path.endswith("gate_proj.weight_q")
    assert indexer_gate.target_path.endswith("weights_proj.weight")
    assert not any(key.endswith("weight_scale_inv") for key in mappings)


@pytest.mark.parametrize("moe_backend", ["fused_v2", "fused_rs"])
def test_glm52_static_per_channel_mappings_load_linear_and_fused_moe_scales_directly(
    moe_backend,
):
    model = object.__new__(Glm5ForCausalLM)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            hidden_size=6144,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=1,
            moe_backend=moe_backend,
            quantization_config=SimpleNamespace(
                is_static_checkpoint=True,
                weight_block_size=None,
            ),
        ),
    )
    mappings = model._create_moe_layer_mappings(
        layer_idx=3,
        target_idx=3,
        is_mlp_layer=False,
        is_static_quant=True,
        has_indexer=True,
    )

    linear_scale = mappings["model.layers.3.self_attn.o_proj.weight_scale_inv"]
    assert linear_scale.target_path.endswith("o_proj.weight_scale")
    assert linear_scale.sharding == (None,)

    routed_scale = mappings["__MOE_EXPERTS__model.layers.3.mlp.w1_scale"]
    assert routed_scale.target_path[0] == "model.layers.3.mlp.w1_scale"
    assert routed_scale.sharding == (("data", "tensor"), None)
    assert routed_scale.reshape == (256, 1, 1, 2048)

    shared_scale = mappings["model.layers.3.mlp.shared_experts.gate_proj.weight_scale_inv"]
    assert shared_scale.target_path == "model.layers.3.mlp.w1_shared_scale"
    assert shared_scale.sharding == (None, None, None)
    assert shared_scale.reshape == (1, 1, 2048)


def test_fused_rs_forward_mode_policy_keeps_decode_on_v2():
    rs_modes = {ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DRAFT_EXTEND}
    for mode in ForwardMode:
        assert _use_fused_rs_for_forward_mode(mode) is (mode in rs_modes)


def test_fused_v2_disable_shared_expert_omits_shared_kernel_arguments(monkeypatch):
    mesh = _cpu_mesh()
    observed = {}

    def fake_fused_ep_moe_v2(*args, **kwargs):
        observed.update(kwargs)
        return jnp.zeros_like(args[1])

    monkeypatch.setattr(
        "sgl_jax.srt.kernels.fused_moe.v2.kernel.fused_ep_moe_v2",
        fake_fused_ep_moe_v2,
    )
    with jax.set_mesh(mesh):
        layer = FusedEPMoEV2(
            hidden_size=4,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=3,
            disable_shared_expert=True,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
        )
        layer(
            jnp.ones((2, 4), dtype=jnp.bfloat16),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.int32),
            block_config=object(),
        )

    assert observed["w1_shared"] is None
    assert observed["w2_shared"] is None
    assert observed["w3_shared"] is None
    assert observed["w1_shared_scale"] is None
    assert observed["w2_shared_scale"] is None
    assert observed["w3_shared_scale"] is None


def test_fused_rs_ep32_64k_per_channel_tuning_is_narrow(monkeypatch):
    monkeypatch.setattr(
        gmm_fused_rs_nodedup,
        "_is_tpu_v7x_ep_size",
        lambda ep_size, expected_ep_size: ep_size == expected_ep_size,
    )
    default = (128, 1024, 1024, 1024, 1024, 2, 2)
    common = dict(
        k1=6144,
        n1=4096,
        k2=2048,
        n2=6144,
        num_current_groups=8,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.float8_e4m3fn,
        rhs_quant_block_size=6144,
        default_block_sizes=default,
        ep_size=32,
        fuse_act="silu",
        fp8_direct_write=False,
    )

    assert gmm_fused_rs_nodedup.get_fused_rs_tuned_block_sizes(
        65536 * 8, **common
    ) == (128, 6144, 2048, 2048, 6144, 1, 1)
    assert (
        gmm_fused_rs_nodedup.get_fused_rs_tuned_block_sizes(32768 * 8, **common)
        == default
    )


def test_fused_rs_shared_expert_matches_reference_on_cpu():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        layer = FusedEPMoERS(
            hidden_size=4,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=3,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
        )
        x = jnp.arange(8, dtype=jnp.bfloat16).reshape(2, 4) / 8
        w1 = jnp.arange(12, dtype=jnp.bfloat16).reshape(4, 3) / 16
        w3 = (jnp.arange(12, dtype=jnp.bfloat16).reshape(4, 3) - 4) / 16
        w2 = (jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4) - 6) / 16
        layer.w1_shared.value = w1
        layer.w3_shared.value = w3
        layer.w2_shared.value = w2

        actual = layer._shared_expert_for_rs(x, enable_act_quant=False)
        expected = (jax.nn.silu(x.astype(jnp.float32) @ w1.astype(jnp.float32))
                    * (x.astype(jnp.float32) @ w3.astype(jnp.float32))) @ w2.astype(
                        jnp.float32
                    )

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0.02,
        atol=0.02,
    )


def test_fused_rs_quantized_shared_expert_matches_per_channel_reference_on_cpu():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        layer = FusedEPMoERS(
            hidden_size=8,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=8,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=4,
            weight_dtype=jnp.float8_e4m3fn,
            dtype=jnp.bfloat16,
            quantization_config=_MoEQuantConfig(),
        )
        x = jnp.asarray(
            [
                [0.25, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
                [-1.0, 0.5, 0.125, -0.25, 1.0, -0.5, -0.125, 0.25],
            ],
            dtype=jnp.bfloat16,
        )
        w1 = (jnp.arange(32, dtype=jnp.float32).reshape(8, 4) - 15).astype(
            jnp.float8_e4m3fn
        )
        w3 = (17 - jnp.arange(32, dtype=jnp.float32).reshape(8, 4)).astype(
            jnp.float8_e4m3fn
        )
        w2 = (jnp.arange(32, dtype=jnp.float32).reshape(4, 8) % 7 - 3).astype(
            jnp.float8_e4m3fn
        )
        s1 = jnp.asarray([[[0.01, 0.02, 0.03, 0.025]]], dtype=jnp.float32)
        s3 = jnp.asarray([[[0.04, 0.015, 0.025, 0.035]]], dtype=jnp.float32)
        s2 = jnp.asarray(
            [[[0.03, 0.01, 0.02, 0.04, 0.025, 0.015, 0.035, 0.02]]],
            dtype=jnp.float32,
        )
        layer.w1_shared.value = w1
        layer.w3_shared.value = w3
        layer.w2_shared.value = w2
        layer.w1_shared_scale.value = s1
        layer.w3_shared_scale.value = s3
        layer.w2_shared_scale.value = s2

        def qlinear(value, weight, scale, *, reserve_v2_scale_slots=False):
            value_f32 = value.astype(jnp.float32)
            value_amax = jnp.max(jnp.abs(value_f32), axis=-1, keepdims=True)
            value_scale = jnp.maximum(
                value_amax / jnp.float32(448.0),
                jnp.float32(1e-12),
            )
            value_q = (value_f32 / value_scale).astype(jnp.float8_e4m3fn)
            if reserve_v2_scale_slots:
                lane_width = value_q.shape[-1] // 4
                channel = jnp.arange(value_q.shape[-1], dtype=jnp.int32)
                reserved = (channel % lane_width) == (lane_width - 1)
                value_q = jnp.where(
                    reserved[None, :], jnp.zeros_like(value_q), value_q
                )
                acc = jnp.zeros(
                    (value_q.shape[0], weight.shape[1]), dtype=jnp.float32
                )
                for lane in range(4):
                    start = lane * lane_width
                    acc += (
                        value_q[:, start : start + lane_width]
                        @ weight[start : start + lane_width, :]
                    ).astype(jnp.float32)
            else:
                acc = (value_q @ weight).astype(jnp.float32)
            return (
                acc
                * (
                    value_scale.astype(jnp.float32)
                    * scale.reshape(1, -1)
                )
            )

        gate = qlinear(x, w1, s1, reserve_v2_scale_slots=True)
        up = qlinear(x, w3, s3, reserve_v2_scale_slots=True)
        intermediate = jax.nn.silu(gate) * up
        expected = None
        for start in range(0, intermediate.shape[1], 2):
            partial = qlinear(
                intermediate[:, start : start + 2],
                w2[start : start + 2, :],
                s2,
            )
            expected = (
                partial.astype(jnp.bfloat16)
                if expected is None
                else (expected.astype(jnp.float32) + partial).astype(jnp.bfloat16)
            )
        actual = layer._shared_expert_for_rs(
            x,
            enable_act_quant=True,
            v2_shared_block_size=2,
        )

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0.02,
        atol=0.002,
    )


def test_fused_rs_routing_metadata_preserves_expert_token_and_topk_slot():
    topk_indices = jnp.asarray(
        [[3, 1, 5], [0, 3, 2], [5, 2, 4], [1, 4, 0]],
        dtype=jnp.int32,
    )
    lhs_indices, group_sizes, output_indices, topk_slots = _compute_rs_routing(
        topk_indices,
        num_experts=6,
        topk=3,
    )
    flat_positions = np.asarray(lhs_indices) * 3 + np.asarray(topk_slots)
    flat_experts = np.asarray(topk_indices).reshape(-1)

    np.testing.assert_array_equal(np.asarray(output_indices), np.asarray(lhs_indices))
    np.testing.assert_array_equal(
        np.asarray(group_sizes),
        np.bincount(flat_experts, minlength=6),
    )
    np.testing.assert_array_equal(
        flat_experts[flat_positions],
        np.sort(flat_experts),
    )


def test_fused_rs_routing_metadata_keeps_invalid_padding_after_valid_groups():
    topk_indices = jnp.asarray(
        [[3, 1, -1], [-1, -1, -1], [0, 3, 2], [5, -1, 4]],
        dtype=jnp.int32,
    )
    lhs_indices, group_sizes, output_indices, topk_slots = _compute_rs_routing(
        topk_indices,
        num_experts=6,
        topk=3,
    )
    flat_positions = np.asarray(lhs_indices) * 3 + np.asarray(topk_slots)
    routed_experts = np.asarray(topk_indices).reshape(-1)[flat_positions]
    valid_count = int(np.asarray(group_sizes).sum())

    np.testing.assert_array_equal(np.asarray(output_indices), np.asarray(lhs_indices))
    np.testing.assert_array_equal(
        np.asarray(group_sizes),
        np.asarray([1, 1, 1, 2, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        routed_experts[:valid_count],
        np.asarray([0, 1, 2, 3, 3, 4, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        routed_experts[valid_count:],
        np.full(routed_experts.shape[0] - valid_count, -1, dtype=np.int32),
    )


def test_fused_rs_precomputed_routes_support_explicit_mesh(monkeypatch):
    mesh = _cpu_mesh()
    token_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("data", "tensor"), None)
    )
    weight_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("data", "tensor"), None, None)
    )

    hidden_states = jax.device_put(
        jnp.ones((16, 4), dtype=jnp.bfloat16), token_sharding
    )
    w1 = jax.device_put(
        jnp.ones((1, 4, 4), dtype=jnp.bfloat16), weight_sharding
    )
    w2 = jax.device_put(
        jnp.ones((1, 4, 4), dtype=jnp.bfloat16), weight_sharding
    )
    topk_weights = jax.device_put(
        jnp.ones((16, 1), dtype=jnp.float32), token_sharding
    )
    topk_indices = jax.device_put(
        jnp.zeros((16, 1), dtype=jnp.int32), token_sharding
    )

    def fake_expert_parallel_gmm_rs(
        hidden_states,
        _w1,
        _w1_scale,
        _w1_bias,
        _w2,
        _w2_scale,
        _w2_bias,
        _topk_weights,
        _topk_indices,
        **_kwargs,
    ):
        return jnp.zeros_like(hidden_states)

    monkeypatch.setattr(
        fused_moe_rs,
        "expert_parallel_gmm_rs",
        fake_expert_parallel_gmm_rs,
    )
    output = fused_moe_rs.fused_moe_func_rs(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        gating_output=None,
        topk=1,
        renormalize=False,
        mesh=mesh,
        activation="silu",
        scoring_fn="softmax",
        topk_weights=topk_weights,
        topk_indices=topk_indices,
    )

    assert output.sharding == token_sharding
    np.testing.assert_array_equal(np.asarray(output), np.zeros((16, 4)))


def test_fused_rs_high_m_index_tile_table_preserves_local_routed_rows():
    topk_indices = jnp.asarray(
        [[3, 1, 5], [0, 3, 2], [5, 2, 4], [1, 4, 0]],
        dtype=jnp.int32,
    )
    lhs_indices, group_sizes, _, topk_slots = _compute_rs_routing(
        topk_indices,
        num_experts=6,
        topk=3,
    )
    packed = lhs_indices * 3 + topk_slots
    group_sizes_np = np.asarray(group_sizes)
    group_starts = np.cumsum(group_sizes_np) - group_sizes_np

    first_local_expert = 2
    num_local_experts = 2
    tile_m = 4
    sublane = 2
    expected_rows = []
    for expert_id in range(
        first_local_expert,
        first_local_expert + num_local_experts,
    ):
        group_start = int(group_starts[expert_id])
        group_end = group_start + int(group_sizes_np[expert_id])
        tile_start = group_start
        while tile_start < group_end:
            row_count = min(tile_m, group_end - tile_start)
            row_ids = list(range(tile_start, tile_start + row_count))
            row_ids.extend([row_ids[-1]] * (tile_m - row_count))
            expected_rows.append(np.asarray(packed)[row_ids])
            first_capacity = tile_m - (group_start % sublane)
            tile_start = (
                group_start + first_capacity
                if tile_start == group_start
                else tile_start + tile_m
            )

    actual = _build_packed_index_tile_table(
        packed,
        group_sizes,
        jnp.asarray([first_local_expert], dtype=jnp.int32),
        num_local_groups=num_local_experts,
        tile_m=tile_m,
        size_lhs_sublane=sublane,
        max_num_gm=len(expected_rows),
    )
    np.testing.assert_array_equal(
        np.asarray(actual)[:, 0, :],
        np.stack(expected_rows),
    )


def test_quantize_on_load_rejects_blockwise_config(tmp_path):
    config_path = tmp_path / "bad-load-time-quant.yaml"
    config_path.write_text(
        """
quantization:
  quantize_on_load: true
  weight_block_size: [128, 128]
  linear:
    rules:
      - module_path: '.*'
        weight_dtype: 'float8_e4m3fn'
  moe:
    weight_dtype: 'float8_e4m3fn'
    activation_dtype: 'float8_e4m3fn'
""".strip()
    )

    with pytest.raises(ValueError, match="per-channel"):
        QuantizationConfig.from_yaml(str(config_path))
