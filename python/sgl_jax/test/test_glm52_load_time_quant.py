from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.fused_moe import FusedEPMoEV2
from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader


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


def test_glm52_static_per_channel_mappings_load_linear_and_fused_moe_scales_directly():
    model = object.__new__(Glm5ForCausalLM)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            hidden_size=6144,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=1,
            moe_backend="fused_v2",
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
