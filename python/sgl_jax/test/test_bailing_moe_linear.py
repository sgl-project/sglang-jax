import jax
import jax.numpy as jnp

from sgl_jax.srt.configs.bailing_hybrid import (
    BailingHybridConfig,
    get_bailing_hybrid_config,
)
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.models.bailing_moe_linear import (
    BailingMoEGQAAttention,
    BailingMoELinearAttention,
    BailingMoELinearDecoderLayer,
    BailingMoeV2_5ForCausalLM,
)
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_group_size=2,
        num_experts=1,
        first_k_dense_replace=0,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        kv_lora_rank=4,
        q_lora_rank=None,
    )
    defaults.update(overrides)
    return BailingHybridConfig(**defaults)


def test_layer_group_size_builds_upstream_linear_and_full_ids():
    cfg = _tiny_config(num_hidden_layers=8, layer_group_size=2)

    assert cfg.linear_layer_ids == [0, 2, 4, 6]
    assert cfg.full_attention_layer_ids == [1, 3, 5, 7]
    assert cfg.layers_block_type == [
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
    ]

    cfg = _tiny_config(num_hidden_layers=16, layer_group_size=8)
    assert cfg.linear_layer_ids == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
    assert cfg.full_attention_layer_ids == [7, 15]


def test_bailing_hybrid_config_exposes_runner_linear_state_params():
    cfg = _tiny_config(num_hidden_layers=8, layer_group_size=2)

    assert get_bailing_hybrid_config(cfg) is cfg
    assert cfg.linear_attn_config == {
        "kda_layers": [0, 2, 4, 6],
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 1,
    }

    state_params = cfg.linear_state_params
    assert state_params.layers == [0, 2, 4, 6]
    assert state_params.num_heads == 2
    assert state_params.head_dim == 8
    assert state_params.conv_kernel_size == 1


def test_bailing_moe_v2_5_resolves_from_bailing_moe_linear_file():
    model_cls, arch = get_model_architecture(
        type(
            "DummyModelConfig",
            (),
            {
                "hf_config": _tiny_config(architectures=["BailingMoeV2_5ForCausalLM"]),
                "model_impl": "auto",
                "model_path": "",
                "revision": None,
            },
        )()
    )

    assert arch == "BailingMoeV2_5ForCausalLM"
    assert model_cls is BailingMoeV2_5ForCausalLM
    assert model_cls.__module__.endswith("bailing_moe_linear")


def test_decoder_layer_selects_linear_attention_and_full_mla():
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    cfg = _tiny_config(layer_group_size=2, full_attention_type="mla")

    with jax.set_mesh(mesh):
        linear_layer = BailingMoELinearDecoderLayer(cfg, mesh=mesh, layer_id=0, dtype=jnp.float32)
        full_layer = BailingMoELinearDecoderLayer(cfg, mesh=mesh, layer_id=1, dtype=jnp.float32)

    assert isinstance(linear_layer.self_attn, BailingMoELinearAttention)
    assert isinstance(full_layer.self_attn, DeepseekV3Attention)


def test_decoder_layer_selects_gqa_fallback_for_non_mla_full_attention():
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    cfg = _tiny_config(layer_group_size=2, full_attention_type="gqa")

    with jax.set_mesh(mesh):
        full_layer = BailingMoELinearDecoderLayer(cfg, mesh=mesh, layer_id=1, dtype=jnp.float32)

    assert isinstance(full_layer.self_attn, BailingMoEGQAAttention)


def test_weight_mapping_contains_linear_mla_dense_and_shared_mlp_keys():
    cfg = _tiny_config(num_hidden_layers=2, layer_group_size=2, full_attention_type="mla")
    model = object.__new__(BailingMoeV2_5ForCausalLM)
    object.__setattr__(model, "config", cfg)
    object.__setattr__(
        model,
        "model",
        type("DummyInnerModel", (), {"decoder_attention_types": [0, 1]})(),
    )

    mappings = model._create_bailing_moe_linear_weight_mappings(
        type("DummyModelConfig", (), {"quantization_config": None})()
    )

    assert mappings["model.layers.0.attention.query_key_value.weight"].target_path == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    assert (
        mappings["model.layers.0.attention.g_norm.weight"].target_path
        == "model.layers.0.self_attn.g_norm.weight"
    )
    assert (
        mappings["model.layers.1.attention.kv_a_proj_with_mqa.weight"].target_path
        == "model.layers.1.self_attn.kv_a_proj.weight"
    )
    assert (
        mappings["model.layers.1.attention.dense.weight"].target_path
        == "model.layers.1.self_attn.o_proj.weight"
    )
    assert (
        mappings["model.layers.0.mlp.gate_proj.weight"].target_path
        == "model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        mappings["model.layers.0.mlp.down_proj.weight"].target_path
        == "model.layers.0.mlp.down_proj.weight"
    )


def test_weight_mapping_static_quant_gla_splits_into_weight_q_and_scale():
    cfg = _tiny_config(num_hidden_layers=2, layer_group_size=2, full_attention_type="mla")
    model = object.__new__(BailingMoeV2_5ForCausalLM)
    object.__setattr__(model, "config", cfg)
    object.__setattr__(
        model,
        "model",
        type("DummyInnerModel", (), {"decoder_attention_types": [0, 1]})(),
    )

    class _FakeQuantCfg:
        is_static_checkpoint = True

    mappings = model._create_bailing_moe_linear_weight_mappings(
        type("DummyModelConfig", (), {"quantization_config": _FakeQuantCfg()})()
    )

    qkv_w = mappings["model.layers.0.attention.query_key_value.weight"]
    assert qkv_w.target_path == [
        "model.layers.0.self_attn.q_proj.weight_q",
        "model.layers.0.self_attn.k_proj.weight_q",
        "model.layers.0.self_attn.v_proj.weight_q",
    ]
    assert qkv_w.sharding == ("tensor", None)
    assert qkv_w.transpose is False

    qkv_s = mappings["model.layers.0.attention.query_key_value.weight_scale"]
    assert qkv_s.target_path == [
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.k_proj.weight_scale",
        "model.layers.0.self_attn.v_proj.weight_scale",
    ]
    assert qkv_s.sharding == ("tensor", None)
    assert qkv_s.transpose is False


def test_weight_mapping_contains_moe_shared_expert_and_gqa_fallback_keys():
    cfg = _tiny_config(
        num_hidden_layers=2,
        layer_group_size=2,
        full_attention_type="gqa",
        num_experts=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        num_experts_per_tok=1,
        n_group=0,
        topk_group=0,
    )
    model = object.__new__(BailingMoeV2_5ForCausalLM)
    object.__setattr__(model, "config", cfg)
    object.__setattr__(
        model,
        "model",
        type("DummyInnerModel", (), {"decoder_attention_types": [0, 1]})(),
    )

    mappings = model._create_bailing_moe_linear_weight_mappings(
        type("DummyModelConfig", (), {"quantization_config": None})()
    )

    assert (
        mappings["model.layers.1.attention.query_key_value.weight"].target_path
        == "model.layers.1.self_attn.qkv_proj.weight"
    )
    assert (
        mappings["model.layers.1.attention.dense.weight"].target_path
        == "model.layers.1.self_attn.dense.weight"
    )
    assert (
        mappings["model.layers.1.mlp.gate.weight"].target_path == "model.layers.1.moe_gate.kernel"
    )
    assert (
        mappings["model.layers.1.mlp.shared_experts.gate_proj.weight"].target_path
        == "model.layers.1.shared_experts.gate_proj.weight"
    )
    assert "__MOE_EXPERTS__model.layers.1.mlp.wi_0" in mappings
