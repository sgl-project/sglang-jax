"""CPU component tests for Ling 3 Tiny/Flash modeling.

These tests use reduced dimensions and synthetic checkpoint key sets. Full
checkpoint loading and numerical smoke tests run separately on TPU.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from sgl_jax.srt.configs.bailing_hybrid import BailingHybridConfig
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.models.bailing_moe_v3 import (
    BailingKDAAttention,
    BailingMLA,
    BailingMoeV3ForCausalLM,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        layer_group_size=2,
        short_conv_kernel_size=4,
        no_kda_lora=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        gated_attention_proj_granularity_type="head_wise",
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
    )
    values.update(overrides)
    return BailingHybridConfig(**values)


def test_config_exposes_ling3_layer_and_state_layout():
    config = _config(num_hidden_layers=8, layer_group_size=4)

    assert config.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert config.full_attention_layer_ids == [3, 7]
    assert config.is_kda_layer(0)
    assert not config.is_kda_layer(3)
    assert config.linear_state_params.conv_kernel_size == 4
    assert config.kda_lower_bound == -5.0


def test_architecture_is_registered_from_single_model_module():
    model_cls, architecture = get_model_architecture(
        SimpleNamespace(
            hf_config=_config(),
            model_impl="auto",
            model_path="",
            revision=None,
        )
    )

    assert architecture == "BailingMoeV3ForCausalLM"
    assert model_cls is BailingMoeV3ForCausalLM
    assert model_cls.__module__.endswith("bailing_moe_v3")


def test_model_builds_kda_mla_and_standard_epmoe():
    config = _config(use_absorbed_mla=False)
    config.ep_size = 1
    config.moe_backend = "epmoe"
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    with jax.set_mesh(mesh):
        model = BailingMoeV3ForCausalLM(config, mesh, dtype=jnp.float32)

    assert isinstance(model.model.layers[0].self_attn, BailingKDAAttention)
    assert isinstance(model.model.layers[1].self_attn, BailingMLA)
    assert model.model.layers[1].self_attn.use_absorbed is False
    assert isinstance(model.model.layers[1].experts, EPMoE)
    assert model.model.layers[1].topk.mesh is mesh


def test_weight_mappings_cover_tiny_q_lora_and_flash_flat_q():
    tiny = object.__new__(BailingMoeV3ForCausalLM)
    object.__setattr__(tiny, "config", _config(q_lora_rank=4, moe_backend="epmoe"))
    tiny_mappings = tiny._create_weight_mappings()
    assert "model.layers.1.attention.q_a_proj.weight" in tiny_mappings
    assert "model.layers.1.attention.q_proj.weight" not in tiny_mappings
    assert (
        tiny_mappings["model.layers.1.mlp.shared_experts.gate_proj.weight"].target_path
        == "model.layers.1.shared_experts.gate_proj.weight"
    )

    flash = object.__new__(BailingMoeV3ForCausalLM)
    object.__setattr__(
        flash,
        "config",
        _config(
            q_lora_rank=None,
            moe_backend="fused_v2",
            expert_swiglu_limit_list=[0, 4],
            share_expert_swiglu_limit_list=[0, 5],
        ),
    )
    flash_mappings = flash._create_weight_mappings()
    assert "model.layers.1.attention.q_proj.weight" in flash_mappings
    assert "model.layers.1.attention.q_a_proj.weight" not in flash_mappings
    assert (
        flash_mappings["model.layers.1.mlp.shared_experts.gate_proj.weight"].target_path
        == "model.layers.1.experts.w1_shared"
    )
