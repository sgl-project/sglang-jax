from types import SimpleNamespace

import jax
import jax.numpy as jnp

from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.models.gemma3n import (
    Gemma3nForCausalLM,
    Gemma3nForConditionalGeneration,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _tiny_config(**overrides):
    num_hidden_layers = overrides.get("num_hidden_layers", 4)
    defaults = dict(
        architectures=["Gemma3nForCausalLM"],
        vocab_size=32,
        vocab_size_per_layer_input=32,
        pad_token_id=0,
        hidden_size=16,
        hidden_size_per_layer_input=4,
        intermediate_size=[32] * num_hidden_layers,
        activation_sparsity_pattern=[0.0] * num_hidden_layers,
        hidden_activation="gelu_pytorch_tanh",
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention", "sliding_attention"] * ((num_hidden_layers + 1) // 2),
        sliding_window=16,
        rope_parameters={
            "full_attention": {"rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
        num_kv_shared_layers=0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        laurel_rank=4,
        altup_num_inputs=2,
        altup_active_idx=0,
        altup_coef_clip=None,
        altup_correct_scale=True,
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    defaults["layer_types"] = defaults["layer_types"][: defaults["num_hidden_layers"]]
    return SimpleNamespace(**defaults)


def _mesh():
    return create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


def test_gemma3n_causal_lm_resolves_from_model_registry():
    model_cls, arch = get_model_architecture(
        type(
            "DummyModelConfig",
            (),
            {
                "hf_config": _tiny_config(architectures=["Gemma3nForCausalLM"]),
                "model_impl": "auto",
                "model_path": "",
                "revision": None,
            },
        )()
    )

    assert arch == "Gemma3nForCausalLM"
    assert model_cls is Gemma3nForCausalLM


def test_gemma3n_conditional_generation_resolves_from_model_registry():
    model_cls, arch = get_model_architecture(
        type(
            "DummyModelConfig",
            (),
            {
                "hf_config": _tiny_config(architectures=["Gemma3nForConditionalGeneration"]),
                "model_impl": "auto",
                "model_path": "",
                "revision": None,
            },
        )()
    )

    assert arch == "Gemma3nForConditionalGeneration"
    assert model_cls is Gemma3nForConditionalGeneration


def test_gemma3n_tiny_model_constructs_text_modules():
    mesh = _mesh()
    cfg = _tiny_config()

    with jax.set_mesh(mesh):
        model = Gemma3nForCausalLM(cfg, mesh=mesh, dtype=jnp.float32)

    assert len(model.model.layers) == cfg.num_hidden_layers
    assert len(model.model.altup_projections) == cfg.altup_num_inputs - 1
    assert model.model.layers[0].self_attn.is_kv_shared_layer is False
    assert not hasattr(model, "lm_head")


def test_gemma3n_conditional_generation_uses_text_config():
    mesh = _mesh()
    text_cfg = _tiny_config()
    cfg = SimpleNamespace(text_config=text_cfg, architectures=["Gemma3nForConditionalGeneration"])

    with jax.set_mesh(mesh):
        model = Gemma3nForConditionalGeneration(cfg, mesh=mesh, dtype=jnp.float32)

    assert model.config is text_cfg
    assert len(model.model.layers) == text_cfg.num_hidden_layers


def test_gemma3n_shared_kv_layers_reference_last_non_shared_same_type():
    mesh = _mesh()
    cfg = _tiny_config(
        num_hidden_layers=4,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        num_kv_shared_layers=2,
    )

    with jax.set_mesh(mesh):
        model = Gemma3nForCausalLM(cfg, mesh=mesh, dtype=jnp.float32)

    layer_2_attn = model.model.layers[2].self_attn
    layer_3_attn = model.model.layers[3].self_attn

    assert layer_2_attn.is_kv_shared_layer
    assert layer_2_attn.kv_shared_layer_index == 0
    assert layer_2_attn.attn.layer_id == 0
    assert layer_2_attn.k_proj is None
    assert layer_2_attn.v_proj is None

    assert layer_3_attn.is_kv_shared_layer
    assert layer_3_attn.kv_shared_layer_index == 1
    assert layer_3_attn.attn.layer_id == 1


def test_gemma3n_weight_mappings_cover_core_and_conditional_aliases():
    mesh = _mesh()
    cfg = _tiny_config(
        num_hidden_layers=4,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        num_kv_shared_layers=2,
        tie_word_embeddings=False,
    )

    with jax.set_mesh(mesh):
        model = Gemma3nForCausalLM(cfg, mesh=mesh, dtype=jnp.float32)

    mappings = model._create_weight_mappings()

    assert (
        mappings["model.embed_tokens_per_layer.weight"].target_path
        == "model.embed_tokens_per_layer.embedding"
    )
    assert (
        mappings["model.per_layer_model_projection.weight"].target_path
        == "model.per_layer_model_projection.weight"
    )
    assert (
        mappings["model.layers.0.altup.prediction_coefs.weight"].target_path
        == "model.layers.0.altup.prediction_coefs.weight"
    )
    assert (
        mappings["model.layers.0.laurel.linear_left.weight"].target_path
        == "model.layers.0.laurel.linear_left.weight"
    )
    assert "model.layers.0.self_attn.k_norm.weight" in mappings
    assert "model.layers.0.self_attn.v_norm.weight" not in mappings

    assert "model.layers.2.self_attn.q_proj.weight" in mappings
    assert "model.layers.2.self_attn.k_proj.weight" not in mappings
    assert "model.layers.2.self_attn.v_proj.weight" not in mappings
    assert "model.layers.2.self_attn.k_norm.weight" not in mappings

    assert (
        mappings["model.language_model.layers.0.self_attn.q_proj.weight"].target_path
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        mappings["language_model.model.layers.0.self_attn.q_proj.weight"].target_path
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert mappings["lm_head.weight"].target_path == "lm_head.embedding"
