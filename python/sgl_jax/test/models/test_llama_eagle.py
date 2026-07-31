import jax
import jax.numpy as jnp
import numpy as np
from jax._src.mesh import AxisType
from jax.sharding import Mesh
from transformers import LlamaConfig


def _mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _tiny_config():
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def test_classic_eagle_architecture_is_registered():
    from sgl_jax.srt.models.registry import ModelRegistry

    model_cls, arch = ModelRegistry.resolve_model_cls(["LlamaForCausalLMEagle"])
    assert arch == "LlamaForCausalLMEagle"
    assert model_cls.__name__ == "LlamaForCausalLMEagle"


def test_draft_architecture_selection_keeps_eagle_and_eagle3_distinct():
    from sgl_jax.srt.configs.model_config import _get_llama_draft_architecture

    assert _get_llama_draft_architecture("EAGLE") == "LlamaForCausalLMEagle"
    assert _get_llama_draft_architecture("eagle") == "LlamaForCausalLMEagle"
    assert _get_llama_draft_architecture("EAGLE3") == "LlamaForCausalLMEagle3"


def test_classic_eagle_uses_two_feature_projection_and_full_vocab():
    from sgl_jax.srt.models.llama_eagle import LlamaForCausalLMEagle

    config = _tiny_config()
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = LlamaForCausalLMEagle(config, mesh=mesh, dtype=jnp.bfloat16)

    assert model.model.fc.weight.shape == (2 * config.hidden_size, config.hidden_size)
    assert model.lm_head.embedding.shape == (config.vocab_size, config.hidden_size)
    assert model.hot_token_ids is None


def test_classic_eagle_weight_mapping_skips_shared_embed_head_and_first_input_norm():
    from sgl_jax.srt.models.llama_eagle import LlamaForCausalLMEagle

    config = _tiny_config()
    model = object.__new__(LlamaForCausalLMEagle)
    object.__setattr__(model, "config", config)

    mappings = model._create_eagle_weight_mappings()

    assert "fc.weight" in mappings
    assert "fc.bias" in mappings
    assert "layers.0.input_layernorm.weight" not in mappings
    assert "model.embed_tokens.weight" not in mappings
    assert "lm_head.weight" not in mappings
    assert mappings["layers.0.self_attn.q_proj.weight"].target_path == (
        "model.layers.0.self_attn.q_proj.weight"
    )
