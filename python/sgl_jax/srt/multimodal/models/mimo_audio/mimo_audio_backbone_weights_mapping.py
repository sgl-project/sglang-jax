from transformers import PretrainedConfig

from sgl_jax.srt.utils.weight_utils import WeightMapping


def _create_qwen2_layer_mappings(prefix: str, target_prefix: str) -> dict[str, WeightMapping]:
    mappings = {
        f"{prefix}.self_attn.q_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_proj.weight",
            transpose=True,
        ),
        f"{prefix}.self_attn.q_proj.bias": WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_proj.bias",
            sharding=(),
        ),
        f"{prefix}.self_attn.k_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.self_attn.k_proj.weight",
            transpose=True,
        ),
        f"{prefix}.self_attn.k_proj.bias": WeightMapping(
            target_path=f"{target_prefix}.self_attn.k_proj.bias",
            sharding=(),
        ),
        f"{prefix}.self_attn.v_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.self_attn.v_proj.weight",
            transpose=True,
        ),
        f"{prefix}.self_attn.v_proj.bias": WeightMapping(
            target_path=f"{target_prefix}.self_attn.v_proj.bias",
            sharding=(),
        ),
        f"{prefix}.self_attn.o_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.self_attn.o_proj.weight",
            transpose=True,
        ),
        f"{prefix}.mlp.gate_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.mlp.gate_proj.weight",
            transpose=True,
        ),
        f"{prefix}.mlp.up_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.mlp.up_proj.weight",
            transpose=True,
        ),
        f"{prefix}.mlp.down_proj.weight": WeightMapping(
            target_path=f"{target_prefix}.mlp.down_proj.weight",
            transpose=True,
        ),
        f"{prefix}.input_layernorm.weight": WeightMapping(
            target_path=f"{target_prefix}.input_layernorm.scale",
            sharding=(),
        ),
        f"{prefix}.post_attention_layernorm.weight": WeightMapping(
            target_path=f"{target_prefix}.post_attention_layernorm.scale",
            sharding=(),
        ),
    }
    return mappings


def to_mappings(config: PretrainedConfig) -> dict[str, WeightMapping]:
    mappings = {}
    mappings["model.embed_tokens.weight"] = WeightMapping(
        target_path="model.embed_tokens.embedding",
        sharding=("tensor", None),
        transpose=False,
    )
    mappings["model.norm.weight"] = WeightMapping(
        target_path="model.norm.scale",
        sharding=(),
    )
    mappings.update(_create_qwen2_layer_mappings("model.layers.*", "model.layers.*"))
    mappings["local_transformer.norm.weight"] = WeightMapping(
        target_path="patch_decoder.norm.scale",
        sharding=(),
    )
    mappings.update(_create_qwen2_layer_mappings("local_transformer.layers.*", "patch_decoder.layers.*"))
    mappings["input_local_transformer.norm.weight"] = WeightMapping(
        target_path="patch_encoder.norm.scale",
        sharding=(),
    )
    mappings.update(
        _create_qwen2_layer_mappings("input_local_transformer.layers.*", "patch_encoder.layers.*")
    )
    mappings["lm_head.weight"] = WeightMapping(
        target_path="lm_head.embedding",
        transpose=False,
    )
    mappings["hidden_states_downcast.weight"] = WeightMapping(
        target_path="hidden_states_downcast.weight",
        transpose=True,
    )
    mappings["speech_group_downcast.weight"] = WeightMapping(
        target_path="speech_group_downcast.weight",
        transpose=True,
    )
    audio_channels = getattr(config, "audio_channels", 8)
    for i in range(audio_channels):
        mappings[f"speech_embeddings.{i}.weight"] = WeightMapping(
            target_path=f"speech_embeddings.{i}.embedding",
            transpose=False,
        )
        mappings[f"local_transformer_lm_heads.{i}.weight"] = WeightMapping(
            target_path=f"patch_decoder_lm_heads.{i}.weight",
            transpose=True,
        )

    return mappings
