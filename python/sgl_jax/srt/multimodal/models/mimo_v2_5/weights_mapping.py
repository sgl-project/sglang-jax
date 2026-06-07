"""Weight mappings for the MiMo-V2.5 embed-stage model.

Split out of ``embedding.py`` (design doc §5). Each builder returns a
``{hf_key: WeightMapping}`` dict; :func:`build_embedding_weight_mappings`
assembles the full map for ``MiMoV2_5Embedding`` (text embedding + audio tower).
The vision tower mappings are intentionally omitted this round.
"""

from __future__ import annotations

from sgl_jax.srt.utils.weight_utils import WeightMapping


def build_text_embed_mapping() -> dict[str, WeightMapping]:
    return {
        "model.embed_tokens.weight": WeightMapping(
            target_path="text_embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        )
    }


def build_speech_embeddings_mapping(num_channels: int) -> dict[str, WeightMapping]:
    mappings = {}
    for idx in range(num_channels):
        mappings[f"speech_embeddings.{idx}.weight"] = WeightMapping(
            target_path=f"audio_encoder.speech_embeddings.{idx}.embedding",
            sharding=(None, None),
            transpose=False,
        )
    return mappings


def build_input_local_mapping(num_layers: int) -> dict[str, WeightMapping]:
    mappings = {
        "audio_encoder.input_local_transformer.norm.weight": WeightMapping(
            target_path="audio_encoder.input_local_transformer.norm.scale",
            sharding=(None,),
        )
    }
    for i in range(num_layers):
        src = f"audio_encoder.input_local_transformer.layers.{i}"
        dst = f"audio_encoder.input_local_transformer.layers.{i}"
        mappings.update(
            {
                f"{src}.self_attn.q_proj.weight": WeightMapping(
                    target_path=f"{dst}.self_attn.q_proj.weight", transpose=True
                ),
                f"{src}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{dst}.self_attn.q_proj.bias", sharding=(None,)
                ),
                f"{src}.self_attn.k_proj.weight": WeightMapping(
                    target_path=f"{dst}.self_attn.k_proj.weight", transpose=True
                ),
                f"{src}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{dst}.self_attn.k_proj.bias", sharding=(None,)
                ),
                f"{src}.self_attn.v_proj.weight": WeightMapping(
                    target_path=f"{dst}.self_attn.v_proj.weight", transpose=True
                ),
                f"{src}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{dst}.self_attn.v_proj.bias", sharding=(None,)
                ),
                f"{src}.self_attn.o_proj.weight": WeightMapping(
                    target_path=f"{dst}.self_attn.o_proj.weight", transpose=True
                ),
                f"{src}.input_layernorm.weight": WeightMapping(
                    target_path=f"{dst}.input_layernorm.scale", sharding=(None,)
                ),
                f"{src}.post_attention_layernorm.weight": WeightMapping(
                    target_path=f"{dst}.post_attention_layernorm.scale", sharding=(None,)
                ),
                f"{src}.mlp.gate_proj.weight": WeightMapping(
                    target_path=f"{dst}.mlp.gate_proj.weight", transpose=True
                ),
                f"{src}.mlp.up_proj.weight": WeightMapping(
                    target_path=f"{dst}.mlp.up_proj.weight", transpose=True
                ),
                f"{src}.mlp.down_proj.weight": WeightMapping(
                    target_path=f"{dst}.mlp.down_proj.weight", transpose=True
                ),
            }
        )
    return mappings


def build_projection_mapping() -> dict[str, WeightMapping]:
    return {
        "audio_encoder.projection.mlp.0.weight": WeightMapping(
            target_path="audio_encoder.proj_fc1.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        "audio_encoder.projection.mlp.2.weight": WeightMapping(
            target_path="audio_encoder.proj_fc2.weight",
            sharding=("tensor", None),
            transpose=True,
        ),
    }


def build_embedding_weight_mappings(
    *, num_audio_channels: int, num_input_local_layers: int
) -> dict[str, WeightMapping]:
    """Full HF->target weight map for MiMoV2_5Embedding (text + audio tower)."""
    mappings: dict[str, WeightMapping] = {}
    mappings.update(build_text_embed_mapping())
    mappings.update(build_speech_embeddings_mapping(num_audio_channels))
    mappings.update(build_input_local_mapping(num_input_local_layers))
    mappings.update(build_projection_mapping())
    return mappings
