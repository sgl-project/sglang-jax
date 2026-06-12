"""Weight mappings for the MiMo-V2.5 embed-stage model.

Split out of ``embedding.py`` (design doc §5). Each builder returns a
``{hf_key: WeightMapping}`` dict; :func:`build_embedding_weight_mappings`
assembles the text-embedding + audio-tower map for ``MiMoV2_5Embedding``, and
:func:`create_mimo_vision_weight_mappings` builds the vision-tower map (merged in
by ``MiMoV2_5Embedding.load_weights`` when the ViT is present).
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


CONV3D_TORCH_TO_JAX = (2, 3, 4, 1, 0)


def create_mimo_vision_weight_mappings(
    config,
    source_prefix: str = "visual",
    target_prefix: str = "",
) -> dict[str, WeightMapping]:
    from sgl_jax.srt.utils.weight_utils import WeightMapping

    mappings: dict[str, WeightMapping] = {
        f"{source_prefix}.patch_embed.proj.weight": WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.kernel",
            transpose_axes=CONV3D_TORCH_TO_JAX,
        ),
        f"{source_prefix}.merger.ln_q.weight": WeightMapping(
            target_path=f"{target_prefix}merger.ln_q.scale",
            sharding=(),
        ),
        # merger ln_q / mlp are bias-free in the checkpoint (no .bias keys); the merger
        # module is built with use_bias=False to match.
        f"{source_prefix}.merger.mlp.0.weight": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc1.kernel",
            transpose=True,
        ),
        f"{source_prefix}.merger.mlp.2.weight": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc2.kernel",
            transpose=True,
        ),
    }

    for block_idx in range(int(config.depth)):
        source = f"{source_prefix}.blocks.{block_idx}"
        target = f"{target_prefix}blocks.{block_idx}"
        mappings.update(
            {
                f"{source}.norm1.weight": WeightMapping(
                    target_path=f"{target}.norm1.scale",
                    sharding=(),
                ),
                f"{source}.norm2.weight": WeightMapping(
                    target_path=f"{target}.norm2.scale",
                    sharding=(),
                ),
                f"{source}.attn.qkv.weight": WeightMapping(
                    target_path=f"{target}.attn.qkv.kernel",
                    transpose=True,
                ),
                f"{source}.attn.qkv.bias": WeightMapping(
                    target_path=f"{target}.attn.qkv.bias",
                    sharding=(),
                ),
                f"{source}.attn.proj.weight": WeightMapping(
                    target_path=f"{target}.attn.proj.kernel",
                    transpose=True,
                ),
                f"{source}.attn.proj.bias": WeightMapping(
                    target_path=f"{target}.attn.proj.bias",
                    sharding=(),
                ),
                f"{source}.attn.sinks": WeightMapping(
                    target_path=f"{target}.attn.sinks",
                    sharding=(),
                ),
                f"{source}.mlp.gate_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.gate_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.gate_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.gate_proj.bias",
                    sharding=(),
                ),
                f"{source}.mlp.up_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.up_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.up_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.up_proj.bias",
                    sharding=(),
                ),
                f"{source}.mlp.down_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.down_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.down_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.down_proj.bias",
                    sharding=(),
                ),
            }
        )
    return mappings
