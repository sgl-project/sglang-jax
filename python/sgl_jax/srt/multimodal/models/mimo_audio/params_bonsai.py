import re
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.qwen2.params import Transform, TRANSFORM_LINEAR, TRANSFORM_NONE, _stoi, _assign_weights


def _get_qwen2_key_mapping(prefix: str) -> dict[str, tuple[str, Transform]]:
    q_bias_flatten = Transform(reshape=(-1,))
    kv_bias_flatten = Transform(reshape=(-1,))

    return {
        rf"{prefix}\.embed_tokens\.weight": (f"{prefix}.embedder.embedding", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            rf"{prefix}.layers.\1.attn.q_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            rf"{prefix}.layers.\1.attn.k_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            rf"{prefix}.layers.\1.attn.v_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            rf"{prefix}.layers.\1.attn.o_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            rf"{prefix}.layers.\1.attn.q_proj.bias",
            q_bias_flatten,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            rf"{prefix}.layers.\1.attn.k_proj.bias",
            kv_bias_flatten,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            rf"{prefix}.layers.\1.attn.v_proj.bias",
            kv_bias_flatten,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            rf"{prefix}.layers.\1.mlp.gate_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            rf"{prefix}.layers.\1.mlp.up_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            rf"{prefix}.layers.\1.mlp.down_proj.kernel",
            TRANSFORM_LINEAR,
        ),
        rf"{prefix}\.norm\.weight": (f"{prefix}.final_norm.scale", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.input_layernorm\.weight": (
            rf"{prefix}.layers.\1.input_layernorm.scale",
            TRANSFORM_NONE,
        ),
        rf"{prefix}\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            rf"{prefix}.layers.\1.post_attention_layernorm.scale",
            TRANSFORM_NONE,
        ),
    }


def _get_mimo_key_mapping(audio_channels: int) -> dict[str, tuple[str, Transform]]:
    """Generate key mapping for MiMo-specific layers."""
    mapping = {
        r"lm_head\.weight": ("lm_head.kernel", TRANSFORM_LINEAR),
        r"hidden_states_downcast\.weight": ("hidden_states_downcast.kernel", TRANSFORM_LINEAR),
        r"speech_group_downcast\.weight": ("speech_group_downcast.kernel", TRANSFORM_LINEAR),
    }

    for i in range(audio_channels):
        mapping[rf"speech_embeddings\.{i}\.weight"] = (f"speech_embeddings.{i}.embedding", TRANSFORM_NONE)
        mapping[rf"local_transformer_lm_heads\.{i}\.weight"] = (
            f"local_transformer_lm_heads.{i}.kernel",
            TRANSFORM_LINEAR,
        )

    return mapping


def _get_jax_key(mapping: dict[str, tuple[str, Transform]], source_key: str) -> tuple[str | None, Transform | None]:
    """Get JAX key from source key using regex mapping."""
    for pat, (jax_key, transform) in mapping.items():
        match = re.fullmatch(pat, source_key)
        if match:
            result_key = re.sub(pat, jax_key, source_key)
            return result_key, transform
    return None, None


def create_model_with_weights(
    model_path: str,
    config,
    args,
    rngs: nnx.Rngs | None = None,
    dtype: Any = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
    """Create MiMo Audio model and load weights from safetensors."""
    import os
    import json
    from safetensors import safe_open

    if rngs is None:
        rngs = nnx.Rngs(0)

    print("Creating MiMo Audio model with weights...")

    from bonsai.models.mimo_audio.modeling import FlaxMiMoAudioForCausalLM

    model = nnx.eval_shape(lambda: FlaxMiMoAudioForCausalLM(config, args, nnx.Rngs(0), dtype=dtype))
    graph_def, abs_state = nnx.split(model)
    pure_state_dict = abs_state.to_pure_dict()

    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            with safe_open(os.path.join(model_path, shard_file), framework="numpy") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        safetensors_path = os.path.join(model_path, "model.safetensors")
        with safe_open(safetensors_path, framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    full_mapping = {}

    # Main model mapping (model prefix)
    full_mapping.update(_get_qwen2_key_mapping("model"))

    # Local transformer mapping
    full_mapping.update(_get_qwen2_key_mapping("local_transformer"))

    # Input local transformer mapping
    full_mapping.update(_get_qwen2_key_mapping("input_local_transformer"))

    full_mapping.update(_get_mimo_key_mapping(config.audio_channels))

    conversion_errors = []
    loaded_count = 0

    for torch_key, tensor in state_dict.items():
        jax_key, transform = _get_jax_key(full_mapping, torch_key)
        if jax_key is None:
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            tensor_jax = jnp.asarray(tensor, dtype=dtype)
            _assign_weights(keys, tensor_jax, pure_state_dict, torch_key, transform, sharding)
            loaded_count += 1
        except KeyError as e:
            continue
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed: '{torch_key}' -> '{full_jax_key}': {e}")

    if conversion_errors:
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors:\n" + "\n".join(conversion_errors[:10])
        )

    model = nnx.merge(graph_def, pure_state_dict)

    print(f"MiMo Audio model created successfully! ({loaded_count} parameters loaded)")
    return model
