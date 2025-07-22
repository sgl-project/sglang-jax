import functools
import glob
import logging
import math
import os
import re
from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def load_hf_weights(
    model_config: ModelConfig,
    model: nnx.Module,
    mappings: Dict[str, str],
    mesh: Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
):
    def shard_put(x: jax.Array, sharding_names, mesh: jax.sharding.Mesh) -> jax.Array:
        if math.prod(mesh.axis_sizes) == 1:
            return jax.device_put(x, mesh.devices.flatten()[0])
        return jax.device_put(x, NamedSharding(mesh, P(*sharding_names)))

    def get_param(params: nnx.State, path: str) -> nnx.State:
        keys = path.split(".")
        current_level = params
        for key in keys:
            if key.isdigit():
                current_level = current_level[int(key)]
            else:
                if hasattr(current_level, "__contains__") and key in current_level:
                    current_level = current_level[key]
                elif hasattr(current_level, key):
                    current_level = getattr(current_level, key)
                else:
                    raise ValueError(f"{path} is not a valid param path")
        return current_level

    def hf_model_weights_iterator(model_name_or_path: str, framework: str):
        weights_files = glob.glob(os.path.join(model_name_or_path, "*.safetensors"))
        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_name_or_path}.")
        weights_files.sort()

        for st_file in weights_files:
            logger.info(f"Loading weights from {st_file}")
            with jax.default_device(jax.local_devices(backend="cpu")[0]):
                with safe_open(st_file, framework=framework) as f:
                    for name in f.keys():
                        weight_tensor = f.get_tensor(name)
                        yield name, weight_tensor

    sharding_size = mesh.shape["tensor"]
    shard = functools.partial(shard_put, mesh=mesh)

    model_path = model_config.model_path

    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    hidden_size = model_config.hidden_size

    # Pad head_dim for kernel performance.
    head_dim_original = model_config.head_dim
    head_dim = (head_dim_original + 127) // 128 * 128  # Pad to nearest multiple of 128
    head_dim_pad = head_dim - head_dim_original

    # These reshape/transpose rules are for models with separate q_proj/k_proj/v_proj
    # QWen uses c_attn (combined QKV) so these rules don't apply
    reshape_keys = {
        "q_proj": (num_heads, head_dim_original, hidden_size),
        "k_proj": (num_kv_heads, head_dim_original, hidden_size),
        "v_proj": (num_kv_heads, head_dim_original, hidden_size),
        "o_proj": (hidden_size, num_heads, head_dim_original),
    }
    bias_reshape_keys = {
        "q_proj.bias": (num_heads, head_dim_original),
        "k_proj.bias": (num_kv_heads, head_dim_original),
        "v_proj.bias": (num_kv_heads, head_dim_original),
    }
    transpose_keys = {
        "lm_head": (1, 0),
        "gate_proj": (1, 0),
        "up_proj": (1, 0),
        "down_proj": (1, 0),
        "q_proj": (2, 0, 1),
        "k_proj": (2, 0, 1),
        "v_proj": (2, 0, 1),
        "o_proj": (1, 2, 0),
    }

    # QWen-specific transpose rules
    qwen_transpose_keys = {
        "c_attn": (
            1,
            0,
        ),
        "c_proj": (
            1,
            0,
        ),  # HF: (in_features, out_features) -> Our: (out_features, in_features)
        "w1": (1, 0),  # MLP weights
        "w2": (1, 0),  # MLP weights
        "lm_head": (1, 0),  # LM head
    }

    # key: (padding_dim, padding_size)
    pad_keys = {
        "q_proj": (1, sharding_size // num_heads),
        "k_proj": (1, sharding_size // num_kv_heads),
        "v_proj": (1, sharding_size // num_kv_heads),
        "o_proj": (0, sharding_size // num_heads),
    }
    bias_pad_keys = {
        "q_proj.bias": (0, sharding_size // num_heads),
        "k_proj.bias": (0, sharding_size // num_kv_heads),
        "v_proj.bias": (0, sharding_size // num_kv_heads),
    }

    params = nnx.state(model)
    for hf_key, hf_weight in hf_model_weights_iterator(model_path, framework="flax"):
        if hf_key.endswith(".weight"):
            hf_key = hf_key.removesuffix(".weight")

        # Find the corresponding model key using the HF key
        if "layer" in hf_key:
            layer_num = re.search(r"layers\.(\d+)", hf_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", hf_key)
            mapping_result = mappings[layer_key]
            if isinstance(mapping_result[0], list):
                model_keys = mapping_result[0]
                model_sharding = mapping_result[1]
                model_keys = [key.replace("layers.*", f"layers.{layer_num}") for key in model_keys]
            else:
                model_key = mapping_result[0]
                model_sharding = mapping_result[1]
                model_key = re.sub(r"layers\.\*", f"layers.{layer_num}", model_key)
        else:
            mapping_result = mappings[hf_key]
            if isinstance(mapping_result[0], list):
                model_keys = mapping_result[0]
                model_sharding = mapping_result[1]
            else:
                model_key = mapping_result[0]
                model_sharding = mapping_result[1]
        # print(
        #     "DEBUG: before transform | "
        #     f"{hf_key}: {hf_weight.shape}  -->  {model_key}: {model_weight.value.shape} {model_sharding}"
        # )

        # Check if this is a QWen model (uses c_attn, c_proj, w1, w2) vs other models (q_proj, k_proj, etc.)
        is_qwen_layer = any(key in hf_key for key in ["c_attn", "c_proj", "w1", "w2"])
        is_other_model_layer = any(
            key in hf_key
            for key in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )

        if is_qwen_layer:
            # Apply QWen-specific transformations
            for key in qwen_transpose_keys:
                if key in hf_key and not hf_key.endswith(".bias"):  # Don't transpose bias
                    hf_weight = jnp.transpose(hf_weight, qwen_transpose_keys[key])
                    break
            if "c_attn" in hf_key:
                if not hf_key.endswith(".bias"):
                    total_qkv_dim = hf_weight.shape[1]
                    q_dim = num_heads * head_dim_original
                    k_dim = num_kv_heads * head_dim_original
                    v_dim = num_kv_heads * head_dim_original

                    expected_total_dim = q_dim + k_dim + v_dim

                    if total_qkv_dim != expected_total_dim:
                        raise ValueError(
                            f"c_attn dimension mismatch: expected {expected_total_dim} "
                            f"(q:{q_dim} + k:{k_dim} + v:{v_dim}), got {total_qkv_dim}"
                        )

                    q_weight = hf_weight[:, :q_dim]
                    k_weight = hf_weight[:, q_dim : q_dim + k_dim]
                    v_weight = hf_weight[:, q_dim + k_dim : q_dim + k_dim + v_dim]
                    for i, (proj_name, proj_weight) in enumerate(
                        [
                            ("q_proj", q_weight),
                            ("k_proj", k_weight),
                            ("v_proj", v_weight),
                        ]
                    ):
                        proj_model_key = model_keys[i]

                        if head_dim_pad > 0:
                            if proj_name == "q_proj":
                                proj_weight = jnp.reshape(
                                    proj_weight, (hidden_size, num_heads, head_dim_original)
                                )
                                proj_weight = jnp.pad(
                                    proj_weight, ((0, 0), (0, 0), (0, head_dim_pad))
                                )
                                proj_weight = jnp.reshape(
                                    proj_weight, (hidden_size, num_heads * head_dim)
                                )
                            else:
                                proj_weight = jnp.reshape(
                                    proj_weight, (hidden_size, num_kv_heads, head_dim_original)
                                )
                                proj_weight = jnp.pad(
                                    proj_weight, ((0, 0), (0, 0), (0, head_dim_pad))
                                )
                                proj_weight = jnp.reshape(
                                    proj_weight, (hidden_size, num_kv_heads * head_dim)
                                )
                        proj_weight = proj_weight.astype(dtype)
                        proj_model_weight = get_param(params, proj_model_key)
                        proj_model_weight.value = shard(proj_weight, model_sharding)

                else:
                    total_qkv_dim = hf_weight.shape[0]
                    q_dim = num_heads * head_dim_original
                    k_dim = num_kv_heads * head_dim_original
                    v_dim = num_kv_heads * head_dim_original

                    expected_total_dim = q_dim + k_dim + v_dim
                    if total_qkv_dim != expected_total_dim:
                        raise ValueError(
                            f"c_attn bias dimension mismatch: expected {expected_total_dim} "
                            f"(q:{q_dim} + k:{k_dim} + v:{v_dim}), got {total_qkv_dim}"
                        )

                    q_bias = hf_weight[:q_dim]
                    k_bias = hf_weight[q_dim : q_dim + k_dim]
                    v_bias = hf_weight[q_dim + k_dim : q_dim + k_dim + v_dim]

                    for i, (proj_name, proj_bias) in enumerate(
                        [
                            ("q_proj", q_bias),
                            ("k_proj", k_bias),
                            ("v_proj", v_bias),
                        ]
                    ):
                        proj_bias_key = model_keys[i]
                        if head_dim_pad > 0:
                            if proj_name == "q_proj":
                                proj_bias = jnp.reshape(proj_bias, (num_heads, head_dim_original))
                                proj_bias = jnp.pad(proj_bias, ((0, 0), (0, head_dim_pad)))
                                proj_bias = jnp.reshape(proj_bias, (num_heads * head_dim,))
                            else:
                                proj_bias = jnp.reshape(
                                    proj_bias, (num_kv_heads, head_dim_original)
                                )
                                proj_bias = jnp.pad(proj_bias, ((0, 0), (0, head_dim_pad)))
                                proj_bias = jnp.reshape(proj_bias, (num_kv_heads * head_dim,))

                        proj_bias = proj_bias.astype(dtype)

                        proj_model_bias = get_param(params, proj_bias_key)
                        proj_model_bias.value = shard(proj_bias, model_sharding)

                continue

        elif is_other_model_layer:
            if hf_key.endswith(".bias"):
                for key in bias_reshape_keys:
                    if key in hf_key:
                        hf_weight = jnp.reshape(hf_weight, bias_reshape_keys[key])
                        if head_dim_pad > 0:
                            hf_weight = jnp.pad(hf_weight, ((0, 0), (0, head_dim_pad)))
                        break
            else:
                for key in reshape_keys:
                    if key in hf_key:
                        hf_weight = jnp.reshape(hf_weight, reshape_keys[key])
                        if head_dim_pad > 0:
                            if "o_proj" in key:
                                hf_weight = jnp.pad(hf_weight, ((0, 0), (0, 0), (0, head_dim_pad)))
                            else:
                                hf_weight = jnp.pad(hf_weight, ((0, 0), (0, head_dim_pad), (0, 0)))
                        break
                for key in transpose_keys:
                    if key in hf_key:
                        hf_weight = jnp.transpose(hf_weight, transpose_keys[key])
                        break

            # Pad num-kv-heads
            if hf_key.endswith(".bias"):
                for key, value in bias_pad_keys.items():
                    dim = value[0]
                    dim_size = value[1]
                    if key in hf_key and dim_size != 0:
                        hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                        break
            else:
                for key, value in pad_keys.items():
                    dim = value[0]
                    dim_size = value[1]
                    if key in hf_key and dim_size != 0:
                        hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                        break

        if "model_key" in locals():
            model_weight = get_param(params, model_key)
            if head_dim_pad == 0:
                if model_weight.value.shape != hf_weight.shape:
                    print(
                        f"ERROR: Shape mismatch for {hf_key} -> {model_key}: "
                        f"model expects {model_weight.value.shape}, "
                        f"got {hf_weight.shape}, "
                        f"sharding: {model_sharding}"
                    )
                assert model_weight.value.shape == hf_weight.shape

            # Convert to dtype
            hf_weight = hf_weight.astype(dtype)

            # Update the model weight
            model_weight.value = shard(hf_weight, model_sharding)

    nnx.update(model, params)
