import functools
import glob
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WeightMapping:
    target_path: Union[str, List[str]]
    sharding: Optional[Tuple] = None
    transpose: bool = False
    reshape: Optional[Tuple] = None
    head_dim_padding: bool = False
    kv_head_padding: bool = False

    def __post_init__(self):
        if self.sharding is None:
            self.sharding = self._infer_default_sharding()

    def _infer_default_sharding(self) -> Tuple:
        if isinstance(self.target_path, list):
            path = self.target_path[0]
        else:
            path = self.target_path

        if any(pattern in path for pattern in ["embedding", "lm_head"]):
            return (None, None)
        elif any(
            pattern in path
            for pattern in [
                "q_proj",
                "k_proj",
                "v_proj",
                "w1",
                "w2",
                "gate_proj",
                "up_proj",
            ]
        ):
            return (None, "tensor")
        elif any(pattern in path for pattern in ["c_proj", "o_proj", "down_proj"]):
            return ("tensor", None)
        elif "bias" in path or "weight" in path:
            return (None,)
        else:
            return (None,)


class WeightLoader:
    def __init__(
        self,
        model: nnx.Module,
        model_config: ModelConfig,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.model = model
        self.model_config = model_config
        self.mesh = mesh
        self.dtype = dtype

        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = (
            model_config.get_total_num_kv_heads()
        )  # Use original count for replication logic
        self.hidden_size = model_config.hidden_size
        self.head_dim_original = getattr(
            model_config, "head_dim", self.hidden_size // self.num_heads
        )

        self.head_dim = (self.head_dim_original + 127) // 128 * 128
        self.head_dim_pad = self.head_dim - self.head_dim_original

        if hasattr(self.mesh, "shape") and "tensor" in self.mesh.shape:
            self.sharding_size = self.mesh.shape["tensor"]
        else:
            self.sharding_size = 1

    def load_weights_from_safetensors(
        self, weight_mappings: Dict[str, Union[str, List[str], WeightMapping]]
    ):
        params = nnx.state(self.model)

        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        expert_weights = {}

        logger.info(
            f"WeightLoader: Will load layers 0 to {self.model_config.num_hidden_layers - 1}"
        )

        for hf_key, hf_weight in self._iterate_weights():
            if hf_key in regular_mappings:
                mapping = regular_mappings[hf_key]
                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)

                self._process_and_assign_weight(params, hf_key, hf_weight, mapping)
            elif "mlp.experts." in hf_key and hf_key.endswith(".weight"):
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug(f"Skipping excluded MoE expert weight: {hf_key}")
                else:
                    expert_weights[hf_key] = hf_weight.astype(self.dtype)
            else:
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug(f"Skipping excluded layer weight: {hf_key}")
                else:
                    logger.warning(f"No mapping found for weight: {hf_key}")
            nnx.update(self.model, params)

        if moe_mappings:
            self._process_moe_expert_weights(params, moe_mappings, expert_weights)
            nnx.update(self.model, params)

    def _iterate_weights(self):
        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        skipped_files = 0
        with tqdm(weights_files, desc="[LOADING] MODEL WEIGHTS", unit="file") as pbar:
            for st_file in pbar:
                filename = os.path.basename(st_file)
                pbar.set_postfix({"file": filename})

                with jax.default_device(jax.local_devices(backend="cpu")[0]):
                    with safe_open(st_file, framework="flax") as f:
                        needed_keys = []
                        for name in f.keys():
                            if not name.startswith("model.layers."):
                                needed_keys.append(name)
                                continue

                            if not self._is_excluded_layer_weight(name):
                                needed_keys.append(name)

                        if not needed_keys:
                            skipped_files += 1
                            logger.debug(
                                f"Skipping {filename}: 0/{len(f.keys())} weights needed"
                            )
                            continue

                        logger.debug(
                            f"Loading {filename}: {len(needed_keys)}/{len(f.keys())} weights needed"
                        )
                        for name in needed_keys:
                            weight_tensor = f.get_tensor(name)
                            yield name, weight_tensor

        if skipped_files > 0:
            logger.info(
                f"Memory optimization: Skipped {skipped_files}/{len(weights_files)} files with no needed weights"
            )

    def _process_and_assign_weight(
        self,
        params: nnx.State,
        hf_key: str,
        hf_weight: jax.Array,
        mapping: WeightMapping,
    ):
        processed_weight = hf_weight.astype(self.dtype)

        if mapping.transpose and not hf_key.endswith(".bias"):
            processed_weight = jnp.transpose(processed_weight, (1, 0))

        if isinstance(mapping.target_path, list):
            self._handle_split_weight(params, hf_key, processed_weight, mapping)
        else:
            self._handle_single_weight(params, hf_key, processed_weight, mapping)

    def _handle_single_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_path = mapping.target_path
        processed_weight = weight

        if mapping.reshape is not None:
            processed_weight = jnp.reshape(processed_weight, mapping.reshape)

        if mapping.head_dim_padding and self.head_dim_pad > 0:
            processed_weight = self._apply_head_dim_padding(
                processed_weight, hf_key, mapping
            )

        if mapping.kv_head_padding:
            processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

        sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

        try:
            model_param = self._get_param(params, jax_path)
            logger.debug(
                f"Loading {hf_key} -> {jax_path}, shape: {processed_weight.shape}, transpose: {mapping.transpose}"
            )
            model_param.value = sharded_weight
        except Exception as e:
            logger.error(f"Failed to load {hf_key} -> {jax_path}: {str(e)}")
            raise

    def _handle_split_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        if "c_attn" in hf_key:
            self._split_qkv_weight(params, hf_key, weight, mapping)
        else:
            raise ValueError(f"Unknown split weight pattern for {hf_key}")

    def _split_qkv_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_paths = mapping.target_path

        if hf_key.endswith(".bias"):
            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original

            q_bias = weight[:q_dim]
            k_bias = weight[q_dim : q_dim + kv_dim]
            v_bias = weight[q_dim + kv_dim : q_dim + 2 * kv_dim]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                q_bias = jnp.reshape(q_bias, (self.num_heads, self.head_dim_original))
                q_bias = jnp.pad(q_bias, ((0, 0), (0, self.head_dim_pad)))
                q_bias = jnp.reshape(q_bias, (self.num_heads * self.head_dim,))

                k_bias = jnp.reshape(
                    k_bias, (self.num_kv_heads, self.head_dim_original)
                )
                k_bias = jnp.pad(k_bias, ((0, 0), (0, self.head_dim_pad)))
                k_bias = jnp.reshape(k_bias, (self.num_kv_heads * self.head_dim,))

                v_bias = jnp.reshape(
                    v_bias, (self.num_kv_heads, self.head_dim_original)
                )
                v_bias = jnp.pad(v_bias, ((0, 0), (0, self.head_dim_pad)))
                v_bias = jnp.reshape(v_bias, (self.num_kv_heads * self.head_dim,))

            splits = [q_bias, k_bias, v_bias]
        else:

            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original

            if mapping.transpose:
                q_weight = weight[:, :q_dim]
                k_weight = weight[:, q_dim : q_dim + kv_dim]
                v_weight = weight[:, q_dim + kv_dim : q_dim + 2 * kv_dim]
            else:
                q_weight = weight[:q_dim, :]
                k_weight = weight[q_dim : q_dim + kv_dim, :]
                v_weight = weight[q_dim + kv_dim : q_dim + 2 * kv_dim, :]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                if mapping.transpose:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.hidden_size, self.num_heads, self.head_dim_original),
                    )
                    q_weight = jnp.pad(
                        q_weight, ((0, 0), (0, 0), (0, self.head_dim_pad))
                    )
                    q_weight = jnp.reshape(
                        q_weight, (self.hidden_size, self.num_heads * self.head_dim)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    k_weight = jnp.pad(
                        k_weight, ((0, 0), (0, 0), (0, self.head_dim_pad))
                    )
                    k_weight = jnp.reshape(
                        k_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    v_weight = jnp.pad(
                        v_weight, ((0, 0), (0, 0), (0, self.head_dim_pad))
                    )
                    v_weight = jnp.reshape(
                        v_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )
                else:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.num_heads, self.head_dim_original, self.hidden_size),
                    )
                    q_weight = jnp.pad(
                        q_weight, ((0, 0), (0, self.head_dim_pad), (0, 0))
                    )
                    q_weight = jnp.reshape(
                        q_weight, (self.num_heads * self.head_dim, self.hidden_size)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    k_weight = jnp.pad(
                        k_weight, ((0, 0), (0, self.head_dim_pad), (0, 0))
                    )
                    k_weight = jnp.reshape(
                        k_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    v_weight = jnp.pad(
                        v_weight, ((0, 0), (0, self.head_dim_pad), (0, 0))
                    )
                    v_weight = jnp.reshape(
                        v_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

            splits = [q_weight, k_weight, v_weight]

        for split_weight, jax_path in zip(splits, jax_paths):
            processed_weight = split_weight

            if mapping.kv_head_padding and (
                "k_proj" in jax_path or "v_proj" in jax_path
            ):
                processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

            sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

            model_param = self._get_param(params, jax_path)
            model_param.value = sharded_weight
            logger.debug(
                f"Split {hf_key} -> {jax_path}, shape: {processed_weight.shape}"
            )

    def _shard_weight(self, weight: jax.Array, sharding: tuple) -> jax.Array:
        if math.prod(self.mesh.axis_sizes) == 1:
            return jax.device_put(weight, self.mesh.devices.flatten()[0])
        return jax.device_put(weight, NamedSharding(self.mesh, P(*sharding)))

    def _get_param(self, params: nnx.State, path: str) -> nnx.State:
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

    def _apply_head_dim_padding(
        self, weight: jax.Array, hf_key: str, mapping: WeightMapping
    ) -> jax.Array:
        if hf_key.endswith(".bias"):
            if any(proj in hf_key for proj in ["q_proj", "k_proj", "v_proj"]):
                if "q_proj" in hf_key:
                    reshaped = jnp.reshape(
                        weight, (self.num_heads, self.head_dim_original)
                    )
                    padded = jnp.pad(reshaped, ((0, 0), (0, self.head_dim_pad)))
                    return jnp.reshape(padded, (self.num_heads * self.head_dim,))
                else:  # k_proj or v_proj
                    reshaped = jnp.reshape(
                        weight, (self.num_kv_heads, self.head_dim_original)
                    )
                    padded = jnp.pad(reshaped, ((0, 0), (0, self.head_dim_pad)))
                    return jnp.reshape(padded, (self.num_kv_heads * self.head_dim,))
        else:
            if mapping.reshape is not None:
                if "o_proj" in hf_key:
                    padded = jnp.pad(weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                else:
                    padded = jnp.pad(weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                return padded
            else:
                if mapping.transpose:
                    if "q_proj" in hf_key:
                        reshaped = jnp.reshape(
                            weight,
                            (self.hidden_size, self.num_heads, self.head_dim_original),
                        )
                        padded = jnp.pad(
                            reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad))
                        )
                        return jnp.reshape(
                            padded, (self.hidden_size, self.num_heads * self.head_dim)
                        )
                    elif any(proj in hf_key for proj in ["k_proj", "v_proj"]):
                        reshaped = jnp.reshape(
                            weight,
                            (
                                self.hidden_size,
                                self.num_kv_heads,
                                self.head_dim_original,
                            ),
                        )
                        padded = jnp.pad(
                            reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad))
                        )
                        return jnp.reshape(
                            padded,
                            (self.hidden_size, self.num_kv_heads * self.head_dim),
                        )
                    elif "o_proj" in hf_key:
                        reshaped = jnp.reshape(
                            weight,
                            (self.num_heads * self.head_dim_original, self.hidden_size),
                        )
                        padded_reshaped = jnp.reshape(
                            reshaped,
                            (self.num_heads, self.head_dim_original, self.hidden_size),
                        )
                        padded = jnp.pad(
                            padded_reshaped, ((0, 0), (0, self.head_dim_pad), (0, 0))
                        )
                        return jnp.reshape(
                            padded, (self.num_heads * self.head_dim, self.hidden_size)
                        )

        return weight

    def _apply_kv_head_padding(self, weight: jax.Array, hf_key: str) -> jax.Array:
        """Apply KV head padding/replication when tp_size > total_kv_heads."""
        # Only apply when dealing with KV projections and replication is needed
        if any(
            proj in hf_key for proj in ["k_proj", "v_proj"]
        ) and self.model_config.needs_kv_head_replication(self.sharding_size):
            total_kv_heads = self.model_config.get_total_num_kv_heads()
            num_replicas = self.model_config.get_num_kv_head_replicas(
                self.sharding_size
            )
            padding_strategy = self.model_config.get_kv_padding_strategy()

            if padding_strategy == "replicate":
                # GQA models: replicate existing KV heads to maintain attention semantics
                if hf_key.endswith(".bias"):
                    # For bias: replicate each original head
                    replicated_bias_parts = []

                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_bias = weight[start_idx:end_idx]

                        # Replicate this head for all its assigned devices
                        for _ in range(num_replicas):
                            replicated_bias_parts.append(original_head_bias)

                    # Concatenate all replicated parts
                    weight = jnp.concatenate(replicated_bias_parts, axis=0)
                else:
                    # For weight matrix: replicate each original head
                    replicated_weight_parts = []

                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_weight = weight[:, start_idx:end_idx]

                        # Replicate this head for all its assigned devices
                        for _ in range(num_replicas):
                            replicated_weight_parts.append(original_head_weight)

                    # Concatenate all replicated parts along head dimension
                    weight = jnp.concatenate(replicated_weight_parts, axis=1)

            elif padding_strategy == "zero":
                # MHA models: zero padding since all heads are equivalent
                target_heads = total_kv_heads * num_replicas
                target_size = target_heads * self.head_dim

                if hf_key.endswith(".bias"):
                    current_size = weight.shape[0]
                    padding_size = target_size - current_size

                    if padding_size > 0:
                        padding = jnp.zeros((padding_size,), dtype=weight.dtype)
                        weight = jnp.concatenate([weight, padding], axis=0)
                else:
                    current_size = weight.shape[1]
                    padding_size = target_size - current_size

                    if padding_size > 0:
                        padding = jnp.zeros(
                            (weight.shape[0], padding_size), dtype=weight.dtype
                        )
                        weight = jnp.concatenate([weight, padding], axis=1)

        return weight

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])

        is_excluded = layer_num >= self.model_config.num_hidden_layers

        if is_excluded and not hasattr(self, "_debug_count"):
            logger.info(
                f"DEBUG: Excluding layer {layer_num} >= {self.model_config.num_hidden_layers}"
            )
            self._debug_count = True

        return is_excluded

    def _process_moe_expert_weights(
        self,
        params: nnx.State,
        moe_mappings: Dict[str, WeightMapping],
        expert_weights: Dict[str, jax.Array],
    ):
        with tqdm(
            moe_mappings.items(), desc="[STACKING] MOE EXPERTS", unit="layer"
        ) as pbar:
            for moe_key, mapping in pbar:
                layer_name = moe_key.replace("__MOE_EXPERTS__", "")
                pbar.set_postfix({"layer": layer_name})

                if (
                    not isinstance(mapping.target_path, list)
                    or len(mapping.target_path) < 2
                ):
                    logger.warning(f"Invalid MoE mapping for {moe_key}")
                    continue

                target_path = mapping.target_path[0]
                expert_keys = mapping.target_path[1:]

                collected_weights = []
                for expert_key in expert_keys:
                    if expert_key in expert_weights:
                        weight = expert_weights[expert_key]
                        if mapping.transpose and not expert_key.endswith(".bias"):
                            weight = jnp.transpose(weight, (1, 0))
                        collected_weights.append(weight)
                    else:
                        logger.warning(f"Missing expert weight: {expert_key}")

                if len(collected_weights) == len(expert_keys):
                    stacked_weight = jnp.stack(collected_weights, axis=0)

                    device_experts = stacked_weight

                    sharded_weight = self._shard_weight(
                        device_experts, mapping.sharding
                    )
                    model_param = self._get_param(params, target_path)
                    model_param.value = sharded_weight
                else:
                    logger.error(
                        f"Could not collect all expert weights for {target_path}"
                    )
