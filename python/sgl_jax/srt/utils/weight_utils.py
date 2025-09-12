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
        self.num_kv_heads = model_config.num_key_value_heads
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

        self.original_num_kv_heads_per_device = model_config.get_original_num_kv_heads(
            self.sharding_size
        )
        self.padded_num_kv_heads_per_device = (
            model_config.get_num_kv_heads_with_padding(self.sharding_size)
        )
        self.needs_kv_padding = model_config.needs_kv_heads_padding(self.sharding_size)
        self.kv_padding_strategy = model_config.get_kv_padding_strategy()

        self.needs_kv_replication = model_config.needs_kv_head_replication(
            self.sharding_size
        )
        self.num_kv_head_replicas = model_config.get_num_kv_head_replicas(
            self.sharding_size
        )
        self.total_original_kv_heads = model_config.get_total_num_kv_heads()

        if self.needs_kv_padding:
            model_type = "GQA" if model_config.is_gqa_model() else "MHA"
            logger.info(
                f"KV projection weights padding enabled for {model_type} model: "
                f"k_proj/v_proj weights will be padded from {self.original_num_kv_heads_per_device} "
                f"to {self.padded_num_kv_heads_per_device} heads using {self.kv_padding_strategy} strategy"
            )

    def load_weights_from_safetensors(
        self, weight_mappings: Dict[str, Union[str, List[str], WeightMapping]]
    ):
        params = nnx.eval_shape(self.model)

        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        expert_weights = {}

        for hf_key, hf_weight in self._iterate_weights():
            if hf_key in regular_mappings:
                mapping = regular_mappings[hf_key]
                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)

                self._process_and_assign_weight(params, hf_key, hf_weight, mapping)
            elif "mlp.experts." in hf_key and hf_key.endswith(".weight"):
                expert_weights[hf_key] = hf_weight.astype(self.dtype)
            else:
                logger.warning(f"No mapping found for weight: {hf_key}")

        if moe_mappings:
            self._process_moe_expert_weights(params, moe_mappings, expert_weights)
        nnx.update(params)

    def _iterate_weights(self):

        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        for st_file in weights_files:
            logger.info(f"Loading weights from {st_file}")
            with jax.default_device(jax.local_devices(backend="cpu")[0]):
                with safe_open(st_file, framework="flax") as f:
                    for name in f.keys():
                        weight_tensor = f.get_tensor(name)
                        yield name, weight_tensor

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
        """Apply KV head padding/replication logic based on sglang approach."""

        # Handle KV head replication when tp_size >= total_kv_heads
        if (
            any(proj in hf_key for proj in ["k_proj", "v_proj"])
            and self.needs_kv_replication
        ):
            weight = self._prepare_kv_weights_for_replication(weight, hf_key)

        # Handle traditional padding for tiling optimization (when each device has < 2 heads)
        if (
            any(proj in hf_key for proj in ["k_proj", "v_proj"])
            and self.needs_kv_padding
        ):
            weight = self._pad_kv_projection_weight(weight, hf_key)

        return weight

    def _prepare_kv_weights_for_replication(
        self, weight: jax.Array, hf_key: str
    ) -> jax.Array:
        """
        Prepare KV weights for replication across devices.
        When tp_size >= total_kv_heads, we need to replicate original heads.
        This method prepares the weights so that JAX sharding will distribute them correctly.
        """
        if not self.needs_kv_replication:
            return weight

        if hf_key.endswith(".bias"):
            # For bias: prepare replicated bias for all devices
            # Each original head will be replicated across multiple devices
            replicated_bias_parts = []

            for original_head_id in range(self.total_original_kv_heads):
                start_idx = original_head_id * self.head_dim
                end_idx = (original_head_id + 1) * self.head_dim
                original_head_bias = weight[start_idx:end_idx]

                # Replicate this head for all its assigned devices
                for _ in range(self.num_kv_head_replicas):
                    replicated_bias_parts.append(original_head_bias)

            # Concatenate all replicated parts
            replicated_weight = jnp.concatenate(replicated_bias_parts, axis=0)

        else:
            # For weight matrix: prepare replicated weights for all devices
            replicated_weight_parts = []

            for original_head_id in range(self.total_original_kv_heads):
                start_idx = original_head_id * self.head_dim
                end_idx = (original_head_id + 1) * self.head_dim
                original_head_weight = weight[:, start_idx:end_idx]

                # Replicate this head for all its assigned devices
                for _ in range(self.num_kv_head_replicas):
                    replicated_weight_parts.append(original_head_weight)

            # Concatenate all replicated parts along head dimension
            replicated_weight = jnp.concatenate(replicated_weight_parts, axis=1)

        logger.debug(
            f"KV head replication for {hf_key}: {weight.shape} -> {replicated_weight.shape} "
            f"(original_kv_heads={self.total_original_kv_heads}, replicas={self.num_kv_head_replicas})"
        )

        return replicated_weight

    def _pad_kv_projection_weight(self, weight: jax.Array, hf_key: str) -> jax.Array:
        if not self.needs_kv_padding:
            return weight

        if hf_key.endswith(".bias"):
            padding_size = (
                self.padded_num_kv_heads_per_device
                - self.original_num_kv_heads_per_device
            ) * self.head_dim

            if self.kv_padding_strategy == "replicate":
                num_heads_to_add = (
                    self.padded_num_kv_heads_per_device
                    - self.original_num_kv_heads_per_device
                )
                num_original_heads = self.original_num_kv_heads_per_device

                if num_heads_to_add == num_original_heads:
                    interleaved_pieces = []
                    for head_idx in range(num_original_heads):
                        start_idx = head_idx * self.head_dim
                        end_idx = (head_idx + 1) * self.head_dim
                        original_head_bias = weight[start_idx:end_idx]
                        interleaved_pieces.extend(
                            [original_head_bias, original_head_bias]
                        )

                    return jnp.concatenate(interleaved_pieces, axis=0)
                else:
                    padding_pieces = []
                    for i in range(num_heads_to_add):
                        head_idx_to_copy = i % num_original_heads
                        start_idx = head_idx_to_copy * self.head_dim
                        end_idx = (head_idx_to_copy + 1) * self.head_dim
                        head_bias_to_copy = weight[start_idx:end_idx]
                        padding_pieces.append(head_bias_to_copy)

                    if padding_pieces:
                        padding = jnp.concatenate(padding_pieces, axis=0)
                    else:
                        padding = jnp.zeros((0,), dtype=weight.dtype)
            else:
                padding = jnp.zeros((padding_size,), dtype=weight.dtype)

            return jnp.concatenate([weight, padding], axis=0)
        else:
            hidden_size, kv_dim = weight.shape

            original_total_kv_heads = (
                self.original_num_kv_heads_per_device * self.sharding_size
            )
            expected_kv_dim_total = original_total_kv_heads * self.head_dim
            expected_kv_dim_per_device = (
                self.original_num_kv_heads_per_device * self.head_dim
            )

            if kv_dim == expected_kv_dim_total:
                expected_kv_dim = expected_kv_dim_total
            elif kv_dim == expected_kv_dim_per_device:
                expected_kv_dim = expected_kv_dim_per_device
            else:
                assert (
                    False
                ), f"Expected kv_dim={expected_kv_dim_total} (total) or {expected_kv_dim_per_device} (per-device), got {kv_dim}"

            if kv_dim == expected_kv_dim_total:
                padding_size = (
                    self.padded_num_kv_heads_per_device * self.sharding_size
                    - self.original_num_kv_heads_per_device * self.sharding_size
                ) * self.head_dim
            else:
                padding_size = (
                    self.padded_num_kv_heads_per_device
                    - self.original_num_kv_heads_per_device
                ) * self.head_dim

            if self.kv_padding_strategy == "replicate":
                num_heads_to_add = (
                    self.padded_num_kv_heads_per_device
                    - self.original_num_kv_heads_per_device
                )
                if kv_dim == expected_kv_dim_total:
                    num_heads_to_add *= self.sharding_size
                    num_original_heads = (
                        self.original_num_kv_heads_per_device * self.sharding_size
                    )
                else:
                    num_original_heads = self.original_num_kv_heads_per_device

                # For GQA, we want each head to be duplicated in-place
                # E.g., [head_0, head_1, head_2, head_3] -> [head_0, head_0, head_1, head_1, head_2, head_2, head_3, head_3]
                if num_heads_to_add == num_original_heads:
                    # Special case: duplicate each head once (most common for GQA)
                    # Interleave original heads with their copies
                    interleaved_pieces = []
                    for head_idx in range(num_original_heads):
                        start_idx = head_idx * self.head_dim
                        end_idx = (head_idx + 1) * self.head_dim
                        original_head = weight[:, start_idx:end_idx]
                        # Add original head and its copy
                        interleaved_pieces.extend([original_head, original_head])

                    return jnp.concatenate(interleaved_pieces, axis=1)
                else:
                    padding_pieces = []
                    for i in range(num_heads_to_add):
                        head_idx_to_copy = i % num_original_heads
                        start_idx = head_idx_to_copy * self.head_dim
                        end_idx = (head_idx_to_copy + 1) * self.head_dim
                        head_to_copy = weight[:, start_idx:end_idx]
                        padding_pieces.append(head_to_copy)

                    if padding_pieces:
                        padding = jnp.concatenate(padding_pieces, axis=1)
                    else:
                        # No padding needed
                        padding = jnp.zeros((weight.shape[0], 0), dtype=weight.dtype)
            else:  # zero padding
                padding = jnp.zeros((hidden_size, padding_size), dtype=weight.dtype)

            return jnp.concatenate([weight, padding], axis=1)

    def _process_moe_expert_weights(
        self,
        params: nnx.State,
        moe_mappings: Dict[str, WeightMapping],
        expert_weights: Dict[str, jax.Array],
    ):
        logger.info("Stacking expert weights...")

        for moe_key, mapping in moe_mappings.items():
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

                sharded_weight = self._shard_weight(device_experts, mapping.sharding)
                model_param = self._get_param(params, target_path)
                model_param.value = sharded_weight
            else:
                logger.error(f"Could not collect all expert weights for {target_path}")

        logger.info("MoE expert weights processing completed.")
