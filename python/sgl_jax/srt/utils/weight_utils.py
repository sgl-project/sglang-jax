import glob
import logging
import os
from dataclasses import dataclass
from re import M

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WeightMapping:
    target_path: str | list[str]
    sharding: tuple | None = None
    transpose: bool = False
    reshape: tuple | None = None
    head_dim_padding: bool = False
    kv_head_padding: bool = False
    concat_axis: int | None = None

    def __post_init__(self):
        if self.sharding is None:
            self.sharding = self._infer_default_sharding()

    def _infer_default_sharding(self) -> tuple:
        path = self.target_path[0] if isinstance(self.target_path, list) else self.target_path

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

        if hasattr(model_config, "ep_size") and model_config.ep_size > 1:
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // model_config.ep_size
            ep_size = model_config.ep_size
            abstract_mesh = self.mesh.abstract_mesh
            self.moe_abstract_mesh = abstract_mesh.update(
                axis_sizes=(ep_size, tp_size), axis_names=("expert", "tensor")
            )
        else:
            self.moe_abstract_mesh = None

    def load_weights_from_safetensors(
        self, weight_mappings: dict[str, str | list[str] | WeightMapping]
    ):
        params = nnx.state(self.model)

        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        moe_buffer = {}

        logger.info(
            "WeightLoader: Will load layers 0 to %s",
            self.model_config.num_hidden_layers - 1,
        )

        rank = jax.process_index()
        idx = 0
        for hf_key, hf_weight in self._iterate_weights():
            idx += 1
            if rank == 0:
                with open("/tmp/debug.txt", "a") as f:
                    f.write(f"idx: {idx}\n")
                    f.write(f"hf_key: {hf_key}\n")
                    f.write(f"hf_weight.shape: {hf_weight.shape}\n")
            if hf_key in regular_mappings:
                mapping = regular_mappings[hf_key]
                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)
                self._process_and_assign_weight(params, hf_key, hf_weight, mapping)
            elif ("mlp.experts." in hf_key or "block_sparse_moe.experts") and hf_key.endswith(".weight"):
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded MoE expert weight: %s", hf_key)
                    continue

                assigned = False
                for moe_key, mapping in moe_mappings.items():
                    expected_hf_keys = mapping.target_path[1:]  # list of expected HF keys
                    if hf_key in expected_hf_keys:
                        if moe_key not in moe_buffer:
                            moe_buffer[moe_key] = {}
                        if hf_key not in moe_buffer[moe_key]:
                            moe_buffer[moe_key][hf_key] = []
                        moe_buffer[moe_key][hf_key].append(hf_weight)
                        assigned = True

                        if len(moe_buffer[moe_key]) == len(expected_hf_keys):
                            shard_counts = [len(v) for v in moe_buffer[moe_key].values()]
                            if len(set(shard_counts)) != 1 or shard_counts[0] != 8:
                                continue
                            self._process_single_moe_group(
                                params, moe_key, mapping, moe_buffer[moe_key]
                            )
                            del moe_buffer[moe_key]  # free memory
                        break

                if not assigned:
                    # logger.warning("MoE expert weight not assigned to any mapping: %s", hf_key)
                    pass # TODO: add warning, right now just for debugging
            else:
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded layer weight: %s", hf_key)
                else:
                    logger.warning("No mapping found for weight: %s", hf_key)

        if moe_buffer:
            for moe_key in moe_buffer:
                expected = len(moe_mappings[moe_key].target_path[1:])
                got = len(moe_buffer[moe_key])
                logger.error(
                    "MoE group %s incomplete: %s/%s weights loaded", moe_key, got, expected
                )
            raise RuntimeError("Incomplete MoE expert weights detected.")

        nnx.update(self.model, params)
        with open("/tmp/debug.txt", "a") as f:
            f.write("updated params\n")
            f.write(f"params: {params}\n")

    def _process_single_moe_group(
        self,
        params: nnx.State,
        moe_key: str,
        mapping: WeightMapping,
        expert_weights_dict: dict[str, list[jax.Array]],
    ):
        target_path = mapping.target_path[0]
        expected_hf_keys = mapping.target_path[1:]

        collected_weights = []
        for hf_key in expected_hf_keys:
            weights = expert_weights_dict[hf_key]
            # concat weights along axis 0
            if hf_key.split(".")[-2] == "w2":
                logging.info("hf_key: %s", hf_key)
                logging.info("weight shape: %s", weights[0].shape)
                logging.info("mapping.concat_axis: %s", mapping.concat_axis)
                logging.info("transpose: %s", mapping.transpose)
                
            if mapping.concat_axis is not None:
                weights = jnp.concatenate(weights, axis=mapping.concat_axis)
            if mapping.transpose and not hf_key.endswith(".bias"):
                weights = jnp.transpose(weights, (1, 0))
            # logging.info("weights.shape: %s", weights.shape)
            # logging.info("first 100 weights: %s", weights[:10, :100])
            collected_weights.append(weights)

        stacked_weight = jnp.stack(collected_weights, axis=0)  # (num_experts, ...)

        if "expert" in mapping.sharding:
            ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // ep_size

            devices = self.mesh.devices.flatten()
            moe_mesh = jax.sharding.Mesh(
                devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
            )

            sharded_weight = self._shard_weight(stacked_weight, mapping.sharding, mesh=moe_mesh)
        else:
            sharded_weight = self._shard_weight(stacked_weight, mapping.sharding)

        model_param = self._get_param(params, target_path)
        logging.info("param name: %s", target_path)
        # logging.info("model_param.value.shape: %s", model_param.value.shape)
        # logging.info("target_path: %s", target_path)
        model_param.value = sharded_weight.astype(model_param.value.dtype)

        logger.debug("Assigned MoE group %s, shape: %s", moe_key, stacked_weight.shape)

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

                with (
                    jax.default_device(jax.local_devices(backend="cpu")[0]),
                    safe_open(st_file, framework="flax") as f,
                ):
                    needed_keys = []
                    for name in f.keys():  # noqa: SIM118
                        if not name.startswith("model.layers."):
                            needed_keys.append(name)
                            continue

                        if not self._is_excluded_layer_weight(name):
                            needed_keys.append(name)

                    if not needed_keys:
                        skipped_files += 1
                        logger.debug(
                            "Skipping %s: 0/%s weights needed",
                            filename,
                            len(f.keys()),
                        )
                        continue

                    logger.debug(
                        "Loading %s: %s/%s weights needed",
                        filename,
                        len(needed_keys),
                        len(f.keys()),
                    )
                    for name in needed_keys:
                        weight_tensor = f.get_tensor(name)
                        yield name, weight_tensor

        if skipped_files > 0:
            logger.info(
                "Memory optimization: Skipped %s/%s files with no needed weights",
                skipped_files,
                len(weights_files),
            )

    def _process_and_assign_weight(
        self,
        params: nnx.State,
        hf_key: str,
        hf_weight: jax.Array,
        mapping: WeightMapping,
    ):
        processed_weight = hf_weight

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
            processed_weight = self._apply_head_dim_padding(processed_weight, hf_key, mapping)

        if mapping.kv_head_padding:
            processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

        sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

        try:
            model_param = self._get_param(params, jax_path)
            logger.debug(
                "Loading %s -> %s, shape: %s, transpose: %s",
                hf_key,
                jax_path,
                processed_weight.shape,
                mapping.transpose,
            )
            model_param.value = sharded_weight.astype(model_param.value.dtype)
        except Exception as e:
            logger.error("Failed to load %s -> %s: %s", hf_key, jax_path, str(e))
            raise

    def _handle_split_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        self._split_qkv_weight(params, hf_key, weight, mapping)

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

                k_bias = jnp.reshape(k_bias, (self.num_kv_heads, self.head_dim_original))
                k_bias = jnp.pad(k_bias, ((0, 0), (0, self.head_dim_pad)))
                k_bias = jnp.reshape(k_bias, (self.num_kv_heads * self.head_dim,))

                v_bias = jnp.reshape(v_bias, (self.num_kv_heads, self.head_dim_original))
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
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    q_weight = jnp.reshape(
                        q_weight, (self.hidden_size, self.num_heads * self.head_dim)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    k_weight = jnp.reshape(
                        k_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    v_weight = jnp.reshape(
                        v_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )
                else:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.num_heads, self.head_dim_original, self.hidden_size),
                    )
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    q_weight = jnp.reshape(
                        q_weight, (self.num_heads * self.head_dim, self.hidden_size)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    k_weight = jnp.reshape(
                        k_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    v_weight = jnp.reshape(
                        v_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

            splits = [q_weight, k_weight, v_weight]

        for split_weight, jax_path in zip(splits, jax_paths):
            processed_weight = split_weight

            if mapping.kv_head_padding and ("k_proj" in jax_path or "v_proj" in jax_path):
                processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

            sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

            model_param = self._get_param(params, jax_path)
            model_param.value = sharded_weight.astype(model_param.value.dtype)
            logger.debug("Split %s -> %s, shape: %s", hf_key, jax_path, processed_weight.shape)

    def _shard_weight(
        self, weight: jax.Array, sharding_spec: tuple, mesh: jax.sharding.Mesh = None
    ) -> jax.Array:
        if mesh is None:
            mesh = self.mesh
        target_sharding = jax.sharding.NamedSharding(mesh, P(*sharding_spec))

        if (
            getattr(weight, "_committed", False)
            and getattr(weight, "sharding", None) == target_sharding
        ):
            return weight

        if jax.process_count() > 1:

            def make_shard(indices):
                shard = weight[indices]
                return shard

            return jax.make_array_from_callback(
                shape=weight.shape, sharding=target_sharding, data_callback=make_shard
            )
        else:
            return jax.device_put(weight, target_sharding)

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
                    reshaped = jnp.reshape(weight, (self.num_heads, self.head_dim_original))
                    padded = jnp.pad(reshaped, ((0, 0), (0, self.head_dim_pad)))
                    return jnp.reshape(padded, (self.num_heads * self.head_dim,))
                else:  # k_proj or v_proj
                    reshaped = jnp.reshape(weight, (self.num_kv_heads, self.head_dim_original))
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
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
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
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
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
                        padded = jnp.pad(padded_reshaped, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                        return jnp.reshape(
                            padded, (self.num_heads * self.head_dim, self.hidden_size)
                        )

        return weight

    def _apply_kv_head_padding(self, weight: jax.Array, hf_key: str) -> jax.Array:
        """Apply KV head padding/replication when tp_size > total_kv_heads."""
        if any(
            proj in hf_key for proj in ["k_proj", "v_proj"]
        ) and self.model_config.needs_kv_head_replication(self.sharding_size):
            total_kv_heads = self.model_config.get_total_num_kv_heads()
            num_replicas = self.model_config.get_num_kv_head_replicas(self.sharding_size)
            padding_strategy = self.model_config.get_kv_padding_strategy()

            if padding_strategy == "replicate":
                if hf_key.endswith(".bias"):
                    replicated_bias_parts = []
                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_bias = weight[start_idx:end_idx]
                        for _ in range(num_replicas):
                            replicated_bias_parts.append(original_head_bias)
                    weight = jnp.concatenate(replicated_bias_parts, axis=0)
                else:
                    replicated_weight_parts = []
                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_weight = weight[:, start_idx:end_idx]
                        for _ in range(num_replicas):
                            replicated_weight_parts.append(original_head_weight)
                    weight = jnp.concatenate(replicated_weight_parts, axis=1)
            elif padding_strategy == "zero":
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
                        padding = jnp.zeros((weight.shape[0], padding_size), dtype=weight.dtype)
                        weight = jnp.concatenate([weight, padding], axis=1)
        return weight

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])
        return layer_num >= self.model_config.num_hidden_layers
