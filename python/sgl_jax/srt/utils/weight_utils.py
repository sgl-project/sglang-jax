import glob
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
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
    is_eagle3: bool = False
    # MoE weight fusion configuration
    fuse_moe_weights: bool = False
    fuse_gate_up: tuple[str, str] | None = None

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
        self.dummy_mode = getattr(model_config, "_dummy_mode", False)

        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = (
            model_config.get_total_num_kv_heads()
        )  # Use original count for replication logic
        self.hidden_size = model_config.hidden_size
        self.head_dim_original = getattr(
            model_config, "head_dim", self.hidden_size // self.num_heads
        )

        self.head_dim_pad = (self.head_dim_original + 127) // 128 * 128 - self.head_dim_original
        self.head_dim = self.head_dim_original
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
        self,
        weight_mappings: Mapping[str, str | list[str] | WeightMapping],
        safetensors_partition=1,
        dummy=False,
    ):
        """Load weights from safetensors files or generate dummy weights.

        Args:
            weight_mappings: A read-only (covariant) mapping from HF keys to model paths with sharding info(Do not modify this dictionary in place)
            safetensors_partition: Number of safetensors partitions
            dummy: If True, generate random weights instead of loading from files
        """
        params = nnx.state(self.model)

        # Dummy mode: generate random weights using mapping's sharding info
        # Can be explicitly passed or set via model_config._dummy_mode
        if dummy or self.dummy_mode:
            self._load_dummy_weights(params, weight_mappings)
            return

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

        for hf_key, hf_weight in self._iterate_weights():
            if hf_key in regular_mappings:
                if hf_key == "d2t":
                    base = jnp.arange(hf_weight.shape[0], dtype=hf_weight.dtype)
                    hot_ids = (hf_weight + base).astype(jnp.int32)
                    params["hot_token_ids"].value = hot_ids
                    continue
                mapping = regular_mappings[hf_key]
                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)
                self._process_and_assign_weight(params, hf_key, hf_weight, mapping)
            elif (
                "mlp.experts." in hf_key or "block_sparse_moe.experts" in hf_key
            ) and hf_key.endswith(".weight"):
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded MoE expert weight: %s", hf_key)
                    continue

                assigned = False
                for moe_key, mapping in moe_mappings.items():
                    expected_hf_keys = mapping.target_path[1:]  # list of expected HF keys
                    if hf_key in expected_hf_keys:
                        if mapping.fuse_moe_weights:
                            # Fused MoE Logic
                            gate_id, up_id = mapping.fuse_gate_up
                            if gate_id in hf_key:
                                group_type = "gate"
                            elif up_id in hf_key:
                                group_type = "up"
                            else:
                                logger.warning(
                                    "Fused key %s matches neither %s nor %s",
                                    hf_key,
                                    gate_id,
                                    up_id,
                                )
                                continue

                            if moe_key not in moe_buffer:
                                moe_buffer[moe_key] = {"gate": {}, "up": {}}

                            if hf_key not in moe_buffer[moe_key][group_type]:
                                moe_buffer[moe_key][group_type][hf_key] = []

                            moe_buffer[moe_key][group_type][hf_key].append(hf_weight)
                            assigned = True

                            # Check if we have all necessary weights
                            total_captured = len(moe_buffer[moe_key]["gate"]) + len(
                                moe_buffer[moe_key]["up"]
                            )

                            if total_captured == len(expected_hf_keys):
                                # Validate shard counts for ALL weights
                                all_shard_counts = []
                                for g_type in ["gate", "up"]:
                                    for w_list in moe_buffer[moe_key][g_type].values():
                                        all_shard_counts.append(len(w_list))

                                if not all_shard_counts:  # Should not happen if total_captured > 0
                                    continue

                                if len(set(all_shard_counts)) != 1:
                                    continue

                                if mapping.concat_axis is not None:
                                    if all_shard_counts[0] < safetensors_partition:
                                        continue
                                elif all_shard_counts[0] != 1:
                                    continue

                                self._process_fused_moe_group(
                                    params, moe_key, mapping, moe_buffer[moe_key]
                                )
                                del moe_buffer[moe_key]

                        else:
                            # Regular (Non-Fused) MoE Logic
                            if moe_key not in moe_buffer:
                                moe_buffer[moe_key] = {}
                            if hf_key not in moe_buffer[moe_key]:
                                moe_buffer[moe_key][hf_key] = []
                            moe_buffer[moe_key][hf_key].append(hf_weight)
                            assigned = True

                            if len(moe_buffer[moe_key]) == len(expected_hf_keys):
                                shard_counts = [len(v) for v in moe_buffer[moe_key].values()]
                                # Validate all weights have consistent shard counts
                                if len(set(shard_counts)) != 1:
                                    continue

                                # Auto-detect TP sharding:
                                # - Grok-2: concat_axis is set, needs multiple shards (e.g., 8)
                                if mapping.concat_axis is not None:
                                    # TP-sharded weights: need to collect all TP shards
                                    # Expected number of shards = total model files / experts per file
                                    if shard_counts[0] < safetensors_partition:
                                        # Still collecting shards, wait for more
                                        continue
                                else:
                                    # Non-TP-sharded weights: expect exactly 1 copy per expert
                                    if shard_counts[0] != 1:
                                        continue

                                self._process_single_moe_group(
                                    params, moe_key, mapping, moe_buffer[moe_key]
                                )
                                del moe_buffer[moe_key]  # free memory
                        break

                if not assigned:
                    logger.warning("MoE expert weight not assigned to any mapping: %s", hf_key)
            else:
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded layer weight: %s", hf_key)
                else:
                    logger.warning("No mapping found for weight: %s", hf_key)

        if moe_buffer:
            for moe_key in moe_buffer:
                mapping = moe_mappings[moe_key]
                expected = len(mapping.target_path[1:])

                if mapping.fuse_moe_weights:
                    got_gate = len(moe_buffer[moe_key].get("gate", {}))
                    got_up = len(moe_buffer[moe_key].get("up", {}))
                    got = got_gate + got_up
                    shard_counts = []
                    if "gate" in moe_buffer[moe_key]:
                        shard_counts.extend([len(v) for v in moe_buffer[moe_key]["gate"].values()])
                    if "up" in moe_buffer[moe_key]:
                        shard_counts.extend([len(v) for v in moe_buffer[moe_key]["up"].values()])
                else:
                    got = len(moe_buffer[moe_key])
                    shard_counts = (
                        [len(v) for v in moe_buffer[moe_key].values()]
                        if moe_buffer[moe_key]
                        else []
                    )

                logger.error(
                    "MoE group %s incomplete: %s/%s weights loaded, shard_counts=%s, concat_axis=%s",
                    moe_key,
                    got,
                    expected,
                    shard_counts,
                    mapping.concat_axis,
                )
            raise RuntimeError("Incomplete MoE expert weights detected.")

        nnx.update(self.model, params)

        # Final verification: check all fused MoE layers
        self._verify_fused_moe_weights(params, moe_mappings)

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
            # If TP-sharded (e.g., Grok-2), concatenate shards along concat_axis
            if mapping.concat_axis is not None and len(weights) > 1:
                weight = jnp.concatenate(weights, axis=mapping.concat_axis)
            else:
                # Non-TP-sharded (e.g., Qwen3-MoE), expect single weight
                weight = weights[0]

            if mapping.transpose and not hf_key.endswith(".bias"):
                weight = jnp.transpose(weight, (1, 0))
            collected_weights.append(weight)

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
        model_param.value = sharded_weight.astype(model_param.value.dtype)

        logger.debug("Assigned MoE group %s, shape: %s", moe_key, stacked_weight.shape)

    def _process_fused_moe_group(
        self,
        params: nnx.State,
        moe_key: str,
        mapping: WeightMapping,
        grouped_weights: dict[str, dict[str, list[jax.Array]]],
    ):
        """
        Process fused MoE weight groups (gate + up weights).

        Args:
            params: Model parameter state
            moe_key: MoE weight key (e.g., "__MOE_EXPERTS__model.layers.0.block_sparse_moe.experts.w1")
            mapping: Weight mapping configuration
            grouped_weights: Grouped weights dict
                {
                    "gate": {hf_key: [weight_shard1, weight_shard2, ...]},
                    "up": {hf_key: [weight_shard1, weight_shard2, ...]}
                }
        """
        target_path = mapping.target_path[0]
        expected_hf_keys = mapping.target_path[1:]

        # Step 1: Process gate and up weights separately
        # Use the predefined order from expected_hf_keys, not sorting
        gate_weights = []
        up_weights = []

        gate_id, up_id = mapping.fuse_gate_up

        # Separate expected keys into gate and up based on fuse_gate_up config
        for hf_key in expected_hf_keys:
            if gate_id in hf_key:
                # This is a gate weight
                weights = grouped_weights["gate"][hf_key]

                # Concatenate TP shards
                if mapping.concat_axis is not None and len(weights) > 1:
                    weight = jnp.concatenate(weights, axis=mapping.concat_axis)
                else:
                    weight = weights[0]

                # Transpose
                if mapping.transpose:
                    weight = jnp.transpose(weight, (1, 0))

                gate_weights.append(weight)

            elif up_id in hf_key:
                # This is an up weight
                weights = grouped_weights["up"][hf_key]

                # Concatenate TP shards
                if mapping.concat_axis is not None and len(weights) > 1:
                    weight = jnp.concatenate(weights, axis=mapping.concat_axis)
                else:
                    weight = weights[0]

                # Transpose
                if mapping.transpose:
                    weight = jnp.transpose(weight, (1, 0))

                up_weights.append(weight)

        # Step 2: Stack to 3D tensors
        # gate_stacked: (num_experts, hidden_size, intermediate_size)
        # up_stacked: (num_experts, hidden_size, intermediate_size)
        gate_stacked = jnp.stack(gate_weights, axis=0)
        up_stacked = jnp.stack(up_weights, axis=0)

        # Step 3: Fuse to 4D tensor
        # fused_weight: (num_experts, 2, hidden_size, intermediate_size)
        fused_weight = jnp.stack([gate_stacked, up_stacked], axis=1)

        # Step 4: Apply sharding
        if "expert" in mapping.sharding:
            ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // ep_size

            devices = self.mesh.devices.flatten()
            moe_mesh = jax.sharding.Mesh(
                devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
            )

            sharded_weight = self._shard_weight(fused_weight, mapping.sharding, mesh=moe_mesh)
        else:
            sharded_weight = self._shard_weight(fused_weight, mapping.sharding)

        # Step 5: Assign to model parameter
        model_param = self._get_param(params, target_path)
        original_dtype = model_param.value.dtype
        expected_shape = model_param.value.shape

        # Validate shape before assignment
        if fused_weight.shape != expected_shape:
            raise ValueError(
                f"Fused MoE weight shape mismatch for {target_path}: "
                f"expected {expected_shape}, got {fused_weight.shape}"
            )

        model_param.value = sharded_weight.astype(original_dtype)

        # Verify assignment was successful
        actual_shape = model_param.value.shape
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Failed to assign fused MoE weight to {target_path}: shape mismatch"
            )

    def _load_dummy_weights(
        self,
        params: nnx.State,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        seed: int = 1234,
    ):
        logger.info("Generating dummy weights with proper sharding from weight mappings")
        # Separate regular and MOE weights
        regular_mappings = {}
        moe_mappings = {}

        for hf_key, mapping in weight_mappings.items():
            if hf_key.startswith("__MOE_EXPERTS__"):
                moe_mappings[hf_key] = mapping
            else:
                regular_mappings[hf_key] = mapping

        # Process regular weights
        for hf_key, mapping in regular_mappings.items():

            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            target_path = (
                mapping.target_path
                if isinstance(mapping.target_path, str)
                else mapping.target_path[0]
            )

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy weight for %s (parameter not found)", target_path)
                continue

            shape = model_param.value.shape
            dtype = model_param.value.dtype

            # Generate dummy weight with correct sharding
            sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
            sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)

            def make_shard(indices, shape=shape, dtype=dtype):
                # Compute shard shape from global shape and indices
                shard_shape = []
                for dim_size, idx in zip(shape, indices):
                    if isinstance(idx, slice):
                        start, stop, step = idx.indices(dim_size)
                        assert step == 1, f"Non-unit step not supported: {idx}"
                        shard_shape.append(stop - start)
                    else:
                        shard_shape.append(1)
                shard_shape = tuple(shard_shape)

                # Generate random data
                rng = np.random.default_rng(seed)
                if jnp.issubdtype(dtype, jnp.floating):
                    if dtype == jnp.bfloat16:
                        gen_dtype = np.float32
                    else:
                        gen_dtype = {
                            jnp.float16: np.float16,
                            jnp.float32: np.float32,
                            jnp.float64: np.float64,
                        }.get(dtype, np.float32)
                    arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                    return jnp.asarray(arr_np, dtype=dtype)
                else:
                    # Non-floating types, just zeros
                    return jnp.zeros(shard_shape, dtype=dtype)

            dummy_weight = jax.make_array_from_callback(shape, sharding, make_shard)
            model_param.value = dummy_weight
            logger.debug(
                "Generated dummy weight for %s, shape=%s, sharding=%s",
                target_path,
                shape,
                sharding_spec,
            )

        # Process MOE weights
        for moe_key, mapping in moe_mappings.items():
            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            target_path = mapping.target_path[0]

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy MOE weight for %s (parameter not found)", target_path)
                continue

            # Expected shape: (num_experts, ...)
            full_shape = model_param.value.shape
            num_experts = full_shape[0]
            expert_weight_shape = full_shape[1:]
            dtype = model_param.value.dtype

            # Generate dummy weights for all experts
            collected_weights = []
            for expert_idx in range(num_experts):
                # For each expert weight, generate with appropriate sharding
                # Remove "expert" axis from sharding for individual expert weight generation
                if mapping.sharding and "expert" in mapping.sharding:
                    # Expert-parallel sharding: use tensor-only sharding for generation
                    expert_sharding_tuple = tuple(s for s in mapping.sharding if s != "expert")
                else:
                    expert_sharding_tuple = mapping.sharding

                expert_sharding_spec = P(*expert_sharding_tuple) if expert_sharding_tuple else P()
                expert_sharding = jax.sharding.NamedSharding(self.mesh, expert_sharding_spec)

                def make_expert_shard(
                    indices, weight_shape=expert_weight_shape, weight_dtype=dtype, idx=expert_idx
                ):
                    shard_shape = []
                    for dim_size, idx_val in zip(weight_shape, indices):
                        if isinstance(idx_val, slice):
                            start, stop, step = idx_val.indices(dim_size)
                            assert step == 1, f"Non-unit step not supported: {idx_val}"
                            shard_shape.append(stop - start)
                        else:
                            shard_shape.append(1)
                    shard_shape = tuple(shard_shape)

                    rng = np.random.default_rng(seed + idx)
                    if jnp.issubdtype(weight_dtype, jnp.floating):
                        gen_dtype = np.float32 if weight_dtype == jnp.bfloat16 else weight_dtype
                        arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                        return jnp.asarray(arr_np, dtype=weight_dtype)
                    else:
                        return jnp.zeros(shard_shape, dtype=weight_dtype)

                expert_weight = jax.make_array_from_callback(
                    expert_weight_shape, expert_sharding, make_expert_shard
                )
                collected_weights.append(expert_weight)

            # Stack all expert weights: (num_experts, ...)
            stacked_weight = jnp.stack(collected_weights, axis=0)

            # Apply final sharding with expert axis if needed
            if mapping.sharding and "expert" in mapping.sharding:
                # Use MOE mesh with expert parallelism
                ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                if ep_size > 1:
                    world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
                    tp_size = world_size // ep_size

                    devices = self.mesh.devices.flatten()
                    moe_mesh = jax.sharding.Mesh(
                        devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                    )
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(moe_mesh, final_sharding_spec)
                else:
                    # No expert parallelism, use regular mesh
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)
            else:
                final_sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
                final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)

            # Reshard to final sharding
            sharded_weight = jax.device_put(stacked_weight, final_sharding)
            model_param.value = sharded_weight.astype(dtype)

            logger.debug(
                "Generated dummy MOE weight for %s, shape=%s, num_experts=%s, sharding=%s",
                target_path,
                full_shape,
                num_experts,
                mapping.sharding,
            )

        nnx.update(self.model, params)
        logger.info("Dummy weights generated successfully!")

    def _iterate_weights(self):
        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        skipped_files = 0

        platform = os.getenv("JAX_PLATFORMS", None)
        backend = "cpu" if platform != "proxy" else "proxy"
        with tqdm(weights_files, desc="[LOADING] MODEL WEIGHTS", unit="file") as pbar:
            for st_file in pbar:
                filename = os.path.basename(st_file)
                pbar.set_postfix({"file": filename})

                with (
                    jax.default_device(jax.local_devices(backend=backend)[0]),
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

        # Apply output_multiplier_scale to lm_head weights (matching PyTorch implementation)
        if "lm_head" in hf_key and hasattr(self.model_config.hf_config, "output_multiplier_scale"):
            logger.info(
                "Applying output_multiplier_scale (%.2f) to %s",
                self.model_config.hf_config.output_multiplier_scale,
                hf_key,
            )
            processed_weight = processed_weight.astype(jnp.float32)
            processed_weight = (
                processed_weight * self.model_config.hf_config.output_multiplier_scale
            )

        if mapping.reshape is not None:
            processed_weight = jnp.reshape(processed_weight, mapping.reshape)

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
                processed_weight = self._apply_kv_head_padding(processed_weight, jax_path)

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
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_heads,
                                self.head_dim_original,
                            ),
                        )
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                        return jnp.reshape(
                            padded,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_heads * self.head_dim,
                            ),
                        )
                    elif any(proj in hf_key for proj in ["k_proj", "v_proj"]):
                        reshaped = jnp.reshape(
                            weight,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_kv_heads,
                                self.head_dim_original,
                            ),
                        )
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                        return jnp.reshape(
                            padded,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_kv_heads * self.head_dim,
                            ),
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

    def _verify_fused_moe_weights(
        self, params: nnx.State, moe_mappings: dict[str, WeightMapping]
    ) -> None:
        """Verify that all fused MoE weights were loaded correctly."""
        # Get all fused w1 mappings
        fused_w1_mappings = {
            k: v for k, v in moe_mappings.items() if getattr(v, "fuse_moe_weights", False)
        }

        # Get corresponding w2 mappings (same layer, but w2 instead of w1)
        w2_mappings = {}
        for k in fused_w1_mappings:
            w2_key = k.replace(".w1", ".w2")
            if w2_key in moe_mappings:
                w2_mappings[w2_key] = moe_mappings[w2_key]

        if not fused_w1_mappings:
            return

        all_verified = True
        verified_count = 0

        # Verify w1 and w2 weights
        for _, mapping in fused_w1_mappings.items():
            target_path = mapping.target_path[0]
            try:
                model_param = self._get_param(params, target_path)
                weight_shape = model_param.value.shape
                weight_values = model_param.value

                if (
                    len(weight_shape) != 4
                    or weight_shape[1] != 2
                    or jnp.all(weight_values == 0)
                    or jnp.any(jnp.isnan(weight_values))
                ):
                    logger.error("✗ %s: Invalid or corrupted weights", target_path)
                    all_verified = False
                else:
                    verified_count += 1
            except (KeyError, AttributeError, ValueError) as e:
                logger.error("✗ %s: Failed to access - %s", target_path, str(e))
                all_verified = False

        for _, mapping in w2_mappings.items():
            target_path = mapping.target_path[0]
            try:
                model_param = self._get_param(params, target_path)
                weight_shape = model_param.value.shape
                weight_values = model_param.value

                if (
                    len(weight_shape) != 3
                    or jnp.all(weight_values == 0)
                    or jnp.any(jnp.isnan(weight_values))
                ):
                    logger.error("✗ %s (w2): Invalid or corrupted weights", target_path)
                    all_verified = False
                else:
                    verified_count += 1
            except (KeyError, AttributeError, ValueError) as e:
                logger.error("✗ %s (w2): Failed to access - %s", target_path, str(e))
                all_verified = False

        if all_verified:
            logger.info("✓ Fused MoE weights verified: %d layers", verified_count // 2)
        else:
            raise RuntimeError("Fused MoE weight verification failed")
