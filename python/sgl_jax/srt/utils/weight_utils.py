import glob
import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WeightMetadata:
    """Metadata for a weight tensor without loading the actual data."""

    file_path: str
    shape: tuple
    dtype: str
    hf_key: str


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

    def _scan_weight_metadata(self) -> dict[str, WeightMetadata]:
        """
        Scan all safetensors files and build a metadata map.

        This is a fast operation that only reads file headers, not actual weight data.
        All processes scan all files to build a complete global view.

        Returns:
            A dictionary mapping weight keys to their metadata
        """
        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        metadata_map = {}
        process_index = jax.process_index()

        logger.info(
            "Process %d: Scanning %d files for weight metadata...",
            process_index,
            len(weights_files),
        )

        for st_file in weights_files:
            with safe_open(st_file, framework="flax") as f:
                # Get all keys and their metadata
                for hf_key in f.keys():  # noqa: SIM118
                    # Skip excluded layers
                    if hf_key.startswith("model.layers.") and self._is_excluded_layer_weight(
                        hf_key
                    ):
                        continue

                    # Get tensor metadata from header (no I/O)
                    tensor_slice = f.get_slice(hf_key)
                    shape = tensor_slice.get_shape()
                    dtype = str(tensor_slice.get_dtype())

                    metadata_map[hf_key] = WeightMetadata(
                        file_path=st_file,
                        shape=shape,
                        dtype=dtype,
                        hf_key=hf_key,
                    )

        logger.info(
            "Process %d: Found %d weights across all files", process_index, len(metadata_map)
        )
        return metadata_map

    def _determine_responsible_process(self, file_path: str, weights_files: list[str]) -> int:
        """
        Determine which process is responsible for loading weights from a given file.

        Uses round-robin distribution: file at index i is handled by process (i % process_count)
        """
        try:
            file_index = weights_files.index(file_path)
            return file_index % jax.process_count()
        except ValueError:
            # Fallback: hash the filename
            return hash(file_path) % jax.process_count()

    def _load_and_broadcast_weight(
        self, metadata: WeightMetadata, responsible_process: int
    ) -> jax.Array:
        """
        Load a weight from disk and broadcast it to all processes.

        Args:
            metadata: Weight metadata including file path and shape
            responsible_process: The process that should load from disk

        Returns:
            The weight tensor, available on all processes
        """
        process_index = jax.process_index()

        if process_index == responsible_process:
            # This process is responsible: load the weight from disk
            # Calculate weight size for I/O monitoring
            num_elements = 1
            for dim in metadata.shape:
                num_elements *= dim

            bytes_per_element = 2  # Assume bf16/fp16 (most common)
            if "F32" in metadata.dtype or "I32" in metadata.dtype:
                bytes_per_element = 4
            elif "I64" in metadata.dtype:
                bytes_per_element = 8

            weight_size_mb = (num_elements * bytes_per_element) / (1024 * 1024)

            logger.debug(
                "Process %d: Loading %s (%.2f MB) from %s",
                process_index,
                metadata.hf_key,
                weight_size_mb,
                os.path.basename(metadata.file_path),
            )

            with (
                jax.default_device(jax.local_devices(backend="cpu")[0]),
                safe_open(metadata.file_path, framework="flax") as f,
            ):
                weight = f.get_tensor(metadata.hf_key)
                # Broadcast to all other processes
                return broadcast_one_to_all(weight, is_source=True)
        else:
            # This process receives the weight from responsible_process
            # We need to receive with the correct shape/dtype
            return broadcast_one_to_all(None, is_source=False)

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

        process_count = jax.process_count()
        process_index = jax.process_index()

        if process_count > 1:
            # Multi-process mode: scan metadata first, then load with distribution
            logger.info("Process %d: Multi-process loading mode enabled", process_index)

            # Phase 1: All processes scan file metadata (fast, no weight I/O)
            metadata_map = self._scan_weight_metadata()

            # Phase 2: Build file list for determining responsible processes
            model_path = self.model_config.model_path
            weights_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

            # I/O statistics tracking
            total_weights_loaded_by_me = 0
            total_weights = len(metadata_map)

            # Phase 3: Load weights with broadcasting
            # Process weights in deterministic order for consistency across processes
            for hf_key in sorted(metadata_map.keys()):
                metadata = metadata_map[hf_key]

                # Determine which process should load this weight
                responsible_process = self._determine_responsible_process(
                    metadata.file_path, weights_files
                )

                # Track I/O statistics
                if responsible_process == process_index:
                    total_weights_loaded_by_me += 1

                # Load and broadcast the weight
                hf_weight = self._load_and_broadcast_weight(metadata, responsible_process)

                # Now all processes have this weight, process it normally
                if hf_key in regular_mappings:
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
                        expected_hf_keys = mapping.target_path[1:]
                        if hf_key in expected_hf_keys:
                            if moe_key not in moe_buffer:
                                moe_buffer[moe_key] = {}
                            if hf_key not in moe_buffer[moe_key]:
                                moe_buffer[moe_key][hf_key] = []
                            moe_buffer[moe_key][hf_key].append(hf_weight)
                            assigned = True

                            if len(moe_buffer[moe_key]) == len(expected_hf_keys):
                                shard_counts = [len(v) for v in moe_buffer[moe_key].values()]
                                if len(set(shard_counts)) != 1:
                                    continue

                                if mapping.concat_axis is not None:
                                    if shard_counts[0] < 2:
                                        continue
                                else:
                                    if shard_counts[0] != 1:
                                        continue

                                self._process_single_moe_group(
                                    params, moe_key, mapping, moe_buffer[moe_key]
                                )
                                del moe_buffer[moe_key]
                            break

                    if not assigned:
                        logger.warning("MoE expert weight not assigned to any mapping: %s", hf_key)
                else:
                    if self._is_excluded_layer_weight(hf_key):
                        logger.debug("Skipping excluded layer weight: %s", hf_key)
                    else:
                        logger.warning("No mapping found for weight: %s", hf_key)

            # Log I/O statistics
            io_reduction_ratio = (
                1.0 - (total_weights_loaded_by_me / total_weights) if total_weights > 0 else 0
            )
            logger.info(
                "Process %d: I/O optimization complete - loaded %d/%d weights from disk (%.1f%% reduction)",
                process_index,
                total_weights_loaded_by_me,
                total_weights,
                io_reduction_ratio * 100,
            )

        else:
            # Single process mode: use original fast path
            for hf_key, hf_weight, _source_process in self._iterate_weights():
                if hf_key in regular_mappings:
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
                                    # TP-sharded weights: need multiple shards to concatenate
                                    if shard_counts[0] < 2:
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
                got = len(moe_buffer[moe_key])
                shard_counts = (
                    [len(v) for v in moe_buffer[moe_key].values()] if moe_buffer[moe_key] else []
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

    def _iterate_weights(self):
        """
        Iterate through weights with multi-process optimization.

        In multi-process mode (process_count > 1):
        - Each process reads only its assigned files (process_index::process_count)
        - Weights are yielded with source_process info for later broadcasting
        - This reduces I/O by ~N× where N is the number of processes

        In single-process mode:
        - Behaves the same as before for backward compatibility
        """
        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        process_count = jax.process_count()
        process_index = jax.process_index()

        # Multi-process optimization: partition files across processes
        if process_count > 1:
            my_files = weights_files[process_index::process_count]
            logger.info(
                "Multi-process I/O optimization enabled: Process %d/%d loading %d/%d files",
                process_index,
                process_count,
                len(my_files),
                len(weights_files),
            )
        else:
            my_files = weights_files

        skipped_files = 0
        # Only show progress bar on process 0 in multi-process mode
        show_progress = process_index == 0 or process_count == 1

        with tqdm(
            my_files, desc="[LOADING] MODEL WEIGHTS", unit="file", disable=not show_progress
        ) as pbar:
            for st_file in pbar:
                filename = os.path.basename(st_file)
                if show_progress:
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
                            "Process %d: Skipping %s: 0/%s weights needed",
                            process_index,
                            filename,
                            len(f.keys()),
                        )
                        continue

                    logger.debug(
                        "Process %d: Loading %s: %s/%s weights needed",
                        process_index,
                        filename,
                        len(needed_keys),
                        len(f.keys()),
                    )
                    for name in needed_keys:
                        weight_tensor = f.get_tensor(name)
                        yield name, weight_tensor, process_index  # Include source process

        if skipped_files > 0:
            logger.info(
                "Process %d: Skipped %d/%d files with no needed weights",
                process_index,
                skipped_files,
                len(my_files),
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
