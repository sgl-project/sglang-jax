import glob
import logging
import os
import pickle
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import multihost_utils
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


class SequentialSafetensorManager:
    """
    Manages open file handles during a weight loading session to prevent
    repeated opening/parsing of safetensors headers.
    """

    def __init__(self):
        self.handles = {}

    def get_handle(self, filename):
        if filename not in self.handles:
            # Keep the file open. framework="np" is crucial for JAX interop.
            # device="cpu" ensures we don't accidentally alloc on GPU/TPU here.
            self.handles[filename] = safe_open(filename, framework="np", device="cpu")
        return self.handles[filename]

    def close_all(self):
        # safe_open objects don't strictly require close() as they rely on RAII/GC,
        # but clearing references ensures we don't hold descriptors.
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


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
        # Use original count for replication logic
        self.num_kv_heads = model_config.get_total_num_kv_heads()
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

    def _scan_weight_info(self) -> dict[str, list[dict]]:
        """
        Scan all safetensors files to build a mapping from HF key to file info.
        """
        # 1. Host 0 does the heavy lifting (Scanning)
        if jax.process_index() == 0:
            model_path = self.model_config.model_path
            weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

            if len(weights_files) == 0:
                raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

            weights_files.sort()
            weight_info = {}

            logger.info(
                "Scanning metadata for %s model files (single host only)...", len(weights_files)
            )
            # Use tqdm only on master to avoid log spam
            iterator = tqdm(weights_files, desc="Scanning Metadata", unit="file")

            for st_file in iterator:
                with safe_open(st_file, framework="flax", device="cpu") as f:
                    for key in f.keys():  # noqa: SIM118
                        slice_info = f.get_slice(key)
                        info = {
                            "file": st_file,
                            "shape": tuple(slice_info.get_shape()),
                            "dtype": slice_info.get_dtype(),
                        }
                        if key not in weight_info:
                            weight_info[key] = []
                        weight_info[key].append(info)

            # Serialize the result
            serialized_data = pickle.dumps(weight_info)
            # Convert to uint8 array for JAX broadcasting
            data_np = np.frombuffer(serialized_data, dtype=np.uint8)
            data_len = np.array(len(data_np), dtype=np.int32)
        else:
            # Other hosts just wait
            logger.info("Waiting for metadata broadcast from other host...")
            data_len = np.array(0, dtype=np.int32)
            data_np = None

        # 2. Broadcast the length of the data first
        data_len = multihost_utils.broadcast_one_to_all(
            data_len, is_source=(jax.process_index() == 0)
        )

        # 3. Prepare buffer on receivers
        if jax.process_index() != 0:
            data_np = np.empty(data_len.item(), dtype=np.uint8)

        # 4. Broadcast the actual serialized data
        synced_data = multihost_utils.broadcast_one_to_all(
            data_np, is_source=(jax.process_index() == 0)
        )

        # 5. Deserialize
        synced_bytes = np.array(synced_data).tobytes()
        weight_info = pickle.loads(synced_bytes)

        if jax.process_index() != 0:
            logger.info("Metadata received. Total keys: %s", len(weight_info))

        return weight_info

    def _create_lazy_tensors(
        self,
        hf_key: str,
        infos: list[dict],
        file_manager: SequentialSafetensorManager,
        target_sharding: jax.sharding.NamedSharding = None,
    ) -> list[jax.Array]:
        """
        Create a list of JAX arrays that lazy load data from safetensors via callback.
        Supports 'Global Loading' via target_sharding to avoid redundant I/O.
        """
        lazy_arrays = []

        for info in infos:
            shape = info["shape"]
            st_dtype = info["dtype"]

            dtype_map = {
                "BF16": jnp.bfloat16,
                "F16": jnp.float16,
                "F32": jnp.float32,
                "I64": jnp.int64,
                "I32": jnp.int32,
                "BOOL": jnp.bool_,
            }
            target_dtype = dtype_map.get(st_dtype, jnp.float32)

            filename = info["file"]

            if target_sharding is not None:
                # Load only what this host needs (Global Loading)
                sharding = target_sharding
            else:
                # Fallback: Load full tensor on every host (Replicated)
                sharding = jax.sharding.NamedSharding(self.mesh, P())

            def _make_load_slice(fname=filename, fm=file_manager):
                def _load_slice(index):
                    # index is the slice required by the current host based on 'sharding'
                    f = fm.get_handle(fname)
                    return f.get_slice(hf_key)[index]

                return _load_slice

            lazy_array = jax.make_array_from_callback(shape, sharding, _make_load_slice()).astype(
                target_dtype
            )

            lazy_arrays.append(lazy_array)

        return lazy_arrays

    def _create_split_lazy_tensor(
        self,
        hf_key: str,
        infos: list[dict],
        file_manager: SequentialSafetensorManager,
        concat_axis: int,
        target_sharding: jax.sharding.NamedSharding = None,
    ) -> jax.Array:
        """
        Lazy loader for TP-Split weights (e.g., Grok Attention/MLP).
        Instead of loading ALL shards on EVERY host, it calculates overlap
        and only reads the specific file(s) containing the requested slice.
        """
        # 1. Build the "Map": Calculate start/end offsets for each file
        # Sort by filename to ensure correct order (part-00001, part-00002...)
        sorted_infos = sorted(infos, key=lambda x: x["file"])

        cumulative_start = 0
        file_intervals = []  # List of (start, end, info)

        # Assume all shards have same shape except on concat_axis
        base_shape = list(sorted_infos[0]["shape"])

        for info in sorted_infos:
            shape = info["shape"]
            length = shape[concat_axis]
            start = cumulative_start
            end = start + length
            file_intervals.append((start, end, info))
            cumulative_start = end

        # 2. Determine Global Shape
        global_shape = list(base_shape)
        global_shape[concat_axis] = cumulative_start
        global_shape = tuple(global_shape)

        st_dtype = sorted_infos[0]["dtype"]
        dtype_map = {
            "BF16": jnp.bfloat16,
            "F16": jnp.float16,
            "F32": jnp.float32,
            "I64": jnp.int64,
            "I32": jnp.int32,
            "BOOL": jnp.bool_,
        }
        target_dtype = dtype_map.get(st_dtype, jnp.float32)

        if target_sharding is None:
            sharding = jax.sharding.NamedSharding(self.mesh, P())
        else:
            sharding = target_sharding

        # 3. Define Smart Stitching Callback
        def _smart_load_slice(index):
            # index is the slice required by JAX.
            # We need to intersect this slice with the physical file intervals.
            slice_on_axis = index[concat_axis]

            # Normalize slice
            req_start, req_stop, req_step = slice_on_axis.indices(global_shape[concat_axis])
            assert req_step == 1, "Strided access not supported in split loader yet"

            collected_chunks = []

            for f_start, f_end, info in file_intervals:
                # Calculate Intersection: [req_start, req_stop) AND [f_start, f_end)
                intersect_start = max(req_start, f_start)
                intersect_end = min(req_stop, f_end)

                if intersect_start < intersect_end:
                    local_start = intersect_start - f_start
                    local_end = intersect_end - f_start

                    # Construct read index for this file
                    file_read_index = list(index)
                    file_read_index[concat_axis] = slice(local_start, local_end)
                    file_read_index = tuple(file_read_index)

                    # Read directly
                    f = file_manager.get_handle(info["file"])
                    chunk = f.get_slice(hf_key)[file_read_index]
                    collected_chunks.append(chunk)

            if not collected_chunks:
                return np.zeros((0,) * len(global_shape), dtype=np.float32)

            if len(collected_chunks) == 1:
                # Perfect match (1-to-1 mapping), no copy needed
                return collected_chunks[0]
            else:
                # Cross-file boundary (rare if TP matches), needs stitching
                return np.concatenate(collected_chunks, axis=concat_axis)

        return jax.make_array_from_callback(global_shape, sharding, _smart_load_slice).astype(
            target_dtype
        )

    def _create_stacked_moe_lazy_tensor(
        self,
        expected_hf_keys: list[str],
        weight_info: dict,
        file_manager: SequentialSafetensorManager,
        do_transpose: bool = False,
        target_sharding: jax.sharding.NamedSharding = None,
    ) -> jax.Array:
        """
        Lazy oader for MoE weights:
        1. Global Loading: Directly shards data to avoid redundant I/O.
        2. CPU Transpose: Performs transpose on CPU to reduce TPU memory peak.
        3. Pre-allocation: Uses np.empty instead of np.stack to save memory.
        4. Threaded I/O: High concurrency for GCS throughput.
        """

        # 1. Get base info from the first expert
        first_key = expected_hf_keys[0]
        # assume len(infos) == 1 for standard MoE
        info = weight_info[first_key][0]

        single_expert_shape = info["shape"]
        st_dtype = info["dtype"]

        dtype_map = {
            "BF16": jnp.bfloat16,
            "F16": jnp.float16,
            "F32": jnp.float32,
            "I64": jnp.int64,
            "I32": jnp.int32,
            "BOOL": jnp.bool_,
        }
        target_dtype = dtype_map.get(st_dtype, jnp.float32)

        num_experts = len(expected_hf_keys)

        # Determine the final stacked shape, considering optional transpose
        if do_transpose:
            if len(single_expert_shape) >= 2:
                final_single_shape = list(single_expert_shape)
                final_single_shape[-1], final_single_shape[-2] = (
                    final_single_shape[-2],
                    final_single_shape[-1],
                )
                final_single_shape = tuple(final_single_shape)
            else:
                final_single_shape = single_expert_shape
        else:
            final_single_shape = single_expert_shape

        stacked_shape = (num_experts, *final_single_shape)

        if target_sharding is None:
            sharding = jax.sharding.NamedSharding(self.mesh, P())
        else:
            sharding = target_sharding

        MAX_WORKERS = 128

        def _load_stacked_slice(index):
            expert_slice = index[0]
            inner_slice = index[1:]

            start, stop, step = expert_slice.indices(num_experts)
            expert_indices = list(range(start, stop, step))
            sliced_num_experts = len(expert_indices)

            if sliced_num_experts == 0:
                return np.zeros((0, *[1] * len(inner_slice)), dtype=np.float32)

            first_idx = expert_indices[0]
            first_hf_key = expected_hf_keys[first_idx]

            fname_first = weight_info[first_hf_key][0]["file"]
            f_first = file_manager.get_handle(fname_first)

            # Read raw data
            first_data = f_first.get_slice(first_hf_key)[:]

            # Process first data (transpose if needed) to determine final buffer properties
            first_data_processed = np.transpose(first_data) if do_transpose else first_data

            # Slice according to inner_slice to determine the exact shape required
            first_final_chunk = first_data_processed[inner_slice]

            out_shape = (sliced_num_experts, *first_final_chunk.shape)
            out_array = np.empty(out_shape, dtype=first_final_chunk.dtype)

            # Fill the first slot
            out_array[0] = first_final_chunk

            def load_one_expert(args):
                i, expert_idx = args
                hf_key = expected_hf_keys[expert_idx]
                fname = weight_info[hf_key][0]["file"]
                f = file_manager.get_handle(fname)

                data = f.get_slice(hf_key)[:]
                if do_transpose:
                    data = np.transpose(data)
                out_array[i] = data[inner_slice]

            tasks = []
            for i, expert_idx in enumerate(expert_indices[1:], start=1):
                tasks.append((i, expert_idx))

            if tasks:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    list(executor.map(load_one_expert, tasks))

            return out_array

        return jax.make_array_from_callback(stacked_shape, sharding, _load_stacked_slice).astype(
            target_dtype
        )

    def load_weights_from_safetensors(
        self,
        weight_mappings: Mapping[str, str | list[str] | WeightMapping],
        safetensors_partition=1,
        dummy=False,
    ):
        """Load weights using JAX lazy evaluation and parallel I/O."""
        params = nnx.state(self.model)

        if dummy or self.dummy_mode:
            self._load_dummy_weights(params, weight_mappings)
            return

        # 1. Build index
        weight_info = self._scan_weight_info()

        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        logger.info("Starting parallel weight loading via JAX Lazy Loader...")

        with SequentialSafetensorManager() as file_manager:
            # 2. Process Regular Weights (Lazy Pull)
            for hf_key, mapping in tqdm(regular_mappings.items(), desc="Loading Regular Weights"):
                if hf_key not in weight_info:
                    if hf_key == "d2t":
                        logger.warning("Weight %s not found in safetensors index.", hf_key)
                        continue
                    if self._is_excluded_layer_weight(hf_key):
                        logger.debug("Skipping excluded layer weight: %s", hf_key)
                        continue
                    else:
                        logger.warning("No file found for weight: %s", hf_key)
                        continue

                infos = weight_info[hf_key]

                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)

                is_split_weight = len(infos) > 1 and mapping.concat_axis is not None

                can_optimize = (
                    isinstance(mapping.target_path, str)
                    and mapping.reshape is None
                    and not mapping.kv_head_padding
                    and not mapping.head_dim_padding
                    and mapping.sharding is not None
                )

                if can_optimize:
                    try:
                        if mapping.transpose and len(mapping.sharding) == 2:
                            # Swap: (dim0, dim1) -> (dim1, dim0)
                            sharding_tuple = mapping.sharding[::-1]
                        else:
                            sharding_tuple = mapping.sharding

                        spec = P(*sharding_tuple)
                        final_sharding = jax.sharding.NamedSharding(self.mesh, spec)

                        lazy_weight = None

                        if is_split_weight:
                            lazy_weight = self._create_split_lazy_tensor(
                                hf_key,
                                infos,
                                file_manager,
                                concat_axis=mapping.concat_axis,
                                target_sharding=final_sharding,
                            )
                        else:
                            lazy_arrays = self._create_lazy_tensors(
                                hf_key,
                                infos,
                                file_manager,
                                target_sharding=final_sharding,
                            )
                            lazy_weight = lazy_arrays[0]

                        if mapping.transpose:
                            lazy_weight = jnp.transpose(lazy_weight, (1, 0))

                        if "lm_head" in hf_key and hasattr(
                            self.model_config.hf_config, "output_multiplier_scale"
                        ):
                            lazy_weight = (
                                lazy_weight.astype(jnp.float32)
                                * self.model_config.hf_config.output_multiplier_scale
                            )

                        target_path = mapping.target_path
                        model_param = self._get_param(params, target_path)
                        model_param.value = lazy_weight.astype(model_param.value.dtype)

                        mode_str = "Split-Stitch" if is_split_weight else "Direct"
                        logger.debug(
                            "Fast Loading %s -> %s (%s), shape: %s",
                            hf_key,
                            target_path,
                            mode_str,
                            lazy_weight.shape,
                        )
                        continue

                    except Exception as e:
                        logger.warning(
                            "Fast load failed for %s, falling back to slow path. Error: %s",
                            hf_key,
                            str(e),
                        )
                lazy_arrays = self._create_lazy_tensors(
                    hf_key,
                    infos,
                    file_manager,
                    target_sharding=None,
                )

                if len(lazy_arrays) > 1 and mapping.concat_axis is not None:
                    lazy_weight = jnp.concatenate(lazy_arrays, axis=mapping.concat_axis)
                else:
                    lazy_weight = lazy_arrays[0]

                if hf_key == "d2t":
                    base = jnp.arange(lazy_weight.shape[0], dtype=lazy_weight.dtype)
                    hot_ids = (lazy_weight + base).astype(jnp.int32)
                    params["hot_token_ids"].value = hot_ids
                    continue

                self._process_and_assign_weight(params, hf_key, lazy_weight, mapping)
            # 3. Process MoE Weights (Lazy Pull)
            for moe_key, mapping in tqdm(moe_mappings.items(), desc="Loading MoE Weights"):
                expected_hf_keys = mapping.target_path[1:]

                group_complete = True
                is_tp_split = False

                # Validation pass
                for hf_key in expected_hf_keys:
                    if hf_key not in weight_info:
                        if self._is_excluded_layer_weight(hf_key):
                            logger.debug("Skipping excluded MoE expert weight: %s", hf_key)
                        else:
                            logger.warning("MoE expert weight %s not found.", hf_key)
                        group_complete = False
                        break

                    infos = weight_info[hf_key]

                    # Check for TP split (Grok style)
                    if mapping.concat_axis is not None:
                        if len(infos) > 1:
                            is_tp_split = True

                        if len(infos) < safetensors_partition:
                            logger.warning(
                                "Incomplete shards for %s: expected %s, found %s",
                                hf_key,
                                safetensors_partition,
                                len(infos),
                            )
                            group_complete = False
                            break

                if not group_complete:
                    continue

                # OPTIMIZATION: Use Stacked Loader if no TP split
                if not is_tp_split and mapping.concat_axis is None:
                    # 1. Pre-construct target sharding
                    if "expert" in mapping.sharding:
                        ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                        world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get(
                            "tensor", 1
                        )
                        tp_size = world_size // ep_size

                        devices = self.mesh.devices.flatten()
                        # Construct MoE specific mesh
                        moe_mesh = jax.sharding.Mesh(
                            devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                        )
                        final_sharding = jax.sharding.NamedSharding(moe_mesh, P(*mapping.sharding))
                    else:
                        # Standard Sharding
                        final_sharding = jax.sharding.NamedSharding(self.mesh, P(*mapping.sharding))

                    # 2. Call creator
                    stacked_weight = self._create_stacked_moe_lazy_tensor(
                        expected_hf_keys,
                        weight_info,
                        file_manager,
                        do_transpose=mapping.transpose,  # CPU transpose
                        target_sharding=final_sharding,  # Global loading
                    )

                    # 3. Direct assignment
                    target_path = mapping.target_path[0]
                    model_param = self._get_param(params, target_path)
                    model_param.value = stacked_weight.astype(model_param.value.dtype)

                    logger.debug(
                        "Assigned MoE group %s, shape: %s",
                        moe_key,
                        stacked_weight.shape,
                    )
                else:
                    collected_experts = []
                    moe_mesh = self.mesh
                    if "expert" in mapping.sharding:
                        ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                        world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get(
                            "tensor", 1
                        )
                        tp_size = world_size // ep_size
                        devices = self.mesh.devices.flatten()
                        moe_mesh = jax.sharding.Mesh(
                            devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                        )

                    for hf_key in expected_hf_keys:
                        infos = weight_info[hf_key]

                        expert_weight = self._create_split_lazy_tensor(
                            hf_key,
                            infos,
                            file_manager,
                            concat_axis=mapping.concat_axis,
                            target_sharding=None,
                        )

                        if mapping.transpose:
                            expert_weight = jnp.transpose(expert_weight, (1, 0))

                        collected_experts.append(expert_weight)

                    stacked_weight = jnp.stack(collected_experts, axis=0)
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(moe_mesh, final_sharding_spec)

                    sharded_weight = jax.device_put(stacked_weight, final_sharding)

                    target_path = mapping.target_path[0]
                    model_param = self._get_param(params, target_path)
                    model_param.value = sharded_weight.astype(model_param.value.dtype)

                    logger.debug(
                        "Assigned MoE group %s (Grok Split-Stitch), shape: %s",
                        moe_key,
                        stacked_weight.shape,
                    )

        nnx.update(self.model, params)
        logger.info("All weights loaded successfully.")

    def _load_dummy_weights(
        self,
        params: nnx.State,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        seed: int = 1234,
    ):
        logger.info("Generating dummy weights with proper sharding from weight mappings")
        regular_mappings = {}
        moe_mappings = {}

        for hf_key, mapping in weight_mappings.items():
            if hf_key.startswith("__MOE_EXPERTS__"):
                moe_mappings[hf_key] = mapping
            else:
                regular_mappings[hf_key] = mapping

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

            sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
            sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)

            def make_shard(indices, shape=shape, dtype=dtype):
                shard_shape = []
                for dim_size, idx in zip(shape, indices):
                    if isinstance(idx, slice):
                        start, stop, step = idx.indices(dim_size)
                        assert step == 1, f"Non-unit step not supported: {idx}"
                        shard_shape.append(stop - start)
                    else:
                        shard_shape.append(1)
                shard_shape = tuple(shard_shape)

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
                    return jnp.zeros(shard_shape, dtype=dtype)

            dummy_weight = jax.make_array_from_callback(shape, sharding, make_shard)
            model_param.value = dummy_weight
            logger.debug(
                "Generated dummy weight for %s, shape=%s, sharding=%s",
                target_path,
                shape,
                sharding_spec,
            )

        for moe_key, mapping in moe_mappings.items():
            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            target_path = mapping.target_path[0]

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy MOE weight for %s (parameter not found)", target_path)
                continue

            full_shape = model_param.value.shape
            num_experts = full_shape[0]
            expert_weight_shape = full_shape[1:]
            dtype = model_param.value.dtype

            collected_weights = []
            for expert_idx in range(num_experts):
                if mapping.sharding and "expert" in mapping.sharding:
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

            stacked_weight = jnp.stack(collected_weights, axis=0)

            if mapping.sharding and "expert" in mapping.sharding:
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
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)
            else:
                final_sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
                final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)

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
        # Since 'weight' is already a Lazy JAX Array (backed by a callback),
        # using device_put here is necessary when we are NOT using the "Global Loading"
        # optimization path. It will trigger the slice/distribute logic lazily.
        # However, for the optimized path, we skip this method entirely.
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
