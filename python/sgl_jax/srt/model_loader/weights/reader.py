import logging
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from .source import (
    _SAFETENSORS_DTYPE_TO_JAX,
    WeightSource,
    _reinterpret_dtype_if_needed,
    coordinate_error,
)
from .specs import WeightSpec

logger = logging.getLogger(__name__)


class WeightReader(ABC):
    """Materialize declared reads without binding or modifying model parameters.

    All ranks call these methods in plan order. Implementations read only
    addressable shards, retain host buffers through H2D completion, and coordinate
    read failures before global array assembly, never inside I/O callbacks.
    Returned arrays have completed their local transfers. Later layout transforms
    and parameter assignment belong to TensorLayout and WeightLoader.
    """

    @abstractmethod
    def read(
        self,
        source: WeightSource,
        name: str,
        spec: WeightSpec,
        sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        """Read one tensor, or stack experts (including their declared transpose).

        Ordinary tensor layout transforms run after this read. ``concat_axis``
        joins checkpoint fragments before either kind of output is assembled.
        """

    @abstractmethod
    def read_host_group(
        self,
        source: WeightSource,
        spec: WeightSpec,
        targets: tuple[jax.ShapeDtypeStruct, ...],
        sharding: jax.sharding.NamedSharding,
    ) -> tuple[jax.Array, ...]:
        """Apply a host recipe to local expert intervals and place its outputs."""


class JaxShardReader(WeightReader):
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def read(self, source, name, spec, sharding=None):
        if spec.sources:
            args = (list(spec.sources), source.metadata, source)
            kwargs = dict(
                do_transpose=spec.transpose,
                target_sharding=sharding,
                physical_to_logical_map=spec.physical_to_logical_map,
            )
            if spec.concat_axis is not None:
                return self._read_split_experts(*args, spec.concat_axis, **kwargs)
            return self._read_experts(*args, **kwargs)
        infos = source.metadata[name]
        if spec.concat_axis is not None and len(infos) > 1:
            return self._read_split_tensor(name, infos, source, spec.concat_axis, sharding)
        if len(infos) != 1:
            raise ValueError(f"Multiple checkpoint fragments need concat_axis: {name}")
        return self._read_tensor(name, infos[0], source, sharding)

    def read_host_group(self, source, spec, targets, sharding):
        """Prefused experts: read local expert intervals, convert on host once,
        and upload every output's local shards without a full-model host copy.
        """
        groups = {}
        for output, param in enumerate(targets):
            for device, index in sharding.addressable_devices_indices_map(param.shape).items():
                expert = index[0]
                key = (expert.start, expert.stop, expert.step)
                groups.setdefault(key, []).append((output, device, index))
        uploaded = [{} for _ in targets]
        error = None
        try:
            budget = int(os.environ.get("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", str(4 << 30)))
            for key, assignments in groups.items():
                expert = slice(*key)
                input_bytes = sum(
                    len(range(*expert.indices(source.metadata[name][0]["shape"][0])))
                    * int(np.prod(source.metadata[name][0]["shape"][1:]))
                    * np.dtype(
                        _SAFETENSORS_DTYPE_TO_JAX[source.metadata[name][0]["dtype"]]
                    ).itemsize
                    for name in spec.sources
                )
                # Input, conversion copies and all outputs. Prefused gate/up
                # splits preserve total element count; dtype conversion may double it.
                required = input_bytes * 8
                if required > budget:
                    raise ValueError(f"Host recipe needs up to {required} bytes, budget={budget}")
                inputs = [self._read_host_input(source, name, (expert,)) for name in spec.sources]
                outputs = spec.host_recipe(inputs)
                if len(outputs) != len(targets):
                    raise ValueError("Host recipe output count does not match targets")
                local = []
                for output, device, index in assignments:
                    value = outputs[output][(slice(None), *index[1:])]
                    value = np.asarray(value, dtype=targets[output].dtype)
                    local.append(jax.device_put(value, device))
                    uploaded[output][device] = local[-1]
                jax.block_until_ready(local)
        except Exception as exc:
            error = exc
        coordinate_error(error, "host recipe")
        outputs = []
        for param, arrays in zip(targets, uploaded):
            devices = [d for d in sharding.mesh.devices.flat if d in arrays]
            outputs.append(
                jax.make_array_from_single_device_arrays(
                    param.shape, sharding, [arrays[d] for d in devices]
                )
            )
        return tuple(outputs)

    @staticmethod
    def _read_host_input(source, name, index):
        infos = source.metadata[name]
        if len(infos) != 1:
            raise ValueError(f"Recipe must explicitly handle split input: {name}")
        return source.read_tensor(infos[0]["file"], name, index)

    def _assemble(self, shape, sharding, callback):
        # Read and upload only addressable shards. Single-device uploads cannot
        # wait for a peer's collective. Read failures are coordinated BEFORE any
        # global layout operation; never synchronize inside a callback/thread.
        arrays, error = [], None
        assignments = sharding.addressable_devices_indices_map(shape)
        groups = {}
        for device, index in assignments.items():
            key = tuple((part.start, part.stop, part.step) for part in index)
            groups.setdefault(key, (index, []))[1].append(device)
        try:
            uploaded = {}
            for index, devices in groups.values():
                value = callback(index)
                copies = [jax.device_put(value, device) for device in devices]
                # Retain host owners until transfers finish; duplicate replicas
                # reuse one read. This wait involves only single-device arrays.
                jax.block_until_ready(copies)
                uploaded.update(zip(devices, copies))
            arrays = [uploaded[d] for d in sharding.mesh.devices.flat if d in assignments]
        except Exception as exc:
            error = exc
        coordinate_error(error, "read")
        return jax.make_array_from_single_device_arrays(shape, sharding, arrays)

    def _normalize_physical_to_logical_map(
        self,
        physical_to_logical_map: np.ndarray | None,
        num_logical_experts: int,
        context: str,
    ) -> np.ndarray | None:
        if physical_to_logical_map is None:
            return None

        map_np = np.asarray(physical_to_logical_map, dtype=np.int64)
        if map_np.ndim != 1:
            raise ValueError(
                f"{context}: expected 1D physical_to_logical_map, got shape={map_np.shape}"
            )
        if map_np.size == 0:
            raise ValueError(f"{context}: physical_to_logical_map is empty")

        min_idx = int(np.min(map_np))
        max_idx = int(np.max(map_np))
        if min_idx < 0 or max_idx >= num_logical_experts:
            raise ValueError(
                f"{context}: invalid physical_to_logical_map range [{min_idx}, {max_idx}] "
                f"for num_logical_experts={num_logical_experts}"
            )

        sample = map_np[: min(10, map_np.size)].tolist()
        logger.debug(
            "%s: p2l_map physical=%d logical=%d unique=%d sample=%s",
            context,
            map_np.size,
            num_logical_experts,
            np.unique(map_np).size,
            sample,
        )
        return map_np

    def _read_tensor(self, name, info, source, sharding):
        if sharding is None:
            sharding = jax.sharding.NamedSharding(self.mesh, P())
        return self._assemble(
            info["shape"], sharding, lambda index: source.read_tensor(info["file"], name, index)
        )

    def _read_split_tensor(
        self,
        hf_key: str,
        infos: list[dict],
        file_manager: WeightSource,
        concat_axis: int,
        target_sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        """
        Read TP-split weights (e.g., Grok Attention/MLP).
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
        target_dtype = _SAFETENSORS_DTYPE_TO_JAX.get(st_dtype, jnp.float32)

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
                    chunk = file_manager.read_tensor(info["file"], hf_key, file_read_index)
                    collected_chunks.append(chunk)

            if not collected_chunks:
                return np.zeros((0,) * len(global_shape), dtype=target_dtype)

            if len(collected_chunks) == 1:
                # Perfect match (1-to-1 mapping), no copy needed
                result = collected_chunks[0]
            else:
                # Cross-file boundary (rare if TP matches), needs stitching
                result = np.concatenate(collected_chunks, axis=concat_axis)
            return _reinterpret_dtype_if_needed(result, target_dtype)

        return self._assemble(global_shape, sharding, _smart_load_slice).astype(target_dtype)

    def _read_split_experts(
        self,
        expected_hf_keys: list[str],
        weight_infos: dict[str, list[dict]],
        file_manager: WeightSource,
        concat_axis: int,
        do_transpose: bool = False,
        target_sharding: jax.sharding.NamedSharding | None = None,
        physical_to_logical_map: np.ndarray | None = None,
    ) -> jax.Array:
        """
        Lazy loader for TP-Split MOE weights (e.g., Grok MOE).
        """
        num_logical_experts = len(expected_hf_keys)
        physical_to_logical_map = self._normalize_physical_to_logical_map(
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=num_logical_experts,
            context="split_moe_loader",
        )
        num_physical_experts = (
            len(physical_to_logical_map)
            if physical_to_logical_map is not None
            else num_logical_experts
        )

        # 1. Build file intervals for each expert
        expert_file_intervals = []
        expert_global_shapes = []

        first_hf_key = expected_hf_keys[0]
        first_infos = weight_infos[first_hf_key]
        sorted_first_infos = sorted(first_infos, key=lambda x: x["file"])

        st_dtype = sorted_first_infos[0]["dtype"]
        target_dtype = _SAFETENSORS_DTYPE_TO_JAX.get(st_dtype, jnp.float32)

        for hf_key in expected_hf_keys:
            infos = weight_infos[hf_key]
            sorted_infos = sorted(infos, key=lambda x: x["file"])
            cumulative_start = 0
            file_intervals = []
            base_shape = list(sorted_infos[0]["shape"])
            for info in sorted_infos:
                shape = info["shape"]
                length = shape[concat_axis]
                start = cumulative_start
                end = start + length
                file_intervals.append((start, end, info))
                cumulative_start = end
            global_shape = list(base_shape)
            global_shape[concat_axis] = cumulative_start
            expert_file_intervals.append(file_intervals)
            expert_global_shapes.append(tuple(global_shape))

        single_expert_shape = expert_global_shapes[0]
        output_shape = single_expert_shape[::-1] if do_transpose else single_expert_shape
        stacked_shape = (num_physical_experts, *output_shape)
        sharding = target_sharding or jax.sharding.NamedSharding(self.mesh, P())

        def _load_single_expert_slice(expert_idx, inner_index):
            hf_key = expected_hf_keys[expert_idx]
            if do_transpose:
                inner_index = inner_index[::-1]
            file_intervals = expert_file_intervals[expert_idx]
            expert_shape = expert_global_shapes[expert_idx]
            slice_on_axis = inner_index[concat_axis]
            req_start, req_stop, req_step = slice_on_axis.indices(expert_shape[concat_axis])
            assert req_step == 1
            collected_chunks = []
            for f_start, f_end, info in file_intervals:
                intersect_start = max(req_start, f_start)
                intersect_end = min(req_stop, f_end)
                if intersect_start < intersect_end:
                    file_read_index = list(inner_index)
                    file_read_index[concat_axis] = slice(
                        intersect_start - f_start, intersect_end - f_start
                    )
                    chunk = file_manager.read_tensor(info["file"], hf_key, tuple(file_read_index))
                    collected_chunks.append(chunk)
            if not collected_chunks:
                return np.zeros((0,) * len(expert_shape), dtype=target_dtype)
            if len(collected_chunks) > 1:
                result = np.concatenate(collected_chunks, axis=concat_axis)
            else:
                result = collected_chunks[0]
            result = _reinterpret_dtype_if_needed(result, target_dtype)
            return result.T if do_transpose else result

        MAX_WORKERS = int(os.environ.get("SGLANG_MOE_LOAD_WORKERS", "16"))

        def _load_stacked_slice(index):
            expert_slice = index[0]
            inner_slice = index[1:]
            start, stop, step = expert_slice.indices(num_physical_experts)
            physical_indices = list(range(start, stop, step))
            if not physical_indices:
                return np.zeros((0, *[1] * len(inner_slice)), dtype=target_dtype)

            if physical_to_logical_map is not None:
                logical_indices = [int(physical_to_logical_map[p]) for p in physical_indices]
                if physical_indices[0] == 0:
                    sample_size = min(10, len(physical_indices))
                    sample_map = {
                        p: logical_indices[i] for i, p in enumerate(physical_indices[:sample_size])
                    }
                    logger.debug("Cloning split-experts map (sample): %s", sample_map)
            else:
                logical_indices = physical_indices

            # Build task list: (logical_idx, list of physical positions that need it)
            logical_to_positions: dict[int, list[int]] = {}
            for phys_pos, log_idx in enumerate(logical_indices):
                if log_idx not in logical_to_positions:
                    logical_to_positions[log_idx] = []
                logical_to_positions[log_idx].append(phys_pos)

            inner_shape = tuple(
                len(range(*sl.indices(dim))) for sl, dim in zip(inner_slice, output_shape)
            )
            expert_bytes = math.prod(inner_shape) * np.dtype(target_dtype).itemsize
            output_bytes = len(physical_indices) * expert_bytes
            budget = int(os.environ.get("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", str(4 << 30)))
            # Split fragments and concatenation can coexist for each worker.
            workers = min(
                MAX_WORKERS,
                len(logical_to_positions),
                (budget - output_bytes) // max(1, 2 * expert_bytes),
            )
            if workers < 1:
                raise ValueError(
                    f"Split expert shard needs {output_bytes + 2 * expert_bytes} host bytes, budget={budget}"
                )
            # Pre-load first expert to determine shape
            first_log_idx = logical_indices[0]
            first_data = _load_single_expert_slice(first_log_idx, tuple(inner_slice))

            out_array = np.empty((len(physical_indices), *first_data.shape), dtype=target_dtype)
            for pos in logical_to_positions[first_log_idx]:
                out_array[pos] = first_data
            del first_data

            # Load remaining unique experts in parallel and fill positions
            remaining_logical = [
                log_idx for log_idx in logical_to_positions if log_idx != first_log_idx
            ]

            def load_and_fill_expert(log_idx):
                data = _load_single_expert_slice(log_idx, tuple(inner_slice))
                for pos in logical_to_positions[log_idx]:
                    out_array[pos] = data

            if remaining_logical:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(load_and_fill_expert, remaining_logical))

            return out_array

        result = self._assemble(stacked_shape, sharding, _load_stacked_slice)
        if result.dtype != target_dtype:
            result = result.astype(target_dtype)
        return result

    def _read_experts(
        self,
        expected_hf_keys,
        weight_info,
        file_manager,
        do_transpose=False,
        target_sharding=None,
        physical_to_logical_map=None,
    ):
        infos = [weight_info[key][0] for key in expected_hf_keys]
        shape, storage_dtype = tuple(infos[0]["shape"]), infos[0]["dtype"]
        if any(tuple(info["shape"]) != shape or info["dtype"] != storage_dtype for info in infos):
            raise ValueError("All experts in a loading group must have the same shape and dtype")
        dtype = _SAFETENSORS_DTYPE_TO_JAX[storage_dtype]
        placement = self._normalize_physical_to_logical_map(
            physical_to_logical_map, len(infos), "expert group"
        )
        if placement is None:
            placement = np.arange(len(infos))
        sharding = target_sharding or jax.sharding.NamedSharding(self.mesh, P())

        def axis_size(axis):
            names = (axis,) if isinstance(axis, str) else axis or ()
            return math.prod(sharding.mesh.shape[name] for name in names)

        unsharded = all(axis_size(axis) == 1 for axis in sharding.spec[1:])
        deferred = do_transpose and len(shape) == 2 and unsharded
        final_shape = (
            (*shape[:-2], shape[-1], shape[-2]) if do_transpose and not deferred else shape
        )
        stacked_shape = (len(placement), *final_shape)
        # Mapping specs may include singleton axes introduced by scale reshape.
        read_sharding = jax.sharding.NamedSharding(
            sharding.mesh, P(*sharding.spec[: len(stacked_shape)])
        )
        expert_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        budget = int(os.environ.get("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", str(4 << 30)))
        workers = int(os.environ.get("SGLANG_MOE_LOAD_WORKERS", "16"))
        use_bulk = (
            unsharded
            and (not do_transpose or deferred)
            and (
                file_manager.prefers_bulk
                or (
                    (deferred or os.environ.get("SGLANG_MOE_BULK_READ") == "1")
                    and expert_bytes >= 1 << 20
                )
            )
        )
        use_bulk = use_bulk and all("byte_offset" in info for info in infos)

        def read_expert(logical, index):
            info = infos[logical]
            source_index = index[::-1] if do_transpose and not deferred else index
            value = file_manager.read_tensor(info["file"], expected_hf_keys[logical], source_index)
            value = _reinterpret_dtype_if_needed(value, dtype)
            return value.T if do_transpose and not deferred else value

        if not use_bulk:

            def callback(index):
                physical = list(range(*index[0].indices(len(placement))))
                logical = [int(placement[i]) for i in physical]
                unique = sorted(set(logical))
                inner_shape = tuple(
                    len(range(*sl.indices(dim))) for sl, dim in zip(index[1:], final_shape)
                )
                required = (
                    (len(physical) + min(workers, len(unique)))
                    * math.prod(inner_shape)
                    * np.dtype(dtype).itemsize
                )
                if required > budget:
                    raise ValueError(f"Expert shard needs {required} host bytes, budget={budget}")
                output = np.empty((len(physical), *inner_shape), dtype=dtype)
                positions = {
                    key: [i for i, value in enumerate(logical) if value == key] for key in unique
                }

                def fill(key):
                    value = read_expert(key, index[1:])
                    for position in positions[key]:
                        output[position] = value

                with ThreadPoolExecutor(max_workers=min(workers, max(1, len(unique)))) as executor:
                    list(executor.map(fill, unique))
                return output

            result = self._assemble(stacked_shape, read_sharding, callback)
        else:
            error = None
            try:
                assignments = read_sharding.addressable_devices_indices_map(stacked_shape)
                devices = [d for d in read_sharding.mesh.devices.flat if d in assignments]
                physical = {
                    dev: list(range(*assignments[dev][0].indices(len(placement))))
                    for dev in devices
                }

                def estimate(batch):
                    unique = {int(placement[i]) for dev in batch for i in physical[dev]}
                    # Owned input copies, merged range scratch, assembly and H2D owners.
                    return (3 * len(unique) + sum(len(physical[d]) for d in batch)) * expert_bytes

                batches, batch = [], []
                for dev in devices:
                    if estimate([dev]) > budget:
                        raise ValueError(
                            f"Expert shard working set {estimate([dev])} exceeds host budget {budget}"
                        )
                    if batch and estimate([*batch, dev]) > budget:
                        batches.append(batch)
                        batch = []
                    batch.append(dev)
                if batch:
                    batches.append(batch)
                arrays = []
                for batch in batches:
                    logical = sorted({int(placement[i]) for dev in batch for i in physical[dev]})
                    raw = file_manager.read_ranges(
                        [(infos[i]["file"], infos[i]["byte_offset"], expert_bytes) for i in logical]
                    )
                    values = {i: value.view(dtype).reshape(shape) for i, value in zip(logical, raw)}
                    # Keep owners until every local device_put has completed. These
                    # waits cannot depend on another process's collective submission.
                    owners = [
                        np.stack([values[int(placement[i])] for i in physical[dev]])
                        for dev in batch
                    ]
                    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                        uploaded = list(
                            executor.map(lambda pair: jax.device_put(*pair), zip(owners, batch))
                        )
                    jax.block_until_ready(uploaded)
                    arrays.extend(uploaded)
            except Exception as exc:
                error = exc
            coordinate_error(error, "expert read")
            result = jax.make_array_from_single_device_arrays(stacked_shape, read_sharding, arrays)
        if deferred:
            with jax.set_mesh(sharding.mesh):
                result = jnp.transpose(result, (0, 2, 1))
                result = jax.sharding.reshard(result, sharding)
        return result
