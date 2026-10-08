"""E-owned DeepSeek V4 MoE checkpoint loading.

M scans the checkpoint once and passes only this layer's assigned entries.
Published FP8 uses the shared parallel shard reader; original MXFP4 uses the
bounded incremental conversion path from epic/dsv4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, SingleDeviceSharding

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.model_loader.weights.loader import WeightLoader
from sgl_jax.srt.model_loader.weights.source import LocalSource
from sgl_jax.srt.model_loader.weights.specs import WeightSpec
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    convert_mxfp4_pair_from_reader,
)

_PROJECTIONS = (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo"))
_SHARED = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))
_ITEMSIZE = {"I8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1, "F32": 4, "BF16": 2}
STATIC_EXPERT_FORMAT = "sglang-jax-deepseek-v4-expert-fp8-per-channel-v1"
logger = logging.getLogger(__name__)


class _AssignedMoESource(LocalSource):
    """Reuse shared reads with M's validated inventory, without another scan."""

    prefers_bulk = False

    def __init__(self, assigned, source=None):
        super().__init__()
        self.source = source
        self.assigned = assigned
        self._weight_info_cache = assigned
        self.payload_keys = set()
        self.range_keys = {
            (entry["file"], entry["byte_offset"], entry["byte_size"]): key
            for key, entries in assigned.items()
            for entry in entries
        }

    @property
    def metadata(self):
        return self.assigned

    def _validate_payload(self, name, value):
        # Check FP8 in its stored dtype; do not expand full experts to FP32.
        if not np.isfinite(value).all() or (name.endswith(".scale") and np.any(value <= 0)):
            raise ValueError(f"{name}: invalid static FP8 weight or scale")

    def read_tensor(self, filename, name, index):
        if name not in self.assigned or self.assigned[name][0]["file"] != filename:
            raise ValueError(f"{name}: read outside assigned MoE inventory")
        reader = self.source if self.source is not None else super()
        value = reader.read_tensor(filename, name, index)
        self._validate_payload(name, value)
        with self._lock:
            self.payload_keys.add(name)
        return value

    def read_ranges(self, ranges):
        keys = [self.range_keys[request] for request in ranges]
        reader = self.source if self.source is not None else super()
        values = reader.read_ranges(ranges)
        for key, value in zip(keys, values):
            dtype = (
                ml_dtypes.float8_e4m3fn
                if self.assigned[key][0]["dtype"] == "F8_E4M3"
                else np.float32
            )
            self._validate_payload(key, value.view(dtype))
        self.payload_keys.update(keys)
        return values

    def release(self, filenames):
        # The framework session keeps handles reusable across layers. The public
        # loader drains transfers before this call; final close belongs to M.
        if self.source is None:
            super().release(filenames)

    def close(self):
        if self.source is None:
            super().close()

    @property
    def identity(self):
        return self.source.identity if self.source is not None else super().identity


def _load_static_routed(layer, assigned, physical_to_logical, weight_source=None):
    """Use public WeightLoader mappings for published FP8, as in epic/dsv4.

    Large EP-local weights use bounded bulk reads; small scales use cached
    safetensors handles. TensorLayout supplies the kernel's 4D scale layout.
    """
    mesh = layer.experts.moe_mesh
    routed = {key: entries for key, entries in assigned.items() if ".ffn.experts." in key}
    with _AssignedMoESource(routed, weight_source) as source:
        mappings = {}
        for projection, target in _PROJECTIONS:
            keys = tuple(
                f"layers.{layer.layer_id}.ffn.experts.{i}.{projection}"
                for i in range(layer.num_experts)
            )
            for suffix in ("weight", "scale"):
                path = target if suffix == "weight" else target + "_scale"
                spec = getattr(layer.experts, path).value.sharding.spec
                # Compact [E,N] scales must be read with their two source axes;
                # TensorLayout expands them to [E,1,1,N] after stacking.
                if suffix == "scale":
                    spec = jax.sharding.PartitionSpec(spec[0], spec[-1])
                mappings[keys[0] + "." + suffix] = WeightSpec(
                    target_path=path,
                    sharding=tuple(spec),
                    sources=tuple(key + "." + suffix for key in keys),
                    transpose=suffix == "weight",
                    physical_to_logical_map=physical_to_logical,
                )
        started = time.monotonic()
        with jax.set_mesh(mesh):
            WeightLoader(layer.experts, SimpleNamespace(), mesh, source=source).load(
                mappings, validate_checkpoint_coverage=True
            )
        jax.block_until_ready(
            tuple(
                getattr(layer.experts, path).value
                for _, target in _PROJECTIONS
                for path in (target, target + "_scale")
            )
        )
        logger.info(
            "Loaded DeepSeek V4 layer %d static FP8 experts via WeightLoader in %.1fs",
            layer.layer_id + 1,
            time.monotonic() - started,
        )
        return source.payload_keys


@dataclass(frozen=True)
class MoELoadReport:
    # The complete E-owned partition is claimed once after metadata validation.
    consumed_keys: frozenset[str]
    # Payload bytes read by this process; EP may place other experts on remote hosts.
    local_payload_keys: frozenset[str]
    converted_pairs: int
    max_conversion_error: float
    peak_host_bytes_calculated: int
    peak_memory_method: str = "converter overlapping-array allocation account; not RSS or HBM"


def expected_moe_keys(layer) -> set[str]:
    """The exact E-owned source partition for one backbone layer."""
    prefix = f"layers.{layer.layer_id}.ffn."
    keys = {prefix + "gate.weight"}
    keys.add(prefix + ("gate.tid2eid" if layer.is_hash_layer else "gate.bias"))
    for source, _ in _SHARED:
        keys.update(
            (prefix + f"shared_experts.{source}.weight", prefix + f"shared_experts.{source}.scale")
        )
    for expert_id in range(layer.num_experts):
        for source, _ in _PROJECTIONS:
            stem = prefix + f"experts.{expert_id}.{source}"
            keys.update((stem + ".weight", stem + ".scale"))
    return keys


def _entry(assigned: dict[str, list[dict]], key: str, dtype: str, shape: tuple[int, ...]) -> dict:
    entries = assigned[key]
    if len(entries) != 1:
        raise ValueError(f"{key}: expected exactly one source tensor")
    entry = entries[0]
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * _ITEMSIZE[dtype]
    if entry.get("dtype") != dtype or tuple(entry.get("shape", ())) != shape:
        raise ValueError(
            f"{key}: expected {dtype} {shape}, got {entry.get('dtype')} {entry.get('shape')}"
        )
    if (
        entry.get("byte_size") != expected_bytes
        or not isinstance(entry.get("byte_offset"), int)
        or entry["byte_offset"] < 0
    ):
        raise ValueError(f"{key}: invalid byte offset or byte count")
    if not entry.get("file"):
        raise ValueError(f"{key}: missing source file")
    return entry


def _read(entry: dict, dtype, *, source=None, key=None) -> np.ndarray:
    if source is not None:
        value = source.read_tensor(entry["file"], key, slice(None))
        return np.asarray(value).view(dtype).reshape(entry["shape"])
    with open(entry["file"], "rb") as source:
        source.seek(entry["byte_offset"])
        raw = source.read(entry["byte_size"])
    if len(raw) != entry["byte_size"]:
        raise ValueError(f"{entry['file']}: truncated tensor payload")
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def _read_rows(source, entry: dict, rows: slice) -> np.ndarray:
    start, stop, step = rows.indices(entry["shape"][0])
    if step != 1:
        raise ValueError("MXFP4 row reads require a unit step")
    columns = entry["shape"][1]
    source.seek(entry["byte_offset"] + start * columns)
    raw = source.read((stop - start) * columns)
    if len(raw) != (stop - start) * columns:
        raise ValueError(f"{entry['file']}: truncated tensor payload")
    return np.frombuffer(raw, dtype=np.uint8).reshape(stop - start, columns)


def _assign(param, value, key: str, *, mesh: jax.sharding.Mesh) -> None:
    old = param.value
    if value.shape != old.shape:
        raise ValueError(f"{key}: decoded shape {value.shape} != parameter {old.shape}")
    if not np.isfinite(np.asarray(value, np.float32)).all():
        raise ValueError(f"{key}: non-finite checkpoint value")
    # JAXModelLoader builds the graph with nnx.eval_shape. Its prototype
    # sharding retains the partition spec but has no addressable devices.
    sharding = NamedSharding(mesh, old.sharding.spec)
    param.value = jax.device_put(np.asarray(value, dtype=old.dtype), sharding)
    param.value.block_until_ready()


def _e8m0_scale(entry: dict, key: str, *, source=None) -> np.ndarray:
    codes = _read(entry, np.uint8, source=source, key=key)
    if np.any(codes == 255):
        raise ValueError(f"{key}: reserved F8_E8M0 scale code 255")
    values = np.ldexp(np.ones(codes.shape, np.float32), codes.astype(np.int16) - 127)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{key}: invalid F8_E8M0 scale")
    return values


def _on_device(call, device, *args):
    local_mesh = jax.sharding.Mesh(
        np.asarray([device]), ("loader",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(local_mesh):
        return call(*args)


def _empty_shards(param, mesh):
    old = param.value
    sharding = NamedSharding(mesh, old.sharding.spec)
    shards = {}
    for device, index in sharding.addressable_devices_indices_map(old.shape).items():
        local_shape = tuple(len(range(*sl.indices(n))) for sl, n in zip(index, old.shape))
        placement = SingleDeviceSharding(device)
        initialize = jax.jit(
            lambda shape=local_shape, dtype=old.dtype: jnp.zeros(shape, dtype),
            out_shardings=placement,
        )
        update = jax.jit(
            lambda buffer, row, offset: jax.lax.dynamic_update_slice(
                buffer, row[None], (offset,) + (0,) * (buffer.ndim - 1)
            ),
            donate_argnums=(0,),
            out_shardings=placement,
        )
        shards[device] = [index, _on_device(initialize, device), update]
    return sharding, shards


def _physical_to_logical_for_layer(locations, layer_id: int) -> np.ndarray:
    mapping = locations.physical_to_logical_map
    if not getattr(mapping, "is_fully_addressable", True):
        if not mapping.sharding.is_fully_replicated:
            raise ValueError("V4 expert placement must be replicated across hosts")
        # ExpertLocationMetadata stores this map with P(None). Each host can
        # read its complete local replica without fetching the global array.
        mapping = mapping.addressable_data(0)
    return np.asarray(mapping[layer_id], dtype=np.int32)


def load_moe_weights(
    layer,
    assigned: dict[str, list[dict]],
    *,
    expert_format: str | None = None,
    row_chunk_size: int = 128,
    weight_source=None,
) -> MoELoadReport:
    """Validate and consume only M's E-owned inventory for this layer.

    M selects the original MXFP4 or published static expert-FP8 format.
    Both paths populate EPMoE [E,K,N] weights and [E,1,1,N] scales,
    using public FP8 assembly or incremental MXFP4 conversion.
    """
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")
    if expert_format not in (None, STATIC_EXPERT_FORMAT):
        raise ValueError(f"unsupported V4 expert format: {expert_format}")
    if expert_format == STATIC_EXPERT_FORMAT and not layer.static_fp8:
        raise ValueError("static V4 expert format requires static FP8 shared-expert parameters")
    expected = expected_moe_keys(layer)
    if set(assigned) != expected:
        missing, unexpected = expected - set(assigned), set(assigned) - expected
        raise ValueError(
            f"MoE inventory mismatch: missing={sorted(missing)[:4]}, unexpected={sorted(unexpected)[:4]}"
        )
    if layer.experts.replicate_experts:
        raise ValueError("V4 FP8 loading does not support replicated experts")
    locations = get_global_expert_location_metadata()
    physical_to_logical = (
        np.arange(layer.num_experts, dtype=np.int32)
        if locations is None
        else _physical_to_logical_for_layer(locations, layer.layer_id)
    )
    if physical_to_logical.shape != (layer.experts.num_experts,) or (
        np.any(physical_to_logical < 0)
        or np.any(physical_to_logical >= layer.num_experts)
        or set(physical_to_logical.tolist()) != set(range(layer.num_experts))
    ):
        raise ValueError("V4 physical-to-logical expert placement is incomplete or invalid")

    prefix = f"layers.{layer.layer_id}.ffn."
    local_payload_keys = set()
    gate_key = prefix + "gate.weight"
    gate_dtype = assigned[gate_key][0].get("dtype") if len(assigned[gate_key]) == 1 else None
    if gate_dtype not in ("F32", "BF16"):
        raise ValueError(f"{gate_key}: unsupported gate dtype {gate_dtype}")
    _entry(assigned, gate_key, gate_dtype, (layer.num_experts, layer.hidden_size))
    if layer.is_hash_layer:
        route_key = prefix + "gate.tid2eid"
        route_dtype = assigned[route_key][0].get("dtype") if len(assigned[route_key]) == 1 else None
        if route_dtype not in ("I32", "I64"):
            raise ValueError(f"{route_key}: routing table must be integer")
        _entry(assigned, route_key, route_dtype, (layer.vocab_size, layer.top_k))
    else:
        route_key = prefix + "gate.bias"
        _entry(assigned, route_key, "F32", (layer.num_experts,))

    for source, target in _SHARED:
        stem = prefix + f"shared_experts.{source}"
        linear = getattr(layer.shared_experts, target)
        if layer.static_fp8:
            out_size, in_size = linear.weight_q.value.shape
            if linear.weight_q.value.dtype != jnp.float8_e4m3fn or (
                linear.weight_scale.value.shape != (in_size // 128, 1, out_size)
                or linear.weight_scale.value.dtype != jnp.float32
            ):
                raise ValueError(f"{stem}: incompatible static shared-expert parameters")
        else:
            in_size, out_size = linear.weight.value.shape
        _entry(assigned, stem + ".weight", "F8_E4M3", (out_size, in_size))
        _entry(
            assigned, stem + ".scale", "F8_E8M0", ((out_size + 127) // 128, (in_size + 127) // 128)
        )
    for source, target in _PROJECTIONS:
        parameter = getattr(layer.experts, target).value
        if parameter.dtype != jnp.float8_e4m3fn:
            raise ValueError("routed experts must have resident E4M3FN weights")
        in_size, out_size = parameter.shape[1:]
        if in_size % 32:
            raise ValueError(f"{source}: K must be divisible by 32")
        scale = getattr(layer.experts, target + "_scale").value
        if scale.shape != (parameter.shape[0], 1, 1, out_size) or scale.dtype != jnp.float32:
            raise ValueError(f"{source}: invalid resident scale shape or dtype")
        for expert_id in range(layer.num_experts):
            stem = prefix + f"experts.{expert_id}.{source}"
            if expert_format == STATIC_EXPERT_FORMAT:
                _entry(assigned, stem + ".weight", "F8_E4M3", (out_size, in_size))
                _entry(assigned, stem + ".scale", "F32", (out_size,))
            else:
                _entry(assigned, stem + ".weight", "I8", (out_size, in_size // 2))
                _entry(assigned, stem + ".scale", "F8_E8M0", (out_size, in_size // 32))

    # Ordinary and shared projections are read one at a time. Routed
    # projections below are converted by row chunk and local expert.
    started = time.monotonic()
    gate_numpy = np.float32 if gate_dtype == "F32" else ml_dtypes.bfloat16
    _assign(
        layer.gate.kernel,
        _read(assigned[gate_key][0], gate_numpy, source=weight_source, key=gate_key).T,
        gate_key,
        mesh=layer.mesh,
    )
    local_payload_keys.add(gate_key)
    if layer.is_hash_layer:
        route_numpy = np.int32 if route_dtype == "I32" else np.int64
        layer.load_hash_table(
            _read(assigned[route_key][0], route_numpy, source=weight_source, key=route_key)
        )
    else:
        _assign(
            layer.gate.bias,
            _read(assigned[route_key][0], np.float32, source=weight_source, key=route_key),
            route_key,
            mesh=layer.mesh,
        )
    local_payload_keys.add(route_key)
    logger.info(
        "Loaded DeepSeek V4 layer %d MoE gate/routing in %.1fs",
        layer.layer_id + 1,
        time.monotonic() - started,
    )

    started = time.monotonic()
    for source, target in _SHARED:
        stem = prefix + f"shared_experts.{source}"
        linear = getattr(layer.shared_experts, target)
        weight = _read(
            assigned[stem + ".weight"][0], np.uint8, source=weight_source, key=stem + ".weight"
        ).view(ml_dtypes.float8_e4m3fn)
        scale = _e8m0_scale(assigned[stem + ".scale"][0], stem + ".scale", source=weight_source)
        local_payload_keys.update((stem + ".weight", stem + ".scale"))
        if layer.static_fp8:
            _assign(linear.weight_q, weight, stem + ".weight", mesh=layer.mesh)
            expanded = np.repeat(scale, 128, axis=0)[: weight.shape[0], :].T[:, None, :]
            _assign(linear.weight_scale, expanded, stem + ".scale", mesh=layer.mesh)
        else:
            block = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            dequantized = weight.astype(np.float32) * block[: weight.shape[0], : weight.shape[1]]
            _assign(linear.weight, dequantized.T, stem + ".weight", mesh=layer.mesh)
    logger.info(
        "Loaded DeepSeek V4 layer %d shared experts in %.1fs",
        layer.layer_id + 1,
        time.monotonic() - started,
    )

    if expert_format == STATIC_EXPERT_FORMAT:
        local_payload_keys.update(
            _load_static_routed(layer, assigned, physical_to_logical, weight_source)
        )
        return MoELoadReport(
            consumed_keys=frozenset(expected),
            local_payload_keys=frozenset(local_payload_keys),
            converted_pairs=0,
            max_conversion_error=0.0,
            peak_host_bytes_calculated=0,
            peak_memory_method="shared shard reader inflight target; no conversion allocation account",
        )

    converted_pairs = 0
    max_error = 0.0
    peak_host_bytes = 0
    for source, target in _PROJECTIONS:
        weight_param = getattr(layer.experts, target)
        scale_param = getattr(layer.experts, target + "_scale")
        buffers = [
            (param, is_scale, *_empty_shards(param, layer.experts.moe_mesh))
            for param, is_scale in ((weight_param, False), (scale_param, True))
        ]
        for physical_id, expert_id in enumerate(physical_to_logical):
            local = any(
                physical_id in range(*index[0].indices(weight_param.value.shape[0]))
                for index, _, _ in buffers[0][3].values()
            )
            if not local:
                continue
            stem = prefix + f"experts.{expert_id}.{source}"
            wk, sk = stem + ".weight", stem + ".scale"
            weight_entry, scale_entry = assigned[wk][0], assigned[sk][0]
            with (
                open(weight_entry["file"], "rb") as weight_source,
                open(scale_entry["file"], "rb") as scale_source,
            ):
                converted = convert_mxfp4_pair_from_reader(
                    weight_name=wk,
                    scale_name=sk,
                    weight_shape=weight_entry["shape"],
                    scale_shape=scale_entry["shape"],
                    weight_dtype=weight_entry["dtype"],
                    scale_dtype=scale_entry["dtype"],
                    read_weight_rows=lambda rows, source=weight_source, entry=weight_entry: _read_rows(
                        source, entry, rows
                    ),
                    read_scale_rows=lambda rows, source=scale_source, entry=scale_entry: _read_rows(
                        source, entry, rows
                    ),
                    row_chunk_size=row_chunk_size,
                    strict=True,
                )
            weight, scale = converted.weight_fp8, converted.scale_fp32
            converted_pairs += 1
            max_error = max(max_error, converted.report.max_abs_error)
            peak_host_bytes = max(peak_host_bytes, converted.report.calculated_peak_host_bytes)
            local_payload_keys.update((wk, sk))
            for param, is_scale, _, shards in buffers:
                value = scale[None, None, :] if is_scale else weight.T
                if value.shape != param.value.shape[1:]:
                    raise ValueError(
                        f"{stem}: decoded expert shape {value.shape} != {param.value.shape[1:]}"
                    )
                for device, shard in shards.items():
                    index, array, update = shard
                    owned = range(*index[0].indices(param.value.shape[0]))
                    if physical_id not in owned:
                        continue
                    placement = SingleDeviceSharding(device)
                    row = jax.device_put(
                        np.ascontiguousarray(value[index[1:]], dtype=param.value.dtype), placement
                    )
                    offset = jax.device_put(np.int32(owned.index(physical_id)), placement)
                    shard[1] = _on_device(update, device, array, row, offset)
            for _, _, _, shards in buffers:
                for _, array, _ in shards.values():
                    array.block_until_ready()
        for param, _, sharding, shards in buffers:
            param.value = jax.make_array_from_single_device_arrays(
                param.value.shape, sharding, [array for _, array, _ in shards.values()]
            )
    return MoELoadReport(
        consumed_keys=frozenset(expected),
        local_payload_keys=frozenset(local_payload_keys),
        converted_pairs=converted_pairs,
        max_conversion_error=max_error,
        peak_host_bytes_calculated=peak_host_bytes,
    )
