"""E-owned DeepSeek V4 MoE checkpoint loading.

M scans the checkpoint once and passes only this layer's assigned entries.
The routed-expert device assembly follows the incremental epic/dsv4 loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, SingleDeviceSharding

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    convert_mxfp4_pair_from_reader,
)

_PROJECTIONS = (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo"))
_SHARED = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))
_ITEMSIZE = {"I8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1, "F32": 4, "BF16": 2}
STATIC_EXPERT_FORMAT = "sglang-jax-deepseek-v4-expert-fp8-per-channel-v1"


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


def _read(entry: dict, dtype) -> np.ndarray:
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


def _e8m0_scale(entry: dict, key: str) -> np.ndarray:
    codes = _read(entry, np.uint8)
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
) -> MoELoadReport:
    """Validate and consume only M's E-owned inventory for this layer.

    M selects the original MXFP4 or published static expert-FP8 format.
    Both paths populate EPMoE [E,K,N] weights and [E,1,1,N] scales,
    processing one local expert at a time.
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
    gate_numpy = np.float32 if gate_dtype == "F32" else ml_dtypes.bfloat16
    _assign(
        layer.gate.kernel, _read(assigned[gate_key][0], gate_numpy).T, gate_key, mesh=layer.mesh
    )
    local_payload_keys.add(gate_key)
    if layer.is_hash_layer:
        route_numpy = np.int32 if route_dtype == "I32" else np.int64
        layer.load_hash_table(_read(assigned[route_key][0], route_numpy))
    else:
        _assign(
            layer.gate.bias, _read(assigned[route_key][0], np.float32), route_key, mesh=layer.mesh
        )
    local_payload_keys.add(route_key)

    for source, target in _SHARED:
        stem = prefix + f"shared_experts.{source}"
        linear = getattr(layer.shared_experts, target)
        weight = _read(assigned[stem + ".weight"][0], np.uint8).view(ml_dtypes.float8_e4m3fn)
        scale = _e8m0_scale(assigned[stem + ".scale"][0], stem + ".scale")
        local_payload_keys.update((stem + ".weight", stem + ".scale"))
        if layer.static_fp8:
            _assign(linear.weight_q, weight, stem + ".weight", mesh=layer.mesh)
            expanded = np.repeat(scale, 128, axis=0)[: weight.shape[0], :].T[:, None, :]
            _assign(linear.weight_scale, expanded, stem + ".scale", mesh=layer.mesh)
        else:
            block = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            dequantized = weight.astype(np.float32) * block[: weight.shape[0], : weight.shape[1]]
            _assign(linear.weight, dequantized.T, stem + ".weight", mesh=layer.mesh)

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
            if expert_format == STATIC_EXPERT_FORMAT:
                weight = _read(assigned[wk][0], ml_dtypes.float8_e4m3fn)
                scale = _read(assigned[sk][0], np.float32)
                if not np.isfinite(np.asarray(weight, np.float32)).all() or (
                    not np.isfinite(scale).all() or np.any(scale <= 0)
                ):
                    raise ValueError(f"{stem}: invalid static FP8 weight or scale")
            else:
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
