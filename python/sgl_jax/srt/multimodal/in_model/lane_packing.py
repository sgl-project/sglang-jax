"""Lane packing and output restoration for multimodal encoders."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem


class _Encoder(Protocol):
    mesh: Mesh | None

    def encode(self, *inputs: Any) -> jax.Array: ...


def get_grid_thw(item: MultimodalDataItem) -> tuple[int, int, int]:
    value = item.get("image_grid_thw")
    if value is None:
        value = item.get("video_grid_thw")
    return tuple(int(entry) for entry in np.asarray(value).reshape(3))


def put_sharded_batch(value: Any, mesh: Mesh | None, batch_axis: Any):
    if mesh is None:
        return jax.tree.map(jnp.asarray, value)
    return jax.device_put(value, NamedSharding(mesh, PartitionSpec(batch_axis)))


def encoder_num_lanes(mesh: Mesh | None, tensor_parallel: bool) -> int:
    if mesh is None:
        return 1
    data_size = int(mesh.shape.get("data", 1))
    tensor_size = int(mesh.shape.get("tensor", 1))
    return data_size * (1 if tensor_parallel else tensor_size)


@functools.cache
def _replicate_fn(mesh: Mesh, ndim: int):
    # Cache the jitted function to avoid creating a new lambda and triggering
    # redundant tracing or compilation for the same mesh and ndim.
    spec = NamedSharding(mesh, PartitionSpec())
    return jax.jit(lambda a: jax.sharding.reshard(a, spec))


def replicate_across_mesh(array: ArrayLike, mesh: Mesh) -> jax.Array:
    """Replicate an array across a mesh, returning a jax.Array with the right sharding."""
    spec = NamedSharding(mesh, PartitionSpec())
    if not isinstance(array, jax.Array):
        return jax.device_put(array, spec)
    # JAX canonicalizes a rank-N replicated output to ``P(None, ..., None)``
    # even when it was requested with ``P()``.  Compare the effective layout
    # instead of the syntactic PartitionSpec so an already-replicated encoder
    # result does not compile an identity reshard on the first real request.
    if array.sharding.is_fully_replicated and array.sharding.device_set == spec.device_set:
        return array
    return _replicate_fn(mesh, array.ndim)(array)


def balance_lanes(item_lengths: list[int] | tuple[int, ...], num_lanes: int) -> list[list[int]]:
    """Greedily balance items over ``num_lanes`` lanes by descending length."""
    lanes: list[list[int]] = [[] for _ in range(num_lanes)]
    loads = [0] * num_lanes
    for index in sorted(range(len(item_lengths)), key=lambda i: (-item_lengths[i], i)):
        lane = min(range(num_lanes), key=lambda i: (loads[i], i))
        lanes[lane].append(index)
        loads[lane] += item_lengths[index]
    return lanes


@dataclass(frozen=True)
class PackedLanes:
    features: np.ndarray  # [num_lanes, cap, *feature_shape]
    valid: np.ndarray  # [num_lanes], filled input length per lane
    output_indices: np.ndarray
    lanes: list[list[int]]
    cap: int


def _bucket_capacity(length: int, buckets: tuple[int, ...], unit: int) -> int:
    """Smallest ``unit``-aligned bucket that fits ``length`` (power-of-two fallback)."""
    return next(
        (bucket for bucket in buckets if bucket >= length and bucket % unit == 0),
        ((1 << (length - 1).bit_length()) + unit - 1) // unit * unit,
    )


def pack_lanes(
    items: list[MultimodalDataItem],
    num_lanes: int,
    *,
    buckets: tuple[int, ...],
    merge_unit: int,
    dtype: np.dtype | type = np.float32,
) -> PackedLanes:
    features_np = [np.asarray(item.feature) for item in items]
    lengths = [feature.shape[0] for feature in features_np]
    lanes = balance_lanes(lengths, num_lanes)
    lane_loads = [sum(lengths[index] for index in lane) for lane in lanes]
    cap = _bucket_capacity(max(lane_loads), buckets, merge_unit)
    features = np.zeros((num_lanes, cap, *features_np[0].shape[1:]), dtype=dtype)
    valid = np.zeros(num_lanes, dtype=np.int32)
    output_cap = cap // merge_unit
    output_starts = np.zeros(len(items), dtype=np.int32)

    for lane_index, lane in enumerate(lanes):
        input_offset = 0
        output_offset = 0
        for item_index in lane:
            feature = features_np[item_index]
            end = input_offset + feature.shape[0]
            features[lane_index, input_offset:end] = feature
            out_len = feature.shape[0] // merge_unit
            output_starts[item_index] = lane_index * output_cap + output_offset
            input_offset = end
            output_offset += out_len
        valid[lane_index] = input_offset

    output_indices = np.full(num_lanes * output_cap, -1, dtype=np.int32)
    cursor = 0
    for length, source_start in zip(lengths, output_starts, strict=True):
        output_len = length // merge_unit
        output_indices[cursor : cursor + output_len] = source_start + np.arange(
            output_len, dtype=np.int32
        )
        cursor += output_len
    return PackedLanes(features, valid, output_indices, lanes, cap)


def pack_vision_inputs(
    items: list[MultimodalDataItem],
    *,
    num_lanes: int,
    buckets: tuple[int, ...],
    merge_unit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = pack_lanes(
        items,
        num_lanes,
        buckets=buckets,
        merge_unit=merge_unit,
    )
    grid_thw = np.zeros(
        (num_lanes, max(map(len, packed.lanes)), 3),
        dtype=np.int32,
    )
    for lane_index, lane in enumerate(packed.lanes):
        for item_offset, item_index in enumerate(lane):
            grid_thw[lane_index, item_offset] = get_grid_thw(items[item_index])
    return packed.features, grid_thw, packed.output_indices


def _restore_input_order(
    output: jax.Array,
    indices: jax.Array,
    mask: jax.Array,
    *,
    out_sharding: NamedSharding | None = None,
) -> jax.Array:
    output = output.reshape(-1, output.shape[-1])
    output = (
        output[indices]
        if out_sharding is None
        else output.at[indices].get(out_sharding=out_sharding)
    )
    return jnp.where(mask[:, None], output, jnp.zeros((), output.dtype))


_restore_input_order_jit = jax.jit(_restore_input_order)


@functools.cache
def _restore_input_order_mesh_jit(mesh: Mesh):
    out_sharding = NamedSharding(mesh, PartitionSpec())
    return jax.jit(
        functools.partial(_restore_input_order, out_sharding=out_sharding),
        out_shardings=out_sharding,
    )


def restore_encoder_output(
    output: jax.Array,
    output_indices: np.ndarray,
    mesh: Mesh | None,
) -> jax.Array:
    output_indices = np.asarray(output_indices, dtype=np.int32)
    mask = output_indices >= 0
    indices = np.maximum(output_indices, 0)

    if mesh is None:
        return _restore_input_order_jit(output, indices, mask)
    return _restore_input_order_mesh_jit(mesh)(output, indices, mask)


def run_dp_sharded_encoder(
    encoder: _Encoder,
    items: list[MultimodalDataItem],
    *,
    num_lanes: int,
    buckets: tuple[int, ...],
    merge_unit: int,
) -> jax.Array:
    patches, grid_thw, output_indices = pack_vision_inputs(
        items,
        num_lanes=num_lanes,
        buckets=buckets,
        merge_unit=merge_unit,
    )
    output = encoder.encode(patches, grid_thw)
    return restore_encoder_output(output, output_indices, encoder.mesh)
