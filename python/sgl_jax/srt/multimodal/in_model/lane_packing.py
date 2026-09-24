"""Lane packing and output restoration for multimodal encoders."""

from __future__ import annotations

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numba import njit, types

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import VisionInputSpec
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.jax_utils import canonicalize_sharding


def get_grid_thw(item: MultimodalDataItem) -> tuple[int, int, int]:
    value = item.get("image_grid_thw")
    if value is None:
        value = item.get("video_grid_thw")
    return tuple(int(entry) for entry in np.asarray(value).reshape(3))


def _validate_vision_items(items: list[MultimodalDataItem], merge_unit: int) -> None:
    for item_index, item in enumerate(items):
        feature = item.feature
        # When the Processor creates an Item, if the feature is None, it should not create the Item.
        if feature is None or feature.ndim == 0:
            raise ValueError(f"Vision item {item_index} feature must have a patch dimension.")

        feature_patches = int(feature.shape[0])
        grid_patches = math.prod(get_grid_thw(item))
        placeholder_patches = (
            sum(end - start for start, end in item.placeholder_ranges or ()) * merge_unit
        )
        if not feature_patches == grid_patches == placeholder_patches:
            raise ValueError(
                f"Vision item {item_index} patch counts must match: "
                f"feature rows={feature_patches}, grid_thw product={grid_patches}, "
                f"placeholder tokens * merge_unit={placeholder_patches}."
            )


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


def _bucket_capacity(length: int, unit: int) -> int:
    """Round the number of merge units up to a power of two."""
    if length < 0 or unit <= 0:
        raise ValueError("patch count must be nonnegative and merge unit positive")
    units = max(1, (length + unit - 1) // unit)
    return (1 << (units - 1).bit_length()) * unit


# Sizes are runtime values; compile once at import, before serving requests.
@njit(
    types.int32[::1](
        types.Array(types.int32, 1, "C", readonly=True),
        types.Array(types.int32, 1, "C", readonly=True),
        types.int32,
        types.int32,
    ),
    nogil=True,
    cache=True,
)
def _build_output_indices(lengths, output_starts, output_size, merge_unit):
    output_indices = np.full(output_size, -1, dtype=np.int32)
    cursor = 0
    for item_index in range(lengths.size):
        output_len = lengths[item_index] // merge_unit
        source_start = output_starts[item_index]
        for index in range(output_len):
            output_indices[cursor + index] = source_start + index
        cursor += output_len
    return output_indices


def pack_lanes(
    items_per_lane: list[list[MultimodalDataItem]],
    *,
    merge_unit: int,
    input_sharding: NamedSharding,
    dtype: np.dtype | type | None = None,
) -> tuple[jax.Array, jax.Array, list[list[int]]]:

    num_lanes = len(items_per_lane)
    items = [item for lane in items_per_lane for item in lane]
    lanes = []
    offset = 0
    for lane in items_per_lane:
        lanes.append(list(range(offset, offset + len(lane))))
        offset += len(lane)
    with jax.profiler.TraceAnnotation("encoder_pack_layout"):
        if not items:
            raise ValueError("cannot pack an empty multimodal batch")
        item_features = [np.asarray(item.feature) for item in items]
        lengths = np.asarray([feature.shape[0] for feature in item_features], dtype=np.int32)
        lane_loads = [sum(int(lengths[index]) for index in lane) for lane in lanes]
        cap = _bucket_capacity(max(lane_loads), merge_unit)
        feature_shape = item_features[0].shape[1:]
        if dtype is None:
            dtype = np.result_type(*(feature.dtype for feature in item_features))

    with jax.profiler.TraceAnnotation("encoder_pack_allocate"):
        features = np.empty((num_lanes, cap, *feature_shape), dtype=dtype)
        output_cap = cap // merge_unit
        output_starts = np.empty(len(items), dtype=np.int32)

    with jax.profiler.TraceAnnotation("encoder_pack_copy"):
        for lane_index, lane in enumerate(lanes):
            input_offset = output_offset = 0
            for item_index in lane:
                feature = item_features[item_index]
                end = input_offset + feature.shape[0]
                features[lane_index, input_offset:end] = feature
                output_starts[item_index] = lane_index * output_cap + output_offset
                input_offset = end
                output_offset += feature.shape[0] // merge_unit

    with jax.profiler.TraceAnnotation("encoder_pack_output_indices"):
        output_indices = _build_output_indices(
            lengths, output_starts, num_lanes * output_cap, merge_unit
        )

    shard_shape = input_sharding.shard_shape(features.shape)
    if shard_shape[1:] != features.shape[1:]:
        raise ValueError("Lane packing requires sharding only along the lane dimension.")
    flat_sharding = canonicalize_sharding(
        input_sharding.update(spec=PartitionSpec(*input_sharding.spec[:1]))
    )
    indices_sharding = canonicalize_sharding(NamedSharding(input_sharding.mesh, PartitionSpec()))

    with jax.profiler.TraceAnnotation("encoder_pack_device_put"):
        features, output_indices = jax.device_put(
            (features.reshape(-1), output_indices), (flat_sharding, indices_sharding)
        )
    return features, output_indices, lanes


def pack_vision_inputs(
    items_per_lane: list[list[MultimodalDataItem]],
    *,
    merge_unit: int,
    input_sharding: NamedSharding,
    dtype: np.dtype | type | None = None,
) -> tuple[jax.Array, jax.Array, np.ndarray]:
    num_lanes = len(items_per_lane)
    items = [item for lane in items_per_lane for item in lane]
    with jax.profiler.TraceAnnotation("encoder_pack_validate"):
        _validate_vision_items(items, merge_unit)
    features, output_indices, lanes = pack_lanes(
        items_per_lane,
        merge_unit=merge_unit,
        input_sharding=input_sharding,
        dtype=dtype,
    )
    with jax.profiler.TraceAnnotation("encoder_pack_lane_metadata"):
        grid_thw = np.zeros(
            (num_lanes, max(map(len, lanes)), 3),
            dtype=np.int32,
        )
        for lane_index, lane in enumerate(lanes):
            for item_offset, item_index in enumerate(lane):
                grid_thw[lane_index, item_offset] = get_grid_thw(items[item_index])
    return features, output_indices, grid_thw


def pack_2d_position_inputs(
    items_per_lane: list[list[MultimodalDataItem]],
    *,
    merge_unit: int,
    input_sharding: NamedSharding,
    dtype: np.dtype | type | None = None,
) -> tuple[jax.Array, jax.Array, np.ndarray, np.ndarray]:
    """Pack vision inputs that carry explicit per-patch 2D positions."""
    num_lanes = len(items_per_lane)
    items = [item for lane in items_per_lane for item in lane]
    item_positions = []
    for item_index, item in enumerate(items):
        positions_value = item.get("pixel_position_ids")
        if positions_value is None:
            raise ValueError(f"Vision item {item_index} is missing pixel_position_ids.")
        positions = np.asarray(positions_value, dtype=np.int32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"Vision item {item_index} pixel_position_ids must have shape "
                f"[patches, 2], got {positions.shape}."
            )
        feature = item.feature
        if feature is None or feature.ndim == 0:
            raise ValueError(f"Vision item {item_index} feature must have a patch dimension.")
        item_length = int(feature.shape[0])
        if len(positions) != item_length:
            raise ValueError(
                f"Vision item {item_index} patch and position counts must match: "
                f"feature rows={item_length}, position rows={len(positions)}."
            )
        if item_length % merge_unit:
            raise ValueError(
                f"Vision item {item_index} patch count {item_length} must be divisible "
                f"by merge unit {merge_unit}."
            )
        if np.any(positions < 0):
            raise ValueError(f"Vision item {item_index} pixel_position_ids must be non-negative.")
        item_positions.append(positions)

    features, output_indices, lanes = pack_lanes(
        items_per_lane,
        merge_unit=merge_unit,
        input_sharding=input_sharding,
        dtype=dtype,
    )
    capacity = output_indices.size * merge_unit // num_lanes
    position_ids = np.full(
        (num_lanes, capacity, 2),
        -1,
        dtype=np.int32,
    )
    patch_counts = np.zeros(
        (num_lanes, max(map(len, lanes))),
        dtype=np.int32,
    )
    for lane_index, lane in enumerate(lanes):
        offset = 0
        for item_offset, item_index in enumerate(lane):
            positions = item_positions[item_index]
            item_length = len(positions)
            end = offset + item_length
            position_ids[lane_index, offset:end] = positions
            patch_counts[lane_index, item_offset] = item_length
            offset = end
    return features, output_indices, position_ids, patch_counts


def _restore_input_order(
    output: jax.Array,
    output_indices: jax.Array,
    *,
    out_sharding: NamedSharding,
) -> jax.Array:
    output = output.reshape(-1, output.shape[-1])
    # Replicate lane outputs before indexing, rather than all-reducing a full
    # reordered output from every lane. Keep sharded consumers on their path.
    if out_sharding.is_fully_replicated:
        output = jax.sharding.reshard(output, out_sharding)
    mask = output_indices >= 0
    indices = jnp.maximum(output_indices, 0)
    output = output.at[indices].get(out_sharding=out_sharding)
    return jnp.where(mask[:, None], output, jnp.zeros((), output.dtype))


_restore_input_order_jit = jax.jit(_restore_input_order, static_argnames=("out_sharding",))


def restore_encoder_output(
    output: jax.Array,
    output_indices: jax.Array,
    output_sharding: NamedSharding,
) -> jax.Array:
    """Restore lane-packed output to item order.

    Args:
        output: Encoder output shaped ``[num_lanes * output_capacity, hidden_size]``.
        output_indices: Gather indices shaped ``[num_lanes * output_capacity]``;
            negative entries denote padding rows.
        output_sharding: Sharding for the restored output.

    Returns:
        An array shaped ``[num_lanes * output_capacity, hidden_size]`` in item
        order with ``output_sharding`` and padding rows zeroed.
    """
    output_sharding = canonicalize_sharding(output_sharding)
    return _restore_input_order_jit(
        output,
        output_indices,
        out_sharding=output_sharding,
    )


def run_mrope_vision_model(
    vision_model,
    items_per_lane: list[list[MultimodalDataItem]],
    *,
    mesh: Mesh,
    num_lanes: int,
    merge_unit: int,
    rope_type: Literal["rope_3d", "rope_2d", "rope_2d_packed"],
    input_sharding: NamedSharding,
    output_sharding: NamedSharding,
) -> jax.Array:
    """Pack vision items into lanes, run the encoder, and restore item order.

    Items are already assigned to lanes by the caller. Each lane is padded to
    the same bucketed patch capacity. ``rope_3d`` and ``rope_2d`` use per-item
    THW grids; ``rope_2d_packed`` uses explicit per-patch 2D positions and
    per-item patch counts. The encoder converts these inputs into its metadata
    before running on the packed patches under ``mesh``.

    Args:
        vision_model: Callable encoder exposing ``prepare_metadata``.
        items_per_lane: Vision items grouped by lane, including empty lanes.
        mesh: Device mesh used for encoder execution.
        num_lanes: Expected number of lanes in ``items_per_lane``.
        merge_unit: Number of input patches per output token.
        rope_type: Selects the positional metadata format for the encoder.
        input_sharding: Input layout, sharded only along the lane dimension.
        output_sharding: Layout of the restored encoder output.

    Returns:
        Embeddings shaped ``[num_lanes * capacity // merge_unit, hidden_size]``,
        where ``capacity`` is the selected patch capacity per lane. Valid rows
        follow the order of items flattened from ``items_per_lane``; trailing
        padding rows are zeroed, and the result uses ``output_sharding``.
    """

    if len(items_per_lane) != num_lanes:
        raise ValueError("item lane count does not match the encoder topology")
    if rope_type == "rope_2d_packed":
        patches, output_indices, position_ids, patch_counts = pack_2d_position_inputs(
            items_per_lane,
            merge_unit=merge_unit,
            input_sharding=input_sharding,
        )
        metadata_args = (position_ids, patch_counts)
    elif rope_type in ("rope_3d", "rope_2d"):
        patches, output_indices, grid_thw = pack_vision_inputs(
            items_per_lane,
            merge_unit=merge_unit,
            input_sharding=input_sharding,
        )
        metadata_args = (grid_thw,)
    else:
        raise ValueError(f"Unsupported vision RoPE type: {rope_type}")

    capacity = output_indices.size * merge_unit // num_lanes
    with jax.profiler.TraceAnnotation("encoder_build_vision_metadata"):
        metadata = vision_model.prepare_metadata(
            *metadata_args, capacity=capacity, sharding=input_sharding
        )
    with jax.set_mesh(mesh):
        with jax.profiler.TraceAnnotation("encoder_vision_dispatch"):
            output = vision_model(patches, **metadata)
        return restore_encoder_output(output, output_indices, output_sharding)


def mrope_vision_dummy_inputs(
    spec: VisionInputSpec,
    patch_paddings=None,
    *,
    num_lanes: int = 1,
    modality: Modality = Modality.IMAGE,
):
    """Yield one fake image per vision bucket; image/video share this encoder."""
    merge_size = spec.spatial_merge_size
    merge_unit = merge_size**2
    capacities = sorted(
        {
            _bucket_capacity(patches, merge_unit)
            for patches in resolve_vision_patch_buckets(patch_paddings)
        }
    )
    for capacity in capacities:
        y, x = np.indices((merge_size, capacity // merge_size), dtype=np.int32)
        item = MultimodalDataItem(
            modality=modality,
            feature=np.zeros((capacity, spec.patch_dim), dtype=spec.dtype),
            placeholder_ranges=[(0, capacity // merge_unit)],
            model_specific_data={
                "pixel_position_ids": np.stack((x, y), axis=-1).reshape(-1, 2),
                ("video_grid_thw" if modality == Modality.VIDEO else "image_grid_thw"): np.asarray(
                    (1, merge_size, capacity // merge_size), dtype=np.int32
                ),
            },
        )
        yield modality, [[item]] + [[] for _ in range(num_lanes - 1)]
