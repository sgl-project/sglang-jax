"""Encoder-agnostic lane packing for the in-model multimodal contract.

Every in-model encoder (vision, audio, ...) turns a list of items into the
packed ``[num_lanes, cap, ...]`` layout that :class:`PackedMultimodalEmbedding`
requires: items are balanced across a fixed number of lanes (one per encoder
device), each lane is zero-padded to a bucketed capacity, and every item records
where its output lands via ``placements[i] = Placement(lane, offset, length)``.

That bookkeeping is identical across modalities; only a few knobs vary (how long
an item is, how many output tokens it produces, and the bucket ladder).  This
module owns the mechanical part so each encoder supplies just those knobs and
keeps its own metadata packing.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import (
    PackedMultimodalEmbedding,
    Placement,
)


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
    if array.sharding == spec:
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
    """Host-side packing result, before the features are encoded.

    ``lanes`` is exposed so the encoder can pack its own (modality-specific)
    per-lane metadata against the same lane assignment -- keeping a single source
    of truth for which item lives in which lane.
    """

    features: np.ndarray  # [num_lanes, cap, *feature_shape]
    valid: np.ndarray  # [num_lanes], filled input length per lane
    placements: tuple[Placement, ...]
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
    feature_of: Callable[[MultimodalDataItem], ArrayLike] = lambda item: item.feature,
    dtype: np.dtype | type = np.float32,
) -> PackedLanes:
    # I extracted the pack from qwen2.5VL and qwen3VL into a function,
    # but this might not be a good practice.
    # Gemma4 has a fixed shape, so there is clearly a better approach.
    features_np = [np.asarray(feature_of(item)) for item in items]
    lengths = [feature.shape[0] for feature in features_np]
    lanes = balance_lanes(lengths, num_lanes)
    lane_loads = [sum(lengths[index] for index in lane) for lane in lanes]
    cap = _bucket_capacity(max(lane_loads), buckets, merge_unit)
    features = np.zeros((num_lanes, cap, *features_np[0].shape[1:]), dtype=dtype)
    valid = np.zeros(num_lanes, dtype=np.int32)
    placements: list[Placement | None] = [None] * len(items)

    def fill_lane(lane_index: int) -> None:
        input_offset = 0
        output_offset = 0
        for item_index in lanes[lane_index]:
            feature = features_np[item_index]
            end = input_offset + feature.shape[0]
            features[lane_index, input_offset:end] = feature
            out_len = feature.shape[0] // merge_unit
            placements[item_index] = Placement(lane_index, output_offset, out_len)
            input_offset = end
            output_offset += out_len
        valid[lane_index] = input_offset

    for lane_index in range(num_lanes):
        fill_lane(lane_index)
    assert all(placement is not None for placement in placements)
    return PackedLanes(features, valid, tuple(placements), lanes, cap)


def pack_batch(
    items: list[MultimodalDataItem],
    num_lanes: int,
    *,
    buckets: tuple[int, ...],
    merge_unit: int,
    put_batch: Callable[[np.ndarray], jax.Array],
    pack_metadata: Callable[[list[MultimodalDataItem]], Any] | None = None,
    empty_metadata: Callable[[int], Any] | None = None,
    pad_metadata: Callable[[Any, int], Any] | None = None,
    feature_of: Callable[[MultimodalDataItem], ArrayLike] = lambda item: item.feature,
    dtype: np.dtype | type = np.float32,
):
    """Pack items into lanes and move features (+ per-lane metadata) to device.

    This is the device-side template shared by every in-model encoder's
    ``_batch_items``: it balances/buckets via :func:`pack_lanes`, kicks the
    (largest) features H2D transfer off *before* building metadata so the copy
    overlaps the metadata CPU work, then pads each lane's metadata to ``cap`` and
    stacks it into a ``[num_lanes, ...]`` pytree.

    Encoders supply only their knobs: ``merge_unit``, the ``put_batch`` device
    placement, and (for metadata-bearing encoders) the ``pack_metadata`` /
    ``empty_metadata`` / ``pad_metadata`` trio -- the same per-lane metadata is
    packed against ``pack_lanes``' lane assignment, keeping one source of truth
    for which item lives in which lane.

    Returns ``(features, metadata, valid, placements)`` when the metadata
    callbacks are given, else ``(features, valid, placements)`` for encoders that
    carry no per-lane metadata (e.g. audio codes).
    """
    packed = pack_lanes(
        items,
        num_lanes,
        buckets=buckets,
        merge_unit=merge_unit,
        feature_of=feature_of,
        dtype=dtype,
    )
    features = put_batch(packed.features)
    if pack_metadata is None:
        return features, put_batch(packed.valid), packed.placements

    dummy_metadata = pad_metadata(empty_metadata(packed.cap), packed.cap)
    metadata = [dummy_metadata] * len(packed.lanes)
    for lane_index, lane in enumerate(packed.lanes):
        if not lane:
            continue
        lane_items = [items[index] for index in lane]
        metadata[lane_index] = pad_metadata(pack_metadata(lane_items), packed.cap)
    metadata = jax.tree.map(lambda *values: np.stack(values), *metadata)
    return (
        features,
        jax.tree.map(put_batch, metadata),
        put_batch(packed.valid),
        packed.placements,
    )


def to_packed_embedding(
    output: jax.Array,
    placements: tuple[Placement, ...],
    mesh: Mesh | None,
    deepstack_dim: int = 0,
) -> PackedMultimodalEmbedding:
    """Wrap an encoder's packed output in :class:`PackedMultimodalEmbedding`.

    ``output`` must be in the packed ``[num_lanes, cap, F]`` layout produced by
    :func:`pack_lanes` (``F == (1 + deepstack_dim) * H``, deepstack planes
    already concatenated onto the trailing axis) -- the leading two dims are read
    back as ``num_lanes`` and ``cap``, and ``placements`` indexes into it. When
    ``mesh`` is set, the output is replicated across the mesh before wrapping.
    """
    if mesh is not None:
        with jax.set_mesh(mesh):
            output = replicate_across_mesh(output, mesh)
    num_lanes, cap = int(output.shape[0]), int(output.shape[1])
    return PackedMultimodalEmbedding(output, placements, num_lanes, cap, deepstack_dim)
