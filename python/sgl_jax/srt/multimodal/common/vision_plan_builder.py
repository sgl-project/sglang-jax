"""Architecture-agnostic in-model vision encode/merge planning.

A model plugs into this framework by registering a :class:`VisionEncoderPlugin`
under its HF architecture name (:func:`register_vision_encoder`). This builder
turns scheduled requests into fixed-shape ``[dp, tp]`` encoder batches and
token-merge routes, delegating only the model-specific metadata math to the
plugin; the encode-input pytree (:class:`VisionEncodeInputs`) is framework-owned
and shared across models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
    register_in_model_plan_builder,
)
from sgl_jax.srt.multimodal.common.mm_plan import (
    DeviceMergePlan,
    ModalityEmbedBatch,
    MultimodalEmbedPlan,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo


def _ceil_to_bucket(value: int, buckets: Sequence[int] | None) -> int:
    """Smallest bucket >= value; the natural value if buckets is empty or value exceeds all."""
    if buckets:
        for bucket in buckets:
            if bucket >= value:
                return bucket
    return value


@register_pytree_node_class
@dataclass
class VisionEncodeInputs:
    """Framework-owned ViT encode inputs for one ``[dp, tp]`` encode round.

    ``patches`` and ``valid`` are built by the plan builder; ``meta`` is the
    model-specific metadata pytree returned by the plugin's ``stack_metadata``.
    The model's encode body reads these three fields. Every array leaf shares
    the leading ``[dp, tp]`` axes, matching the ``encode_inputs`` pytree contract
    of ``ModalityEmbedBatch``.
    """

    patches: Any
    valid: Any
    meta: Any

    def tree_flatten(self):
        return (self.patches, self.valid, self.meta), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class VisionEncoderPlugin(ABC):
    """Base class each model subclasses to plug its vision encoder into the builder.

    The builder owns all request/lane/merge orchestration and the encode-input
    pytree; a subclass only contributes its metadata math and a couple of shape
    constants. Subclasses must set the two class attributes below and implement
    the abstract methods; missing any raises ``TypeError`` at instantiation.
    """

    # Modality under which merged encoder rows land (e.g. ``Modality.IMAGE``).
    output_modality: Modality

    # Trailing dim of one patch feature row (second axis of a real ``item.feature``).
    feature_dim: int

    @abstractmethod
    def get_metadata(self, items: Sequence[MultimodalDataItem]) -> Any:
        """Per-lane request metadata (no dp/tp axis) for ``items`` in order."""

    @abstractmethod
    def dummy_metadata(self, patch_k: int) -> Any:
        """Valid single-lane metadata filling ``patch_k`` patches, for warmup."""

    @abstractmethod
    def stack_metadata(self, lane_metadata: Sequence[Any | None], patch_k: int) -> Any:
        """Pad and stack per-lane metadata into leading ``[dp*tp, ...]`` arrays."""

    @abstractmethod
    def get_num_output_tokens(self, num_input_patches: int) -> int:
        """Return the number of output tokens produced from the input patches."""


@dataclass(frozen=True)
class MergeSlice:
    """Host-only mapping from part of one task's output to current token rows."""

    feature_start: int
    token_start: int
    length: int


@dataclass(frozen=True)
class _EncodeTask:
    task_id: int
    item: MultimodalDataItem
    encoded_rows: int
    merge_slices: tuple[MergeSlice, ...]

    @property
    def patch_rows(self) -> int:
        return np.asarray(self.item.feature).shape[0]


def _assign_tp_lanes(tasks: list[_EncodeTask], tp_size: int) -> list[list[_EncodeTask]]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if tp_size == 1:
        return [list(tasks)]

    lanes: list[list[_EncodeTask]] = [[] for _ in range(tp_size)]
    loads = [0] * tp_size
    for task in sorted(tasks, key=lambda task: (-task.patch_rows, task.task_id)):
        rank = min(range(tp_size), key=lambda candidate: (loads[candidate], candidate))
        lanes[rank].append(task)
        loads[rank] += task.patch_rows
    return lanes


def _vision_items(req) -> list[tuple[int, MultimodalDataItem]]:
    mm_inputs = req.mm_inputs
    if mm_inputs is None or isinstance(mm_inputs, dict) and "mm_items" not in mm_inputs:
        return []
    if not isinstance(mm_inputs, MultimodalInputs):
        raise TypeError(
            "vision plan builder expects req.mm_inputs to be MultimodalInputs, "
            f"got {type(mm_inputs).__name__}."
        )
    if any(not isinstance(item, MultimodalDataItem) for item in mm_inputs.mm_items):
        bad = next(item for item in mm_inputs.mm_items if not isinstance(item, MultimodalDataItem))
        raise TypeError(
            "vision plan builder expects mm_items to contain "
            f"MultimodalDataItem, got {type(bad).__name__}."
        )
    return [
        (item_index, item)
        for item_index, item in enumerate(mm_inputs.mm_items)
        if item.is_image() or item.is_video()
    ]


def _build_task(
    item: MultimodalDataItem,
    task_id: int,
    dp_rank: int,
    req_base: int,
    chunk_start: int,
    chunk_end: int,
    per_dp_token: int,
) -> _EncodeTask | None:
    feature = None if item.feature is None else np.asarray(item.feature)
    if feature is None:
        raise ValueError(f"Vision item in dp_rank {dp_rank} is missing feature.")
    if feature.ndim != 2 or not feature.shape[0]:
        raise ValueError(
            f"Vision item feature must be a non-empty 2D patch array, got shape={feature.shape} "
            f"in dp_rank {dp_rank}."
        )
    if not item.placeholder_ranges:
        raise ValueError(f"Vision item in dp_rank {dp_rank} has no placeholder ranges.")

    slices = []
    encoded_rows = 0
    previous_end = -1
    for start, end in item.placeholder_ranges:
        length = end - start
        if length <= 0:
            raise ValueError(
                f"Vision placeholder range must be non-empty, got start={start}, end={end}."
            )
        if start < previous_end:
            raise ValueError(
                "Vision placeholder token is assigned more than once: "
                f"previous_end={previous_end}, start={start}."
            )
        previous_end = end

        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            token_start = req_base + overlap_start - chunk_start
            merge_length = overlap_end - overlap_start
            if token_start < 0 or token_start + merge_length > per_dp_token:
                raise ValueError(
                    "Vision placeholder chunk is outside its packed rank slot: "
                    f"dp_rank={dp_rank}, req_base={req_base}, "
                    f"chunk=({chunk_start}, {chunk_end}), range=({start}, {end}), "
                    f"per_dp_token={per_dp_token}."
                )
            slices.append(
                MergeSlice(
                    encoded_rows + overlap_start - start,
                    token_start,
                    merge_length,
                )
            )
        encoded_rows += length

    return (
        _EncodeTask(
            task_id,
            item,
            encoded_rows,
            tuple(slices),
        )
        if slices
        else None
    )


def _collect_encode_tasks(
    reqs_info: list[ScheduleReqsInfo] | None,
    dp_size: int,
    per_dp_token: int,
) -> list[list[_EncodeTask]]:
    tasks_by_dp: list[list[_EncodeTask]] = [[] for _ in range(dp_size)]
    task_id = 0
    for dp_rank, info in enumerate((reqs_info or [])[:dp_size]):
        req_base = 0
        for req_index, req in enumerate(info.reqs or []):
            prefix_len = (
                info.prefix_lens[req_index]
                if info.prefix_lens is not None and req_index < len(info.prefix_lens)
                else len(getattr(req, "prefix_indices", []) or [])
            )
            extend_len = (
                info.extend_lens[req_index]
                if info.extend_lens is not None and req_index < len(info.extend_lens)
                else int(getattr(req, "extend_input_len", 0) or 0)
            )
            if prefix_len < 0 or extend_len < 0:
                raise ValueError(
                    "Vision chunk bounds must be non-negative, "
                    f"got prefix_len={prefix_len}, extend_len={extend_len}."
                )

            for _, item in _vision_items(req):
                task = _build_task(
                    item,
                    task_id,
                    dp_rank,
                    req_base,
                    prefix_len,
                    prefix_len + extend_len,
                    per_dp_token,
                )
                if task is not None:
                    tasks_by_dp[dp_rank].append(task)
                    task_id += 1
            req_base += extend_len
    return tasks_by_dp


def _reshape_metadata(meta, dp_size: int, tp_size: int):
    def reshape(value):
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != dp_size * tp_size:
            raise ValueError(
                "Vision metadata leaves must be lane-leading after stack_metadata: "
                f"expected first dimension={dp_size * tp_size}, got shape={value.shape}."
            )
        return value.reshape(dp_size, tp_size, *value.shape[1:])

    return jax.tree.map(reshape, meta)


def _build_encode_arrays(
    tasks_by_dp: list[list[_EncodeTask]],
    plugin: VisionEncoderPlugin,
    dp_size: int,
    tp_size: int,
    patch_buckets: Sequence[int] | None,
) -> tuple[Any, dict[int, tuple[int, int, int]], np.ndarray, int]:
    """Assign tasks to ``[dp,tp]`` encode lanes and stack their patches/metadata.

    Returns the encode-input pytree, a ``task_id -> (dp, tp, feature_base)``
    placement map (feature_base = row offset of the task within its lane's
    concatenated encode output), and the ``[dp,tp]`` per-lane valid row counts.
    """
    all_tasks = [task for tasks in tasks_by_dp for task in tasks]
    if not all_tasks:
        raise ValueError("Vision encode batch requires at least one task.")

    for task in all_tasks:
        expected_rows = plugin.get_num_output_tokens(task.patch_rows)
        if task.encoded_rows != expected_rows:
            raise ValueError(
                "Vision placeholder rows must match encoder output rows: "
                f"task_id={task.task_id}, patches={task.patch_rows}, "
                f"placeholder_rows={task.encoded_rows}, encoder_rows={expected_rows}."
            )

    lanes_by_dp = [_assign_tp_lanes(tasks, tp_size) for tasks in tasks_by_dp]
    lane_features: list[list[np.ndarray | None]] = [[None] * tp_size for _ in range(dp_size)]
    lane_metadata: list[Any] = []
    placements: dict[int, tuple[int, int, int]] = {}

    for dp_rank, lanes in enumerate(lanes_by_dp):
        for tp_rank, tasks in enumerate(lanes):
            if not tasks:
                lane_metadata.append(None)
                continue
            base = 0
            for task in tasks:
                placements[task.task_id] = dp_rank, tp_rank, base
                base += task.encoded_rows
            lane_features[dp_rank][tp_rank] = np.concatenate(
                [np.asarray(task.item.feature) for task in tasks]
            )
            lane_metadata.append(plugin.get_metadata([task.item for task in tasks]))

    real_features = [feature for lanes in lane_features for feature in lanes if feature is not None]
    natural_patch_k = max(feature.shape[0] for feature in real_features)
    feature_dim = real_features[0].shape[1]
    patch_k = _ceil_to_bucket(natural_patch_k, patch_buckets)

    patches = np.zeros((dp_size, tp_size, patch_k, feature_dim), dtype=np.float32)
    valid = np.zeros((dp_size, tp_size), dtype=np.int32)
    for dp_rank, feature_lanes in enumerate(lane_features):
        for tp_rank, feature in enumerate(feature_lanes):
            if feature is not None:
                valid[dp_rank, tp_rank] = feature.shape[0]
                patches[dp_rank, tp_rank, : feature.shape[0]] = feature

    meta = plugin.stack_metadata(lane_metadata, patch_k)
    meta = _reshape_metadata(meta, dp_size, tp_size)
    return VisionEncodeInputs(patches, valid, meta), placements, valid, patch_k


def _build_modality_batch(
    tasks_by_dp: list[list[_EncodeTask]],
    plugin: VisionEncoderPlugin,
    dp_size: int,
    tp_size: int,
    per_dp_token: int,
    patch_buckets: Sequence[int] | None = None,
    merge_buckets: Sequence[int] | None = None,
) -> ModalityEmbedBatch:
    encode_inputs, placements, _, patch_k = _build_encode_arrays(
        tasks_by_dp, plugin, dp_size, tp_size, patch_buckets
    )

    natural_source_capacity = max(
        plugin.get_num_output_tokens(patch_k),
        max(
            (
                placements[task.task_id][2] + task.encoded_rows
                for tasks in tasks_by_dp
                for task in tasks
            ),
            default=1,
        ),
    )
    configured_capacity = max(merge_buckets) if merge_buckets else natural_source_capacity
    source_capacity = max(natural_source_capacity, configured_capacity)

    # Destination-driven routing: position i in the final axis writes token i.
    # Padding to the per-DP token bucket removes the independent merge dimension
    # from the downstream JIT cache key.
    src_idx = np.zeros((dp_size, tp_size, per_dp_token), dtype=np.int32)
    mask = np.zeros_like(src_idx, dtype=np.bool_)
    for task in (task for tasks in tasks_by_dp for task in tasks):
        dp_rank, tp_rank, feature_base = placements[task.task_id]
        for part in task.merge_slices:
            begin = part.token_start
            end = begin + part.length
            if mask[dp_rank, :, begin:end].any():
                raise ValueError(
                    "Vision token row is assigned more than once: "
                    f"dp_rank={dp_rank}, token_range=({begin}, {end})."
                )
            src_idx[dp_rank, tp_rank, begin:end] = np.arange(
                feature_base + part.feature_start,
                feature_base + part.feature_start + part.length,
            )
            mask[dp_rank, tp_rank, begin:end] = True

    return ModalityEmbedBatch(
        encode_inputs,
        DeviceMergePlan(src_idx, mask),
        source_capacity=source_capacity,
    )


class InModelVisionPlanBuilder:
    def __init__(
        self,
        plugin: VisionEncoderPlugin,
        patch_buckets: Sequence[int] | None = None,
        merge_buckets: Sequence[int] | None = None,
    ) -> None:
        self.plugin = plugin
        self.patch_buckets = tuple(patch_buckets) if patch_buckets else None
        self.merge_buckets = tuple(merge_buckets) if merge_buckets else None

    def build(
        self,
        reqs_info: list[ScheduleReqsInfo] | None,
        dp_size: int,
        per_dp_token: int,
        tp_size: int,
    ) -> MultimodalEmbedPlan | None:
        tasks = _collect_encode_tasks(reqs_info, dp_size, per_dp_token)
        if not any(tasks):
            return None
        batch = _build_modality_batch(
            tasks,
            self.plugin,
            dp_size,
            tp_size,
            per_dp_token,
            self.patch_buckets,
            self.merge_buckets,
        )
        return {self.plugin.output_modality: batch}

    def dummy_plan(
        self,
        dp_size: int,
        tp_size: int,
        patch_bucket: int,
        per_dp_token: int,
    ) -> MultimodalEmbedPlan:
        """Zero-filled plan at fixed bucket shapes, matching runtime padded shapes.

        Lane 0 carries a full ``patch_bucket`` synthetic single-image metadata so
        ``encode_jit`` sees the bucketed patch dimension; the remaining lanes are
        dummy. The merge plan is all-masked at ``per_dp_token`` width, so driving
        this plan through the runtime embed path warms ``encode_jit`` and the
        token-shaped merge kernel without touching any real token rows.

        Dummy plans exercise the normal model encode/merge boundary.
        """
        lane_metadata: list[Any] = [None] * (dp_size * tp_size)
        lane_metadata[0] = self.plugin.dummy_metadata(patch_bucket)

        feature_dim = self.plugin.feature_dim
        patches = np.zeros((dp_size, tp_size, patch_bucket, feature_dim), dtype=np.float32)
        valid = np.zeros((dp_size, tp_size), dtype=np.int32)

        meta = self.plugin.stack_metadata(lane_metadata, patch_bucket)
        meta = _reshape_metadata(meta, dp_size, tp_size)

        src_idx = np.zeros((dp_size, tp_size, per_dp_token), dtype=np.int32)
        mask = np.zeros_like(src_idx, dtype=np.bool_)

        natural_source_capacity = self.plugin.get_num_output_tokens(patch_bucket)
        configured_capacity = (
            max(self.merge_buckets) if self.merge_buckets else natural_source_capacity
        )
        source_capacity = max(natural_source_capacity, configured_capacity)

        batch = ModalityEmbedBatch(
            VisionEncodeInputs(patches, valid, meta),
            DeviceMergePlan(src_idx, mask),
            source_capacity=source_capacity,
        )
        return {self.plugin.output_modality: batch}


def register_vision_encoder(
    arch: str,
    plugin_factory: Callable[[ModelConfig], VisionEncoderPlugin],
) -> None:
    """Register ``plugin_factory`` for ``arch`` behind an ``InModelVisionPlanBuilder``."""

    def builder_factory(model_config: ModelConfig) -> InModelVisionPlanBuilder:
        return InModelVisionPlanBuilder(plugin_factory(model_config))

    register_in_model_plan_builder(arch, builder_factory)
