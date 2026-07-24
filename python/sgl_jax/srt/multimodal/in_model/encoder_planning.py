"""Architecture-agnostic in-model encode/merge planning.

A model plugs into this framework by registering an :class:`EncoderPlugin`
under its HF architecture name (:func:`register_encoder`). This builder turns
scheduled requests into fixed-shape ``[dp, tp]`` encoder batches and token-merge
routes, delegating only the model-specific metadata math to the plugin; the
encode-input pytree (:class:`EncodeInputs`) is framework-owned and shared across
models and modalities.

The builder is modality-neutral: a plugin declares which modalities it consumes
(``input_modalities``) and which modality its merged rows land under
(``output_modality``). Vision plugins select image/video items; an audio plugin
selects audio items — the request/lane/merge orchestration below is identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.plan import (
    DeviceMergePlan,
    ModalityEmbedBatch,
    MultimodalEmbedPlan,
)
from sgl_jax.srt.multimodal.in_model.registry import register_in_model_plan_builder

if TYPE_CHECKING:
    from collections.abc import Callable

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo


def _ceil_to_bucket(value: int, buckets: Sequence[int] | None) -> int:
    if buckets:
        for bucket in buckets:
            if bucket >= value:
                return bucket
    return value


@register_pytree_node_class
@dataclass
class EncodeInputs:
    """Framework-owned encoder inputs for one ``[dp, tp]`` encode round.

    ``features`` and ``valid`` are built by the plan builder; ``meta`` is the
    model-specific metadata pytree the framework stacks from the plugin's
    per-lane ``pad_metadata`` output.
    The model's encode body reads these three fields. Every array leaf shares
    the leading ``[dp, tp]`` axes, matching the ``encode_inputs`` pytree contract
    of ``ModalityEmbedBatch``. ``features`` is ``[dp, tp, input_capacity,
    feature_dim]`` — one feature vector per input position (a ViT patch, an audio
    frame, ...).
    """

    features: Any
    valid: Any
    meta: Any

    def tree_flatten(self):
        return (self.features, self.valid, self.meta), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class EncoderPlugin(ABC):
    """Base class each model subclasses to plug an encoder into the builder.

    The builder owns all request/lane/merge orchestration and the encode-input
    pytree; a subclass only contributes its modality selection, its metadata
    math, and a couple of shape constants. Subclasses must set the class
    attributes below and implement the abstract methods; missing any raises
    ``TypeError`` at instantiation.

    ``input_len`` is the length of the feature sequence fed to the encoder (one
    row per input position); ``output_len`` (a.k.a. output tokens) is what the
    encoder + merge produces from it.
    """

    # Modalities this plugin consumes from a request's ``mm_items``.
    input_modalities: tuple[Modality, ...]

    # Modality under which merged encoder rows land (e.g. ``Modality.IMAGE``).
    output_modality: Modality

    # Trailing dim of one feature row (second axis of a real ``item.feature``).
    feature_dim: int

    def select_items(self, req) -> list[tuple[int, MultimodalDataItem]]:
        """Pick this plugin's ``mm_items`` (with their index) from ``req``.

        Default: filter by ``item.modality in self.input_modalities``. Override
        only for selection logic that ``input_modalities`` cannot express.
        """
        mm_inputs = req.mm_inputs
        if mm_inputs is None or isinstance(mm_inputs, dict) and "mm_items" not in mm_inputs:
            return []
        if not isinstance(mm_inputs, MultimodalInputs):
            raise TypeError(
                "encoder plan builder expects req.mm_inputs to be MultimodalInputs, "
                f"got {type(mm_inputs).__name__}."
            )
        if any(not isinstance(item, MultimodalDataItem) for item in mm_inputs.mm_items):
            bad = next(
                item for item in mm_inputs.mm_items if not isinstance(item, MultimodalDataItem)
            )
            raise TypeError(
                "encoder plan builder expects mm_items to contain "
                f"MultimodalDataItem, got {type(bad).__name__}."
            )
        return [
            (item_index, item)
            for item_index, item in enumerate(mm_inputs.mm_items)
            if item.modality in self.input_modalities
        ]

    @abstractmethod
    def get_metadata(self, items: Sequence[MultimodalDataItem]) -> Any:
        """Metadata for one lane's ``items``, in order (single lane, no batch axis)."""

    @abstractmethod
    def dummy_metadata(self, input_capacity: int) -> Any:
        """Native (unpadded) metadata for a synthetic lane of ``input_capacity`` rows.
        Used to fill empty lanes and to warm ``encode_jit`` during precompile.
        """

    @abstractmethod
    def pad_metadata(self, meta: Any, input_capacity: int) -> Any:
        """Pad one lane's native metadata (from :meth:`get_metadata`) to
        ``input_capacity`` rows.
        """

    @abstractmethod
    def get_num_output_tokens(self, input_len: int) -> int:
        """Return the number of output tokens produced from ``input_len`` input rows."""


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
    output_len: int
    merge_slices: tuple[MergeSlice, ...]

    @property
    def input_len(self) -> int:
        return np.asarray(self.item.feature).shape[0]


def _assign_tp_lanes(tasks: list[_EncodeTask], tp_size: int) -> list[list[_EncodeTask]]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if tp_size == 1:
        return [list(tasks)]

    lanes: list[list[_EncodeTask]] = [[] for _ in range(tp_size)]
    loads = [0] * tp_size
    for task in sorted(tasks, key=lambda task: (-task.input_len, task.task_id)):
        rank = min(range(tp_size), key=lambda candidate: (loads[candidate], candidate))
        lanes[rank].append(task)
        loads[rank] += task.input_len
    return lanes


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
        raise ValueError(f"Multimodal item in dp_rank {dp_rank} is missing feature.")
    if feature.ndim != 2 or not feature.shape[0]:
        raise ValueError(
            f"Multimodal item feature must be a non-empty 2D array, got shape={feature.shape} "
            f"in dp_rank {dp_rank}."
        )
    if not item.placeholder_ranges:
        raise ValueError(f"Multimodal item in dp_rank {dp_rank} has no placeholder ranges.")

    slices = []
    output_len = 0
    previous_end = -1
    for start, end in item.placeholder_ranges:
        length = end - start
        if length <= 0:
            raise ValueError(f"Placeholder range must be non-empty, got start={start}, end={end}.")
        if start < previous_end:
            raise ValueError(
                "Placeholder token is assigned more than once: "
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
                    "Placeholder chunk is outside its packed rank slot: "
                    f"dp_rank={dp_rank}, req_base={req_base}, "
                    f"chunk=({chunk_start}, {chunk_end}), range=({start}, {end}), "
                    f"per_dp_token={per_dp_token}."
                )
            slices.append(
                MergeSlice(
                    output_len + overlap_start - start,
                    token_start,
                    merge_length,
                )
            )
        output_len += length

    return (
        _EncodeTask(
            task_id,
            item,
            output_len,
            tuple(slices),
        )
        if slices
        else None
    )


def _collect_encode_tasks(
    reqs_info: list[ScheduleReqsInfo] | None,
    plugin: EncoderPlugin,
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
                    "Multimodal chunk bounds must be non-negative, "
                    f"got prefix_len={prefix_len}, extend_len={extend_len}."
                )

            for _, item in plugin.select_items(req):
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
                "Metadata leaves must be lane-leading after _stack_metadata: "
                f"expected first dimension={dp_size * tp_size}, got shape={value.shape}."
            )
        return value.reshape(dp_size, tp_size, *value.shape[1:])

    return jax.tree.map(reshape, meta)


def _stack_metadata(
    plugin: EncoderPlugin, lane_metadata: Sequence[Any | None], input_capacity: int
) -> Any:
    """Pad every lane to ``input_capacity`` and stack into a leading ``[dp*tp, ...]`` axis.

    Empty lanes (``None``) are filled with ``plugin.dummy_metadata`` before
    padding, so the plugin only ever implements single-lane padding
    (:meth:`EncoderPlugin.pad_metadata`). All lanes end at identical shapes, so
    the per-leaf ``np.stack`` is a pure structural op over the registered
    metadata pytree — the framework never reads a field.
    """
    padded = [
        plugin.pad_metadata(
            meta if meta is not None else plugin.dummy_metadata(input_capacity),
            input_capacity,
        )
        for meta in lane_metadata
    ]
    return jax.tree.map(lambda *leaves: np.stack(leaves), *padded)


def _build_encode_arrays(
    tasks_by_dp: list[list[_EncodeTask]],
    plugin: EncoderPlugin,
    dp_size: int,
    tp_size: int,
    input_buckets: Sequence[int] | None,
) -> tuple[Any, dict[int, tuple[int, int, int]], np.ndarray, int]:
    """Assign tasks to ``[dp,tp]`` encode lanes and stack their features/metadata.

    Returns the encode-input pytree, a ``task_id -> (dp, tp, feature_base)``
    placement map (feature_base = row offset of the task within its lane's
    concatenated encode output), and the ``[dp,tp]`` per-lane valid row counts.
    """
    all_tasks = [task for tasks in tasks_by_dp for task in tasks]
    if not all_tasks:
        raise ValueError("Encode batch requires at least one task.")

    for task in all_tasks:
        expected_rows = plugin.get_num_output_tokens(task.input_len)
        if task.output_len != expected_rows:
            raise ValueError(
                "Placeholder rows must match encoder output rows: "
                f"task_id={task.task_id}, input_len={task.input_len}, "
                f"placeholder_rows={task.output_len}, encoder_rows={expected_rows}."
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
                base += task.output_len
            lane_features[dp_rank][tp_rank] = np.concatenate(
                [np.asarray(task.item.feature) for task in tasks]
            )
            lane_metadata.append(plugin.get_metadata([task.item for task in tasks]))

    real_features = [feature for lanes in lane_features for feature in lanes if feature is not None]
    natural_input_capacity = max(feature.shape[0] for feature in real_features)
    feature_dim = real_features[0].shape[1]
    input_capacity = _ceil_to_bucket(natural_input_capacity, input_buckets)

    features = np.zeros((dp_size, tp_size, input_capacity, feature_dim), dtype=np.float32)
    valid = np.zeros((dp_size, tp_size), dtype=np.int32)
    for dp_rank, feature_lanes in enumerate(lane_features):
        for tp_rank, feature in enumerate(feature_lanes):
            if feature is not None:
                valid[dp_rank, tp_rank] = feature.shape[0]
                features[dp_rank, tp_rank, : feature.shape[0]] = feature

    meta = _stack_metadata(plugin, lane_metadata, input_capacity)
    meta = _reshape_metadata(meta, dp_size, tp_size)
    return EncodeInputs(features, valid, meta), placements, valid, input_capacity


def _build_modality_batch(
    tasks_by_dp: list[list[_EncodeTask]],
    plugin: EncoderPlugin,
    dp_size: int,
    tp_size: int,
    per_dp_token: int,
    input_buckets: Sequence[int] | None = None,
    merge_buckets: Sequence[int] | None = None,
) -> ModalityEmbedBatch:
    encode_inputs, placements, _, input_capacity = _build_encode_arrays(
        tasks_by_dp, plugin, dp_size, tp_size, input_buckets
    )

    natural_source_capacity = max(
        plugin.get_num_output_tokens(input_capacity),
        max(
            (
                placements[task.task_id][2] + task.output_len
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
                    "Token row is assigned more than once: "
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


class InModelEncoderPlanBuilder:
    def __init__(
        self,
        plugin: EncoderPlugin,
        input_buckets: Sequence[int] | None = None,
        merge_buckets: Sequence[int] | None = None,
    ) -> None:
        self.plugin = plugin
        self.input_buckets = tuple(input_buckets) if input_buckets else None
        self.merge_buckets = tuple(merge_buckets) if merge_buckets else None

    def build(
        self,
        reqs_info: list[ScheduleReqsInfo] | None,
        dp_size: int,
        per_dp_token: int,
        tp_size: int,
    ) -> MultimodalEmbedPlan | None:
        tasks = _collect_encode_tasks(reqs_info, self.plugin, dp_size, per_dp_token)
        if not any(tasks):
            return None
        batch = _build_modality_batch(
            tasks,
            self.plugin,
            dp_size,
            tp_size,
            per_dp_token,
            self.input_buckets,
            self.merge_buckets,
        )
        return {self.plugin.output_modality: batch}

    def dummy_plan(
        self,
        dp_size: int,
        tp_size: int,
        input_bucket: int,
        per_dp_token: int,
    ) -> MultimodalEmbedPlan:
        """Zero-filled plan at fixed bucket shapes, matching runtime padded shapes.

        Lane 0 carries a full ``input_bucket`` synthetic single-item metadata so
        ``encode_jit`` sees the bucketed input dimension; the remaining lanes are
        dummy. The merge plan is all-masked at ``per_dp_token`` width, so driving
        this plan through the runtime embed path warms ``encode_jit`` and the
        token-shaped merge kernel without touching any real token rows.

        Dummy plans exercise the normal model encode/merge boundary.
        """
        lane_metadata: list[Any] = [None] * (dp_size * tp_size)
        lane_metadata[0] = self.plugin.dummy_metadata(input_bucket)

        feature_dim = self.plugin.feature_dim
        features = np.zeros((dp_size, tp_size, input_bucket, feature_dim), dtype=np.float32)
        valid = np.zeros((dp_size, tp_size), dtype=np.int32)

        meta = _stack_metadata(self.plugin, lane_metadata, input_bucket)
        meta = _reshape_metadata(meta, dp_size, tp_size)

        src_idx = np.zeros((dp_size, tp_size, per_dp_token), dtype=np.int32)
        mask = np.zeros_like(src_idx, dtype=np.bool_)

        natural_source_capacity = self.plugin.get_num_output_tokens(input_bucket)
        configured_capacity = (
            max(self.merge_buckets) if self.merge_buckets else natural_source_capacity
        )
        source_capacity = max(natural_source_capacity, configured_capacity)

        batch = ModalityEmbedBatch(
            EncodeInputs(features, valid, meta),
            DeviceMergePlan(src_idx, mask),
            source_capacity=source_capacity,
        )
        return {self.plugin.output_modality: batch}


def register_encoder(
    arch: str,
    plugin_factory: Callable[[ModelConfig], EncoderPlugin],
) -> None:
    """Register ``plugin_factory`` for ``arch`` behind an ``InModelEncoderPlanBuilder``."""

    def builder_factory(model_config: ModelConfig) -> InModelEncoderPlanBuilder:
        return InModelEncoderPlanBuilder(plugin_factory(model_config))

    register_in_model_plan_builder(arch, builder_factory)
