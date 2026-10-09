from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.embedding_pool import (
    EmbeddingPool,
    EmbeddingPoolEntry,
)
from sgl_jax.srt.multimodal.in_model.interface import (
    AudioCodeInputSpec,
    InModelMultimodalContract,
)
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    balance_lanes,
    mrope_vision_dummy_inputs,
)
from sgl_jax.srt.multimodal.in_model.mm_utils import (
    ItemTask,
    MergeMapping,
    apply_gather,
    collect_item_tasks,
    gather_merge,
    prepare_input_embeddings,
    split_embeddings,
)


@dataclass
class MultimodalBatch:
    per_lane_tasks: list[dict[Modality, list[ItemTask]]]
    cached_tasks: list[ItemTask]


def build_multimodal_batch(
    reqs_info: list | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
    embedding_pool: EmbeddingPool | None = None,
    num_encoder_lanes: int = 1,
) -> MultimodalBatch | None:
    """Check cache presence and balance expected misses for this prefill chunk."""
    if num_encoder_lanes < 1:
        raise ValueError("num_encoder_lanes must be positive")
    if reqs_info is None or not model_config.is_in_model_multimodal:
        return None

    grouped: dict[Modality, list[ItemTask]] = {}
    for task in collect_item_tasks(reqs_info, dp_size, per_dp_token):
        grouped.setdefault(task.item.modality, []).append(task)

    if not grouped:
        return None
    result = MultimodalBatch([{} for _ in range(num_encoder_lanes)], [])
    for modality, tasks in grouped.items():
        misses = []
        for task in tasks:
            item = task.item
            if item.hash is None:
                item.set_pad_value()
            if embedding_pool is not None and embedding_pool.contains(item.hash):
                result.cached_tasks.append(task)
            else:
                misses.append(task)
        if misses:
            lengths = [int(task.item.feature.shape[0]) for task in misses]
            lanes = balance_lanes(lengths, num_encoder_lanes)
            for lane_id, indices in enumerate(lanes):
                if indices:
                    result.per_lane_tasks[lane_id][modality] = [misses[i] for i in sorted(indices)]
    return result


def _build_pool_gather_indices(
    tasks: list[ItemTask],
    entries: list[EmbeddingPoolEntry],
    page_size: int,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each destination token to a flat row in the embedding pool."""

    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    for task, entry in zip(tasks, entries, strict=True):
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            token = mapping.source_start + np.arange(length, dtype=np.int32)
            span = slice(dst, dst + length)
            pos_idx[span] = entry.page_ids[token // page_size] * page_size + token % page_size
            mask[span] = True
    return pos_idx, mask


def _gather_from_pool(
    running: jax.Array,
    pool: EmbeddingPool,
    tasks: list[ItemTask],
    entries: list[EmbeddingPoolEntry],
    mesh: Mesh | None,
) -> jax.Array:
    """Overlay cache hits by gathering from the pool's paged buffers."""

    pos_idx, mask = _build_pool_gather_indices(tasks, entries, pool.page_size, running.shape[0])
    return apply_gather(
        running,
        pool.pages.reshape(-1, pool.hidden),
        pos_idx,
        mask,
        mesh,
    )


def _cache_unfinished_items(
    pool: EmbeddingPool | None,
    packed: jax.Array,
    tasks: list[ItemTask],
    item_offsets: list[int],
) -> None:
    if pool is None:
        return
    write_mask = [task.has_unmerged_tail for task in tasks]
    if not any(write_mask):
        return
    pool.write_packed(
        [task.item.hash for task in tasks],
        packed,
        [task.output_len for task in tasks],
        write_mask=write_mask,
        item_offsets=item_offsets,
    )


def precompile_multimodal_encoder(
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
    token_buckets: Sequence[int] = (),
    *,
    num_lanes: int = 1,
    patch_paddings=None,
    audio_token_buckets: Sequence[int] = (16, 64, 256, 1024),
) -> tuple[int, ...]:
    """Warm encoders, merge and cache kernels; return observed output capacities."""
    encode_funcs = multimodal_model.get_multimodal_encode_funcs()
    observed_capacities = set()
    for modality, encode in encode_funcs.items():
        if modality in (Modality.IMAGE, Modality.MULTI_IMAGES, Modality.VIDEO):
            spec = multimodal_model.vision_input_spec
            if spec is None:
                raise ValueError(f"Missing input spec for registered encoder {modality}")
            dummy_inputs = mrope_vision_dummy_inputs(
                spec, patch_paddings, num_lanes=num_lanes, modality=modality
            )
        elif modality == Modality.AUDIO and isinstance(
            multimodal_model.audio_input_spec, AudioCodeInputSpec
        ):
            spec = multimodal_model.audio_input_spec
            dummy_inputs = []
            for tokens in audio_token_buckets:
                item = MultimodalDataItem(
                    modality=modality,
                    feature=np.zeros((tokens * spec.group_size, spec.channels), np.int32),
                    placeholder_ranges=[(0, tokens)],
                )
                dummy_inputs.append((modality, [[item]] + [[] for _ in range(num_lanes - 1)]))
        else:
            raise NotImplementedError(f"No dummy input builder for {modality}")

        for _, items_per_lane in dummy_inputs:
            output = jax.block_until_ready(encode(items_per_lane))
            observed_capacities.add(output.shape[0])
    capacities = tuple(sorted(observed_capacities))
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError(f"invalid multimodal packed capacities: {capacities}")
    if embedding_pool is not None:
        for capacity in capacities:
            embedding_pool.precompile_packed_write(capacity)

    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        for num_tokens in token_buckets:
            input_ids = jnp.zeros(
                num_tokens,
                jnp.int32,
                device=(NamedSharding(mesh, PartitionSpec("data")) if mesh is not None else None),
            )
            running, hidden = prepare_input_embeddings(input_ids, multimodal_model)
            deepstack_dim = multimodal_model.deepstack_visual_layers

            item = MultimodalDataItem(modality=Modality.IMAGE)
            for capacity in capacities or [num_tokens]:
                length = min(num_tokens, capacity)
                task = ItemTask(item, length, [MergeMapping(0, 0, length)])
                packed = jnp.zeros((capacity, running.shape[-1]), running.dtype)
                if mesh is not None:
                    packed = jax.device_put(packed, NamedSharding(mesh, PartitionSpec()))
                running = gather_merge(
                    token_embeddings=running,
                    encoder_embeddings=packed,
                    merge_tasks=[task],
                    mesh=mesh,
                )
                jax.block_until_ready(running)

            if embedding_pool is not None:
                length = min(num_tokens, embedding_pool.page_size)
                task = ItemTask(item, length, [MergeMapping(0, 0, length)])
                entry = EmbeddingPoolEntry(np.asarray([0], dtype=np.int32), length)
                running = _gather_from_pool(running, embedding_pool, [task], [entry], mesh)
                jax.block_until_ready(running)

            jax.block_until_ready(split_embeddings(running, hidden, deepstack_dim, mesh))
    return capacities


def embed_multimodal_inputs(
    multimodal_batch: MultimodalBatch | None,
    input_ids: jax.Array,
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None, bool]:
    """Merge lane-packed encoder outputs directly into the token stream."""
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = multimodal_model.get_input_embeddings()(input_ids)
        hidden = running.shape[-1]
        deepstack_dim = multimodal_model.deepstack_visual_layers
        if deepstack_dim and multimodal_batch is None:
            deepstack = jnp.zeros(
                (deepstack_dim, running.shape[0], hidden),
                dtype=running.dtype,
                out_sharding=NamedSharding(
                    mesh,
                    PartitionSpec(None, "data", None),
                ),
            )
            return running, deepstack, False
        if deepstack_dim:
            running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))

        if multimodal_batch is not None:
            per_lane_tasks = [dict(lane) for lane in multimodal_batch.per_lane_tasks]
            hits, entries = [], []
            expired: dict[Modality, list[ItemTask]] = {}
            for task in multimodal_batch.cached_tasks:
                entry = (
                    embedding_pool.lookup(task.item.hash) if embedding_pool is not None else None
                )
                if entry is None:
                    expired.setdefault(task.item.modality, []).append(task)
                else:
                    hits.append(task)
                    entries.append(entry)
            # Dispatch all cache reads before encoder outputs can overwrite pool pages.
            if hits:
                running = _gather_from_pool(running, embedding_pool, hits, entries, mesh)

            # Scheduling only checks presence. Rebalance misses if those entries were evicted.
            for modality, tasks in expired.items():
                tasks = [task for lane in per_lane_tasks for task in lane.get(modality, [])] + tasks
                lanes = balance_lanes(
                    [int(task.item.feature.shape[0]) for task in tasks],
                    len(per_lane_tasks),
                )
                for lane, indices in zip(per_lane_tasks, lanes, strict=True):
                    lane[modality] = [tasks[i] for i in sorted(indices)]

            encode_funcs = multimodal_model.get_multimodal_encode_funcs()
            modalities = dict.fromkeys(modality for lane in per_lane_tasks for modality in lane)
            for modality in modalities:
                tasks = []
                items_per_lane = []
                for lane in per_lane_tasks:
                    lane_tasks = lane.get(modality, [])
                    items_per_lane.append([task.item for task in lane_tasks])
                    tasks.extend(lane_tasks)
                encode = encode_funcs.get(modality)
                if encode is None:
                    raise ValueError(f"no embedding function for modality {modality}")
                packed = encode(items_per_lane)
                if packed.shape[0] % len(items_per_lane):
                    raise ValueError("encoder output capacity must be divisible by lane count")
                lane_capacity = packed.shape[0] // len(items_per_lane)
                item_offsets = []
                for lane_index, lane in enumerate(per_lane_tasks):
                    offset = lane_index * lane_capacity
                    for task in lane.get(modality, []):
                        item_offsets.append(offset)
                        offset += task.output_len
                    if offset > (lane_index + 1) * lane_capacity:
                        raise ValueError("encoder item lengths exceed lane capacity")
                running = gather_merge(
                    token_embeddings=running,
                    encoder_embeddings=packed,
                    merge_tasks=tasks,
                    mesh=mesh,
                    item_offsets=item_offsets,
                )
                _cache_unfinished_items(embedding_pool, packed, tasks, item_offsets)

        input_embedding, deepstack = split_embeddings(running, hidden, deepstack_dim, mesh)
        apply_for_deepstack = deepstack_dim > 0 and multimodal_batch is not None
        return input_embedding, deepstack, apply_for_deepstack
