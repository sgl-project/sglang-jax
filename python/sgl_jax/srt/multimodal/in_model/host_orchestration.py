"""Resolve multimodal items and merge replicated encoder outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.embedding_pool import (
    EmbeddingPool,
    EmbeddingPoolEntry,
)
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    PackedMultimodalEmbedding,
    Placement,
)


@dataclass(frozen=True)
class _MergeMapping:
    source_start: int
    destination_start: int
    length: int


@dataclass(frozen=True)
class ItemTask:
    item: MultimodalDataItem
    output_len: int
    merge_mappings: tuple[_MergeMapping, ...]


_MultimodalBatch = dict[Modality, tuple[ItemTask, ...]]


def _build_item_task(
    item: MultimodalDataItem,
    token_base: int,
    chunk_start: int,
    chunk_end: int,
) -> ItemTask | None:
    mappings: list[_MergeMapping] = []
    output_len = 0
    for start, end in item.placeholder_ranges or ():
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            mappings.append(
                _MergeMapping(
                    source_start=output_len + overlap_start - start,
                    destination_start=token_base + overlap_start - chunk_start,
                    length=overlap_end - overlap_start,
                )
            )
        output_len += end - start
    return ItemTask(item, output_len, tuple(mappings)) if mappings else None


def build_multimodal_batch(
    reqs_info: list | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
) -> _MultimodalBatch | None:
    """Build tasks for placeholders visible in this prefill chunk."""

    grouped: dict[Modality, list[ItemTask]] = {}
    for dp_rank, info in enumerate((reqs_info or ())[:dp_size]):
        request_base = dp_rank * per_dp_token
        for req_index, req in enumerate(info.reqs or ()):
            prefix_len = (
                info.prefix_lens[req_index]
                if info.prefix_lens is not None
                else len(getattr(req, "prefix_indices", ()))
            )
            extend_len = (
                info.extend_lens[req_index]
                if info.extend_lens is not None
                else getattr(req, "extend_input_len", 0)
            )
            if isinstance(req.mm_inputs, MultimodalInputs):
                for item in req.mm_inputs.mm_items:
                    task = _build_item_task(
                        item,
                        request_base,
                        prefix_len,
                        prefix_len + extend_len,
                    )
                    if task is not None:
                        grouped.setdefault(item.modality, []).append(task)
            request_base += extend_len

    if not grouped:
        return None
    if not ModelRegistry.is_in_model_multimodal(model_config.hf_config.architectures):
        return None
    return {modality: tuple(tasks) for modality, tasks in grouped.items()}


@partial(jax.jit, static_argnames=("deepstack_dim", "out_sharding", "ds_final_sharding"))
def _gather_overlay(
    running: jax.Array,
    ds_running: jax.Array | None,
    source: jax.Array,
    lane_idx: jax.Array,
    pos_idx: jax.Array,
    mask: jax.Array,
    *,
    deepstack_dim: int,
    out_sharding: NamedSharding | None,
    ds_final_sharding: NamedSharding | None,
) -> tuple[jax.Array, jax.Array | None]:
    """Overlay tokens (and deepstack) with a single gather from a packed source.

    ``source`` is ``[num_lanes, cap, (1 + deepstack_dim) * H]`` (replicated);
    ``lane_idx``/``pos_idx``/``mask`` are ``[T]`` token-sharded. One advanced-index
    gather over ``(lane, pos)`` yields ``[T, F]`` (pinned by ``out_sharding`` to
    ``running``'s token-sharded layout), which is then split on the feature axis:
    ``[..., :H]`` overlays the token stream and ``[..., H:]`` -- reshaped to
    ``[T, D, H]`` and transposed to ``[D, T, H]`` -- overlays the deepstack buffer.
    Because deepstack shares the primary's ``(lane, pos)`` indexing, one gather
    serves both. Shapes depend only on the fixed row count and the bucketed
    ``cap`` -- never on image size or request packing.
    """
    if out_sharding is None:
        gathered = source[lane_idx, pos_idx]  # [T, F]
    else:
        gathered = source.at[lane_idx, pos_idx].get(out_sharding=out_sharding)

    hidden = running.shape[-1]
    running = jnp.where(mask[:, None], gathered[..., :hidden], running)

    if deepstack_dim:
        ds = gathered[..., hidden:].reshape(gathered.shape[0], deepstack_dim, hidden)  # [T, D, H]
        ds = jnp.transpose(ds, (1, 0, 2))  # [D, T, H]
        if ds_final_sharding is not None:
            ds = jax.sharding.reshard(ds, ds_final_sharding)
        ds_running = jnp.where(mask[None, :, None], ds, ds_running)
    return running, ds_running


def _build_gather_indices(
    tasks: tuple[ItemTask, ...],
    placements: tuple[Placement, ...],
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each destination token to a ``(row, position)`` in the packed output."""

    lane_idx = np.zeros(num_tokens, dtype=np.int32)
    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    for task, (row, offset, _length) in zip(tasks, placements, strict=True):
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            span = slice(dst, dst + length)
            lane_idx[span] = row
            pos_idx[span] = offset + mapping.source_start + np.arange(length, dtype=np.int32)
            mask[span] = True
    return lane_idx, pos_idx, mask


def _place_token_vector(vector: np.ndarray, running: jax.Array, mesh: Mesh | None) -> jax.Array:
    """Shard a ``[T]`` index/mask vector like ``running``'s token axis."""

    if mesh is not None and isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        return jax.device_put(vector, NamedSharding(mesh, PartitionSpec(token_spec)))
    return jnp.asarray(vector)


def _apply_gather(
    running: jax.Array,
    deepstack: jax.Array | None,
    source: ArrayLike,
    deepstack_dim: int,
    lane_idx: np.ndarray,
    pos_idx: np.ndarray,
    mask: np.ndarray,
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    """Overlay tokens (and deepstack) gathered from a ``[rows, cap, F]`` source.

    ``source`` is either the encoder's packed output or the embedding pool's
    paged buffer; in both cases ``(lane_idx, pos_idx)`` index the leading
    ``(row, position)`` axes and the trailing feature axis carries the primary
    embedding plus ``deepstack_dim`` deepstack planes, so one gather kernel
    serves the fresh-encode and the cache-hit paths.
    """
    if not mask.any():
        return running, deepstack

    lane_dev = _place_token_vector(lane_idx, running, mesh)
    pos_dev = _place_token_vector(pos_idx, running, mesh)
    mask_dev = _place_token_vector(mask, running, mesh)
    sharded = mesh is not None and isinstance(running.sharding, NamedSharding)
    out_sharding = running.sharding if sharded else None
    src = jnp.asarray(source)

    ds_final_sharding = None
    if deepstack_dim:
        token_spec = running.sharding.spec[0] if sharded else None
        if deepstack is None:
            zeros = jnp.zeros((deepstack_dim, *running.shape), dtype=src.dtype)
            if sharded:
                deepstack = jax.device_put(
                    zeros, NamedSharding(mesh, PartitionSpec(None, token_spec, None))
                )
            else:
                deepstack = zeros
        ds_final_sharding = deepstack.sharding if sharded else None

    return _gather_overlay(
        running,
        deepstack,
        src,
        lane_dev,
        pos_dev,
        mask_dev,
        deepstack_dim=deepstack_dim,
        out_sharding=out_sharding,
        ds_final_sharding=ds_final_sharding,
    )


def _gather_merge(
    running: jax.Array,
    deepstack: jax.Array | None,
    packed: PackedMultimodalEmbedding,
    tasks: tuple[ItemTask, ...],
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    """
    Returns the updated ``(running, deepstack)`` buffers: same shapes as the
    inputs, with only this batch's ``tasks`` token positions overlaid (all other
    positions -- earlier passes and plain-text tokens -- left untouched), so the
    result threads into the next gather.
    """

    lane_idx, pos_idx, mask = _build_gather_indices(tasks, packed.placements, running.shape[0])
    return _apply_gather(
        running, deepstack, packed.output, packed.deepstack_dim, lane_idx, pos_idx, mask, mesh
    )


def _build_pool_gather_indices(
    tasks: Sequence[ItemTask],
    entries: Sequence[EmbeddingPoolEntry],
    page_size: int,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each destination token to a ``(page, offset)`` in the embedding pool."""

    lane_idx = np.zeros(num_tokens, dtype=np.int32)
    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    for task, entry in zip(tasks, entries, strict=True):
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            token = mapping.source_start + np.arange(length, dtype=np.int32)
            span = slice(dst, dst + length)
            lane_idx[span] = entry.page_ids[token // page_size]
            pos_idx[span] = token % page_size
            mask[span] = True
    return lane_idx, pos_idx, mask


def _gather_from_pool(
    running: jax.Array,
    deepstack: jax.Array | None,
    pool: EmbeddingPool,
    tasks: Sequence[ItemTask],
    entries: Sequence[EmbeddingPoolEntry],
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    """Overlay cache hits by gathering from the pool's paged buffers."""

    lane_idx, pos_idx, mask = _build_pool_gather_indices(
        tasks, entries, pool.page_size, running.shape[0]
    )
    return _apply_gather(
        running, deepstack, pool.pages, pool.deepstack_dim, lane_idx, pos_idx, mask, mesh
    )


def _write_misses_to_pool(
    pool: EmbeddingPool,
    packed: PackedMultimodalEmbedding,
    tasks: Sequence[ItemTask],
) -> None:
    """Store freshly-encoded items without slicing away the encoder bucket."""

    pool.write_packed(
        tuple(task.item.hash for task in tasks),
        packed.output,
        packed.placements,
    )


def embed_multimodal_inputs(
    multimodal_batch: _MultimodalBatch,
    input_ids: jax.Array,
    input_embedding: Callable[[jax.Array], jax.Array],
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Merge encoder embeddings into the token stream via the packed contract.

    The encoder's packed output (``PackedMultimodalEmbedding``) is the sole
    embedding contract: the merge gathers straight from it (or, on a cache hit,
    from the paged embedding pool) with a bounded number of compiled shapes,
    never keyed on image size or request packing.
    """
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = input_embedding(input_ids)
        deepstack = None
        encode_funcs = multimodal_model.get_multimodal_encode_funcs()
        for modality, tasks in multimodal_batch.items():
            encode_func = encode_funcs.get(modality)
            if encode_func is None:
                raise ValueError(
                    f"no packed embedding function for modality {modality}; "
                    "in-model multimodal models must expose one"
                )

            if embedding_pool is None:
                packed = encode_func([task.item for task in tasks])
                running, deepstack = _gather_merge(running, deepstack, packed, tasks, mesh)
                continue

            # Pool present: split hits from misses by item hash. Misses run the
            # encoder and merge from its packed output (same cost as the no-pool
            # path) then are written back for reuse; hits merge straight from the
            # pool's paged buffers.
            hit_tasks: list[ItemTask] = []
            hit_entries: list[EmbeddingPoolEntry] = []
            miss_tasks: list[ItemTask] = []
            for task in tasks:
                if task.item.hash is None:
                    task.item.set_pad_value()
                entry = embedding_pool.lookup(task.item.hash)
                if entry is not None:
                    hit_tasks.append(task)
                    hit_entries.append(entry)
                else:
                    miss_tasks.append(task)
            # Consume hits before a miss write is allowed to evict their pages.
            if hit_tasks:
                running, deepstack = _gather_from_pool(
                    running, deepstack, embedding_pool, hit_tasks, hit_entries, mesh
                )
            if miss_tasks:
                packed = encode_func([task.item for task in miss_tasks])
                running, deepstack = _gather_merge(
                    running, deepstack, packed, tuple(miss_tasks), mesh
                )
                _write_misses_to_pool(embedding_pool, packed, miss_tasks)
        return running, deepstack
