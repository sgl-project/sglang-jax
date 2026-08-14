from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

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
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract


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

    @property
    def has_unmerged_tail(self) -> bool:
        last = self.merge_mappings[-1]
        return last.source_start + last.length < self.output_len


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


@partial(jax.jit, static_argnames=("out_sharding",))
def _gather_overlay(
    running: jax.Array,
    source: jax.Array,
    pos_idx: jax.Array,
    mask: jax.Array,
    *,
    out_sharding: NamedSharding | None,
) -> jax.Array:
    if out_sharding is None:
        gathered = source[pos_idx]
    else:
        gathered = source.at[pos_idx].get(out_sharding=out_sharding)
    return jnp.where(mask[:, None], gathered, running)


def _build_gather_indices(
    tasks: tuple[ItemTask, ...],
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each destination token to its item-ordered packed source row."""

    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    source_offset = 0
    for task in tasks:
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            span = slice(dst, dst + length)
            pos_idx[span] = source_offset + mapping.source_start + np.arange(length, dtype=np.int32)
            mask[span] = True
        source_offset += task.output_len
    return pos_idx, mask


def _place_token_vector(vector: np.ndarray, running: jax.Array, mesh: Mesh | None) -> jax.Array:
    """Shard a ``[T]`` index/mask vector like ``running``'s token axis."""

    if mesh is not None and isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        return jax.device_put(vector, NamedSharding(mesh, PartitionSpec(token_spec)))
    return jnp.asarray(vector)


def _apply_gather(
    running: jax.Array,
    source: jax.Array,
    pos_idx: np.ndarray,
    mask: np.ndarray,
    mesh: Mesh | None,
) -> jax.Array:
    if not mask.any():
        return running

    pos_dev = _place_token_vector(pos_idx, running, mesh)
    mask_dev = _place_token_vector(mask, running, mesh)
    sharded = mesh is not None and isinstance(running.sharding, NamedSharding)
    out_sharding = running.sharding if sharded else None
    return _gather_overlay(
        running,
        source,
        pos_dev,
        mask_dev,
        out_sharding=out_sharding,
    )


def _gather_merge(
    running: jax.Array,
    packed: jax.Array,
    tasks: tuple[ItemTask, ...],
    mesh: Mesh | None,
) -> jax.Array:
    expected_width = running.shape[-1]
    min_capacity = sum(task.output_len for task in tasks)
    if packed.ndim != 2 or packed.shape[1] != expected_width or packed.shape[0] < min_capacity:
        raise ValueError(
            f"packed embeddings must be [capacity, {expected_width}] with capacity >= "
            f"{min_capacity}, got {packed.shape}"
        )
    pos_idx, mask = _build_gather_indices(tasks, running.shape[0])
    return _apply_gather(running, packed, pos_idx, mask, mesh)


def _build_pool_gather_indices(
    tasks: Sequence[ItemTask],
    entries: Sequence[EmbeddingPoolEntry],
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
    tasks: Sequence[ItemTask],
    entries: Sequence[EmbeddingPoolEntry],
    mesh: Mesh | None,
) -> jax.Array:
    """Overlay cache hits by gathering from the pool's paged buffers."""

    pos_idx, mask = _build_pool_gather_indices(tasks, entries, pool.page_size, running.shape[0])
    return _apply_gather(
        running,
        pool.pages.reshape(-1, pool.feature_width),
        pos_idx,
        mask,
        mesh,
    )


def _write_misses_to_pool(
    pool: EmbeddingPool,
    packed: jax.Array,
    tasks: Sequence[ItemTask],
) -> None:
    write_mask = tuple(task.has_unmerged_tail for task in tasks)
    if not any(write_mask):
        return
    pool.write_packed(
        tuple(task.item.hash for task in tasks),
        packed,
        tuple(task.output_len for task in tasks),
        write_mask=write_mask,
    )


def _split_embeddings(
    running: jax.Array,
    hidden: int,
    deepstack_dim: int,
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    if not deepstack_dim:
        return running, None
    running, deepstack = jnp.split(running, (hidden,), axis=-1)
    deepstack = deepstack.reshape(running.shape[0], deepstack_dim, hidden).transpose(1, 0, 2)
    if isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        deepstack = jax.sharding.reshard(
            deepstack,
            NamedSharding(mesh, PartitionSpec(None, token_spec, None)),
        )
    return running, deepstack


def precompile_multimodal_inputs(
    input_ids: jax.Array,
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Warm merge kernels and return a multimodal-shaped forward input."""
    capacities = tuple(map(int, multimodal_model.get_multimodal_embedding_packed_capacities()))
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError(f"invalid multimodal packed capacities: {capacities}")

    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = multimodal_model.get_input_embeddings()(input_ids)
        num_tokens, hidden = running.shape
        deepstack_dim = multimodal_model.deepstack_visual_layers
        if deepstack_dim:
            running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))

        item = MultimodalDataItem(modality=Modality.IMAGE)
        for capacity in capacities or (num_tokens,):
            length = min(num_tokens, capacity)
            task = ItemTask(item, length, (_MergeMapping(0, 0, length),))
            packed = jnp.zeros((capacity, running.shape[-1]), running.dtype)
            if mesh is not None:
                packed = jax.device_put(packed, NamedSharding(mesh, PartitionSpec()))
            running = _gather_merge(running, packed, (task,), mesh)
            jax.block_until_ready(running)

        if embedding_pool is not None:
            length = min(num_tokens, embedding_pool.page_size)
            task = ItemTask(item, length, (_MergeMapping(0, 0, length),))
            entry = EmbeddingPoolEntry(np.asarray([0], dtype=np.int32), length)
            running = _gather_from_pool(running, embedding_pool, (task,), (entry,), mesh)
            jax.block_until_ready(running)

    return _split_embeddings(running, hidden, deepstack_dim, mesh)


def precompile_multimodal_components(
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> None:
    multimodal_model.precompile_multimodal()
    if embedding_pool is not None:
        for capacity in multimodal_model.get_multimodal_embedding_packed_capacities():
            embedding_pool.precompile_packed_write(capacity)


def embed_multimodal_inputs(
    multimodal_batch: _MultimodalBatch,
    input_ids: jax.Array,
    input_embedding: Callable[[jax.Array], jax.Array],
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Merge padded, item-ordered encoder outputs into the token stream."""
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = input_embedding(input_ids)
        hidden = running.shape[-1]
        deepstack_dim = multimodal_model.deepstack_visual_layers
        if deepstack_dim:
            running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))

        encode_funcs = multimodal_model.get_multimodal_encode_funcs()
        for modality, tasks in multimodal_batch.items():
            encode_func = encode_funcs.get(modality)
            if encode_func is None:
                raise ValueError(
                    f"no embedding function for modality {modality}; "
                    "in-model multimodal models must expose one"
                )

            if embedding_pool is None:
                packed = encode_func([task.item for task in tasks])
                running = _gather_merge(running, packed, tasks, mesh)
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
                running = _gather_from_pool(running, embedding_pool, hit_tasks, hit_entries, mesh)
            if miss_tasks:
                packed = encode_func([task.item for task in miss_tasks])
                running = _gather_merge(running, packed, tuple(miss_tasks), mesh)
                _write_misses_to_pool(embedding_pool, packed, miss_tasks)

        return _split_embeddings(running, hidden, deepstack_dim, mesh)
