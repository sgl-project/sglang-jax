"""Shared multimodal token mapping and embedding merge utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.modality_enum import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract


@dataclass(frozen=True)
class MergeMapping:
    source_start: int
    destination_start: int
    length: int


@dataclass(frozen=True)
class ItemTask:
    item: MultimodalDataItem
    output_len: int
    merge_mappings: list[MergeMapping]

    @property
    def has_unmerged_tail(self) -> bool:
        last = self.merge_mappings[-1]
        return last.source_start + last.length < self.output_len


def build_item_task(
    item: MultimodalDataItem,
    token_base: int,
    chunk_start: int,
    chunk_end: int,
) -> ItemTask | None:
    mappings: list[MergeMapping] = []
    output_len = 0
    for start, end in item.placeholder_ranges or []:
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            mappings.append(
                MergeMapping(
                    source_start=output_len + overlap_start - start,
                    destination_start=token_base + overlap_start - chunk_start,
                    length=overlap_end - overlap_start,
                )
            )
        output_len += end - start
    return ItemTask(item, output_len, mappings) if mappings else None


def collect_item_tasks(reqs_info: list | None, dp_size: int, per_dp_token: int) -> list[ItemTask]:
    """Locate media rows intersecting this batch's prefill windows."""
    tasks = []
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
                    task = build_item_task(
                        item,
                        request_base,
                        prefix_len,
                        prefix_len + extend_len,
                    )
                    if task is not None:
                        tasks.append(task)
            request_base += extend_len

    return tasks


@partial(jax.jit, static_argnames=("out_sharding",))
def gather_overlay(
    running: jax.Array,
    source: jax.Array,
    pos_idx: jax.Array,
    mask: jax.Array,
    *,
    out_sharding: NamedSharding | None,
) -> jax.Array:
    if source.ndim > 2:
        # Gather the requested rows before reshaping. Flattening the fixed pool
        # outside this JIT can reformat the entire registered buffer.
        indices = (pos_idx // source.shape[1], pos_idx % source.shape[1])
    else:
        indices = pos_idx
    if out_sharding is None:
        gathered = source[indices]
    else:
        gathered = source.at[indices].get(out_sharding=out_sharding)
    gathered = gathered.reshape(pos_idx.shape[0], -1)
    # Raiden rows may include trailing physical tile padding.
    gathered = gathered[:, : running.shape[-1]]
    return jnp.where(mask[:, None], gathered, running)


def build_gather_indices(tasks: list[ItemTask], num_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """Map destination tokens to item-ordered packed rows."""
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


def place_token_vector(vector: np.ndarray, running: jax.Array, mesh: Mesh | None) -> jax.Array:
    """Shard a ``[T]`` index/mask vector like ``running``'s token axis."""

    if mesh is not None and isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        return jax.device_put(vector, NamedSharding(mesh, PartitionSpec(token_spec)))
    return jnp.asarray(vector)


def apply_gather(
    running: jax.Array,
    source: jax.Array,
    pos_idx: np.ndarray,
    mask: np.ndarray,
    mesh: Mesh | None,
) -> jax.Array:
    if not mask.any():
        return running

    pos_dev = place_token_vector(pos_idx, running, mesh)
    mask_dev = place_token_vector(mask, running, mesh)
    sharded = mesh is not None and isinstance(running.sharding, NamedSharding)
    out_sharding = running.sharding if sharded else None
    return gather_overlay(
        running,
        source,
        pos_dev,
        mask_dev,
        out_sharding=out_sharding,
    )


def gather_merge(
    running: jax.Array, packed: jax.Array, tasks: list[ItemTask], mesh: Mesh | None
) -> jax.Array:
    """Merge contiguous, item-ordered embeddings."""
    width = running.shape[-1]
    capacity = sum(task.output_len for task in tasks)
    if packed.ndim != 2 or packed.shape[1] != width or packed.shape[0] < capacity:
        raise ValueError(
            f"packed embeddings must be [capacity, {width}] with capacity >= "
            f"{capacity}, got {packed.shape}"
        )
    pos_idx, mask = build_gather_indices(tasks, running.shape[0])
    return apply_gather(running, packed, pos_idx, mask, mesh)


def prepare_input_embeddings(input_ids: jax.Array, model: InModelMultimodalContract):
    """Reserve trailing columns for deepstack features, when present."""
    running = model.get_input_embeddings()(input_ids)
    hidden = running.shape[-1]
    if model.deepstack_visual_layers:
        running = jnp.pad(running, ((0, 0), (0, hidden * model.deepstack_visual_layers)))
    return running, hidden


def split_embeddings(
    running: jax.Array,
    hidden: int,
    deepstack_dim: int,
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    if not deepstack_dim:
        return running, None
    running, deepstack = jnp.split(running, [hidden], axis=-1)
    deepstack = deepstack.reshape(running.shape[0], deepstack_dim, hidden).transpose(1, 0, 2)
    if isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        deepstack = jax.sharding.reshard(
            deepstack,
            NamedSharding(mesh, PartitionSpec(None, token_spec, None)),
        )
    return running, deepstack
