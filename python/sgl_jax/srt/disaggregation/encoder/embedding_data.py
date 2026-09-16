from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.mm_utils import (
    ItemTask,
    apply_gather,
    build_gather_indices,
    collect_item_tasks,
    prepare_input_embeddings,
    split_embeddings,
)

# Adapted for JAX from SGLang's encoder receiver data structures:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/encode_receiver.py


_MODALITY_GRID_KEYS = {
    Modality.IMAGE: ("image_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}


class EmbeddingLifetime(Protocol):
    """Keep received storage alive until both its owner and device readers finish."""

    def record_read(self, result: jax.Array) -> None: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReceivedEmbeddings:
    buffer: jax.Array
    row_indices: np.ndarray
    width: int
    sessions: list[EmbeddingLifetime]

    def slice_rows(self, start: int, end: int) -> ReceivedEmbeddings:
        """Select logical rows without copying device data or transferring ownership."""
        return ReceivedEmbeddings(
            self.buffer, self.row_indices[start:end], self.width, self.sessions
        )

    @staticmethod
    def concatenate(embeddings: list[ReceivedEmbeddings]) -> ReceivedEmbeddings:
        """Combine row locations in one buffer, retaining all receive sessions."""
        first = embeddings[0]
        if len(embeddings) == 1:
            return first
        if any(e.buffer is not first.buffer or e.width != first.width for e in embeddings):
            raise ValueError("Received embeddings must share one pool and width")
        sessions = {id(session): session for e in embeddings for session in e.sessions}
        return ReceivedEmbeddings(
            first.buffer,
            np.concatenate([e.row_indices for e in embeddings]),
            first.width,
            list(sessions.values()),
        )


def release_received_embeddings(mm_inputs) -> None:
    """End request ownership; sessions wait for outstanding device reads."""
    for item in getattr(mm_inputs, "mm_items", ()):
        if isinstance(item.precomputed_embeddings, ReceivedEmbeddings):
            for session in item.precomputed_embeddings.sessions:
                session.release()
            item.precomputed_embeddings = None


@dataclass
class ReceivedEmbeddingBatch:
    # Capture the sources at scheduling time; requests retain session ownership.
    items: list[tuple[ItemTask, ReceivedEmbeddings]]


@dataclass(slots=True)
class EmbeddingData:
    """One encoder part: identity, reconstruction metadata, and transfer descriptor."""

    req_id: str
    num_parts: int
    part_idx: int
    modality: Modality
    grid_dim: Any = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    error_msg: str | None = None
    item_hashes: list[int] | None = None
    second_per_grid_ts: list[float] | None = None
    transfer: dict[str, Any] = field(default_factory=dict)

    @property
    def transfer_id(self) -> str:
        return self.transfer["transfer_id"]


class MultiModalEmbeddingData:
    def __init__(self, num_parts: int) -> None:
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        self.num_parts = num_parts
        self._parts: list[tuple[EmbeddingData, ReceivedEmbeddings] | None] = [None] * num_parts

    def add(self, data: EmbeddingData, embedding: ReceivedEmbeddings) -> None:
        if data.num_parts != self.num_parts:
            raise ValueError("inconsistent num_parts")
        if not 0 <= data.part_idx < self.num_parts:
            raise ValueError(f"invalid part_idx: {data.part_idx}")
        if self._parts[data.part_idx] is not None:
            raise ValueError(f"duplicate part_idx: {data.part_idx}")
        self._parts[data.part_idx] = (data, embedding)

    @property
    def ready(self) -> bool:
        return all(part is not None for part in self._parts)

    def has_part(self, part_idx: int) -> bool:
        return 0 <= part_idx < self.num_parts and self._parts[part_idx] is not None

    def get_embedding(self) -> dict[Modality, ReceivedEmbeddings]:
        if not self.ready:
            raise RuntimeError("embedding parts are incomplete")
        parts = [part for part in self._parts if part is not None]
        grouped: dict[Modality, list[ReceivedEmbeddings]] = {}
        for data, embedding in parts:
            grouped.setdefault(data.modality, []).append(embedding)
        return {
            modality: ReceivedEmbeddings.concatenate(embeddings)
            for modality, embeddings in grouped.items()
        }

    def get_mm_extra_meta(self) -> dict[str, Any]:
        result = {}
        parts = [part for part in self._parts if part is not None]
        for modality, (key, flatten) in _MODALITY_GRID_KEYS.items():
            values = []
            for data, _ in parts:
                if data.modality != modality or data.grid_dim is None:
                    continue
                value = np.asarray(data.grid_dim)
                if flatten:
                    value = value.reshape(-1)
                elif value.ndim == 0:
                    value = value.reshape(1)
                values.append(value)
            if values:
                result[key] = values[0] if len(values) == 1 else np.concatenate(values)

        item_hashes: dict[Modality, list[int]] = {}
        for data, _ in parts:
            values = data.item_hashes
            if values is None:
                raise ValueError("encoder metadata is missing media hashes")
            item_hashes.setdefault(data.modality, []).extend(map(int, values))
        result["item_hashes"] = item_hashes

        second_per_grid_ts = []
        for data, _ in parts:
            if data.modality == Modality.VIDEO:
                values = data.second_per_grid_ts
                if values is not None:
                    second_per_grid_ts.extend(np.asarray(values).ravel().tolist())
        if second_per_grid_ts:
            result["second_per_grid_ts"] = second_per_grid_ts
        return result

    def release(self) -> None:
        """Release received parts when their request is cancelled before admission."""
        for part in self._parts:
            if part is not None:
                for session in part[1].sessions:
                    session.release()


def build_received_embedding_batch(
    reqs_info: list | None, dp_size: int, per_dp_token: int
) -> ReceivedEmbeddingBatch:
    items = []
    for task in collect_item_tasks(reqs_info, dp_size, per_dp_token):
        source = task.item.precomputed_embeddings
        if source is None:
            raise ValueError("Disaggregated input is missing encoder embeddings")
        items.append((task, source))
    return ReceivedEmbeddingBatch(items)


def embed_received_inputs(
    batch: ReceivedEmbeddingBatch,
    input_ids: jax.Array,
    model: InModelMultimodalContract,
) -> tuple[jax.Array, jax.Array | None, bool]:
    mesh = model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running, hidden = prepare_input_embeddings(input_ids, model)
        grouped = {}
        for task, source in batch.items:
            if source.width != running.shape[-1] or len(source.row_indices) < task.output_len:
                raise ValueError("Received embedding dimensions do not match the media item")
            grouped.setdefault((id(source.buffer), source.width), []).append((task, source))

        for items in grouped.values():
            tasks = [task for task, _ in items]
            received = ReceivedEmbeddings.concatenate(
                [source.slice_rows(0, task.output_len) for task, source in items]
            )
            positions, mask = build_gather_indices(tasks, running.shape[0])
            positions[mask] = received.row_indices[positions[mask]]
            running = apply_gather(running, received.buffer, positions, mask, mesh)
            for session in received.sessions:
                session.record_read(running)

        embeddings, deepstack = split_embeddings(
            running, hidden, model.deepstack_visual_layers, mesh
        )
        return embeddings, deepstack, model.deepstack_visual_layers > 0 and bool(batch.items)


def precompile_received_embeddings(
    receive_buffer: jax.Array,
    multimodal_model: InModelMultimodalContract,
    token_buckets: list[int],
) -> None:
    """Warm actual receive-buffer shapes before the backend can write into them."""
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        for num_tokens in token_buckets:
            input_ids = jnp.zeros(
                num_tokens,
                jnp.int32,
                device=(NamedSharding(mesh, PartitionSpec("data")) if mesh is not None else None),
            )
            running = multimodal_model.get_input_embeddings()(input_ids)
            hidden = running.shape[-1]
            deepstack_dim = multimodal_model.deepstack_visual_layers
            if deepstack_dim:
                running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))
            running = apply_gather(
                running,
                receive_buffer,
                np.zeros(num_tokens, np.int32),
                np.ones(num_tokens, np.bool_),
                mesh,
            )
            jax.block_until_ready(split_embeddings(running, hidden, deepstack_dim, mesh))
