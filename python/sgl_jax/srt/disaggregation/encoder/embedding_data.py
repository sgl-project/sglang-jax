from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.common.modality_enum import Modality

# Adapted for JAX from SGLang's encoder receiver data structures:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/encode_receiver.py


_MODALITY_GRID_KEYS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}


class EmbeddingData:
    """Adapted from sglang.srt.disaggregation.encode_receiver.EmbeddingData for JAX."""

    def __init__(
        self,
        req_id: str,
        num_parts: int,
        part_idx: int,
        grid_dim: Any,
        modality: Modality,
        embedding: jax.Array | None = None,
        embedding_shape: list[int] | tuple[int, ...] | None = None,
        shape: list[int] | tuple[int, ...] | None = None,
        dtype: Any = None,
        error_msg: str | None = None,
        error_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.grid_dim = grid_dim
        self.modality = modality
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else dtype
        resolved_shape = embedding_shape if embedding_shape is not None else shape
        self.shape = (
            resolved_shape
            if resolved_shape is not None
            else list(embedding.shape) if embedding is not None else None
        )
        self.error_msg = error_msg
        self.error_code = error_code
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self) -> Any:
        return self.grid_dim

    def get_embedding(self) -> jax.Array | None:
        return self.embedding

    def copy_without_embedding(self) -> EmbeddingData:
        copied = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            embedding_shape=self.shape,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        for key, value in vars(self).items():
            if not key.startswith("_") and key != "embedding":
                setattr(copied, key, value)
        return copied

    def __repr__(self) -> str:
        return (
            f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, "
            f"part_idx={self.part_idx}, error_msg={self.error_msg})"
        )


class MultiModalEmbeddingData:
    def __init__(self, num_parts: int) -> None:
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        self.num_parts = num_parts
        self._parts: list[tuple[EmbeddingData, jax.Array] | None] = [None] * num_parts

    def add(self, data: EmbeddingData, embedding: jax.Array) -> None:
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

    def get_embedding(self, is_concat: bool = False):
        if not self.ready:
            raise RuntimeError("embedding parts are incomplete")
        parts = [part for part in self._parts if part is not None]
        if not is_concat:
            return [embedding for _, embedding in parts]

        grouped: dict[Modality, list[jax.Array]] = {}
        for data, embedding in parts:
            grouped.setdefault(data.modality, []).append(embedding)
        return {
            modality: jnp.concatenate(embeddings, axis=0)
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
                value = jnp.asarray(data.grid_dim)
                if flatten:
                    value = value.reshape(-1)
                elif value.ndim == 0:
                    value = value.reshape(1)
                values.append(value)
            if values:
                result[key] = jnp.concatenate(values)

        second_per_grid_ts = []
        for data, _ in parts:
            if data.modality == Modality.VIDEO:
                values = getattr(data, "second_per_grid_ts", None)
                if values is not None:
                    second_per_grid_ts.extend(jnp.asarray(values).reshape(-1).tolist())
        if second_per_grid_ts:
            result["second_per_grid_ts"] = second_per_grid_ts
        return result
