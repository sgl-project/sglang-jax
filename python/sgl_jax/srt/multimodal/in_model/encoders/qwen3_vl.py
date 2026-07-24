from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.encoder_planning import (
    EncoderPlugin,
    register_encoder,
)


@register_pytree_node_class
@dataclass
class Qwen3VLVisionMetadata:
    pos_indices: Any
    pos_weights: Any
    rotary_pos_emb: Any
    cu_seqlens: Any

    def tree_flatten(self):
        return (self.pos_indices, self.pos_weights, self.rotary_pos_emb, self.cu_seqlens), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _grid(item) -> tuple[int, int, int]:
    value = item.get("image_grid_thw")
    if value is None:
        value = item.get("video_grid_thw")
    array = np.asarray(value)
    if array.size != 3:
        raise ValueError(f"Qwen3-VL expects one grid_thw row per item, got {array.shape}.")
    return tuple(int(x) for x in array.reshape(3))


def _merge_order(x: np.ndarray, t: int, h: int, w: int, merge: int) -> np.ndarray:
    return (
        np.broadcast_to(x, (t, *x.shape))
        .reshape(t, h // merge, merge, w // merge, merge, *x.shape[2:])
        .transpose(0, 1, 3, 2, 4, *range(5, x.ndim + 3))
        .reshape(t * h * w, *x.shape[2:])
    )


class Qwen3VLVisionEncoderPlugin(EncoderPlugin):
    input_modalities = (Modality.IMAGE, Modality.MULTI_IMAGES, Modality.VIDEO)
    output_modality = Modality.IMAGE

    def __init__(self, model_config):
        config = model_config.hf_config.vision_config
        self.patch_size = int(config.patch_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(config.in_channels)
        self.merge = int(config.spatial_merge_size)
        self.num_grid = int(config.num_position_embeddings**0.5)
        head_dim = int(config.hidden_size) // int(config.num_heads)
        rotary_dim = head_dim // 2
        self.inv_freq = 1.0 / (
            10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
        )
        self.feature_dim = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        self.spatial_merge_unit = self.merge * self.merge

    def _metadata(self, grid) -> Qwen3VLVisionMetadata:
        t, h, w = grid
        if h % self.merge or w % self.merge:
            raise ValueError(f"Qwen3-VL grid {grid} is not divisible by merge={self.merge}.")

        ys = np.linspace(0, self.num_grid - 1, h, dtype=np.float32)
        xs = np.linspace(0, self.num_grid - 1, w, dtype=np.float32)
        y0, x0 = ys.astype(np.int32), xs.astype(np.int32)
        y1, x1 = np.minimum(y0 + 1, self.num_grid - 1), np.minimum(x0 + 1, self.num_grid - 1)
        dy, dx = ys - y0, xs - x0
        indices = np.stack(
            [
                y0[:, None] * self.num_grid + x0[None, :],
                y0[:, None] * self.num_grid + x1[None, :],
                y1[:, None] * self.num_grid + x0[None, :],
                y1[:, None] * self.num_grid + x1[None, :],
            ]
        )
        weights = np.stack(
            [
                (1 - dy[:, None]) * (1 - dx[None, :]),
                (1 - dy[:, None]) * dx[None, :],
                dy[:, None] * (1 - dx[None, :]),
                dy[:, None] * dx[None, :],
            ]
        )
        indices = np.stack([_merge_order(x[..., None], t, h, w, self.merge)[:, 0] for x in indices])
        weights = np.stack([_merge_order(x[..., None], t, h, w, self.merge)[:, 0] for x in weights])

        rows, cols = np.indices((h, w))
        coords = _merge_order(np.stack((rows, cols), axis=-1), t, h, w, self.merge)
        rotary = np.concatenate(
            (coords[:, :1] * self.inv_freq, coords[:, 1:] * self.inv_freq), axis=-1
        ).astype(np.float32)
        cu = np.arange(h * w, t * h * w + 1, h * w, dtype=np.int32)
        return Qwen3VLVisionMetadata(indices, weights.astype(np.float32), rotary, cu)

    def get_metadata(self, items) -> Qwen3VLVisionMetadata:
        metas = []
        offset = 0
        for item in items:
            grid = _grid(item)
            rows = int(np.prod(grid))
            if np.asarray(item.feature).shape[0] != rows:
                raise ValueError("Qwen3-VL feature rows do not match grid_thw.")
            if sum(end - start for start, end in item.placeholder_ranges or []) != (
                rows // self.spatial_merge_unit
            ):
                raise ValueError("Qwen3-VL placeholder rows do not match grid_thw.")
            meta = self._metadata(grid)
            meta.cu_seqlens = meta.cu_seqlens + offset
            metas.append(meta)
            offset += rows
        return Qwen3VLVisionMetadata(
            *(
                np.concatenate(
                    [getattr(meta, field) for meta in metas],
                    axis=-1 if field in {"pos_indices", "pos_weights"} else 0,
                )
                for field in ("pos_indices", "pos_weights", "rotary_pos_emb", "cu_seqlens")
            )
        )

    def dummy_metadata(self, input_capacity: int) -> Qwen3VLVisionMetadata:
        if input_capacity <= 0 or input_capacity % self.spatial_merge_unit:
            raise ValueError("Qwen3-VL patch bucket must be a positive spatial-merge multiple.")
        return self._metadata((1, self.merge, input_capacity // self.merge))

    def pad_metadata(self, meta, input_capacity):
        if input_capacity % self.spatial_merge_unit:
            raise ValueError("Qwen3-VL patch bucket must be a positive spatial-merge multiple.")
        capacity = input_capacity // self.spatial_merge_unit
        n = meta.rotary_pos_emb.shape[0]
        if n > input_capacity or meta.cu_seqlens.shape[0] > capacity:
            raise ValueError("Qwen3-VL metadata exceeds the patch bucket.")
        pos_indices = np.zeros((4, input_capacity), dtype=np.int32)
        pos_weights = np.zeros((4, input_capacity), dtype=np.float32)
        rotary = np.zeros((input_capacity, meta.rotary_pos_emb.shape[-1]), dtype=np.float32)
        cu = np.full(capacity, input_capacity, dtype=np.int32)
        pos_indices[:, :n] = meta.pos_indices
        pos_weights[:, :n] = meta.pos_weights
        rotary[:n] = meta.rotary_pos_emb
        cu[: meta.cu_seqlens.shape[0]] = meta.cu_seqlens
        return Qwen3VLVisionMetadata(pos_indices, pos_weights, rotary, cu)

    def get_num_output_tokens(self, input_len: int) -> int:
        if input_len % self.spatial_merge_unit:
            raise ValueError("Qwen3-VL patch count must be divisible by the merge unit.")
        return input_len // self.spatial_merge_unit


def register_qwen3vl_vision_encoder() -> None:
    register_encoder("Qwen3VLForConditionalGeneration", Qwen3VLVisionEncoderPlugin)
