"""Qwen2.5-VL vision-metadata builder.

Produces the arch-specific pytree (``window_index / cu_window_seqlens /
rotary_pos_emb``) that the vision tower consumes, and pads-by-role across DP
ranks. Runs on host in numpy; the resulting numpy arrays are moved to devices
by ``embed_plan.device_put_plan`` in a later stage.

The numeric geometry mirrors ``qwen2_5_vit.Qwen2_5_VisionTransformer`` so the
in-model path and the pre-computed path stay bit-identical.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np

from sgl_jax.srt.multimodal.common.vision_metadata import (
    register_vision_metadata_builder,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Qwen25VLVisionMetadata:
    window_index: (
        Any  # [num_feature_rows]        int32 (single) / [dp, feat_k]        int32 (stacked)
    )
    cu_window_seqlens: (
        Any  # [n_windows+1]        int32 (single) / [dp, L]             int32 (stacked)
    )
    rotary_pos_emb: (
        Any  # [num_patches, rot_dim]  float32 (single) / [dp, patch_k, rot_dim] (stacked)
    )


def _extract_grid(item: Any) -> np.ndarray:
    grid = getattr(item, "image_grid_thw", None)
    if grid is None:
        raise ValueError("Qwen2.5-VL vision metadata: item missing image_grid_thw")
    arr = np.asarray(grid, dtype=np.int32)
    if arr.shape == (1, 3):
        return arr[0]
    if arr.shape == (3,):
        return arr
    raise ValueError(
        f"Qwen2.5-VL vision metadata: image_grid_thw must be shape (3,) or (1,3), got {arr.shape}"
    )


def _rotary_pos_emb_thw(
    t: int, h: int, w: int, spatial_merge_size: int, rotary_dim: int, theta: float = 10000.0
) -> np.ndarray:
    sms = spatial_merge_size
    hpos, wpos = np.indices((h, w))
    hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).ravel()
    wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).ravel()
    pos_ids = np.stack([hpos, wpos], axis=-1)  # [h*w, 2]
    pos_ids = np.tile(pos_ids, (t, 1))  # [t*h*w, 2]

    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    seq = np.arange(max(h, w), dtype=np.float32)
    freqs = np.outer(seq, inv_freq)  # [max, rotary_dim/2]

    picked = freqs[pos_ids]  # [t*h*w, 2, rotary_dim/2]
    return picked.reshape(pos_ids.shape[0], -1)  # [t*h*w, rotary_dim]


def _window_index_thw(
    t: int, h: int, w: int, spatial_merge_size: int, patch_size: int, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    sms = spatial_merge_size
    merger_window = window_size // sms // patch_size
    llm_h, llm_w = h // sms, w // sms

    index = np.arange(t * llm_h * llm_w, dtype=np.int32).reshape(t, llm_h, llm_w)
    pad_h = merger_window - llm_h % merger_window
    pad_w = merger_window - llm_w % merger_window
    nwh = (llm_h + pad_h) // merger_window
    nww = (llm_w + pad_w) // merger_window

    padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
    padded = padded.reshape(t, nwh, merger_window, nww, merger_window)
    padded = padded.transpose(0, 1, 3, 2, 4).reshape(t, nwh * nww, merger_window, merger_window)
    seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)  # [t*nwh*nww]

    flat = padded.reshape(-1)
    valid = np.nonzero(flat != -100)[0]
    window_index = flat[valid].astype(np.int32)
    cu = (np.cumsum(seqlens) * (sms * sms)).astype(np.int32)
    return window_index, cu


class Qwen25VLVisionMetadataBuilder:
    def __init__(self, vision_cfg: Any) -> None:
        self.vision_cfg = vision_cfg
        self.spatial_merge_size = int(vision_cfg.spatial_merge_size)
        self.patch_size = int(vision_cfg.patch_size)
        self.window_size = int(vision_cfg.window_size)
        head_dim = int(vision_cfg.hidden_size) // int(vision_cfg.num_heads)
        self.rotary_dim = head_dim // 2

    def get_metadata(self, item: Any) -> Qwen25VLVisionMetadata:
        t, h, w = (int(v) for v in _extract_grid(item))
        sms = self.spatial_merge_size
        sms2 = sms * sms

        rope = _rotary_pos_emb_thw(t, h, w, sms, self.rotary_dim)  # [t*h*w, rot_dim]
        rope = rope.reshape(rope.shape[0] // sms2, sms2, -1)  # [num_feat, sms2, rot_dim]

        window_index, cu_window = _window_index_thw(t, h, w, sms, self.patch_size, self.window_size)
        rope = rope[window_index, :, :].reshape(-1, rope.shape[-1])  # [num_patches, rot_dim]
        cu_window = np.concatenate([np.array([0], dtype=np.int32), cu_window])

        return Qwen25VLVisionMetadata(
            window_index=window_index,
            cu_window_seqlens=cu_window,
            rotary_pos_emb=rope.astype(np.float32),
        )

    def stack_metadata(
        self, metas: list[Qwen25VLVisionMetadata | None], patch_k: int
    ) -> Qwen25VLVisionMetadata:
        sms2 = self.spatial_merge_size * self.spatial_merge_size
        if patch_k % sms2 != 0:
            raise ValueError(
                f"stack_metadata: patch_k={patch_k} not divisible by spatial_merge_size**2={sms2}"
            )
        feat_k = patch_k // sms2
        dp = len(metas)
        rot_dim = self.rotary_dim

        cu_len = max((int(m.cu_window_seqlens.shape[0]) for m in metas if m is not None), default=2)
        cu_len = max(cu_len, 2)

        out_window = np.empty((dp, feat_k), dtype=np.int32)
        out_rope = np.zeros((dp, patch_k, rot_dim), dtype=np.float32)
        out_cu = np.empty((dp, cu_len), dtype=np.int32)

        for rank, m in enumerate(metas):
            if m is None:
                out_window[rank] = np.arange(feat_k, dtype=np.int32)
                out_cu[rank, 0] = 0
                out_cu[rank, 1:] = patch_k
                continue

            wi = np.asarray(m.window_index, dtype=np.int32)
            n = wi.shape[0]
            if n > feat_k:
                raise ValueError(
                    f"stack_metadata: rank {rank} has {n} feature rows but feat_k={feat_k}"
                )
            out_window[rank, :n] = wi
            if n < feat_k:
                out_window[rank, n:] = np.arange(n, feat_k, dtype=np.int32)

            rope = np.asarray(m.rotary_pos_emb, dtype=np.float32)
            if rope.shape[0] > patch_k:
                raise ValueError(
                    f"stack_metadata: rank {rank} has {rope.shape[0]} patches but patch_k={patch_k}"
                )
            out_rope[rank, : rope.shape[0]] = rope

            cu = np.asarray(m.cu_window_seqlens, dtype=np.int32)
            if cu.shape[0] > cu_len:
                raise ValueError(
                    f"stack_metadata: rank {rank} cu_window_seqlens length {cu.shape[0]} "
                    f"exceeds bucket {cu_len}"
                )
            out_cu[rank, : cu.shape[0]] = cu
            if cu.shape[0] < cu_len:
                out_cu[rank, cu.shape[0] :] = patch_k

        return Qwen25VLVisionMetadata(
            window_index=out_window,
            cu_window_seqlens=out_cu,
            rotary_pos_emb=out_rope,
        )


register_vision_metadata_builder(
    "Qwen2_5_VLForConditionalGeneration", Qwen25VLVisionMetadataBuilder
)
