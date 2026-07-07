"""Qwen2.5-VL concrete vision metadata and builder.

Split out of the main model file ``models/qwen2_5_vl.py`` but still Qwen2.5-VL
arch code. This module registers its builder at import time; the main model
file imports it at top level, so ``ModelRegistry`` loading Qwen2.5-VL triggers
the registration (see ``multimodal/common/vision_metadata.py``). The main model
file only consumes the produced ``meta`` in its ViT encode body.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem
from sgl_jax.srt.multimodal.common.vision_metadata import (
    register_vision_metadata_builder,
)


@register_pytree_node_class
@dataclass
class Qwen25VLVisionMetadata:
    """Qwen2.5-VL ViT aux for one DP encode round.

    Common code treats this as an opaque pytree. The Qwen ViT encode body reads
    the concrete fields, in flatten order:
    ``window_index`` / ``cu_window_seqlens`` / ``rotary_pos_emb``.
    """

    window_index: Any
    cu_window_seqlens: Any
    rotary_pos_emb: Any

    def tree_flatten(self):
        children = (self.window_index, self.cu_window_seqlens, self.rotary_pos_emb)
        aux_data = {}  # static sizes/roles may live here; runtime does not depend on it
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _item_grid_thw(item: MultimodalDataItem) -> tuple:
    """First ``(t, h, w)`` of this vision item's grid metadata as a python-int tuple.

    The builder pulls its geometry from the item here, so the common
    ``get_metadata(item)`` interface stays arch-agnostic.
    """
    grid = item.get("image_grid_thw")
    grid_key = "image_grid_thw"
    if grid is None:
        grid = item.get("video_grid_thw")
        grid_key = "video_grid_thw"
    if grid is None:
        raise ValueError("Vision item is missing image_grid_thw/video_grid_thw metadata.")
    arr = np.asarray(grid)
    if arr.size == 0:
        raise ValueError(f"Vision item {grid_key} metadata is empty.")
    if arr.ndim > 1 and arr.shape[0] != 1:
        raise ValueError(
            f"Vision item must carry exactly one {grid_key} row, " f"got shape={arr.shape}."
        )
    row = arr if arr.ndim == 1 else arr[0]
    row = np.asarray(row).reshape(-1)
    if row.shape[0] != 3:
        raise ValueError(f"Vision item {grid_key} must contain (t, h, w), got {row}.")
    return (int(row[0]), int(row[1]), int(row[2]))


def _placeholder_rows(item) -> int:
    return sum(int(end) - int(start) + 1 for start, end in item.offsets or [])


class Qwen25VLVisionMetadataBuilder:
    """Host-side builder for Qwen2.5-VL ViT aux.

    Contract:
    - ``get_metadata(item)`` consumes one image item carrying ``image_grid_thw``
      and returns native-size metadata for that image.
    - ``stack_metadata(metas, patch_k)`` consumes one native metadata object per
      DP rank (or ``None`` for a dummy lane) and returns a pad-stacked metadata
      pytree for one encode round.

    The scheduler owns pixels/valid/merge indices; this builder owns the Qwen
    metadata roles and their padding semantics.
    """

    def __init__(self, model_config):
        hf_config = getattr(model_config, "hf_config", None)
        vision_cfg = getattr(hf_config, "vision_config", None)
        if vision_cfg is None:
            raise ValueError(
                "Qwen2.5-VL vision metadata builder requires "
                "model_config.hf_config.vision_config."
            )
        self.vision_cfg = vision_cfg
        self.patch_size = int(getattr(vision_cfg, "patch_size", 14))
        self.window_size = int(getattr(vision_cfg, "window_size", 112))
        self.spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        num_heads = int(getattr(vision_cfg, "num_heads", 16))
        hidden_size = int(getattr(vision_cfg, "hidden_size", 1280))
        head_dim = hidden_size // num_heads
        # rotary dim = head_dim // 2.
        self.rotary_dim = head_dim // 2
        self.theta = float(getattr(vision_cfg, "rope_theta", 10000.0))
        self.spatial_merge_unit = self.spatial_merge_size**2

    # ---- ported host-only algorithms (numpy) ----------------------------------
    def _rotary_pos_emb_full(self, seq_len: int) -> np.ndarray:
        inv_freq = 1.0 / (
            self.theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim)
        )
        seq = np.arange(seq_len, dtype=np.float32)
        return np.outer(seq, inv_freq)  # [seq_len, rotary_dim//2]

    def _rotary_pos_emb_thw(self, t, h, w) -> np.ndarray:
        sms = self.spatial_merge_size
        hpos_ids, wpos_ids = np.indices((h, w))
        hpos_ids = hpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
        pos_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = np.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self._rotary_pos_emb_full(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        return rotary_pos_emb  # [merge_units, sms^2, rot_dim]

    def _window_index_thw(self, grid_t, grid_h, grid_w):
        sms = self.spatial_merge_size
        vit_merger_window_size = self.window_size // sms // self.patch_size

        llm_grid_h = grid_h // sms
        llm_grid_w = grid_w // sms

        index = np.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = (
            vit_merger_window_size - llm_grid_h % vit_merger_window_size
        ) % vit_merger_window_size
        pad_w = (
            vit_merger_window_size - llm_grid_w % vit_merger_window_size
        ) % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_window = np.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_window = cu_seqlens_window.astype(np.int32)
        return index_new.astype(np.int32), cu_seqlens_window

    def get_metadata(self, item) -> Qwen25VLVisionMetadata:
        """Build native-size Qwen metadata for one image item.

        Input contract: ``item`` must provide exactly one ``image_grid_thw`` row.
        Output contract: all fields are native-size numpy arrays for this image:
        ``window_index`` is a spatial-merge-unit permutation,
        ``cu_window_seqlens`` is cumulative window boundaries, and
        ``rotary_pos_emb`` is per-patch 2D rope already gathered into window
        order. Full-attention boundaries are derived later from ``valid``.
        """
        t, h, w = _item_grid_thw(item)
        sms = self.spatial_merge_size
        if h % sms != 0 or w % sms != 0:
            raise ValueError(
                "Qwen2.5-VL image_grid_thw height/width must be divisible by "
                f"spatial_merge_size={sms}, got grid={(t, h, w)}."
            )

        expected_patch_rows = t * h * w
        feature = item.get("feature")
        if feature is None or int(np.asarray(feature).shape[0]) != expected_patch_rows:
            actual = None if feature is None else int(np.asarray(feature).shape[0])
            raise ValueError(
                "Qwen2.5-VL image feature rows must match image_grid_thw: "
                f"actual feature rows={actual}, expected={expected_patch_rows}, "
                f"grid={(t, h, w)}."
            )

        expected_feature_rows = t * (h // sms) * (w // sms)
        actual_placeholder_rows = _placeholder_rows(item)
        if actual_placeholder_rows != expected_feature_rows:
            raise ValueError(
                "Qwen2.5-VL placeholder rows must match spatial-merged grid rows: "
                f"actual placeholder rows={actual_placeholder_rows}, "
                f"expected={expected_feature_rows}, grid={(t, h, w)}, "
                f"spatial_merge_size={sms}."
            )

        window_index, cu_window_seqlens = self._window_index_thw(t, h, w)

        # rope: gather into window order, matching the vision transformer layout.
        rope_units = self._rotary_pos_emb_thw(t, h, w)  # [merge_units, sms^2, rot_dim]
        rope = rope_units[window_index, :, :]
        rope = rope.reshape(-1, rope.shape[-1]).astype(np.float32)  # [patches, rot_dim]

        return Qwen25VLVisionMetadata(
            window_index=window_index.astype(np.int32),
            cu_window_seqlens=cu_window_seqlens.astype(np.int32),
            rotary_pos_emb=rope,
        )

    def stack_metadata(self, metas, patch_k):
        """Pad and stack native per-rank metadata for one encode round.

        ``metas[r]`` is this rank's round-k native-size
        :class:`Qwen25VLVisionMetadata`, or ``None`` for a dummy lane (rank owns
        no image in this round). ``patch_k`` is the round's cross-rank patch-row
        bucket and is used as the sentinel for cumulative boundaries.

        Padding contract:
        - ``window_index`` stays a valid permutation via identity tail padding.
        - ``cu_window_seqlens`` uses ``patch_k`` sentinel tail padding.
        - ``rotary_pos_emb`` uses zero row padding to ``patch_k``.
        """
        present = [m for m in metas if m is not None]
        units_k = max(int(m.window_index.shape[0]) for m in present)
        win_k = max(int(m.cu_window_seqlens.shape[0]) for m in present)
        rot_dim = int(present[0].rotary_pos_emb.shape[-1])
        window_indices, cu_window_seqlens_rows, rotary_rows = [], [], []
        for m in metas:
            if m is None:  # dummy lane
                window_indices.append(np.arange(units_k, dtype=np.int32))  # identity perm
                cu_window_seqlens_rows.append(np.full(win_k, patch_k, dtype=np.int32))
                rotary_rows.append(np.zeros((patch_k, rot_dim), dtype=np.float32))
            else:
                # window_index: true values + arange continuation for the pad tail.
                w = np.arange(units_k, dtype=np.int32)
                n_units = int(m.window_index.shape[0])
                w[:n_units] = np.asarray(m.window_index, dtype=np.int32)
                # cu_window_seqlens: true values + patch_k sentinel tail.
                c = np.full(win_k, patch_k, dtype=np.int32)
                n_win = int(m.cu_window_seqlens.shape[0])
                c[:n_win] = np.asarray(m.cu_window_seqlens, dtype=np.int32)
                # rotary_pos_emb: true rows + zero pad to patch_k.
                r = np.zeros((patch_k, rot_dim), dtype=np.float32)
                rp = np.asarray(m.rotary_pos_emb, dtype=np.float32)
                r[: int(rp.shape[0])] = rp
                window_indices.append(w)
                cu_window_seqlens_rows.append(c)
                rotary_rows.append(r)
        return Qwen25VLVisionMetadata(
            np.stack(window_indices),
            np.stack(cu_window_seqlens_rows),
            np.stack(rotary_rows),
        )


register_vision_metadata_builder(
    "Qwen2_5_VLForConditionalGeneration", Qwen25VLVisionMetadataBuilder
)
