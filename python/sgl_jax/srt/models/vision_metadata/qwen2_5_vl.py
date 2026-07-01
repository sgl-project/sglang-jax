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

from sgl_jax.srt.multimodal.common.vision_metadata import (
    register_vision_metadata_builder,
)


@register_pytree_node_class
@dataclass
class Qwen25VLVisionMetadata:
    """Per-round Qwen2.5-VL vision aux, scheduler-computed, threaded into encode JIT.

    Per-arch registered pytree: common code treats the encode ``meta`` as an
    OPAQUE pytree and never names these fields;
    ``Qwen25VLVisionMetadataBuilder`` (below) constructs it and only the ViT
    encode body interprets it. Children flatten order is fixed:
    ``window_index``, ``cu_window_seqlens``, ``rotary_pos_emb``.

    Fields hold one round's pad-stacked-across-ranks ViT aux:

    - ``window_index``:      ``[dp, merge_units]`` int -- unit-granularity
      window-order permutation (identity-padded across ranks so ``argsort``
      un-permute holds).
    - ``cu_window_seqlens``: ``[dp, max_windows]`` int -- cumulative window
      boundaries on the window-reordered layout (no leading 0; last real value =
      true patch count; sentinel-padded to ``max_windows``). Per-patch segment
      ids are computed inside the ViT forward via
      ``searchsorted(cu_window_seqlens, arange(seq))`` (host no longer
      pre-derives them).
    - ``rotary_pos_emb``:    ``[dp, patch_k, rot_dim]`` -- per-patch 2D rope
      (zero-padded).
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


def _item_grid_thw(item: Any) -> tuple:
    """First ``(t, h, w)`` of this item's ``image_grid_thw`` as a python-int tuple.

    Qwen-specific: the builder pulls its geometry FROM the item here, so the
    common ``get_metadata(item)`` interface stays arch-agnostic.

    # TODO(stage1): one round handles ONE image per rank, so we read row 0 only;
    #   multi-image-per-item packing needs the full grid list.
    """
    md = getattr(item, "model_specific_data", None)
    grid = md.get("image_grid_thw") if isinstance(md, dict) else getattr(md, "image_grid_thw", None)
    if grid is None:
        grid = getattr(item, "image_grid_thw", None)
        if grid is None and isinstance(item, dict):
            grid = item.get("image_grid_thw")
    if grid is None:
        return (0, 0, 0)
    arr = np.asarray(grid)
    row = arr if arr.ndim == 1 else arr[0]
    return (int(row[0]), int(row[1]), int(row[2]))


class Qwen25VLVisionMetadataBuilder:
    """Per-arch, config-only ViT aux builder.

    Pure numpy, NO weights, NO model instance: the scheduler instantiates this
    from ``model_config.vision_config`` and calls :meth:`get_metadata` once per
    image to produce a :class:`Qwen25VLVisionMetadata` (``window_index`` /
    ``cu_window_seqlens`` / ``rotary_pos_emb``). Reuses the same window/rope
    algorithm the ViT used host-side (ported off ``jnp`` to ``np``);
    ``cu_window_seqlens`` is carried as-is (cumulative window boundaries) and
    converted to per-patch segment ids INSIDE the ViT forward (``searchsorted``),
    not here.

    :meth:`get_metadata` produces single-image native metadata (pulling the
    grid from the item). :meth:`stack_metadata` owns the per-arch cross-rank
    pad-by-role and stack into the round-loop bucket.
    """

    def __init__(self, vision_cfg):
        self.vision_cfg = vision_cfg
        self.patch_size = int(getattr(vision_cfg, "patch_size", 14))
        self.window_size = int(getattr(vision_cfg, "window_size", 112))
        self.spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        self.fullatt_block_indexes = list(
            getattr(vision_cfg, "fullatt_block_indexes", [7, 15, 23, 31])
        )
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
        """One ``MultimodalDataItem`` -> :class:`Qwen25VLVisionMetadata`.

        Pulls the Qwen ``image_grid_thw`` ``(t, h, w)`` from ``item`` (arch-
        specific), then:

        - ``window_index``: unit-granularity window-order permutation (length
          ``t * (h//sms) * (w//sms)`` = merge_units).
        - ``rotary_pos_emb``: per-patch 2D rope, ALREADY gathered into window
          order (``rope[window_index]`` flattened), length ``t*h*w`` patches.
        - ``cu_window_seqlens``: cumulative window boundaries on the
          window-reordered layout (``cumsum(seqlens) * spatial_merge_unit``; no
          leading 0; last value = true patch count). Per-patch segment ids
          (``searchsorted(cu_window_seqlens, patch, side="right")``) are NOT
          pre-derived here -- computed inside ``compute_hidden_states`` (in the
          encode JIT), right before the flash kernel.

        ``cu_seqlens`` (whole-image boundary) is NOT produced: round-loop single
        image -> full-att = full attention + ``valid`` padding mask.
        """
        t, h, w = _item_grid_thw(item)

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
        """Cross-rank pad-by-role + stack of single-image metas -> ``[dp, ...]``.

        ``metas[r]`` is this rank's round-k native-size
        :class:`Qwen25VLVisionMetadata`, or ``None`` for a dummy lane (rank owns
        < k+1 images). ``patch_k`` is the round's cross-rank max patch-row bucket
        (drives the cu sentinel and rope pad length). Bucket sizes are the
        cross-rank max of each role's native length:

        - ``window_index`` is a PERMUTATION over ``units_k`` (= merge_units):
          identity-fill pad slots with ``arange(native_units, units_k)`` so the
          full row stays a valid permutation (the ViT ``argsort``-un-permutes; a
          non-permutation would corrupt that reverse-scatter). Dummy lanes get a
          full ``arange(units_k)`` identity row.
        - ``cu_window_seqlens`` (cumulative boundaries): sentinel = ``patch_k``
          for pad slots. cu's last real value = true patch count <= ``patch_k``,
          so a ``patch_k`` sentinel keeps the row non-decreasing AND > every real
          patch index, hence the forward's ``searchsorted(side="right")`` never
          counts a sentinel window for any real patch. Dummy lanes get an
          all-``patch_k`` row (irrelevant -- masked by ``valid`` in the forward).
        - ``rotary_pos_emb`` (per-patch values): 0 for pad patches; rope is
          padded to ``patch_k`` rows.
        """
        present = [m for m in metas if m is not None]
        units_k = max(int(m.window_index.shape[0]) for m in present)
        win_k = max(int(m.cu_window_seqlens.shape[0]) for m in present)
        rot_dim = int(present[0].rotary_pos_emb.shape[-1])
        wi, cu, rope = [], [], []
        for m in metas:
            if m is None:  # dummy lane
                wi.append(np.arange(units_k, dtype=np.int32))  # identity perm (un-permute safe)
                cu.append(np.full(win_k, patch_k, dtype=np.int32))  # all sentinel
                rope.append(np.zeros((patch_k, rot_dim), dtype=np.float32))
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
                wi.append(w)
                cu.append(c)
                rope.append(r)
        return Qwen25VLVisionMetadata(np.stack(wi), np.stack(cu), np.stack(rope))


register_vision_metadata_builder(
    "Qwen2_5_VLForConditionalGeneration", Qwen25VLVisionMetadataBuilder
)
