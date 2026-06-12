"""Shared assembly of per-modality model inputs from ``mm_items`` (neutral mm CORE).

Relocated from ``multimodal/manager/mm_assembly.py`` in M6-S1: it is shared by the in-model
understanding path (``ModelRunner.encode_mm_reqs``) and -- until M6-S5 removes them -- the staged
embed/vit model runners, so it belongs in the neutral ``mm_core`` layer that ``srt`` may import
directly (the old location forced an ``importlib`` workaround to dodge the srt->multimodal reverse
import). ``mm_items`` is the single source of truth for multimodal features; this helper turns them
into the per-modality host-side kwargs bundle the encoders consume. No model-specific logic: audio
routing (discrete codes vs continuous features) is a generic per-item flag.

Pure numpy (no jax) so it is unit-testable without the model stack. Items may be
``MultimodalDataItem`` objects or their transport dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_item(item):
    """Normalize a transport dict into a MultimodalDataItem (lazy import to stay light)."""
    if isinstance(item, dict):
        from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem

        return MultimodalDataItem.from_dict(item)
    return item


def _concat(features) -> np.ndarray | None:
    arrays = [np.asarray(f) for f in features if f is not None]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def _is_codes_audio(item) -> bool:
    """Discrete-codes audio (e.g. MiMo-V2.5 RVQ) vs continuous features (e.g. Qwen mel).

    Decided by a generic per-item marker carried in model_specific_data, so the
    assembler stays model-agnostic.
    """
    meta = getattr(item, "model_specific_data", None) or {}
    return bool(meta.get("is_codes")) or "token_lengths" in meta


def assemble_mm_inputs(mm_inputs: dict | None) -> dict[str, Any]:
    """Turn an ``mm_inputs`` dict (carrying ``mm_items`` + grids) into a host-side
    per-modality kwargs bundle (numpy arrays / None), generic across models.

    Returned keys mirror the embed model ``__call__`` surface:
    ``pixel_values_images``, ``pixel_values_videos``, ``audio_codes`` (discrete) or
    ``audio_features`` (continuous), ``audio_feature_attention_mask``,
    ``image_grid_thw``, ``video_grid_thw``.
    """
    out: dict[str, Any] = {
        "pixel_values": None,
        "pixel_values_images": None,
        "pixel_values_videos": None,
        "audio_codes": None,
        "audio_features": None,
        "audio_feature_attention_mask": None,
        "image_grid_thw": None,
        "video_grid_thw": None,
    }
    if not isinstance(mm_inputs, dict):
        return out

    images, videos, codes_audio, cont_audio = [], [], [], []
    for raw in mm_inputs.get("mm_items", []) or []:
        item = _as_item(raw)
        if item.is_image():
            images.append(item)
        elif item.is_video():
            videos.append(item)
        elif item.is_audio():
            (codes_audio if _is_codes_audio(item) else cont_audio).append(item)

    image_feats = [it.feature for it in images]
    video_feats = [it.feature for it in videos]
    out["pixel_values_images"] = _concat(image_feats)
    out["pixel_values_videos"] = _concat(video_feats)
    # combined image+video features, for the vit stage which takes a single pixel_values
    out["pixel_values"] = _concat(image_feats + video_feats)
    # A model's audio is either discrete codes or continuous features, not both.
    out["audio_codes"] = _concat([it.feature for it in codes_audio])
    out["audio_features"] = _concat([it.feature for it in cont_audio])
    out["audio_feature_attention_mask"] = mm_inputs.get("audio_feature_attention_mask")
    out["image_grid_thw"] = mm_inputs.get("image_grid_thw")
    out["video_grid_thw"] = mm_inputs.get("video_grid_thw")
    return out
