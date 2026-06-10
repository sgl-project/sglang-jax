"""Qwen2.5-VL multimodal input processor (refactor M3, design §3.5.1).

Productionizes the validated front-end transform (tmp/refactor/frontend_transform.py):
real HF AutoProcessor -> input_ids (placeholders) + pixel_values + grid_thw -> mm_items ->
set_pad_value -> mRoPE -> pad_input_tokens. pad_input_tokens runs AFTER mRoPE because mRoPE
locates vision spans by the *raw* image/video token id in input_ids; it bakes each item's
pad_value into the placeholder rows so the in-model merge()'s isin(input_ids, pad_values)
finds exactly those rows (== the ViT output rows). Registered by HF arch name into the
mm_core ProcessorRegistry; the standard TokenizerManager resolves it for understanding reqs.
"""

from __future__ import annotations

import base64
import io
import logging
import os

import numpy as np

from sgl_jax.srt.mm_core.processor import BaseMultimodalProcessor
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    pad_input_tokens,
)
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_mrope_positions

logger = logging.getLogger(__name__)


def _strip_batch_dim(arr):
    if arr is None:
        return None
    if hasattr(arr, "cpu"):  # torch tensor
        arr = arr.cpu().numpy()
    array = np.asarray(arr)
    if array.ndim > 1 and array.shape[0] == 1:
        return array[0]
    return array


def _to_grid_list(grid_thw):
    """Normalize a grid_thw tensor/array into a list of (t, h, w) int tuples (or None).

    compute_mrope_positions / assemble_mm_inputs expect a list, not a raw np array
    (a bare `grid or []` truth-test on an array is ambiguous). Mirrors
    MultimodalTokenizer._to_grid_list.
    """
    if grid_thw is None:
        return None
    if hasattr(grid_thw, "cpu"):  # torch tensor
        grid_thw = grid_thw.cpu().numpy()
    grid = np.asarray(grid_thw)
    if grid.size == 0:
        return None
    return [tuple(int(x) for x in row) for row in grid.tolist()]


def _to_list(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def _load_image(source):
    """Load one image source into a PIL RGB image (design §3.5.1: loading is the
    processor's job, not the TokenizerManager's). Accepts a pre-loaded PIL image, raw
    bytes, a local path, an http(s) URL, a data: URI, or a bare base64 string. Ported
    from MultimodalTokenizer._load_image_from_source (PIL passthrough added)."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, dict) and "url" in source:
        source = source["url"]
    if hasattr(source, "url"):
        source = source.url
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, str) and os.path.exists(source):
        return Image.open(source).convert("RGB")
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        import requests

        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if isinstance(source, str) and source.startswith("data:") and "base64," in source:
        payload = source.split("base64,", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    try:
        return Image.open(io.BytesIO(base64.b64decode(source, validate=True))).convert("RGB")
    except Exception as exc:
        raise ValueError("Unsupported image source format") from exc


class Qwen2_5_VLProcessor(BaseMultimodalProcessor):
    """Image (+video) understanding processor for Qwen2.5-VL (text-out)."""

    models = ["Qwen2_5_VLForConditionalGeneration"]

    def __init__(self, model_path: str):
        from transformers import AutoConfig, AutoProcessor

        self.hf_processor = AutoProcessor.from_pretrained(model_path)
        self.hf_config = AutoConfig.from_pretrained(model_path)
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)
        self.video_token_id = getattr(self.hf_config, "video_token_id", None)
        self.vision_start_token_id = getattr(self.hf_config, "vision_start_token_id", None)
        vision = getattr(self.hf_config, "vision_config", None)
        self.spatial_merge_size = getattr(vision, "spatial_merge_size", 2)
        self.tokens_per_second = getattr(vision, "tokens_per_second", None)

    def apply_chat_template(self, *args, **kwargs):
        """Delegate to the HF processor's chat template (serving_chat calls this on the
        mm_processor). The HF processor carries the model's chat_template even when the bare
        tokenizer does not (e.g. Qwen3-Omni)."""
        return self.hf_processor.apply_chat_template(*args, **kwargs)

    def process(self, *, images=None, videos=None, audios=None, text=None):
        if audios:
            raise NotImplementedError("Qwen2.5-VL processor does not handle audio")
        # Load raw image sources (URL/path/base64/bytes) -> PIL; pre-loaded PIL pass through.
        if images:
            images = [_load_image(s) for s in images]
        out = self.hf_processor(
            text=[text] if isinstance(text, str) else text,
            images=images or None,
            videos=videos or None,
            return_tensors="pt",
        )
        input_ids = _to_list(out["input_ids"][0])
        pixel_values = _strip_batch_dim(out.get("pixel_values"))
        pixel_values_videos = _strip_batch_dim(out.get("pixel_values_videos"))
        image_grid_thw = _to_grid_list(out.get("image_grid_thw"))
        video_grid_thw = _to_grid_list(out.get("video_grid_thw"))
        second_per_grid_ts = out.get("second_per_grid_ts")

        # mRoPE FIRST -- it scans the raw image/video token ids in input_ids to locate spans.
        mrope_positions = mrope_position_delta = None
        if self.vision_start_token_id is not None and self.image_token_id is not None:
            mrope_positions, mrope_position_delta = compute_mrope_positions(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                vision_start_token_id=self.vision_start_token_id,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                spatial_merge_size=self.spatial_merge_size,
                tokens_per_second=self.tokens_per_second,
            )

        mm_items = []
        if pixel_values is not None:
            mm_items.append(MultimodalDataItem(modality=Modality.IMAGE, feature=pixel_values))
        if pixel_values_videos is not None:
            mm_items.append(
                MultimodalDataItem(modality=Modality.VIDEO, feature=pixel_values_videos)
            )
        for item in mm_items:
            item.set_pad_value()

        # pad_input_tokens AFTER mRoPE: bake per-item pad_value into placeholder rows so the
        # in-model merge() can key on them.
        padded_input_ids = pad_input_tokens(
            input_ids,
            mm_items,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

        return {
            "input_ids": padded_input_ids,
            "mm_inputs": {
                "mm_items": mm_items,
                "im_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "mrope_positions": mrope_positions,
                "mrope_position_delta": mrope_position_delta,
                "second_per_grid_ts": (
                    _to_list(second_per_grid_ts) if second_per_grid_ts is not None else None
                ),
            },
        }
