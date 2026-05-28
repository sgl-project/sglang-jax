# Adapted from sgl-jax stage path MultimodalTokenizer logic and upstream
# sglang QwenVLImageProcessor (Qwen2/Qwen2.5/Qwen3-VL share one processor).
"""Minimal CPU processor for Qwen3-VL Dense (pilot #256).

Thin wrapper around HuggingFace `Qwen3VLProcessor` (AutoProcessor):
  PIL.Image + chat-templated text  ->  pixel_values + image_grid_thw + input_ids
                                  ->  MultimodalInputs (mm_items, mrope_positions, ...)

Pilot scope:
  - Image only (no video, no audio).
  - Single request at a time (batching done at HF level via padding=True).
  - No CUDA IPC, no parallel data loading, no radix cache pad_value optimization.

For the full architecture (BaseMultimodalProcessor with 1200+ lines: data loading
concurrency, radix-cache pad values, validation, video frame sampling, etc.), see
upstream sglang `srt/multimodal/processors/base_processor.py`. We deliberately
stay minimal at pilot stage and revisit in the broader #254 CPU plumbing work.
"""

from __future__ import annotations

import base64
import io
import logging
from urllib.parse import urlparse

from PIL import Image
from transformers import AutoProcessor

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_mrope_positions

logger = logging.getLogger(__name__)


# Token ids carried by Qwen3-VL tokenizer; surfaced here so this processor stays
# self-contained without needing to read hf_config from outside.
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656


def _load_image_from_source(source) -> Image.Image:
    """Load a PIL.Image from URL / data URL / base64 / bytes / file path / PIL.Image / ImageData."""
    # ImageData (from sgl_jax.srt.managers.io_struct) wraps a URL string.
    if hasattr(source, "url") and isinstance(source.url, str):
        source = source.url
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, str):
        if source.startswith("data:"):
            # data:image/...;base64,XXX
            payload = source.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            import requests

            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Local file path
        return Image.open(source).convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source)!r}")


class Qwen3VLProcessor:
    """Minimal Qwen3-VL CPU preprocessing.

    Usage:
        proc = Qwen3VLProcessor(model_path="/models/Qwen3-VL-8B-Instruct",
                                spatial_merge_size=2)
        out = proc.process(prompt_text, image_data=[url_or_pil, ...])
        # out is a dict:
        #   input_ids: list[int]                  -- expanded with image tokens
        #   mm_inputs: MultimodalInputs           -- carries pixel_values / grid_thw / mrope
    """

    def __init__(
        self,
        model_path: str,
        spatial_merge_size: int = 2,
        tokens_per_second: int | float | None = None,
        trust_remote_code: bool = True,
    ):
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.tokenizer = self._processor.tokenizer
        self.spatial_merge_size = spatial_merge_size
        self.tokens_per_second = tokens_per_second
        self.vision_start_token_id = _VISION_START_TOKEN_ID
        self.vision_end_token_id = _VISION_END_TOKEN_ID
        self.image_token_id = _IMAGE_TOKEN_ID
        self.video_token_id = _VIDEO_TOKEN_ID

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process(
        self,
        text: str,
        image_data: list | None = None,
    ) -> dict:
        """Run HF processor + MRoPE; return input_ids and a MultimodalInputs.

        Args:
            text: chat-templated prompt string (e.g. produced by
                `tokenizer.apply_chat_template(..., add_generation_prompt=True)`).
                Must already contain placeholder tokens like
                `<|vision_start|><|image_pad|><|vision_end|>` per image.
            image_data: list of image sources (URL / base64 / bytes / path / PIL).

        Returns:
            dict with keys:
                input_ids: list[int]
                mm_inputs: MultimodalInputs | None  (None if text-only)
        """
        if not image_data:
            # Text-only fast path
            input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            return {"input_ids": input_ids, "mm_inputs": None}

        images = [_load_image_from_source(src) for src in image_data]
        processor_out = self._processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=False,
        )

        # HF returns torch tensors; convert to numpy/list immediately.
        input_ids = processor_out["input_ids"][0].tolist()
        pixel_values = processor_out["pixel_values"].cpu().numpy()
        image_grid_thw_t = processor_out["image_grid_thw"].cpu().numpy()
        # image_grid_thw shape [N_images, 3] -> list of (t, h, w) tuples
        image_grid_thw = [(int(g[0]), int(g[1]), int(g[2])) for g in image_grid_thw_t]

        # MRoPE 3D positions
        mrope_positions, mrope_position_delta = compute_mrope_positions(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            spatial_merge_size=self.spatial_merge_size,
            tokens_per_second=self.tokens_per_second,
        )

        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixel_values,
            model_specific_data={"image_grid_thw": image_grid_thw},
        )
        item.set_pad_value()  # for radix cache hashing

        # Use dict form (matches stage path layout so scheduler.handle_generate_request
        # can use `.get(...)` uniformly; downstream `_collect_mm_inputs` and
        # `ForwardBatch.mm_inputs` are both dict-friendly).
        mm_inputs = {
            "mm_items": [item],
            "im_start_id": self.vision_start_token_id,
            "im_end_id": self.vision_end_token_id,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
            "image_grid_thw": image_grid_thw,
        }
        return {"input_ids": input_ids, "mm_inputs": mm_inputs}

    def apply_chat_template(self, messages, **kwargs) -> str:
        """Convenience pass-through; uses HF processor's chat template."""
        return self._processor.apply_chat_template(messages, **kwargs)
