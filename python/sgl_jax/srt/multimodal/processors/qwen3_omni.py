"""Qwen3-Omni multimodal processor (refactor M5).

The image/video transform is identical to Qwen2.5-VL (same HF processor output keys, mRoPE,
pad_input_tokens), so this subclasses Qwen2_5_VLProcessor and only overrides __init__ to
resolve the nested thinker_config (token ids / spatial_merge_size live under
hf_config.thinker_config for Qwen3-Omni). Registered for the omni arch. Audio (continuous
mel) is a follow-up; process() rejects audio inputs for now.
"""

from __future__ import annotations

import logging

from sgl_jax.srt.multimodal.processors.qwen_vl import Qwen2_5_VLProcessor

logger = logging.getLogger(__name__)


class Qwen3OmniMoeProcessor(Qwen2_5_VLProcessor):
    """Image/video understanding processor for Qwen3-Omni (Thinker, text-out)."""

    models = ["Qwen3OmniMoeForConditionalGeneration"]

    def __init__(self, model_path: str):
        from transformers import AutoConfig, AutoProcessor

        self.hf_processor = AutoProcessor.from_pretrained(model_path)
        self.hf_config = AutoConfig.from_pretrained(model_path)
        # Qwen3-Omni nests vision/token-id config under thinker_config.
        thinker = getattr(self.hf_config, "thinker_config", self.hf_config)
        self.image_token_id = getattr(thinker, "image_token_id", None)
        self.video_token_id = getattr(thinker, "video_token_id", None)
        self.vision_start_token_id = getattr(thinker, "vision_start_token_id", None)
        vision = getattr(thinker, "vision_config", None)
        self.spatial_merge_size = getattr(vision, "spatial_merge_size", 2)
        self.tokens_per_second = getattr(vision, "tokens_per_second", None)
