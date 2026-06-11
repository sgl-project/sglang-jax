"""MiMo-V2.5 in-model multimodal processor (refactor M4).

MiMo-V2.5's vision is Qwen2.5-VL-style, so reuse Qwen2_5_VLProcessor: it is model-path-driven
(AutoProcessor.from_pretrained(model_path) + reads image/video token ids + spatial_merge_size +
vision_start_token_id from the model config), so pointing it at the MiMo-V2.5 checkpoint yields a
working image/video processor + the model's chat template. Registered (models=[...]) for the
in-model arch the model_config remap produces.

Audio: MiMo-V2.5 uses RVQ DISCRETE codes (not continuous mel). The encode pass + embed_mm already
carry mm_audio_codes, but EMITTING the codes here needs the host RVQ codec (the staged
MiMoV25Processor composes MiMoV25AudioCodecProcessor). That audio wiring is a follow-up; image/
video + text work through the Qwen2.5-VL parent.
"""

from __future__ import annotations

from sgl_jax.srt.multimodal.processors.qwen_vl import Qwen2_5_VLProcessor


class MiMoV2_5Processor(Qwen2_5_VLProcessor):
    """Image/video (text-out) processor for in-model MiMo-V2.5. Vision via the Qwen2.5-VL parent."""

    models = ["MiMoV2_5ForConditionalGeneration"]

    def __init__(self, model_path: str):
        # MiMo-V2.5 ships a custom HF processor/config -> needs trust_remote_code=True (the bare
        # Qwen2.5-VL parent defaults to False since its processor is built-in).
        super().__init__(model_path, trust_remote_code=True)
