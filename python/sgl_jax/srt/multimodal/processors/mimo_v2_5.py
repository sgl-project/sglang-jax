"""MiMo-V2.5 in-model multimodal processor (refactor M4).

MiMo-V2.5's vision is Qwen2.5-VL-style, so reuse Qwen2_5_VLProcessor for image/video (it is
model-path-driven: AutoProcessor.from_pretrained(model_path) + reads token ids / spatial_merge_size
/ vision_start_token_id from config). Audio is RVQ DISCRETE codes: run the host RVQ codec
(MiMoV25AudioCodecProcessor, which loads {model_path}/audio_tokenizer and encodes mel -> codes),
expand the single <audio_pad> placeholder to the codec token length, and attach the codes as an
AUDIO mm_item (is_codes meta) so assemble_mm_inputs routes it to audio_codes -> the encode pass ->
embed_mm.encode_audio. Mirrors the staged MiMoV25Processor._merge_audio. Registered (models=[...])
for the in-model arch the model_config remap produces.
"""

from __future__ import annotations

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    pad_input_tokens,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import Qwen2_5_VLProcessor


class MiMoV2_5Processor(Qwen2_5_VLProcessor):
    """Image/video + RVQ-codes audio (text-out) processor for in-model MiMo-V2.5."""

    models = ["MiMoV2_5ForConditionalGeneration"]
    # MiMo's AR uses standard positions (forward_batch.positions, mimo_v2_flash.py), NOT mRoPE (the
    # staged AR config sets mrope:False) -> tell the Qwen2.5-VL parent to skip mrope entirely. The
    # parent's vision-span mrope is both unused by MiMo and, being keyed to the pre-expansion
    # length, would go stale after the audio placeholder expansion below (broke _merge_multimodal).
    uses_mrope = False

    def __init__(self, model_path: str, *, hf_processor=None, hf_config=None, codec=None):
        # MiMo-V2.5 ships a custom HF processor/config -> trust_remote_code=True (the bare
        # Qwen2.5-VL parent defaults to False since its processor is built-in). hf_processor /
        # hf_config / codec are injectable for unit tests (no checkpoint on disk).
        super().__init__(
            model_path,
            trust_remote_code=True,
            hf_processor=hf_processor,
            hf_config=hf_config,
        )
        self._model_path = model_path
        self.audio_token_id = getattr(self.hf_config, "audio_token_id", None)
        if self.audio_token_id is None:
            proc_cfg = getattr(self.hf_config, "processor_config", None)
            if isinstance(proc_cfg, dict):
                self.audio_token_id = proc_cfg.get("audio_token_id", 151669)
            else:
                self.audio_token_id = 151669
        self._codec = codec

    def _get_codec(self):
        if self._codec is None:
            from sgl_jax.srt.models.mimo_v2_5.audio_codec_processor import (
                MiMoV25AudioCodecProcessor,
            )

            self._codec = MiMoV25AudioCodecProcessor(
                model_path=self._model_path, audio_token_id=self.audio_token_id
            )
        return self._codec

    def process(self, *, images=None, videos=None, audios=None, text=None):
        # Vision + text via the Qwen2.5-VL parent (its audio path is gated on audio_token_id being
        # set on the HF call; MiMo audio is codes, handled below -- so pass audios=None here). mrope
        # is already None here because uses_mrope=False (see class docstring), so no clearing needed.
        out = super().process(images=images, videos=videos, audios=None, text=text)
        if not audios:
            return out

        from sgl_jax.srt.models.mimo_v2_5.audio_codec_processor import (
            MiMoV25AudioCodecProcessor,
        )

        payload = self._get_codec().encode(audios)
        ids = out["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids[0]) if (ids and isinstance(ids[0], (list, tuple))) else list(ids)
        # Expand the one-pad-per-audio template placeholder to the codec token length, then guard
        # #placeholders == codes rows (host contract, mirrors the staged _merge_audio).
        ids = MiMoV25AudioCodecProcessor.expand_single_audio_placeholders(ids, payload)
        MiMoV25AudioCodecProcessor.validate_placeholder_count(ids, payload)

        mm = out["mm_inputs"]
        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=np.asarray(payload.codes),
            model_specific_data={
                "is_codes": True,
                "token_lengths": [int(x) for x in payload.token_lengths],
                "group_size": int(payload.group_size),
                "codebook_sizes": payload.codebook_sizes,
            },
        )
        audio_item.set_pad_value()
        mm["mm_items"].append(audio_item)
        mm["audio_token_id"] = self.audio_token_id
        out["input_ids"] = ids
        # Rebuild the radix cache key copy now that audio placeholders are expanded + the audio
        # item exists (Scheme B: pad copy lives only in cache_input_ids).
        mm["cache_input_ids"] = pad_input_tokens(
            ids,
            mm["mm_items"],
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
        )
        return out
