"""MiMo-V2.5 host-side processor (HF-processor shaped).

P1 design (design doc §4.3.5): make MiMo-V2.5 look like a normal HF processor so the
generic ``MultimodalTokenizer`` needs no model special-casing. MiMo-V2.5's real HF
processor is ``Qwen2_5_VLProcessor`` (vision-only); this wrapper composes it for
image/video/text and runs the host RVQ codec for audio, returning a single
``processor_out`` dict with **expanded** ``input_ids`` + ``audio_codes`` (+ meta).

Layering for testability: the vision/text call is a thin lazy ``transformers`` layer;
the audio merge / placeholder expansion / host validation is pure (numpy + codec) and
unit-testable by injecting a fake HF processor and codec.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
    MiMoV25AudioPayload,
)

DEFAULT_AUDIO_TOKEN_ID = 151669

_MIMO_V25_MODEL_TYPES = {"mimo_v2_5", "mimo_v25", "mimo2_5"}


def _cfg_value(config, key, default=None):
    value = getattr(config, key, None)
    if value is not None:
        return value
    processor_config = getattr(config, "processor_config", None)
    if isinstance(processor_config, dict):
        return processor_config.get(key, default)
    if processor_config is not None:
        return getattr(processor_config, key, default)
    return default


def _audio_contract(mm_config) -> tuple[int, int, int]:
    audio_config = getattr(mm_config, "audio_config", None)

    def _get(key, default):
        if isinstance(audio_config, dict):
            return int(audio_config.get(key, default))
        return int(getattr(audio_config, key, default))

    return (_get("audio_channels", 20), _get("speech_vocab_size", 1280), _get("group_size", 4))


class MiMoV25Processor:
    """Composes Qwen2.5-VL vision/text processing + MiMo-V2.5 RVQ audio codec.

    Deliberately exposes no ``feature_extractor`` attribute: that signals the generic
    tokenizer to pass raw audio sources (the codec loads them) rather than pre-loaded
    waveforms (which continuous-audio HF processors like Qwen3-Omni expect).
    """

    @staticmethod
    def matches(mm_config) -> bool:
        """True if this processor should wrap the model's HF processor (MiMo-V2.5 omni)."""
        if mm_config is None:
            return False
        model_type = (
            str(getattr(mm_config, "model_type", "") or "")
            .lower()
            .replace("-", "_")
            .replace(".", "_")
        )
        if model_type in _MIMO_V25_MODEL_TYPES:
            return True
        # shares model_type "mimo_v2" with text-only Pro/Flash; detect by audio capability
        if model_type != "mimo_v2" or getattr(mm_config, "audio_config", None) is None:
            return False
        return _audio_contract(mm_config) == (20, 1280, 4) and (
            _cfg_value(mm_config, "audio_token_id") == 151669
        )

    @classmethod
    def from_hf_processor(cls, mm_config, model_path, hf_processor, *, trust_remote_code=True):
        """Build by wrapping an already-loaded vision HF processor (Qwen2.5-VL)."""
        num_channels, codebook_size, group_size = _audio_contract(mm_config)
        audio_token_id = int(_cfg_value(mm_config, "audio_token_id", DEFAULT_AUDIO_TOKEN_ID))
        return cls(
            model_path,
            audio_token_id=audio_token_id,
            num_channels=num_channels,
            codebook_size=codebook_size,
            group_size=group_size,
            trust_remote_code=trust_remote_code,
            hf_processor=hf_processor,
        )

    def __init__(
        self,
        model_path: str | None = None,
        *,
        audio_token_id: int = DEFAULT_AUDIO_TOKEN_ID,
        num_channels: int = 20,
        codebook_size: int = 1280,
        group_size: int = 4,
        trust_remote_code: bool = True,
        hf_processor: Any = None,  # injectable Qwen2_5_VLProcessor (tests / reuse)
        codec: MiMoV25AudioCodecProcessor | None = None,  # injectable for tests
    ):
        self.model_path = model_path
        self.audio_token_id = audio_token_id
        self.num_channels = num_channels
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.trust_remote_code = trust_remote_code
        self._hf_processor = hf_processor
        self._codec = codec

    # ---- lazy heavy deps (transformers) ----
    @property
    def wrapped_hf_processor(self):
        """The underlying HF processor this composes (Qwen2.5-VL), if already loaded.

        Lets the generic tokenizer sniff the wrapped processor's capabilities (e.g.
        Qwen video preprocessing) without forcing a lazy load or knowing about MiMo.
        """
        return self._hf_processor

    def _get_hf_processor(self):
        if self._hf_processor is None:
            from transformers import AutoProcessor

            self._hf_processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        return self._hf_processor

    def _get_codec(self) -> MiMoV25AudioCodecProcessor:
        if self._codec is None:
            self._codec = MiMoV25AudioCodecProcessor(
                self.model_path,
                audio_token_id=self.audio_token_id,
                trust_remote_code=self.trust_remote_code,
            )
        return self._codec

    def apply_chat_template(self, *args, **kwargs):
        """Delegate chat-template rendering to the wrapped HF processor.

        The OpenAI serving path calls ``mm_processor.apply_chat_template`` for
        multimodal models. MiMo-V2.5 wraps the Qwen2.5-VL processor, so keep that
        processor as the source of truth for audio placeholder text.
        """
        processor = self._get_hf_processor()
        if not hasattr(processor, "apply_chat_template"):
            raise AttributeError(
                "Wrapped MiMo-V2.5 HF processor does not expose apply_chat_template"
            )
        return processor.apply_chat_template(*args, **kwargs)

    # ---- HF-processor-shaped entry ----
    def __call__(
        self,
        images=None,
        videos=None,
        audio=None,
        text: str = "",
        return_tensors=None,
        **kwargs,
    ) -> dict:
        # Vision + text via the real (vision-only) Qwen processor.
        out = dict(
            self._get_hf_processor()(
                images=images or None,
                videos=videos or None,
                text=text or "",
                return_tensors=return_tensors,
                **kwargs,
            )
        )
        # Audio via host RVQ codec, merged into the same processor_out shape.
        if self._has_audio(audio):
            self._merge_audio(out, audio)
        return out

    @staticmethod
    def _has_audio(audio) -> bool:
        if audio is None:
            return False
        if isinstance(audio, (list, tuple)):
            return len(audio) > 0
        return True

    def _encode_audio(self, audio) -> MiMoV25AudioPayload:
        """Raw audio sources → host RVQ codec.encode → payload (codes + meta)."""
        return self._get_codec().encode(audio)

    def _merge_audio(self, out: dict, audio) -> None:
        """Pure merge: expand <audio_pad> in input_ids, host-validate, attach codes+meta."""
        payload = self._encode_audio(audio)
        input_ids = self._extract_input_ids(out)
        if input_ids is None:
            raise ValueError(
                "MiMo-V2.5 audio requests require a prompt that tokenizes to "
                "<audio_pad> placeholders (processor produced no input_ids)."
            )
        input_ids = MiMoV25AudioCodecProcessor.expand_single_audio_placeholders(input_ids, payload)
        # Host-side guard (review D3-6): #placeholders must match codes rows.
        MiMoV25AudioCodecProcessor.validate_placeholder_count(input_ids, payload)
        out["input_ids"] = np.asarray([input_ids], dtype=np.int64)
        out["audio_codes"] = np.asarray(payload.codes)
        # Per-item meta consumed by the tokenizer's generic audio_codes -> mm_item rule.
        out["audio_token_lengths"] = [int(x) for x in payload.token_lengths]
        out["audio_offsets"] = payload.offsets
        out["audio_group_size"] = int(payload.group_size)
        out["audio_codebook_sizes"] = payload.codebook_sizes

    @staticmethod
    def _extract_input_ids(out: dict):
        """Normalize processor input_ids (tensor [1,L] / [[...]] / [...]) to a flat list."""
        ids = out.get("input_ids")
        if ids is None:
            return None
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        # unwrap a single batch dim if present
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return list(ids)
