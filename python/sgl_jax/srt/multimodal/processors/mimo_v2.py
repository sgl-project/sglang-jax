"""Multimodal processor for MiMoV2 VLM checkpoints (vision + audio).

MiMoV2's vision stack is Qwen-VL-derived (``grid_thw`` + spatial merge), so this
reuses :class:`QwenVLProcessor`'s image/video loading, HF-processor call, and
placeholder-range helpers.  It differs in two ways:

* MiMoV2's language model uses standard 1-D RoPE (not mRoPE), so this processor
  does **not** emit ``mrope_positions`` / ``mrope_position_delta`` — leaving them
  unset keeps the scheduler on the standard position path.
* Audio inputs are tokenized to speech codes (either passed pre-encoded, or via
  the MiMoV2 audio tokenizer) and merged as ``Modality.AUDIO`` items whose
  ``feature`` is the ``[T, C]`` code array consumed by the in-model audio tower.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import unquote, urlparse

import numpy as np
import numpy.typing as npt
import requests
from transformers import PretrainedConfig
from transformers.processing_utils import ProcessorMixin

from sgl_jax.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
from sgl_jax.srt.server_args import ServerArgs

IntArray = npt.NDArray[np.int32]
FloatArray = npt.NDArray[np.float32]
AudioSource = (
    str
    | bytes
    | os.PathLike[str]
    | npt.NDArray[Any]
    | tuple[npt.NDArray[Any], int]
    | list[Any]
    | dict[str, Any]
)
AudioInput = AudioSource | list[AudioSource] | None


def _value(config: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, None)
    return (
        value
        if value is not None
        else _value(getattr(config, "processor_config", None), name, default)
    )


class _MiMoAudioCodec:
    """Lazy torch-based waveform → speech-code tokenizer (trust_remote_code)."""

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModel

        path = os.path.join(model_path, "audio_tokenizer")
        try:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        except (KeyError, ValueError):
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            config_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizerConfig", model_path, trust_remote_code=True
            )
            model_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizer", model_path, trust_remote_code=True
            )
            with open(os.path.join(path, "config.json")) as config_file:
                config = config_type(**json.load(config_file))
            self.model = model_type.from_pretrained(path, config=config)
        self.model.eval()
        self.torch = torch
        from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import (
            MiMoAudioProcessor,
        )

        self.processor = MiMoAudioProcessor()

    @staticmethod
    def _waveform(source: AudioSource) -> tuple[FloatArray, int]:
        import soundfile as sf

        if isinstance(source, dict):
            source = source.get("url", source.get("audio_url"))
        if isinstance(source, tuple) and len(source) == 2:
            waveform, sampling_rate = source
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        if isinstance(source, np.ndarray):
            return source.astype(np.float32), 24000
        if isinstance(source, bytes):
            waveform, sampling_rate = sf.read(io.BytesIO(source), dtype="float32")
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if not isinstance(source, str):
            raise ValueError(f"Unsupported MiMoV2 audio source: {type(source).__name__}.")
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            waveform, sampling_rate = sf.read(io.BytesIO(response.content), dtype="float32")
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        if source.startswith("data:") and "base64," in source:
            payload = base64.b64decode(source.split("base64,", 1)[1])
            waveform, sampling_rate = sf.read(io.BytesIO(payload), dtype="float32")
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        if source.startswith("file://"):
            source = unquote(urlparse(source).path)
        if os.path.isfile(source):
            waveform, sampling_rate = sf.read(source, dtype="float32")
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        try:
            payload = base64.b64decode(source, validate=True)
        except ValueError as error:
            raise ValueError("Unsupported MiMoV2 audio source.") from error
        waveform, sampling_rate = sf.read(io.BytesIO(payload), dtype="float32")
        return np.asarray(waveform, dtype=np.float32), int(sampling_rate)

    def encode(self, source: AudioSource) -> IntArray:
        waveform, sampling_rate = self._waveform(source)
        if waveform.ndim == 2:
            axis = 0 if waveform.shape[0] <= 8 < waveform.shape[1] else 1
            waveform = waveform.mean(axis=axis)
        mels, _ = self.processor(waveform, sampling_rate)
        encoder = getattr(self.model, "encoder", self.model)
        parameter = next(encoder.parameters())
        parts = []
        with self.torch.no_grad():
            for start in range(0, mels.shape[1], 6000):
                features = self.torch.from_numpy(mels[:, start : start + 6000]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                lengths = self.torch.tensor(
                    [features.shape[1]], dtype=self.torch.long, device=parameter.device
                )
                codes, _ = encoder.encode(
                    input_features=features, input_lens=lengths, return_codes_only=True
                )
                parts.append(codes)
        return np.asarray(
            self.torch.cat(parts, dim=-1).transpose(0, 1).cpu().numpy(), dtype=np.int32
        )


class MiMoV2Processor(QwenVLProcessor):
    models = (
        "MiMoV2ForConditionalGeneration",
        "MiMoV2FlashForConditionalGeneration",
    )

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        processor: ProcessorMixin,
    ) -> None:
        # Base helpers reach into ``hf_config.vision_config`` with attribute
        # access; normalize a dict sub-config so those keep working.
        if isinstance(getattr(hf_config, "vision_config", None), dict):
            hf_config = copy.copy(hf_config)
            hf_config.vision_config = SimpleNamespace(**hf_config.vision_config)
        super().__init__(hf_config, server_args, processor)

        audio_config = getattr(hf_config, "audio_config", None)
        self._has_audio = audio_config is not None
        if self._has_audio:
            audio_token_id = _config_value(hf_config, "audio_token_id")
            if audio_token_id is None:
                raise ValueError("MiMoV2 audio_token_id is missing from the model config.")
            self.audio_token_id = int(audio_token_id)
            self.audio_channels = int(_value(audio_config, "audio_channels"))
            self.group_size = int(_value(audio_config, "group_size"))
            if self.audio_channels <= 0 or self.group_size <= 0:
                raise ValueError("MiMoV2 audio_channels and group_size must be positive.")
            self.vocab_sizes = self._int_list(
                _value(audio_config, "speech_vocab_size"), self.audio_channels
            )
        self._audio_codec: _MiMoAudioCodec | None = None

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj: GenerateReqInput | EmbeddingReqInput,
        **kwargs,
    ) -> MultimodalInputs:
        if isinstance(input_text, list):
            raise ValueError("MiMoV2 multimodal requests require text input, not input_ids.")

        has_vision = getattr(self.hf_config, "vision_config", None) is not None
        sources = self._audio_sources(getattr(request_obj, "audio_data", None))
        if sources and not self._has_audio:
            raise ValueError("This MiMoV2 checkpoint has no audio encoder.")

        if has_vision:
            output = await self._process_vision(image_data, input_text, request_obj)
        else:
            if self.normalize_data(image_data) or self.normalize_data(
                getattr(request_obj, "video_data", None)
            ):
                raise ValueError("This MiMoV2 checkpoint has no vision encoder.")
            processed = self.processor(text=[input_text], padding=True, return_tensors="pt")
            input_ids = self._to_numpy(processed.get("input_ids"))
            if input_ids is None:
                raise ValueError("MiMoV2 HF processor did not return input_ids.")
            output = MultimodalInputs(mm_items=[], input_ids=input_ids.reshape(-1).tolist())

        if not sources:
            return output
        codes = await asyncio.to_thread(lambda: [self._encode_audio(s) for s in sources])
        self._merge_audio(output, codes)
        return output

    # -- vision -----------------------------------------------------------

    async def _process_vision(self, image_data, input_text, request_obj) -> MultimodalInputs:
        images = await self._load_images_async(image_data)
        video_data = self.normalize_data(getattr(request_obj, "video_data", None))
        video_config = self._build_video_config(request_obj)
        videos = await self._load_videos_async(video_data, video_config)
        processor_kwargs = {}
        if videos:
            from sgl_jax.srt.multimodal.processors.qwen_vl import FPS

            processor_kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "fps": video_config.get("fps", FPS),
            }

        processor_output = self.processor(
            text=[input_text],
            images=images or None,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids_array = self._to_numpy(processor_output.get("input_ids"))
        if input_ids_array is None:
            raise ValueError("MiMoV2 HF processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).tolist()
        pixel_values = self._to_numpy(processor_output.get("pixel_values"))
        pixel_values_videos = self._to_numpy(processor_output.get("pixel_values_videos"))
        image_grid_thw = self._to_grid_list(processor_output.get("image_grid_thw"))
        video_grid_thw = self._to_grid_list(processor_output.get("video_grid_thw"))

        spatial_merge_size = int(_value(self.hf_config.vision_config, "spatial_merge_size", 2))
        image_token_id = self.hf_config.image_token_id
        video_token_id = getattr(self.hf_config, "video_token_id", None)

        image_placeholder_ranges = self._compute_image_placeholder_ranges(
            input_ids=input_ids,
            grids=image_grid_thw,
            image_token_id=image_token_id,
            spatial_merge_size=spatial_merge_size,
        )
        video_placeholder_ranges = self._compute_placeholder_ranges(
            input_ids=input_ids,
            grids=video_grid_thw,
            token_id=video_token_id,
            spatial_merge_size=spatial_merge_size,
            modality_name="VIDEO",
        )

        mm_items = []
        mm_items.extend(
            self._build_items(
                pixel_values,
                image_grid_thw,
                image_placeholder_ranges,
                Modality.IMAGE,
                "image_grid_thw",
            )
        )
        mm_items.extend(
            self._build_items(
                pixel_values_videos,
                video_grid_thw,
                video_placeholder_ranges,
                Modality.VIDEO,
                "video_grid_thw",
            )
        )
        for item in mm_items:
            item.set_pad_value()

        # No mrope_positions/mrope_position_delta: MiMoV2 uses standard 1-D RoPE.
        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_start_id=getattr(self.hf_config, "vision_start_token_id", None),
            im_end_id=getattr(self.hf_config, "vision_end_token_id", None),
            im_token_id=image_token_id,
            video_token_id=video_token_id,
        )

    # -- audio ------------------------------------------------------------

    def _encode_audio(self, source: AudioSource) -> IntArray:
        if isinstance(source, dict) and "codes" in source:
            source = source["codes"]
        array = np.asarray(source) if isinstance(source, (list, np.ndarray)) else None
        if array is not None and array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
            return self._normalize_codes(array)
        if array is not None:
            source = array
        if self._audio_codec is None:
            self._audio_codec = _MiMoAudioCodec(self.server_args.model_path)
        return self._normalize_codes(self._audio_codec.encode(source))

    def _normalize_codes(self, values: npt.ArrayLike) -> IntArray:
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"MiMoV2 audio codes must be 2D, got {values.shape}.")
        if values.shape[1] != self.audio_channels:
            if values.shape[0] == self.audio_channels:
                values = values.T
            else:
                raise ValueError(
                    f"MiMoV2 audio codes require {self.audio_channels} channels, got {values.shape}."
                )
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError("MiMoV2 audio codes must be non-negative integers.")
        for channel, size in enumerate(self.vocab_sizes):
            if np.any(values[:, channel] >= size):
                raise ValueError(
                    f"MiMoV2 audio code on channel {channel} exceeds vocab size {size}."
                )
        return values.astype(np.int32, copy=False)

    @staticmethod
    def _audio_sources(data: AudioInput) -> list[AudioSource]:
        if data is None:
            return []
        if isinstance(data, list) and data and isinstance(data[0], (int, float, np.number)):
            return [data]
        try:
            array = np.asarray(data)
        except ValueError:
            array = None
        if (
            isinstance(data, list)
            and array is not None
            and array.ndim == 2
            and np.issubdtype(array.dtype, np.integer)
        ):
            return [data]
        return cast(list[AudioSource], data) if isinstance(data, list) else [data]

    @staticmethod
    def _int_list(value: int | str | Sequence[int], length: int) -> list[int]:
        if isinstance(value, str):
            values = [int(item) for item in value.split("-")]
        elif isinstance(value, int):
            values = [value]
        else:
            values = [int(item) for item in value]
        if len(values) == 1:
            values *= length
        if len(values) != length:
            raise ValueError(f"Expected {length} values, got {len(values)}.")
        return values

    def _merge_audio(self, output: MultimodalInputs, code_arrays: Sequence[IntArray]) -> None:
        if output.input_ids is None:
            raise ValueError("MiMoV2 processor output is missing input_ids.")
        input_ids = list(output.input_ids)
        items: list[MultimodalDataItem] = []
        cursor = 0
        for values in code_arrays:
            values = np.asarray(values)
            if values.ndim != 2 or not values.shape[0]:
                raise ValueError(
                    f"MiMoV2 audio codes must be non-empty [T, C], got {values.shape}."
                )
            pad = (-values.shape[0]) % self.group_size
            if pad:
                values = np.concatenate((values, np.repeat(values[-1:], pad, axis=0)))
            tokens = values.shape[0] // self.group_size
            try:
                start = input_ids.index(self.audio_token_id, cursor)
            except ValueError as error:
                raise ValueError("MiMoV2 prompt is missing an audio placeholder.") from error
            end = start + 1
            while end < len(input_ids) and input_ids[end] == self.audio_token_id:
                end += 1
            if end - start not in (1, tokens):
                raise ValueError(
                    f"MiMoV2 audio placeholder span has {end - start} tokens, expected 1 or {tokens}."
                )
            input_ids[start:end] = [self.audio_token_id] * tokens
            item = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=values,
                placeholder_ranges=[(start, start + tokens)],
            )
            item.set_pad_value()
            items.append(item)
            cursor = start + tokens

        if self.audio_token_id in input_ids[cursor:]:
            raise ValueError("MiMoV2 prompt has more audio placeholders than audio inputs.")
        output.input_ids = input_ids
        output.audio_token_id = self.audio_token_id
        output.mm_items.extend(items)
        self._refresh_vision_ranges(output)

    def _refresh_vision_ranges(self, output: MultimodalInputs) -> None:
        if output.input_ids is None:
            raise ValueError("MiMoV2 processor output is missing input_ids.")
        spatial_merge_size = int(_value(self.hf_config.vision_config, "spatial_merge_size", 2))
        for modality, token_id, grid_key in (
            (Modality.IMAGE, output.im_token_id, "image_grid_thw"),
            (Modality.VIDEO, output.video_token_id, "video_grid_thw"),
        ):
            items = [item for item in output.mm_items if item.modality is modality]
            if not items:
                continue
            if token_id is None:
                raise ValueError(f"MiMoV2 processor output is missing {grid_key} token id.")
            grids = [tuple(np.asarray(item.get(grid_key)).reshape(-1)) for item in items]
            ranges = self._compute_placeholder_ranges(
                output.input_ids, grids, token_id, spatial_merge_size, modality.name
            )
            for item, placeholder_range in zip(items, ranges):
                item.placeholder_ranges = [placeholder_range]
