"""Audio input handling for the MiMoV2 multimodal processor."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import numpy.typing as npt
import requests

from sgl_jax.srt.configs.mimo import audio_int_list, config_value

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


class _MiMoAudioCodec:
    """Lazy torch-based waveform → speech-code tokenizer (trust_remote_code)."""

    def __init__(self, model_path: str, revision: str | None = None) -> None:
        import torch
        from transformers import AutoModel

        is_local = os.path.isdir(model_path)
        source = os.path.join(model_path, "audio_tokenizer") if is_local else model_path
        load_kwargs = {"revision": revision} if revision is not None else {}
        if not is_local:
            load_kwargs["subfolder"] = "audio_tokenizer"
        try:
            self.model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                **load_kwargs,
            )
        except (KeyError, OSError, ValueError):
            from safetensors.torch import load_file
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            config_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizerConfig",
                model_path,
                revision=revision,
                trust_remote_code=True,
            )
            model_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizer",
                model_path,
                revision=revision,
                trust_remote_code=True,
            )
            if is_local:
                config_path = os.path.join(source, "config.json")
                weights_path = os.path.join(source, "model.safetensors")
            else:
                from transformers.utils.hub import cached_file

                config_path = cached_file(
                    model_path,
                    "audio_tokenizer/config.json",
                    revision=revision,
                )
                weights_path = cached_file(
                    model_path,
                    "audio_tokenizer/model.safetensors",
                    revision=revision,
                )
            with open(config_path) as config_file:
                config = config_type(**json.load(config_file))

            # Match the checkpoint loader: meta loading cannot restore this
            # tokenizer's non-persistent RoPE buffers from its state dict.
            self.model = model_type(config)
            self.model.load_state_dict(load_file(weights_path))
        self.model.to(dtype=torch.bfloat16).eval()
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
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        elif not isinstance(source, str):
            raise ValueError(f"Unsupported MiMoV2 audio source: {type(source).__name__}.")
        elif source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            source = io.BytesIO(response.content)
        elif source.startswith("data:") and "base64," in source:
            source = io.BytesIO(base64.b64decode(source.split("base64,", 1)[1]))
        else:
            if source.startswith("file://"):
                source = unquote(urlparse(source).path)
            if not os.path.isfile(source):
                source = io.BytesIO(base64.b64decode(source, validate=True))

        waveform, sampling_rate = sf.read(source, dtype="float32")
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
                features = self.torch.from_numpy(mels[0, start : start + 6000]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                lengths = self.torch.tensor(
                    [features.shape[0]], dtype=self.torch.long, device=parameter.device
                )
                codes, _ = encoder.encode(
                    input_features=features, input_lens=lengths, return_codes_only=True
                )
                parts.append(codes)
        return np.asarray(
            self.torch.cat(parts, dim=-1).transpose(0, 1).cpu().numpy(), dtype=np.int32
        )


class MiMoV2AudioProcessorMixin:
    """Audio tokenizer, code validation, and grouping for the host processor."""

    def _init_audio_processor(self, hf_config) -> None:
        audio_config = getattr(hf_config, "audio_config", None)
        self._has_audio = audio_config is not None
        if self._has_audio:
            self.audio_channels = int(config_value(audio_config, "audio_channels"))
            self.group_size = int(config_value(audio_config, "group_size"))
            if self.audio_channels <= 0 or self.group_size <= 0:
                raise ValueError("MiMoV2 audio_channels and group_size must be positive.")
            self.vocab_sizes = audio_int_list(
                config_value(audio_config, "speech_vocab_size"), self.audio_channels
            )
        self._audio_codec: _MiMoAudioCodec | None = None

    def _encode_audio(self, source: AudioSource) -> IntArray:
        if isinstance(source, dict) and "codes" in source:
            source = source["codes"]
        array = np.asarray(source) if isinstance(source, (list, np.ndarray)) else None
        if array is not None and array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
            return self._normalize_codes(array)
        if array is not None:
            source = array
        if self._audio_codec is None:
            self._audio_codec = _MiMoAudioCodec(
                self.server_args.model_path,
                getattr(self.server_args, "revision", None),
            )
        return self._normalize_codes(self._audio_codec.encode(source))

    def _encode_audio_groups(self, source: AudioSource) -> IntArray:
        values = self._encode_audio(source)
        if not len(values):
            raise ValueError("MiMoV2 audio codes must not be empty.")
        pad = (-len(values)) % self.group_size
        if pad:
            values = np.concatenate((values, np.repeat(values[-1:], pad, axis=0)))
        return values.reshape(-1, self.group_size, self.audio_channels)

    def _normalize_codes(self, values: npt.ArrayLike) -> IntArray:
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"MiMoV2 audio codes must be 2D, got {values.shape}.")
        if values.shape[1] != self.audio_channels:
            if values.shape[0] == self.audio_channels:
                values = values.T
            else:
                raise ValueError(
                    "MiMoV2 audio codes require "
                    f"{self.audio_channels} channels, got {values.shape}."
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
        if not isinstance(data, list):
            return [data]
        if data and isinstance(data[0], (int, float, np.number)):
            return [data]
        try:
            array = np.asarray(data)
        except ValueError:
            return data
        if array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
            return [data]
        return data
