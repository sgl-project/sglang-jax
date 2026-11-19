"""Qwen2-Audio (HF ``Qwen2AudioProcessor``) adapter for the in-model multimodal contract."""

from __future__ import annotations

import base64
import io
import os
from urllib.parse import unquote, urlparse

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


def _load_waveform(source, sampling_rate: int) -> np.ndarray:
    """Decode an audio source to mono float32 at ``sampling_rate``."""
    import librosa

    source = BaseMultimodalProcessor.unwrap_source(source)
    if isinstance(source, dict):
        source = source.get("url", source.get("audio_url"))
    if isinstance(source, tuple) and len(source) == 2:
        wav, sr = source
        wav = np.asarray(wav, dtype=np.float32)
        return wav if int(sr) == sampling_rate else librosa.resample(wav, orig_sr=int(sr), target_sr=sampling_rate)
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)  # assumed already at sampling_rate
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, str):
        if source.startswith(("http://", "https://")):
            import requests

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
    wav, _ = librosa.load(source, sr=sampling_rate, mono=True)
    return wav.astype(np.float32)


class Qwen2AudioProcessor(BaseMultimodalProcessor):
    models = ("Qwen2AudioForConditionalGeneration",)
    auto_mm_processor_worker_num = 2
    supports_mm_processor_concurrency = True

    @staticmethod
    def _placeholder_runs(input_ids: list[int], token_id: int) -> list[tuple[int, int]]:
        runs, start = [], None
        for index, value in enumerate((*input_ids, None)):
            if value == token_id and start is None:
                start = index
            elif value != token_id and start is not None:
                runs.append((start, index))
                start = None
        return runs

    async def process_mm_data_async(self, image_data, input_text, request_obj, **kwargs):
        if isinstance(input_text, list):
            raise ValueError("Qwen2-Audio multimodal requests require text input, not input_ids.")
        if self.normalize_data(image_data) or self.normalize_data(
            getattr(request_obj, "video_data", None)
        ):
            raise ValueError("Qwen2-Audio only accepts audio inputs.")
        audio_data = self.normalize_data(getattr(request_obj, "audio_data", None))
        return await self.mm_processor_executor.run(self._process_request, audio_data, input_text)

    def _process_request(self, audio_data, input_text, *, processor):
        sr = int(processor.feature_extractor.sampling_rate)
        audios = [_load_waveform(item, sr) for item in audio_data]
        kwargs = dict(text=[input_text], return_tensors="pt")
        if audios:
            kwargs.update(audio=audios, sampling_rate=sr)
        output = processor(**kwargs)
        return self.collect_mm_items_from_processor_output(output, audios=audios)

    def collect_mm_items_from_processor_output(
        self, processor_output, images=None, videos=None, audios=None, **kwargs
    ) -> MultimodalInputs:
        input_ids = (
            self._to_numpy(processor_output.get("input_ids")).reshape(-1).astype(np.int64).tolist()
        )
        if not audios:
            return MultimodalInputs(mm_items=[], input_ids=input_ids)

        features = self._to_numpy(processor_output.get("input_features"))  # [n, mels, frames]
        frame_mask = self._to_numpy(processor_output.get("feature_attention_mask"))
        token_id = int(getattr(self.hf_config, "audio_token_index", None) or self.hf_config.audio_token_id)
        runs = self._placeholder_runs(input_ids, token_id)
        if features is None or len(runs) != len(audios) or len(features) != len(audios):
            raise ValueError(
                f"Qwen2-Audio placeholder/feature/audio counts differ: "
                f"{len(runs)}, {None if features is None else len(features)}, {len(audios)}"
            )
        items = []
        for index, (start, end) in enumerate(runs):
            mel = np.ascontiguousarray(features[index].T.astype(np.float32))  # [frames, mels]
            item = MultimodalDataItem(modality=Modality.AUDIO, feature=mel)
            item.set("num_frames", np.asarray(int(frame_mask[index].sum()), dtype=np.int32))
            item.placeholder_ranges = [(start, end)]
            item.set_pad_value()
            items.append(item)
        return MultimodalInputs(mm_items=items, input_ids=input_ids, audio_token_id=token_id)
