"""MiMoV2 media processing with SGLang's token and feature layout."""

from __future__ import annotations

import io
import re
from types import SimpleNamespace

import numpy as np
import torch

from sgl_jax.srt.configs.mimo import config_value
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    _normalize_image_source,
)
from sgl_jax.srt.multimodal.processors.mimo_processing import MiMoMediaProcessor
from sgl_jax.srt.multimodal.processors.mimo_v2_audio import MiMoV2AudioProcessorMixin
from sgl_jax.srt.multimodal.processors.qwen_vl import smart_nframes


class MiMoV2Processor(MiMoV2AudioProcessorMixin, BaseMultimodalProcessor):
    auto_mm_processor_worker_num = 1
    supports_mm_processor_concurrency = False
    models = ("MiMoV2ForCausalLM", "MiMoV2ForConditionalGeneration")

    # Match whole placeholders so replacing them does not duplicate delimiters.
    _PLACEHOLDERS = {
        "image": r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>",
        "video": r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>",
        "audio": r"<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>",
    }
    _PATTERN = re.compile("|".join(f"(?P<{key}>{value})" for key, value in _PLACEHOLDERS.items()))
    _AUDIO_PLACEHOLDER = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"

    def __init__(self, hf_config, server_args, processor):
        super().__init__(hf_config, server_args, processor)
        self._init_audio_processor(hf_config)

    async def process_mm_data_async(self, image_data, input_text, request_obj, **kwargs):
        if not isinstance(input_text, str):
            raise ValueError("MiMoV2 multimodal requests require text input, not input_ids.")
        return await self.mm_processor_executor.run(
            self._process_mm_data,
            input_text,
            self.normalize_data(image_data),
            self.normalize_data(getattr(request_obj, "video_data", None)),
            self._audio_sources(getattr(request_obj, "audio_data", None)),
            request_obj,
        )

    @staticmethod
    def _preprocess_kwargs(source):
        if isinstance(source, dict):
            return source.get("preprocess_kwargs", {}) or {}
        return getattr(source, "preprocess_kwargs", {}) or {}

    def _video_input(self, source, request_obj):
        overrides = dict(self._preprocess_kwargs(source))
        for key in ("fps", "num_frames"):
            value = getattr(request_obj, key, None)
            if value is not None:
                overrides[key] = value
        fields = {
            key: overrides.get(key)
            for key in (
                "min_pixels",
                "max_pixels",
                "total_max_pixels",
                "fps",
                "num_frames",
                "max_frames",
                "min_frames",
            )
        }
        return SimpleNamespace(
            **fields,
            video=None,
            audio=None,
        )

    def _decode_video(self, source, sampling):
        """Preserve sampled source timestamps; decode both streams with decord."""
        from decord import AudioReader, DECORDError, VideoReader, cpu

        source = self.unwrap_source(source)
        if isinstance(source, tuple):
            frames, timestamps = source
            return (torch.as_tensor(frames).float(), torch.as_tensor(timestamps).float()), None
        if isinstance(source, np.ndarray):
            fps = sampling.get("fps", 2)
            frames = torch.from_numpy(source).permute(0, 3, 1, 2).float()
            return (frames, torch.arange(len(frames), dtype=torch.float32) / fps), None
        payload = _normalize_image_source(source)
        reader_source = lambda: io.BytesIO(payload) if isinstance(payload, bytes) else payload
        reader = VideoReader(reader_source(), ctx=cpu(0))
        video_fps = reader.get_avg_fps()
        sampling = dict(sampling)
        if "num_frames" in sampling:
            sampling["nframes"] = sampling.pop("num_frames")
        count = smart_nframes(sampling, len(reader), video_fps)
        indices = np.unique(np.linspace(0, len(reader) - 1, num=count, dtype=np.int64))
        frames = torch.from_numpy(reader.get_batch(indices.tolist()).asnumpy())
        frames = frames.permute(0, 3, 1, 2).float()
        timestamps = torch.as_tensor(indices, dtype=torch.float32) / video_fps
        audio = None
        if self._has_audio:
            audio_config = config_value(self.hf_config, "audio_config")
            pc = self.hf_config.processor_config
            sample_rate = int(
                config_value(pc, "audio_sampling_rate")
                or config_value(audio_config, "sampling_rate")
                or config_value(audio_config, "sample_rate")
            )
            try:
                audio_reader = AudioReader(reader_source(), sample_rate=sample_rate, mono=True)
            except (RuntimeError, DECORDError) as error:
                if "can't find audio stream" not in str(error).lower():
                    raise
            else:
                audio = (audio_reader[:].asnumpy().reshape(-1), sample_rate)
        return (frames, timestamps), audio

    @staticmethod
    def _token_ranges(input_ids, token_id):
        matches = np.asarray(input_ids) == token_id
        boundaries = np.flatnonzero(np.diff(np.pad(matches.astype(np.int8), (1, 1))))
        return list(zip(boundaries[::2].tolist(), boundaries[1::2].tolist()))

    def _append_result(self, output, result, kind, media):
        ids = result["input_ids"]
        offset = len(output.input_ids)
        output.input_ids.extend(ids)
        if "pixel_values" in result:
            modality = Modality.IMAGE if kind == "image" else Modality.VIDEO
            token_id = media.image_token_id if kind == "image" else media.video_token_id
            _, h, w = map(int, result["thw_grid"])
            ranges = self._token_ranges(ids, token_id)
            pixels = self._to_numpy(result["pixel_values"])
            # Each temporal grid is an independent ViT attention segment. Splitting
            # here retains timestamp gaps and permits scheduler lane assignment.
            for frame, (start, end) in enumerate(ranges):
                item = MultimodalDataItem(
                    modality=modality,
                    feature=pixels[frame * h * w : (frame + 1) * h * w],
                    placeholder_ranges=[(offset + start, offset + end)],
                    model_specific_data={
                        "image_grid_thw" if kind == "image" else "video_grid_thw": np.asarray(
                            [[1, h, w]], dtype=np.int32
                        )
                    },
                )
                item.set_pad_value()
                output.mm_items.append(item)
        segments = result.get("audio_segments", [])
        if segments:
            codes = np.concatenate(segments, axis=0).reshape(
                -1, self.group_size, self.audio_channels
            )
            cursor = 0
            for start, end in self._token_ranges(ids, media.audio_token_id):
                length = end - start
                item = MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=codes[cursor : cursor + length].reshape(-1, self.audio_channels),
                    placeholder_ranges=[(offset + start, offset + end)],
                )
                item.set_pad_value()
                output.mm_items.append(item)
                cursor += length

    def _process_mm_data(self, input_text, images, videos, audios, request_obj, *, processor):
        if audios and not re.search(self._PLACEHOLDERS["audio"], input_text):
            input_text = self._AUDIO_PLACEHOLDER + input_text
        sources = {"image": images, "video": videos, "audio": audios}
        matches = list(self._PATTERN.finditer(input_text))
        for kind, items in sources.items():
            count = sum(match.lastgroup == kind for match in matches)
            if count != len(items):
                raise ValueError(
                    f"{kind} placeholder/data mismatch: {count} placeholders vs {len(items)} inputs"
                )
        if (images or videos) and config_value(self.hf_config, "vision_config") is None:
            raise ValueError("This MiMoV2 checkpoint has no vision encoder.")
        if audios and not self._has_audio:
            raise ValueError("This MiMoV2 checkpoint has no audio encoder.")
        media = MiMoMediaProcessor(self.hf_config, processor.tokenizer)
        output = MultimodalInputs(
            mm_items=[],
            input_ids=[],
            im_token_id=media.image_token_id,
            im_start_id=media.vision_start_token_id,
            im_end_id=media.vision_end_token_id,
            video_token_id=media.video_token_id,
            audio_token_id=media.audio_token_id,
            audio_start_id=media.audio_start_token_id,
            audio_end_id=media.audio_end_token_id,
        )
        iterators = {kind: iter(items) for kind, items in sources.items()}
        cursor = 0
        for match in matches:
            if match.start() > cursor:
                output.input_ids.extend(media.tokenizer.encode(input_text[cursor : match.start()]))
            kind = match.lastgroup
            source = next(iterators[kind])
            if kind == "image":
                overrides = self._preprocess_kwargs(source)
                image_input = SimpleNamespace(
                    image=self.load_image(source),
                    min_pixels=overrides.get("min_pixels"),
                    max_pixels=overrides.get("max_pixels"),
                )
                result = media.process_image_content(image_input)
            elif kind == "audio":
                result = media.process_audio_content(self._encode_audio_groups(source))
            else:
                video_input = self._video_input(source, request_obj)
                video_input.video, video_input.audio = self._decode_video(
                    source, media.prepare_video_kwargs(video_input)
                )
                video_result = media.process_video(video_input)
                if video_input.audio is None:
                    result = media.process_video_content(video_result)
                else:
                    result = media.process_video_audio_content(
                        self._encode_audio_groups(video_input.audio), video_result
                    )
            self._append_result(output, result, kind, media)
            cursor = match.end()
        if cursor < len(input_text):
            output.input_ids.extend(media.tokenizer.encode(input_text[cursor:]))
        return output
