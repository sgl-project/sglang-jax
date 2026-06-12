"""Qwen2.5-VL multimodal input processor (refactor M3, design §3.5.1).

Productionizes the validated front-end transform (tmp/refactor/frontend_transform.py):
real HF AutoProcessor -> input_ids (placeholders) + pixel_values + grid_thw -> mm_items ->
set_pad_value -> mRoPE -> pad_input_tokens. Scheme B (design §5.1.2): input_ids stays clean
(raw image/video/audio token ids); pad_input_tokens produces a separate padded copy
(cache_input_ids) used only for the per-image radix cache key. The in-model merge() locates
placeholder rows by isin(input_ids, [image/video/audio_token_id]) on the clean ids; the
forward + detokenizer never see a pad_value. Registered by HF arch name into the mm_core
ProcessorRegistry; the standard TokenizerManager resolves it for understanding reqs.
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


def _load_audio(source, sampling_rate):
    """Decode an audio source -> mono float32 waveform at sampling_rate. Accepts a pre-loaded
    np waveform, raw bytes, a base64 / data: URI, a local path, or an http(s) URL. Uses
    soundfile (libsndfile) -- robust and avoids librosa's msgpack-version fragility on some
    images; resamples via linear interp when the source rate differs."""
    if isinstance(source, np.ndarray):
        return source.astype("float32")
    import soundfile as sf

    def _decode(raw):
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != sampling_rate and len(data) > 0:
            n = int(round(len(data) * sampling_rate / sr))
            data = np.interp(
                np.linspace(0, len(data), n, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype("float32")
        return data

    if isinstance(source, dict) and "url" in source:
        source = source["url"]
    if hasattr(source, "url"):
        source = source.url
    if isinstance(source, bytes):
        return _decode(source)
    if isinstance(source, str) and source.startswith("data:") and "base64," in source:
        return _decode(base64.b64decode(source.split("base64,", 1)[1]))
    if isinstance(source, str) and os.path.exists(source):
        return _decode(open(source, "rb").read())
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        import requests

        return _decode(requests.get(source, timeout=10).content)
    if isinstance(source, str):
        return _decode(base64.b64decode(source, validate=True))
    raise ValueError("Unsupported audio source format")


class Qwen2_5_VLProcessor(BaseMultimodalProcessor):
    """Image (+video) understanding processor for Qwen2.5-VL (text-out)."""

    models = ["Qwen2_5_VLForConditionalGeneration"]
    # Whether the model consumes mRoPE positions. Qwen2.5-VL/Qwen3-Omni do; subclasses whose AR
    # reads raw forward_batch.positions instead (e.g. MiMo-V2.5) set this False so process() skips
    # the mrope computation entirely rather than computing it and throwing it away.
    uses_mrope = True

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        *,
        hf_processor=None,
        hf_config=None,
    ):
        from transformers import AutoConfig, AutoProcessor

        # hf_processor / hf_config injection is a test seam (mirrors the staged MiMoV25Processor):
        # tests pass fakes to exercise process() without a real checkpoint on disk.
        self.hf_processor = (
            hf_processor
            if hf_processor is not None
            else AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        )
        self.hf_config = (
            hf_config
            if hf_config is not None
            else AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        )
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
        audio_token_id = getattr(self, "audio_token_id", None)
        # Load raw image sources (URL/path/base64/bytes) -> PIL; pre-loaded PIL pass through.
        if images:
            images = [_load_image(s) for s in images]
        proc_kwargs = dict(
            text=[text] if isinstance(text, str) else text,
            images=images or None,
            videos=videos or None,
            return_tensors="pt",
        )
        # Audio (continuous mel) only for processors that declare an audio_token_id (Qwen3-Omni);
        # Qwen2.5-VL leaves it None and skips audio.
        if audios and audio_token_id is not None:
            sr = self.hf_processor.feature_extractor.sampling_rate
            proc_kwargs["audio"] = [_load_audio(s, sr) for s in audios]
        out = self.hf_processor(**proc_kwargs)

        input_ids = _to_list(out["input_ids"][0])
        pixel_values = _strip_batch_dim(out.get("pixel_values"))
        pixel_values_videos = _strip_batch_dim(out.get("pixel_values_videos"))
        image_grid_thw = _to_grid_list(out.get("image_grid_thw"))
        video_grid_thw = _to_grid_list(out.get("video_grid_thw"))
        second_per_grid_ts = out.get("second_per_grid_ts")
        audio_features = _strip_batch_dim(out.get("input_features"))
        audio_feature_attention_mask = _strip_batch_dim(out.get("feature_attention_mask"))

        # mRoPE FIRST -- it scans the raw image/video token ids in input_ids to locate spans.
        # Gated on uses_mrope: a subclass whose AR reads raw positions (MiMo-V2.5) skips this so it
        # isn't computed and then discarded (and can't go stale after later placeholder expansion).
        mrope_positions = mrope_position_delta = None
        if (
            self.uses_mrope
            and self.vision_start_token_id is not None
            and self.image_token_id is not None
        ):
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
        if audio_features is not None:
            mm_items.append(MultimodalDataItem(modality=Modality.AUDIO, feature=audio_features))
        for item in mm_items:
            item.set_pad_value()

        # Scheme B (design §5.1.2): input_ids stays CLEAN (raw placeholder token ids, all
        # in-vocab). pad_input_tokens produces a separate padded copy (per-item pad_value baked
        # into the placeholder rows) that travels only in cache_input_ids and is consumed
        # solely to build the per-image radix cache key. The model forward and the detokenizer
        # never see a pad_value, so no clamp / unpadded-copy bookkeeping is needed.
        cache_input_ids = pad_input_tokens(
            input_ids,
            mm_items,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=audio_token_id,
        )

        return {
            "input_ids": input_ids,
            "mm_inputs": {
                "mm_items": mm_items,
                "cache_input_ids": cache_input_ids,
                "im_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "audio_token_id": audio_token_id,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "audio_feature_attention_mask": audio_feature_attention_mask,
                "mrope_positions": mrope_positions,
                "mrope_position_delta": mrope_position_delta,
                "second_per_grid_ts": (
                    _to_list(second_per_grid_ts) if second_per_grid_ts is not None else None
                ),
            },
        }
