import asyncio
import base64
import dataclasses
import hashlib
import io
import logging
import math
import os
import signal
import tempfile
import time
import uuid
from http import HTTPStatus
from io import BytesIO
from typing import Any
from urllib.request import urlopen

import fastapi
import imageio.v3 as iio
import librosa
import numpy as np
import psutil
import requests
import setproctitle
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sgl_jax.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.manager.io_struct import (
    DataType,
    GenerateMMReqInput,
    GenerateOmniReqInput,
    TokenizedGenerateMMReqInput,
    TokenizedGenerateOmniReqInput,
)
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_mrope_positions
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    dataclass_to_string_truncated,
    kill_itself_when_parent_died,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


# Qwen video preprocessing (ported from sglang).
_QWEN_IMAGE_FACTOR = 28
_QWEN_MAX_RATIO = 200
_QWEN_VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))
_QWEN_VIDEO_MIN_PIXELS = 128 * 28 * 28
_QWEN_VIDEO_MAX_PIXELS = 768 * 28 * 28
_QWEN_FRAME_FACTOR = 2
_QWEN_FPS = 2.0
_QWEN_FPS_MIN_FRAMES = 4
_QWEN_FPS_MAX_FRAMES = 768


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int = _QWEN_IMAGE_FACTOR,
    min_pixels: int = _QWEN_VIDEO_MIN_PIXELS,
    max_pixels: int = _QWEN_VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > _QWEN_MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {_QWEN_MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _smart_nframes(video_config: dict, total_frames: int, video_fps: float) -> int:
    assert not (
        "fps" in video_config and "nframes" in video_config
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in video_config:
        nframes = _round_by_factor(video_config["nframes"], _QWEN_FRAME_FACTOR)
    else:
        fps = video_config.get("fps", _QWEN_FPS)
        min_frames = _ceil_by_factor(
            video_config.get("min_frames", _QWEN_FPS_MIN_FRAMES),
            _QWEN_FRAME_FACTOR,
        )
        max_frames = _floor_by_factor(
            video_config.get("max_frames", min(_QWEN_FPS_MAX_FRAMES, total_frames)),
            _QWEN_FRAME_FACTOR,
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning("smart_nframes: nframes[%s] > total_frames[%s]", nframes, total_frames)
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = _floor_by_factor(nframes, _QWEN_FRAME_FACTOR)
    if not (_QWEN_FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(
            "nframes should in interval [%s, %s], but got %s.",
            _QWEN_FRAME_FACTOR,
            total_frames,
            nframes,
        )
    return int(nframes)


@dataclasses.dataclass
class MMReqState(ReqState):
    """Store the state of a multimodal request."""

    rid: str = ""


class MultimodalTokenizer(TokenizerManager):
    """Tokenization manager for multimodal requests.

    `MultimodalTokenizer` accepts high-level multimodal generation requests
    (`GenerateMMReqInput`), tokenizes text inputs (and prepares image
    references when supported), forwards tokenized requests to the
    scheduler pipeline, and waits for/streams back results. It tracks the
    state of outstanding requests via `MMReqState` and uses a
    `TypeBasedDispatcher` to handle results arriving from the pipeline.
    """

    def __init__(self, server_args, port_args):
        """Initialize tokenizer, processor and result dispatcher.

        Loads an image processor (best-effort), initializes an in-memory
        map `rid_to_state` to track request state objects, and prepares a
        result dispatcher that routes batches of outputs back to
        `_handle_batch_output`.
        """
        super().__init__(server_args, port_args)
        self.mm_processor = None
        self.mm_config = None
        processor_candidates = [server_args.model_path]
        model_basename = os.path.basename(server_args.model_path.rstrip("/"))
        if model_basename in {
            "text_encoder",
            "vision_encoder",
            "language_model",
            "transformer",
            "vae",
            "tokenizer",
        }:
            processor_candidates.append(os.path.dirname(server_args.model_path.rstrip("/")))
        trust_remote_code = server_args.trust_remote_code or server_args.multimodal
        for candidate in processor_candidates:
            try:
                self.mm_processor = AutoProcessor.from_pretrained(
                    candidate,
                    trust_remote_code=trust_remote_code,
                )
                self.mm_config = AutoConfig.from_pretrained(
                    candidate,
                    trust_remote_code=trust_remote_code,
                )
                break
            except Exception as exc:
                logger.warning("Failed to load processor/config from %s: %s", candidate, exc)
        self.rid_to_state: dict[str, MMReqState] = {}
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut, list),
                    self._handle_batch_output,
                ),
                (
                    AbortReq,
                    self._handle_abort_req,
                ),
            ]
        )

    def _handle_batch_output(self, reqs: list | BatchStrOut | BatchEmbeddingOut | BatchTokenIDOut):
        """Handle a batch of outputs returned from the pipeline.

        Marks the corresponding `MMReqState` as finished, sets its event to
        wake any waiters, and stores a simple success meta record. If a
        result arrives for an unknown `rid` it logs a warning.
        """
        if hasattr(reqs, "__len__") and len(reqs) > 0 and self.server_args.log_requests:
            logger.info("handle_batch_output %s, self.rid_to_state %s", reqs, self.rid_to_state)
        if isinstance(reqs, (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut)):
            return super()._handle_batch_output(reqs)

        for req in reqs:
            if req.rid in self.rid_to_state:
                self.rid_to_state[req.rid].finished = True
                self.rid_to_state[req.rid].event.set()
                self.rid_to_state[req.rid].out_list = [{"success": True, "meta_info": {}}]
            else:
                logger.warning(
                    "Received result for unknown request rid=%s. Known rids: %s",
                    req.rid,
                    list(self.rid_to_state.keys()),
                )

    def _handle_abort_req(self, recv_obj: AbortReq):
        """Handle an AbortReq returned from the scheduler.

        When a request is aborted (e.g., removed from the scheduler's queue
        before processing started), the scheduler sends an AbortReq back to
        notify the tokenizer. This method marks the request as finished with
        an abort status and wakes any waiting coroutines.
        """
        if recv_obj.rid not in self.rid_to_state:
            logger.warning(
                "Received abort for unknown request rid=%s. Known rids: %s",
                recv_obj.rid,
                list(self.rid_to_state.keys()),
            )
            return

        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "success": False,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": recv_obj.aborted_message or "Request aborted",
                        "status_code": HTTPStatus.BAD_REQUEST,
                    },
                },
            }
        )
        state.event.set()
        logger.info("Abort completed for rid=%s", recv_obj.rid)

    async def generate_request(
        self,
        obj: GenerateMMReqInput | GenerateOmniReqInput,
        request: fastapi.Request | None = None,
    ):
        """High level API: accept a generation request and stream responses.

        This coroutine tokenizes the input (text and optional image refs),
        sends the tokenized request to the scheduler pipeline, and then
        asynchronously yields results as they arrive (supporting streaming
        if `obj.stream` is True). It respects client disconnects and a
        configured wait timeout.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )
        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)
        async for response in self._wait_one_response(obj, state, request):
            yield response

    async def _tokenize_one_request(self, obj: GenerateMMReqInput | GenerateOmniReqInput):
        """
        Converts text fields to token ids using the configured tokenizer.
        Image preprocessing / references are noted as TODO; when provided
        `input_ids` are passed through unchanged.
        """
        # Support both 'prompt' (multimodal) and 'text' (text-only) fields
        input_text = getattr(obj, "prompt", None) or getattr(obj, "text", None)
        neg_input_text = getattr(obj, "neg_prompt", None) or getattr(obj, "text", None)
        input_ids = getattr(obj, "input_ids", None)
        neg_input_ids = getattr(obj, "neg_input_ids", None)
        mm_inputs = None
        image_data = self._normalize_mm_list(getattr(obj, "image_data", None))
        video_data = self._normalize_mm_list(getattr(obj, "video_data", None))
        audio_data = self._normalize_mm_list(getattr(obj, "audio_data", None))

        if not image_data and not video_data and getattr(obj, "input_reference", None) is not None:
            if obj.data_type == DataType.IMAGE:
                image_data = [obj.input_reference]
            elif obj.data_type == DataType.VIDEO:
                video_data = [obj.input_reference]
        if (image_data or video_data) and self.mm_processor is None:
            raise ValueError(
                "Multimodal inputs provided but processor/config is not available. "
                "Check model_path and trust_remote_code settings."
            )
        if image_data or video_data or audio_data:
            images = [
                self._load_image_from_source(item) for item in image_data
            ]  # note: We did not perform a resize operation
            processor_kwargs = {}
            if video_data and self._is_qwen_video_processor():
                video_config = self._build_qwen_video_config(obj)
                videos = [self._preprocess_qwen_video(item, video_config) for item in video_data]
                processor_kwargs["videos_kwargs"] = {"do_sample_frames": False}
                if "fps" in video_config:
                    processor_kwargs["videos_kwargs"]["fps"] = video_config["fps"]
            else:
                videos = [self._load_video_from_source(item) for item in video_data]
            audios = [self._load_audio_from_source(item) for item in audio_data]
            processor_out = self.mm_processor(
                images=images or None,
                videos=videos or None,
                audio=audios or None,
                text=input_text or "",
                return_tensors="pt",
                **processor_kwargs,
            )
            if "input_ids" in processor_out:
                input_ids = processor_out["input_ids"][0].tolist()

            image_grid_thw = self._to_grid_list(processor_out.get("image_grid_thw"))
            video_grid_thw = self._to_grid_list(processor_out.get("video_grid_thw"))
            second_per_grid_ts = processor_out.get("second_per_grid_ts")
            if second_per_grid_ts is None:
                second_per_grid_ts = processor_out.get("video_second_per_grid")
            pixel_values = self._strip_batch_dim(processor_out.get("pixel_values"))
            pixel_values_videos = self._strip_batch_dim(processor_out.get("pixel_values_videos"))

            mrope_positions = None
            mrope_position_delta = None
            if self.mm_config is not None and input_ids is not None:
                if hasattr(self.mm_config, "thinker_config"):
                    # for qwen3-omni
                    self.mm_config = self.mm_config.thinker_config
                vision_start_token_id = getattr(self.mm_config, "vision_start_token_id", None)
                image_token_id = getattr(self.mm_config, "image_token_id", None)
                video_token_id = getattr(self.mm_config, "video_token_id", None)
                vision_config = getattr(self.mm_config, "vision_config", None)
                spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
                tokens_per_second = getattr(vision_config, "tokens_per_second", None)
                if (
                    vision_start_token_id is not None
                    and image_token_id is not None
                    and spatial_merge_size is not None
                ):
                    mrope_positions, mrope_position_delta = compute_mrope_positions(
                        input_ids=input_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        vision_start_token_id=vision_start_token_id,
                        image_token_id=image_token_id,
                        video_token_id=video_token_id,
                        spatial_merge_size=spatial_merge_size,
                        tokens_per_second=tokens_per_second,
                    )

            mm_items = []
            if pixel_values is not None:
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=np.asarray(pixel_values),
                    )
                )
            if pixel_values_videos is not None:
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.VIDEO,
                        feature=np.asarray(pixel_values_videos),
                    )
                )
            audio_features = processor_out.get("audio_features") or processor_out.get(
                "input_features"
            )
            if audio_features is not None:
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.AUDIO,
                        feature=np.asarray(audio_features),
                    )
                )

            audio_feature_attention_mask = processor_out.get("feature_attention_mask")
            if audio_feature_attention_mask is not None:
                audio_feature_attention_mask = np.asarray(audio_feature_attention_mask)

            for item in mm_items:
                item.set_pad_value()

            if isinstance(second_per_grid_ts, np.ndarray):
                second_per_grid_ts = second_per_grid_ts.tolist()

            mm_inputs = {
                "mm_items": mm_items,
                "im_start_id": getattr(self.mm_config, "vision_start_token_id", None),
                "im_end_id": getattr(self.mm_config, "vision_end_token_id", None),
                "im_token_id": getattr(self.mm_config, "image_token_id", None),
                "video_token_id": getattr(self.mm_config, "video_token_id", None),
                "audio_token_id": getattr(self.mm_config, "audio_token_id", None),
                "mrope_positions": mrope_positions,
                "mrope_position_delta": mrope_position_delta,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "second_per_grid_ts": second_per_grid_ts,
                "audio_feature_attention_mask": audio_feature_attention_mask,
            }
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        if neg_input_ids is None and neg_input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but neg_input_text requires tokenization"
                )
            encoded = self.tokenizer(neg_input_text)
            neg_input_ids = encoded["input_ids"]

        is_omni_req = isinstance(obj, GenerateOmniReqInput) or hasattr(obj, "sampling_params")
        if is_omni_req:
            tokenized_obj = self._create_tokenized_omni_object(obj, input_text, input_ids)
        else:
            tokenized_obj = self._create_tokenized_object(
                obj, input_text, input_ids, neg_input_text, neg_input_ids
            )
        tokenized_obj.mm_inputs = mm_inputs
        return tokenized_obj

    def _normalize_mm_list(self, data: list[str] | str | None) -> list[str]:
        if data is None:
            return []
        return data if isinstance(data, list) else [data]

    def _is_qwen_video_processor(self) -> bool:
        if self.mm_processor is None:
            return False
        return self.mm_processor.__class__.__name__ in {
            "Qwen2_5_VLProcessor",
            "Qwen3OmniMoeProcessor",
        }

    def _build_qwen_video_config(self, obj: GenerateMMReqInput | GenerateOmniReqInput) -> dict:
        video_config: dict[str, Any] = {}
        fps = getattr(obj, "fps", None)
        if fps is not None:
            video_config["fps"] = fps
        nframes = getattr(obj, "num_frames", None)
        if nframes is not None and "fps" not in video_config:
            video_config["nframes"] = nframes
        return video_config

    def _preprocess_qwen_video(
        self, source: str | bytes | np.ndarray, video_config: dict
    ) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return self._resize_video_frames(source, video_config)

        if isinstance(source, dict) and "url" in source:
            source = source["url"]
        if hasattr(source, "url"):
            source = source.url

        # Lazy import to avoid dependency issues on some platforms.
        from decord import VideoReader, cpu

        tmp_path = None
        try:
            ctx = cpu(0)

            if isinstance(source, bytes):
                tmp_path = self._write_temp_video(source)
                vr = VideoReader(tmp_path, ctx=ctx)
            elif isinstance(source, str):
                if os.path.exists(source):
                    vr = VideoReader(source, ctx=ctx)
                elif source.startswith(("http://", "https://")):
                    resp = requests.get(source, timeout=10)
                    resp.raise_for_status()
                    tmp_path = self._write_temp_video(resp.content)
                    vr = VideoReader(tmp_path, ctx=ctx)
                elif source.startswith("data:") and "base64," in source:
                    payload = source.split("base64,", 1)[1]
                    tmp_path = self._write_temp_video(base64.b64decode(payload))
                    vr = VideoReader(tmp_path, ctx=ctx)
                else:
                    tmp_path = self._write_temp_video(base64.b64decode(source, validate=True))
                    vr = VideoReader(tmp_path, ctx=ctx)
            else:
                raise ValueError(f"Unsupported video input type: {type(source)}")

            total_frames = len(vr)
            if total_frames <= 0:
                raise ValueError("Video must have at least one frame")
            video_fps = float(vr.get_avg_fps() or 1.0)
            nframes = _smart_nframes(video_config, total_frames=total_frames, video_fps=video_fps)
            idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
            idx = np.unique(idx)
            video_np = vr.get_batch(idx).asnumpy()
            return self._resize_video_frames(video_np, video_config)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _write_temp_video(self, payload: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(payload)
            return tmp.name

    def _resize_video_frames(self, video_np: np.ndarray, video_config: dict) -> np.ndarray:
        if video_np.ndim != 4:
            raise ValueError(f"Expected video array with 4 dims (T,H,W,C), got {video_np.shape}")

        nframes, height, width = video_np.shape[0], video_np.shape[1], video_np.shape[2]
        min_pixels = video_config.get("min_pixels", _QWEN_VIDEO_MIN_PIXELS)
        total_pixels = video_config.get("total_pixels", _QWEN_VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(
                video_config.get("max_pixels", _QWEN_VIDEO_MAX_PIXELS),
                total_pixels / nframes * _QWEN_FRAME_FACTOR,
            ),
            int(min_pixels * 1.05),
        )

        max_pixels_supposed = video_config.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(
                "The given max_pixels[%s] exceeds limit[%s].",
                max_pixels_supposed,
                max_pixels,
            )
        max_pixels = min(max_pixels_supposed, max_pixels)

        if "resized_height" in video_config and "resized_width" in video_config:
            resized_height, resized_width = _smart_resize(
                video_config["resized_height"],
                video_config["resized_width"],
                factor=_QWEN_IMAGE_FACTOR,
            )
        else:
            resized_height, resized_width = _smart_resize(
                height,
                width,
                factor=_QWEN_IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if resized_height == height and resized_width == width:
            return video_np

        resized_frames = []
        for frame in video_np:
            img = Image.fromarray(frame)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((resized_width, resized_height), resample=Image.BILINEAR)
            resized_frames.append(np.asarray(img))
        return np.stack(resized_frames, axis=0)

    def _load_image_from_source(self, source: str | bytes) -> Image.Image:
        if isinstance(source, dict) and "url" in source:
            source = source["url"]
        if hasattr(source, "url"):
            source = source.url
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if os.path.exists(source):
            return Image.open(source).convert("RGB")
        if source.startswith(("http://", "https://")):
            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        if source.startswith("data:") and "base64," in source:
            payload = source.split("base64,", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(source, validate=True))).convert("RGB")
        except Exception as exc:
            raise ValueError("Unsupported image source format") from exc

    def _load_video_from_source(self, source: str | bytes) -> np.ndarray:
        if isinstance(source, dict) and "url" in source:
            source = source["url"]
        if hasattr(source, "url"):
            source = source.url
        if isinstance(source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            try:
                return iio.imread(tmp_path, index=None)
            finally:
                os.unlink(tmp_path)
        if os.path.exists(source):
            return iio.imread(source, index=None)
        if source.startswith(("http://", "https://")):
            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                return iio.imread(tmp_path, index=None)
            finally:
                os.unlink(tmp_path)
        raise ValueError("Unsupported video source format")

    def _load_audio_from_source(self, source: str | bytes) -> np.ndarray:
        if not hasattr(self.mm_processor, "feature_extractor"):
            return None
        if isinstance(source, dict) and "url" in source:
            source = source["url"]
        if hasattr(source, "url"):
            source = source.url
        if isinstance(source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            try:
                audio_data, _ = librosa.load(
                    tmp_path, self.mm_processor.feature_extractor.sampling_rate
                )
                return audio_data
            finally:
                os.unlink(tmp_path)
        if os.path.exists(source):
            return iio.imread(source, index=None)
        if source.startswith(("http://", "https://")):
            try:
                audio_data, _ = librosa.load(
                    BytesIO(urlopen(source).read()),
                    sr=self.mm_processor.feature_extractor.sampling_rate,
                )
                return audio_data
            finally:
                pass
        raise ValueError("Unsupported audio source format")

    def _hash_payload(self, payload: bytes) -> int:
        digest = hashlib.sha256(payload).digest()[:8]
        return int.from_bytes(digest, byteorder="big", signed=False) % (1 << 31)

    def _hash_mm_items(self, images: list[Image.Image], videos: list[np.ndarray]) -> list[int]:
        pad_values = []
        for image in images:
            pad_values.append(self._hash_payload(image.tobytes()))
        for video in videos:
            pad_values.append(self._hash_payload(video.tobytes()))
        return pad_values

    def _to_grid_list(self, grid_thw: Any) -> list[tuple[int, int, int]] | None:
        if grid_thw is None:
            return None
        grid = np.asarray(grid_thw)
        if grid.size == 0:
            return None
        return [tuple(int(x) for x in row) for row in grid.tolist()]

    def _strip_batch_dim(self, arr: Any) -> np.ndarray | None:
        if arr is None:
            return None
        array = np.asarray(arr)
        if array.ndim > 1 and array.shape[0] == 1:
            return array[0]
        return array

    def _create_tokenized_object(
        self, obj: GenerateMMReqInput, input_text, input_ids, neg_input_text, neg_input_ids
    ):
        """Build `TokenizedGenerateMMReqInput` from the original request.

        Ensures a request id (`rid`) exists, and copies over relevant
        properties such as size, num_frames, data type and save_output flag.
        """
        rid = getattr(obj, "rid", None)
        if rid is None:
            rid = uuid.uuid4().hex

        tokenized_obj = TokenizedGenerateMMReqInput(
            rid=rid,
            prompt=input_text,
            negative_prompt=neg_input_text,
            input_ids=input_ids,
            negative_input_ids=neg_input_ids,
            size=getattr(obj, "size", None),
            num_frames=getattr(obj, "num_frames", None),
            num_inference_steps=getattr(obj, "num_inference_steps", 50),
            data_type=getattr(obj, "data_type", None),
            save_output=getattr(obj, "save_output", True),
        )
        return tokenized_obj

    def _create_tokenized_omni_object(
        self, obj: GenerateOmniReqInput, input_text, input_ids
    ) -> TokenizedGenerateOmniReqInput:
        rid = getattr(obj, "rid", None)
        if rid is None:
            rid = uuid.uuid4().hex

        return TokenizedGenerateOmniReqInput(
            rid=rid,
            prompt=input_text,
            input_ids=input_ids,
            stream=getattr(obj, "stream", False),
            n=getattr(obj, "n", 1),
            sampling_params=getattr(obj, "sampling_params", None),
            stop=getattr(obj, "stop", None),
        )

    def _send_one_request(
        self,
        obj: GenerateMMReqInput,
        tokenized_obj: TokenizedGenerateMMReqInput,
        created_time: float | None = None,
    ):
        """Send a tokenized request into the scheduling pipeline and track it.

        Constructs an `MMReqState` to wait for results and stores it in
        `rid_to_state` keyed by the request id.
        """
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = MMReqState(
            rid=tokenized_obj.rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[tokenized_obj.rid] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateMMReqInput,
        state: MMReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for results for a single request, yielding responses.

        This method waits on `state.event` with a timeout (`self.wait_timeout`),
        handles client disconnects (aborting the request), and yields
        intermediate/final outputs according to `obj.stream`.
        """
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    )


def run_multimodal_tokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_tokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        tokenizer = MultimodalTokenizer(server_args, port_args)
        tokenizer.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalTokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)

    return tokenizer
