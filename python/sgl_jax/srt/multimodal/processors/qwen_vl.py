import asyncio
import base64
import logging
import math
import os
import tempfile

import numpy as np
import requests
from PIL import Image

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_mrope_positions
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Ported from SGLang Qwen-VL video preprocessing."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        ratio = max(height, width) / min(height, width)
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_nframes(video_config: dict, total_frames: int, video_fps: int | float) -> int:
    assert not (
        "fps" in video_config and "nframes" in video_config
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in video_config:
        nframes = round_by_factor(video_config["nframes"], FRAME_FACTOR)
    else:
        fps = video_config.get("fps", FPS)
        min_frames = ceil_by_factor(video_config.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            video_config.get("max_frames", min(FPS_MAX_FRAMES, total_frames)),
            FRAME_FACTOR,
        )
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return int(nframes)


def _resize_video_frames(video: np.ndarray, video_config: dict) -> np.ndarray:
    if video.ndim != 4:
        raise ValueError(f"Expected video array with 4 dims (T,H,W,C), got {video.shape}")

    nframes, height, width = video.shape[0], video.shape[1], video.shape[2]
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in video_config and "resized_width" in video_config:
        resized_height, resized_width = smart_resize(
            video_config["resized_height"],
            video_config["resized_width"],
            factor=IMAGE_FACTOR,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    if resized_height == height and resized_width == width:
        logger.info(
            "Qwen-VL video resize skipped: frames=%s, size=%sx%s, max_pixels=%s",
            nframes,
            height,
            width,
            max_pixels,
        )
        return video

    logger.info(
        "Qwen-VL video resized: frames=%s, original=%sx%s, resized=%sx%s, "
        "min_pixels=%s, max_pixels=%s",
        nframes,
        height,
        width,
        resized_height,
        resized_width,
        min_pixels,
        max_pixels,
    )

    resized_frames = []
    for frame in video:
        img = Image.fromarray(frame)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((resized_width, resized_height), resample=Image.BILINEAR)
        resized_frames.append(np.asarray(img))
    return np.stack(resized_frames, axis=0)


def _write_temp_video(payload: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(payload)
        return tmp.name


def _unwrap_source(source):
    if isinstance(source, dict) and "url" in source:
        return source["url"]
    if hasattr(source, "url"):
        return source.url
    return source


def preprocess_video(source, video_config: dict) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return _resize_video_frames(source, video_config)

    source = _unwrap_source(source)

    from decord import VideoReader, cpu

    tmp_path = None
    try:
        ctx = cpu(0)
        if isinstance(source, bytes):
            tmp_path = _write_temp_video(source)
            vr = VideoReader(tmp_path, ctx=ctx)
        elif isinstance(source, str):
            if os.path.exists(source):
                vr = VideoReader(source, ctx=ctx)
            elif source.startswith(("http://", "https://")):
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                tmp_path = _write_temp_video(response.content)
                vr = VideoReader(tmp_path, ctx=ctx)
            elif source.startswith("data:") and "base64," in source:
                payload = source.split("base64,", 1)[1]
                tmp_path = _write_temp_video(base64.b64decode(payload))
                vr = VideoReader(tmp_path, ctx=ctx)
            else:
                tmp_path = _write_temp_video(base64.b64decode(source, validate=True))
                vr = VideoReader(tmp_path, ctx=ctx)
        else:
            raise ValueError(f"Unsupported video input type: {type(source)}")

        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError("Video must have at least one frame.")
        video_fps = float(vr.get_avg_fps() or 1.0)
        nframes = smart_nframes(video_config, total_frames=total_frames, video_fps=video_fps)
        frame_indices = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
        frame_indices = np.unique(frame_indices)
        logger.info(
            "Qwen-VL video frames sampled: total_frames=%s, fps=%.3f, sampled_frames=%s",
            total_frames,
            video_fps,
            len(frame_indices),
        )
        video = vr.get_batch(frame_indices).asnumpy()
        return _resize_video_frames(video, video_config)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class QwenVLProcessor(BaseMultimodalProcessor):
    models = (
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    )

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        if isinstance(input_text, list):
            # TODO: support multimodal input_ids without decode + retokenize drift.
            raise ValueError(
                "Multimodal input_ids are not supported for Qwen-VL. "
                "Please provide text input instead."
            )

        images = await self._load_images_async(image_data)
        video_data = self.normalize_data(getattr(request_obj, "video_data", None))
        video_config = self._build_video_config(request_obj)
        videos = await self._load_videos_async(video_data, video_config)
        processor_kwargs = {}
        if videos:
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
            raise ValueError("HF multimodal processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).tolist()
        pixel_values = self._to_numpy(processor_output.get("pixel_values"))
        pixel_values_videos = self._to_numpy(processor_output.get("pixel_values_videos"))
        image_grid_thw = self._to_grid_list(processor_output.get("image_grid_thw"))
        video_grid_thw = self._to_grid_list(processor_output.get("video_grid_thw"))
        if images or videos:
            logger.info(
                "Qwen-VL processor output: images=%s, videos=%s, image_grid_thw=%s, "
                "video_grid_thw=%s, pixel_values_shape=%s, pixel_values_videos_shape=%s",
                len(images),
                len(videos),
                image_grid_thw,
                video_grid_thw,
                None if pixel_values is None else pixel_values.shape,
                None if pixel_values_videos is None else pixel_values_videos.shape,
            )
        second_per_grid_ts_value = processor_output.get("second_per_grid_ts")
        if second_per_grid_ts_value is None:
            second_per_grid_ts_value = processor_output.get("video_second_per_grid")
        second_per_grid_ts = self._to_list(second_per_grid_ts_value)

        vision_config = self.hf_config.vision_config
        image_placeholder_ranges = self._compute_image_placeholder_ranges(
            input_ids=input_ids,
            grids=image_grid_thw,
            image_token_id=self.hf_config.image_token_id,
            spatial_merge_size=vision_config.spatial_merge_size,
        )
        video_token_id = getattr(self.hf_config, "video_token_id", None)
        video_placeholder_ranges = self._compute_placeholder_ranges(
            input_ids=input_ids,
            grids=video_grid_thw,
            token_id=video_token_id,
            spatial_merge_size=vision_config.spatial_merge_size,
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

        mrope_positions, mrope_position_delta = compute_mrope_positions(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            vision_start_token_id=self.hf_config.vision_start_token_id,
            image_token_id=self.hf_config.image_token_id,
            video_token_id=video_token_id,
            spatial_merge_size=vision_config.spatial_merge_size,
            tokens_per_second=getattr(vision_config, "tokens_per_second", None),
        )
        mrope_position_delta = np.asarray([[mrope_position_delta]], dtype=np.int32)

        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_start_id=self.hf_config.vision_start_token_id,
            im_end_id=self.hf_config.vision_end_token_id,
            im_token_id=self.hf_config.image_token_id,
            video_token_id=video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    @staticmethod
    def _build_items(features, grids, placeholder_ranges, modality, grid_key):
        if features is None:
            return []
        if not grids:
            raise ValueError(f"Missing {grid_key} metadata for {modality.name} inputs.")
        if len(placeholder_ranges) != len(grids):
            raise ValueError(
                f"{modality.name} placeholder range count does not match grid metadata: "
                f"{len(placeholder_ranges)} != {len(grids)}."
            )

        feature_counts = [int(np.prod(grid)) for grid in grids]
        if sum(feature_counts) != len(features):
            raise ValueError(
                f"{modality.name} feature count does not match grid metadata: "
                f"{len(features)} != {sum(feature_counts)}."
            )

        items = []
        offset = 0
        for count, grid in zip(feature_counts, grids):
            item = MultimodalDataItem(
                modality=modality,
                feature=features[offset : offset + count],
            )
            item.set(grid_key, np.asarray([grid], dtype=np.int32))
            item.placeholder_ranges = [placeholder_ranges[len(items)]]
            items.append(item)
            offset += count
        return items

    async def _load_images_async(self, image_data):
        return await asyncio.gather(
            *(self.load_image_async(item) for item in self.normalize_data(image_data))
        )

    @staticmethod
    def _compute_image_placeholder_ranges(input_ids, grids, image_token_id, spatial_merge_size):
        return QwenVLProcessor._compute_placeholder_ranges(
            input_ids=input_ids,
            grids=grids,
            token_id=image_token_id,
            spatial_merge_size=spatial_merge_size,
            modality_name="IMAGE",
        )

    @staticmethod
    def _compute_placeholder_ranges(input_ids, grids, token_id, spatial_merge_size, modality_name):
        if not grids:
            return []
        if token_id is None:
            raise ValueError(f"{modality_name} token id is not configured.")

        placeholder_ranges = []
        search_start = 0
        for grid in grids:
            token_count = int(np.prod(grid) // (spatial_merge_size**2))
            start = None
            for idx in range(search_start, len(input_ids)):
                if input_ids[idx] == token_id:
                    start = idx
                    break
            if start is None:
                raise ValueError(
                    f"Missing {modality_name} placeholder tokens in processor input_ids."
                )

            end = start + token_count
            if end > len(input_ids) or any(
                input_token_id != token_id for input_token_id in input_ids[start:end]
            ):
                raise ValueError(
                    f"{modality_name} placeholder token span does not match grid metadata."
                )
            placeholder_ranges.append((start, end))
            search_start = end

        return placeholder_ranges

    async def _load_videos_async(self, video_data, video_config):
        return await asyncio.gather(
            *(
                asyncio.to_thread(preprocess_video, item, video_config)
                for item in self.normalize_data(video_data)
            )
        )

    @staticmethod
    def _build_video_config(request_obj):
        video_config = {}
        fps = getattr(request_obj, "fps", None)
        if fps is not None:
            video_config["fps"] = fps
        nframes = getattr(request_obj, "num_frames", None)
        if nframes is not None and "fps" not in video_config:
            video_config["nframes"] = nframes
        return video_config

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _to_grid_list(cls, value):
        if value is None:
            return None
        return [tuple(int(item) for item in row) for row in cls._to_numpy(value).tolist()]

    @classmethod
    def _to_list(cls, value):
        if value is None:
            return None
        return cls._to_numpy(value).reshape(-1).tolist()
