"""MiMo media transforms and token layouts ported from SGLang.

Source: sgl-project/sglang, commit 67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b,
python/sglang/srt/multimodal/processors/mimo_v2.py (Apache-2.0).
The JAX adapter supplies decoded media and grouped audio codes. Keep the
resize, patch ordering, timestamps, and interleaving algorithms in sync.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from sgl_jax.srt.configs.mimo import config_value

_QWEN2VL_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
_QWEN2VL_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


class MiMoMediaProcessor:
    def __init__(self, config, tokenizer):
        vision = config_value(config, "vision_config")
        pc = config.processor_config
        self.tokenizer = tokenizer
        self.patch_size = int(config_value(vision, "patch_size", 16))
        self.merge_size = int(config_value(vision, "spatial_merge_size", 2))
        self.temporal_patch_size = int(config_value(vision, "temporal_patch_size", 2))
        self.use_video_timestamps = bool(config_value(pc, "use_video_timestamps", False))
        self.video_audio_interleave_length = config_value(pc, "video_audio_interleave_length", 0)
        interleave = self.video_audio_interleave_length
        if not math.isfinite(interleave) or (interleave < 0 and interleave != -1):
            raise ValueError("video_audio_interleave_length must be -1 or finite and non-negative")
        self.audio_token_per_second = 25 / int(
            config_value(config_value(config, "audio_config"), "group_size", 4)
        )
        for name in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
            "audio_token_id",
            "audio_start_token_id",
            "audio_end_token_id",
        ):
            setattr(self, name, pc[name])
        factor = self.patch_size * self.merge_size
        self.default_image_processor_kwargs = {
            "min_pixels": config_value(pc, "image_min_pixels") or 4 * factor**2,
            "max_pixels": config_value(pc, "image_max_pixels") or 4096 * factor**2,
        }
        self.default_video_processor_kwargs = {
            "min_pixels": config_value(pc, "video_min_pixels") or 4 * factor**2,
            "max_pixels": config_value(pc, "video_max_pixels") or 4096 * factor**2,
            "total_max_pixels": config_value(pc, "video_total_max_pixels") or 16384 * factor**2,
            "fps": config_value(pc, "fps") or 2,
            "num_frames": config_value(pc, "num_frames"),
            "max_frames": config_value(pc, "max_frames") or 256,
            "min_frames": config_value(pc, "min_frames") or 8,
        }

    def prepare_image_kwargs(self, image: SimpleNamespace):
        return {
            key: default if getattr(image, key) is None else getattr(image, key)
            for key, default in self.default_image_processor_kwargs.items()
        }

    def prepare_video_kwargs(self, video: SimpleNamespace):
        kwargs = {
            key: (
                self.default_video_processor_kwargs[key]
                if getattr(video, key) is None
                else getattr(video, key)
            )
            for key in ("min_pixels", "max_pixels", "total_max_pixels")
        }
        if video.num_frames is not None:
            kwargs["num_frames"] = video.num_frames
        elif video.fps is not None:
            kwargs["fps"] = video.fps
            if video.max_frames is not None:
                kwargs["max_frames"] = video.max_frames
            if video.min_frames is not None:
                kwargs["min_frames"] = video.min_frames
        elif self.default_video_processor_kwargs["num_frames"] is not None:
            kwargs["num_frames"] = self.default_video_processor_kwargs["num_frames"]
        elif self.default_video_processor_kwargs["fps"] is not None:
            kwargs["fps"] = self.default_video_processor_kwargs["fps"]
            if self.default_video_processor_kwargs["max_frames"] is not None:
                kwargs["max_frames"] = self.default_video_processor_kwargs["max_frames"]
            if self.default_video_processor_kwargs["min_frames"] is not None:
                kwargs["min_frames"] = self.default_video_processor_kwargs["min_frames"]
        else:
            raise ValueError("Video sampling strategy not specified")
        return kwargs

    def process_image(self, image: SimpleNamespace):
        kwargs = self.prepare_image_kwargs(image)
        pixels = torch.from_numpy(np.array(image.image.convert("RGB"))).permute(2, 0, 1)
        return self.get_visual_transform_batch(
            pixels.unsqueeze(0), self.patch_size * self.merge_size, **kwargs
        ).squeeze(0)

    def process_video(self, video_input: SimpleNamespace):
        frames, timestamps = video_input.video
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError("MiMoV2 video must contain at least one frame.")
        fps = 1 if len(timestamps) < 2 else 1 / (timestamps[1] - timestamps[0])
        # Compute the clip end before padding: repeated frames do not extend audio.
        video_meta = {"segment_end_time": timestamps[-1] + 1 / fps}
        kwargs = self.prepare_video_kwargs(video_input)
        min_pixels = kwargs["min_pixels"]
        max_pixels = max(
            min_pixels,
            min(
                kwargs["max_pixels"],
                kwargs["total_max_pixels"] * self.temporal_patch_size // num_frames,
            ),
        )
        pad = (-num_frames) % self.temporal_patch_size
        if pad:
            frames = torch.cat((frames, frames[-1:].repeat(pad, 1, 1, 1)))
            timestamps = torch.cat((timestamps, timestamps[-1:].repeat(pad)))
        pixels = self.get_visual_transform_batch(
            frames, self.patch_size * self.merge_size, min_pixels, max_pixels
        )
        patches, grid = self._flatten_visual_inputs(pixels, "video")
        return patches, grid, timestamps, video_meta

    def process_image_content(self, image):
        pixels = self.process_image(image)
        patches, grid = self._flatten_visual_inputs(pixels, "image")
        return {
            "input_ids": [self.vision_start_token_id]
            + [self.image_token_id] * (int(grid.prod()) // self.merge_size**2)
            + [self.vision_end_token_id],
            "pixel_values": patches,
            "thw_grid": grid,
        }

    def process_video_content(self, video_result):
        patches, grid, timestamps, _ = video_result
        times = self._grid_timestamps(timestamps)
        tokens_per_frame = int(grid[1] * grid[2]) // self.merge_size**2
        input_ids = [self.video_start_token_id]
        for timestamp in times:
            input_ids += self.tokenizer.encode(self.format_timestamp(timestamp))
            input_ids += (
                [self.vision_start_token_id]
                + [self.video_token_id] * tokens_per_frame
                + [self.vision_end_token_id]
            )
        return {
            "input_ids": input_ids + [self.video_end_token_id],
            "pixel_values": patches,
            "thw_grid": grid,
        }

    def _grid_timestamps(self, timestamps):
        if not self.use_video_timestamps:
            raise NotImplementedError("MiMoV2 video requires use_video_timestamps.")
        return timestamps[:: self.temporal_patch_size]

    def process_audio_content(self, codes):
        return {
            "input_ids": [self.audio_start_token_id]
            + [self.audio_token_id] * len(codes)
            + [self.audio_end_token_id],
            "audio_segments": [codes],
        }

    def _build_video_audio_units(self, grid, timestamps, video_meta, codes):
        times = self._grid_timestamps(timestamps)
        units = []
        for index, timestamp in enumerate(times):
            start = int(timestamp * self.audio_token_per_second)
            end_time = (
                times[index + 1] if index + 1 < len(times) else video_meta["segment_end_time"]
            )
            end = min(int(end_time * self.audio_token_per_second), len(codes))
            if timestamp < 0 or end <= start:
                raise ValueError("MiMoV2 video time slice has no corresponding audio codes.")
            units.append(
                {
                    "timestamp": timestamp,
                    "num_video_tokens": int(grid[1] * grid[2]) // self.merge_size**2,
                    "segment_audio": codes[start:end],
                }
            )
        return units

    def _build_video_audio_input_ids(self, units):
        if self.video_audio_interleave_length == -1:
            groups = [units]
        elif self.video_audio_interleave_length == 0:
            groups = [[unit] for unit in units]
        else:
            # Retain SGLang's half-open time windows and accumulation order.
            groups = []
            index = 0
            time_ptr = 0
            while index < len(units):
                group = []
                while (
                    index < len(units)
                    and time_ptr <= units[index]["timestamp"]
                    and units[index]["timestamp"] < time_ptr + self.video_audio_interleave_length
                ):
                    group.append(units[index])
                    index += 1
                if group:
                    groups.append(group)
                time_ptr += self.video_audio_interleave_length

        input_ids = [self.video_start_token_id]
        audio_segments = []
        for group in groups:
            # Upstream uses one timestamp per group, including interleave=-1.
            input_ids += self.tokenizer.encode(self.format_timestamp(group[0]["timestamp"]))
            audio_tokens = 0
            for unit in group:
                input_ids += (
                    [self.vision_start_token_id]
                    + [self.video_token_id] * unit["num_video_tokens"]
                    + [self.vision_end_token_id]
                )
                audio_segments.append(unit["segment_audio"])
                audio_tokens += len(unit["segment_audio"])
            input_ids += (
                [self.audio_start_token_id]
                + [self.audio_token_id] * audio_tokens
                + [self.audio_end_token_id]
            )
        return {
            "input_ids": input_ids + [self.video_end_token_id],
            "audio_segments": audio_segments,
        }

    def process_video_audio_content(self, codes, video_result):
        patches, grid, timestamps, video_meta = video_result
        units = self._build_video_audio_units(grid, timestamps, video_meta, codes)
        return {
            **self._build_video_audio_input_ids(units),
            "pixel_values": patches,
            "thw_grid": grid,
        }

    def _flatten_visual_inputs(self, visual: torch.Tensor, visual_type: str):
        if visual_type == "image":
            resized_height, resized_width = visual.shape[-2:]
            patches = visual.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        elif visual_type == "video":
            patches = visual
            resized_height, resized_width = patches.shape[-2:]
        else:
            raise ValueError(f"Unknown visual_type: {visual_type}")

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        patches = patches.contiguous().view(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

        flatten_patches = patches.view(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        thw_grids = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.int32)

        return flatten_patches, thw_grids

    @staticmethod
    def format_timestamp(timestamp: float):
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if min(height, width) < factor:
            scale = factor / min(height, width)
            height = int(round(height * scale))
            width = int(round(width * scale))
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return int(h_bar), int(w_bar)

    @classmethod
    def get_visual_transform_batch(cls, frames, factor, min_pixels, max_pixels):
        height, width = cls.smart_resize(*frames.shape[-2:], factor, min_pixels, max_pixels)
        resized = F.interpolate(
            frames.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        return (resized - _QWEN2VL_PIXEL_MEAN) / _QWEN2VL_PIXEL_STD
