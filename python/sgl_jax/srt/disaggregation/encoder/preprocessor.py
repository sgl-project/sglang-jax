"""Media preprocessing for the standalone encoder, without language-model inputs."""

from __future__ import annotations

import logging

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sgl_jax.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sgl_jax.srt.multimodal.processors.qwen_vl import FPS, preprocess_video

logger = logging.getLogger(__name__)


class EncoderPreprocessor:
    """Load media and produce vision features and grid metadata for encoding."""

    def __init__(self, hf_config, server_args, processor):
        self.hf_config = getattr(hf_config, "thinker_config", hf_config)
        self.spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        self._is_qwen = "qwen" in getattr(self.hf_config, "model_type", "").lower() or any(
            "Qwen" in name for name in getattr(self.hf_config, "architectures", ())
        )
        self.worker_count = getattr(server_args, "mm_processor_worker_num", 0) or (
            2 if self._is_qwen else 1
        )
        if self.worker_count <= 0:
            raise ValueError("Encoder preprocessor worker count must be positive")
        try:
            self._executor = MultimodalProcessorExecutor(processor, self.worker_count)
        except Exception:
            logger.warning("Unable to clone encoder processor; using one worker", exc_info=True)
            self.worker_count = 1
            self._executor = MultimodalProcessorExecutor(processor, 1)

    async def process_mm_items(
        self, mm_items, modality: Modality, *, fps=None, num_frames=None
    ) -> MultimodalInputs:
        if not mm_items:
            raise ValueError("encoder request contains no multimodal items")
        if modality == Modality.IMAGE:
            return await self._executor.run(self._process_image_items, mm_items)
        if modality == Modality.VIDEO and self._is_qwen:
            return await self._executor.run(
                self._process_video_items, mm_items, fps=fps, num_frames=num_frames
            )
        raise ValueError(f"Encoder preprocessor does not support {modality.name.lower()} inputs")

    def _process_video_items(self, sources, *, fps, num_frames, processor) -> MultimodalInputs:
        video_config = {
            "factor": int(self.hf_config.vision_config.patch_size * self.spatial_merge_size)
        }
        if fps is not None:
            video_config["fps"] = fps
        elif num_frames is not None:
            video_config["nframes"] = num_frames
        videos = [
            preprocess_video(BaseMultimodalProcessor.unwrap_source(source), video_config)
            for source in sources
        ]
        output = processor.video_processor(
            videos=videos, do_sample_frames=False, return_tensors="pt"
        )
        inputs = self._encoder_vision_inputs(output, Modality.VIDEO, len(videos))
        seconds = np.float32(
            processor.video_processor.temporal_patch_size / video_config.get("fps", FPS)
        )
        for item in inputs.mm_items:
            item.set("second_per_grid_ts", seconds)
        return inputs

    def _process_image_items(self, image_sources, *, processor) -> MultimodalInputs:
        """Preserve the image processor's patches and grid; skip text processing."""
        images = [BaseMultimodalProcessor.load_image(source) for source in image_sources]
        output = processor.image_processor(images=images, return_tensors=None)
        return self._encoder_vision_inputs(output, Modality.IMAGE, len(images))

    def _encoder_vision_inputs(self, output, modality, item_count) -> MultimodalInputs:
        grid_key = "image_grid_thw" if modality == Modality.IMAGE else "video_grid_thw"
        feature_key = "pixel_values" if modality == Modality.IMAGE else "pixel_values_videos"
        features = BaseMultimodalProcessor._to_numpy(output.get(feature_key))
        grids = BaseMultimodalProcessor._to_grid_list(output.get(grid_key))
        if features is None or len(grids) != item_count:
            raise ValueError(
                f"{modality.name} processor returned incomplete patches or grid metadata"
            )
        counts = [int(np.prod(grid)) for grid in grids]
        if any(count <= 0 or count % self.spatial_merge_size**2 for count in counts):
            raise ValueError(
                f"{modality.name} patch counts must be positive multiples of merge size"
            )
        if sum(counts) != len(features):
            raise ValueError(f"{modality.name} patch count does not match grid metadata")
        items = []
        offset = 0
        for grid, count in zip(grids, counts):
            item = MultimodalDataItem(
                modality=modality,
                feature=features[offset : offset + count],
            )
            item.set(grid_key, np.asarray([grid], dtype=np.int32))
            item.set_pad_value()
            items.append(item)
            offset += count
        return MultimodalInputs(mm_items=items)

    def shutdown(self) -> None:
        self._executor.shutdown()
