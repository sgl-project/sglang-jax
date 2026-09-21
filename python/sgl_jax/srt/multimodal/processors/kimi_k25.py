from __future__ import annotations

import asyncio
import logging

import numpy as np

from sgl_jax.srt.multimodal.common.mecord_compat import install_mecord_shim
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

logger = logging.getLogger(__name__)

# Kimi-K2.5's remote processor code runs `from mecord import VideoReader` at
# module import time, so the decord-backed shim has to be in `sys.modules`
# before AutoProcessor pulls that code in. TokenizerManager imports the
# processor package before it builds the HF processor, which makes module
# import the only reliably-early hook. No-op when a real mecord is installed.
install_mecord_shim()

DEFAULT_VIDEO_PLACEHOLDER = "<|kimi_k25_video_placeholder|>"
DEFAULT_MERGE_KERNEL_SIZE = (2, 2)


class KimiK25Processor(BaseMultimodalProcessor):
    """Standard-serving-path processor for Kimi-K2.5.

    Kimi differs from the Qwen-VL family in three ways that shape this class:

    1. Its HF processor takes one ordered ``medias`` list rather than separate
       ``images`` / ``videos`` arguments, and it does its own decoding and frame
       sampling. Raw sources are handed over untouched.
    2. It reports every media item - a still image, or one chunk of a video - in
       a single ``grid_thws`` tensor, and emits exactly one media placeholder
       token per grid. There is no separate video tensor.
    3. The vision tower average-pools a chunk's frames, so a grid ``(t, h, w)``
       yields ``h * w / merge_area`` visual tokens *independently of t*. This is
       why Kimi cannot reuse Qwen's `lane_packing` helper, which asserts that
       placeholder count times the merge unit equals the patch count.
    """

    models = ("KimiK25ForConditionalGeneration",)

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> MultimodalInputs:
        if getattr(request_obj, "audio_data", None) is not None:
            raise ValueError("Kimi-K2.5 does not support audio inputs.")
        if isinstance(input_text, list):
            raise ValueError(
                "Multimodal input_ids are not supported for Kimi-K2.5. "
                "Please provide text input instead."
            )

        image_sources = self.normalize_data(image_data)
        video_sources = self.normalize_data(getattr(request_obj, "video_data", None))
        # Kimi decodes and frame-samples videos inside its remote processor code,
        # which is slow and blocking. Hand the whole thing to a worker thread
        # rather than the base class's executor: the executor clones the HF
        # processor, and Kimi's carries unpicklable remote-code state.
        return await asyncio.to_thread(
            self._process_mm_data, input_text, image_sources, video_sources
        )

    def _process_mm_data(
        self,
        input_text,
        image_sources,
        video_sources,
    ) -> MultimodalInputs:
        # Images are decoded here because the remote code expects PIL objects,
        # but video sources stay raw: the processor samples them itself.
        medias = [{"type": "image", "image": self.load_image(source)} for source in image_sources]
        medias.extend(
            {"type": "video", "video": self.unwrap_source(source), "first_frame_timestamp": 0.0}
            for source in video_sources
        )
        input_text = self._ensure_video_placeholders(input_text, len(video_sources))

        processor_output = self.processor(
            text=input_text or "",
            medias=medias,
            return_tensors="pt",
        )
        return self.collect_mm_items_from_processor_output(processor_output)

    def collect_mm_items_from_processor_output(
        self,
        processor_output,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        **kwargs,
    ) -> MultimodalInputs:
        del images, videos, audios, kwargs

        input_ids_array = self._to_numpy(processor_output.get("input_ids"))
        if input_ids_array is None:
            raise ValueError("Kimi-K2.5 processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).tolist()

        grids = self._to_grid_list(processor_output.get("grid_thws"))
        pixel_values = self._strip_batch_dim(self._to_numpy(processor_output.get("pixel_values")))

        if not grids:
            if pixel_values is not None and pixel_values.size:
                raise ValueError("Kimi-K2.5 processor returned pixel values without grid metadata.")
            return MultimodalInputs(mm_items=[], input_ids=input_ids)

        input_ids, placeholder_ranges = self._expand_media_placeholders(input_ids, grids)
        mm_items = self._build_items(pixel_values, grids, placeholder_ranges)
        for item in mm_items:
            item.set_pad_value()

        logger.debug(
            "Kimi-K2.5 processor output: grids=%s, pixel_values_shape=%s, visual_tokens=%s",
            len(grids),
            None if pixel_values is None else pixel_values.shape,
            sum(end - start for start, end in placeholder_ranges),
        )

        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_token_id=self._media_placeholder_token_id(),
        )

    def _build_items(
        self,
        pixel_values: np.ndarray | None,
        grids: list[tuple[int, int, int]],
        placeholder_ranges: list[tuple[int, int]],
    ) -> list[MultimodalDataItem]:
        """Split the concatenated patch tensor into one item per grid.

        The in-model orchestration keys everything - chunked-prefill clipping,
        embedding-pool hits, per-item packed offsets - off individual items, so
        the single all-media item the MultiStage path used is not usable here.
        """
        if pixel_values is None:
            raise ValueError("Kimi-K2.5 processor returned grid metadata without pixel values.")

        patch_counts = [t * h * w for t, h, w in grids]
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError(
                "Kimi-K2.5 patch count does not match grid metadata: "
                f"{pixel_values.shape[0]} != {sum(patch_counts)}."
            )

        items: list[MultimodalDataItem] = []
        offset = 0
        for count, grid, placeholder_range in zip(
            patch_counts, grids, placeholder_ranges, strict=True
        ):
            item = MultimodalDataItem(
                # Kimi tags video chunks as images: its processor emits a single
                # `grid_thws`/`pixel_values` pair and never `pixel_values_videos`.
                # A video is just a run of grids with t > 1, and the tower path
                # is byte-for-byte identical, so there is nothing to distinguish.
                modality=Modality.IMAGE,
                feature=pixel_values[offset : offset + count],
            )
            item.set("image_grid_thw", np.asarray([grid], dtype=np.int32))
            item.placeholder_ranges = [placeholder_range]
            items.append(item)
            offset += count
        return items

    def _expand_media_placeholders(
        self,
        input_ids: list[int],
        grids: list[tuple[int, int, int]],
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Expand each media placeholder into one token per visual token.

        Kimi's template emits a single placeholder per media item. The expanded
        count is ``h * w / merge_area`` and does not depend on ``t``, because the
        vision tower average-pools a chunk's frames into one spatial grid.

        Returns the expanded ids and the half-open ``[start, end)`` range each
        grid occupies in them. Those ranges are what drives the whole in-model
        merge, so a mismatch is fatal here rather than a warning: under-counting
        silently corrupted embeddings on the MultiStage path.
        """
        token_id = self._media_placeholder_token_id()
        if token_id is None:
            raise ValueError(
                "Kimi-K2.5 config does not define media_placeholder_token_id; "
                "cannot locate visual token spans."
            )

        merge_area = self._merge_area()
        expanded: list[int] = []
        placeholder_ranges: list[tuple[int, int]] = []
        grid_index = 0

        for token in input_ids:
            if token != token_id:
                expanded.append(token)
                continue
            if grid_index >= len(grids):
                raise ValueError(
                    f"Kimi-K2.5 prompt has more media placeholders than grids ({len(grids)})."
                )
            _, height, width = grids[grid_index]
            num_visual_tokens = (height * width) // merge_area
            if num_visual_tokens <= 0:
                raise ValueError(f"Kimi-K2.5 grid {grids[grid_index]} yields no visual tokens.")
            start = len(expanded)
            expanded.extend([token] * num_visual_tokens)
            placeholder_ranges.append((start, start + num_visual_tokens))
            grid_index += 1

        if grid_index != len(grids):
            raise ValueError(
                f"Kimi-K2.5 prompt has {grid_index} media placeholders but the processor "
                f"returned {len(grids)} grids."
            )
        return expanded, placeholder_ranges

    def _ensure_video_placeholders(self, text: str | None, num_videos: int) -> str | None:
        """Make sure the prompt carries one video placeholder per video.

        The chat template emits these automatically, but a raw prompt sent with
        video_data will not have them. The processor asserts that the count
        matches, and without a placeholder the per-chunk prompts (which carry
        the media tokens) would never reach the text.
        """
        if num_videos <= 0:
            return text

        placeholder = (
            getattr(self.processor, "video_placeholder", None)
            or getattr(self.hf_config, "video_placeholder", None)
            or DEFAULT_VIDEO_PLACEHOLDER
        )
        text = text or ""
        missing = num_videos - text.count(placeholder)
        if missing > 0:
            text = placeholder * missing + text
        return text

    def _media_placeholder_token_id(self) -> int | None:
        return getattr(self.hf_config, "media_placeholder_token_id", None) or getattr(
            self.hf_config, "image_token_id", None
        )

    def _merge_area(self) -> int:
        """Number of patches merged into one visual token."""
        vision_config = getattr(self.hf_config, "vision_config", None)
        merge_kernel_size = getattr(vision_config, "merge_kernel_size", None) or getattr(
            self.hf_config, "merge_kernel_size", None
        )
        if merge_kernel_size is None:
            merge_kernel_size = DEFAULT_MERGE_KERNEL_SIZE
        if isinstance(merge_kernel_size, int):
            return merge_kernel_size * merge_kernel_size
        area = 1
        for value in merge_kernel_size:
            area *= int(value)
        return area

    @staticmethod
    def _strip_batch_dim(array: np.ndarray | None) -> np.ndarray | None:
        """Drop the singleton batch axis Kimi's processor puts on pixel_values."""
        if array is None:
            return None
        if array.ndim > 1 and array.shape[0] == 1:
            return array[0]
        return array
