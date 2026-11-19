"""InternVL (HF ``InternVLProcessor``) adapter for the in-model multimodal contract."""

from __future__ import annotations

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


def crops_to_patches(crops: np.ndarray, patch: int) -> np.ndarray:
    """``[n, C, H, W]`` crops -> ``[n * (H/p) * (W/p), C * p * p]`` row-major patches.

    Each patch vector is flattened in (C, kh, kw) order to match the Conv2d
    patch-embedding weight.
    """
    n, c, h, w = crops.shape
    x = crops.reshape(n, c, h // patch, patch, w // patch, patch)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x.reshape(n * (h // patch) * (w // patch), c * patch * patch))


class InternVLProcessor(BaseMultimodalProcessor):
    models = ("InternVLForConditionalGeneration",)
    auto_mm_processor_worker_num = 2
    supports_mm_processor_concurrency = True

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if str(getattr(value, "dtype", "")) == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

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
            raise ValueError("InternVL multimodal requests require text input, not input_ids.")
        if self.normalize_data(getattr(request_obj, "video_data", None)):
            raise ValueError("InternVL video inputs are not supported yet.")
        if self.normalize_data(getattr(request_obj, "audio_data", None)):
            raise ValueError("InternVL has no audio inputs.")
        return await self.mm_processor_executor.run(self._process_request, image_data, input_text)

    def _process_request(self, image_data, input_text, *, processor):
        images = [self.load_image(item) for item in self.normalize_data(image_data)]
        return self.process_and_combine_mm_data(
            input_text, images=images, processor=processor, return_tensors="pt"
        )

    def collect_mm_items_from_processor_output(
        self, processor_output, images=None, videos=None, audios=None, **kwargs
    ) -> MultimodalInputs:
        input_ids = (
            self._to_numpy(processor_output.get("input_ids")).reshape(-1).astype(np.int64).tolist()
        )
        if not images:
            return MultimodalInputs(mm_items=[], input_ids=input_ids)

        pixel_values = self._to_numpy(processor_output.get("pixel_values"))
        if pixel_values is None or pixel_values.ndim != 4:
            raise ValueError("InternVL processor must return pixel_values [crops, C, H, W].")

        vision = self.hf_config.vision_config
        patch = vision.patch_size[0] if isinstance(vision.patch_size, list) else vision.patch_size
        grid_h, grid_w = pixel_values.shape[2] // patch, pixel_values.shape[3] // patch
        tokens_per_crop = int(self.hf_config.image_seq_length)
        merge_unit = int(round(1 / float(self.hf_config.downsample_ratio))) ** 2
        if grid_h * grid_w != tokens_per_crop * merge_unit:
            raise ValueError(
                f"InternVL crop grid {grid_h}x{grid_w} does not match "
                f"image_seq_length={tokens_per_crop} x merge_unit={merge_unit}."
            )

        image_token_id = int(self.hf_config.image_token_id)
        runs = self._placeholder_runs(input_ids, image_token_id)
        if len(runs) != len(images):
            raise ValueError(
                f"InternVL image placeholder count does not match image count: "
                f"{len(runs)} != {len(images)}"
            )

        items, offset = [], 0
        for start, end in runs:
            if (end - start) % tokens_per_crop:
                raise ValueError(f"InternVL placeholder span {end - start} is not whole crops")
            num_crops = (end - start) // tokens_per_crop
            crops = pixel_values[offset : offset + num_crops].astype(np.float32)
            if len(crops) != num_crops:
                raise ValueError("InternVL processor returned fewer crops than placeholders")
            offset += num_crops
            item = MultimodalDataItem(
                modality=Modality.IMAGE, feature=crops_to_patches(crops, patch)
            )
            item.set("image_grid_thw", np.asarray([[num_crops, grid_h, grid_w]], dtype=np.int32))
            item.placeholder_ranges = [(start, end)]
            item.set_pad_value()
            items.append(item)
        if offset != len(pixel_values):
            raise ValueError(
                f"InternVL processor returned {len(pixel_values)} crops for {offset} placeholders"
            )

        tok = getattr(self, "processor", None)
        tok = getattr(tok, "tokenizer", None)
        return MultimodalInputs(
            mm_items=items,
            input_ids=input_ids,
            im_start_id=tok.convert_tokens_to_ids("<img>") if tok is not None else None,
            im_end_id=tok.convert_tokens_to_ids("</img>") if tok is not None else None,
            im_token_id=image_token_id,
        )
