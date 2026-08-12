"""Gemma 4 image processor adapter for the in-model multimodal contract."""

from __future__ import annotations

import asyncio

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class Gemma4Processor(BaseMultimodalProcessor):
    models = ("Gemma4ForConditionalGeneration",)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            # NumPy has no portable bfloat16 representation. Pixel values are
            # host-side inputs, so float32 is the lossless interchange format.
            if str(getattr(value, "dtype", "")) == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _placeholder_runs(input_ids: list[int], token_id: int) -> list[tuple[int, int]]:
        runs = []
        start = None
        for index, value in enumerate((*input_ids, None)):
            if value == token_id and start is None:
                start = index
            elif value != token_id and start is not None:
                runs.append((start, index))
                start = None
        return runs

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> MultimodalInputs:
        if isinstance(input_text, list):
            raise ValueError("Gemma 4 multimodal requests require text input, not input_ids.")
        if self.normalize_data(getattr(request_obj, "video_data", None)):
            raise ValueError("Gemma 4 video inputs are not supported yet.")
        if self.normalize_data(getattr(request_obj, "audio_data", None)):
            raise ValueError("Gemma 4 audio inputs are not supported yet.")

        images = await self._load_images(image_data)
        processor_output = self.processor(
            text=[input_text],
            images=images or None,
            padding=True,
            return_tensors="pt",
        )
        input_ids_array = self._to_numpy(processor_output.get("input_ids"))
        if input_ids_array is None:
            raise ValueError("Gemma 4 processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).astype(np.int64).tolist()

        pixel_values = self._to_numpy(processor_output.get("pixel_values"))
        pixel_position_ids = self._to_numpy(processor_output.get("image_position_ids"))
        if pixel_position_ids is None:
            # vLLM's Gemma 4 input contract uses this compatibility name.
            pixel_position_ids = self._to_numpy(processor_output.get("pixel_position_ids"))
        image_token_id = int(self.hf_config.image_token_id)
        if not images:
            return MultimodalInputs(mm_items=[], input_ids=input_ids)
        if pixel_values is None or pixel_position_ids is None:
            raise ValueError("Gemma 4 processor must return pixel_values and image_position_ids.")
        if pixel_values.ndim == 2:
            pixel_values = pixel_values[None]
        if pixel_position_ids.ndim == 2:
            pixel_position_ids = pixel_position_ids[None]
        if pixel_values.shape[0] != len(images) or pixel_position_ids.shape[0] != len(images):
            raise ValueError(
                "Gemma 4 processor image batch does not match the request: "
                f"{pixel_values.shape[0]}, {pixel_position_ids.shape[0]} != {len(images)}"
            )

        pooling_unit = int(self.hf_config.vision_config.pooling_kernel_size) ** 2
        placeholder_runs = self._placeholder_runs(input_ids, image_token_id)
        if len(placeholder_runs) != len(images):
            raise ValueError(
                "Gemma 4 image placeholder count does not match image count: "
                f"{len(placeholder_runs)} != {len(images)}"
            )

        items = []
        for index, placeholder_range in enumerate(placeholder_runs):
            positions = np.asarray(pixel_position_ids[index], dtype=np.int32)
            if pixel_values[index].shape[0] != positions.shape[0]:
                raise ValueError(
                    "Gemma 4 patch and position capacities do not match: "
                    f"{pixel_values[index].shape[0]} != {positions.shape[0]}."
                )
            padded = np.all(positions == -1, axis=-1)
            valid = np.all(positions >= 0, axis=-1)
            if not np.all(padded | valid):
                raise ValueError(f"Gemma 4 image {index} contains malformed position ids.")
            positions = positions[valid]
            patches = np.asarray(pixel_values[index])[valid]
            if len(patches) == 0 or len(patches) % pooling_unit:
                raise ValueError(
                    f"Gemma 4 image {index} has invalid patch count {len(patches)}; "
                    f"expected a positive multiple of {pooling_unit}."
                )
            output_length = len(patches) // pooling_unit
            if placeholder_range[1] - placeholder_range[0] != output_length:
                raise ValueError(
                    f"Gemma 4 image {index} produces {output_length} visual tokens, "
                    f"but its placeholder span has length "
                    f"{placeholder_range[1] - placeholder_range[0]}."
                )
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=patches,
                model_specific_data={"pixel_position_ids": positions},
            )
            item.placeholder_ranges = [placeholder_range]
            item.set_pad_value()
            items.append(item)

        return MultimodalInputs(
            mm_items=items,
            input_ids=input_ids,
            im_start_id=getattr(self.hf_config, "boi_token_id", None),
            im_end_id=getattr(self.hf_config, "eoi_token_id", None),
            im_token_id=image_token_id,
        )

    async def _load_images(self, image_data):
        return await asyncio.gather(
            *(self.load_image_async(item) for item in self.normalize_data(image_data))
        )
