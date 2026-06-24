import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_mrope_positions
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


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
        images = [self.load_image(item) for item in self.normalize_data(image_data)]
        if not images:
            return None

        if isinstance(input_text, list):
            input_text = self.processor.tokenizer.decode(input_text)

        processor_output = self.processor(
            text=[input_text],
            images=images,
            padding=True,
            return_tensors="np",
        )

        input_ids_array = self._to_numpy(processor_output.get("input_ids"))
        if input_ids_array is None:
            raise ValueError("HF multimodal processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).tolist()
        pixel_values = self._to_numpy(processor_output.get("pixel_values"))
        image_grid_thw = self._to_grid_list(processor_output.get("image_grid_thw"))

        mm_items = self._build_items(pixel_values, image_grid_thw)
        for item in mm_items:
            item.set_pad_value()

        vision_config = self.hf_config.vision_config
        mrope_positions, mrope_position_delta = compute_mrope_positions(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            vision_start_token_id=self.hf_config.vision_start_token_id,
            image_token_id=self.hf_config.image_token_id,
            video_token_id=None,
            spatial_merge_size=vision_config.spatial_merge_size,
            tokens_per_second=None,
        )

        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_start_id=self.hf_config.vision_start_token_id,
            im_end_id=getattr(self.hf_config, "vision_end_token_id", None),
            im_token_id=self.hf_config.image_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    @staticmethod
    def _build_items(features, grids):
        if features is None:
            return []
        if not grids:
            raise ValueError("Missing image_grid_thw metadata for IMAGE inputs.")

        feature_counts = [int(np.prod(grid)) for grid in grids]
        if sum(feature_counts) != len(features):
            raise ValueError(
                f"IMAGE feature count does not match grid metadata: "
                f"{len(features)} != {sum(feature_counts)}."
            )

        items = []
        offset = 0
        for count, grid in zip(feature_counts, grids):
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=features[offset : offset + count],
            )
            item.set("image_grid_thw", np.asarray([grid], dtype=np.int32))
            items.append(item)
            offset += count
        return items

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
