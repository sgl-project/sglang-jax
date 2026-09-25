"""Reconstruct language inputs from independently encoded multimodal parts."""

from __future__ import annotations

import numpy as np

from sgl_jax.srt.disaggregation.encoder.embedding_data import ReceivedEmbeddings
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)


class DisaggregatedInputMixin:
    """Reconstruct language-model inputs from disaggregated encoder outputs."""

    def process_encoder_images(self, image_sources, *, processor) -> MultimodalInputs:
        """Preserve the image processor's patches and grid; skip text processing."""
        images = [self.load_image(source) for source in image_sources]
        output = processor.image_processor(images=images, return_tensors=None)
        features = self._to_numpy(output.get("pixel_values"))
        grids = self._to_grid_list(output.get("image_grid_thw"))
        if features is None or len(grids) != len(images):
            raise ValueError("Image processor returned incomplete patches or grid metadata")
        counts = [int(np.prod(grid)) for grid in grids]
        if sum(counts) != len(features):
            raise ValueError("Image patch count does not match grid metadata")
        items = []
        offset = 0
        for grid, count in zip(grids, counts):
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=features[offset : offset + count],
                # Encoder-local spans carry lengths required by lane packing.
                placeholder_ranges=[
                    (
                        offset // self.spatial_merge_size**2,
                        (offset + count) // self.spatial_merge_size**2,
                    )
                ],
            )
            item.set("image_grid_thw", np.asarray([grid], dtype=np.int32))
            item.set_pad_value()
            items.append(item)
            offset += count
        return MultimodalInputs(mm_items=items)


    # EPD input reconstruction adapted from SGLang:
    # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/processors/base_processor.py
    @property
    def spatial_merge_size(self) -> int:
        return self.hf_config.vision_config.spatial_merge_size

    def build_input_ids(
        self,
        prompt,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_seq_lens=None,
    ):
        """Expand one placeholder per multimodal item into its encoded length."""
        if not isinstance(prompt, list):
            prompt = self.processor.tokenizer(prompt)["input_ids"]

        grids = {
            Modality.IMAGE: self._to_grid_list(image_grid_thw),
            Modality.VIDEO: self._to_grid_list(video_grid_thw),
        }
        audio_lengths = (
            []
            if audio_seq_lens is None
            else self._to_numpy(audio_seq_lens).reshape(-1).astype(int).tolist()
        )
        token_ids = {
            Modality.IMAGE: getattr(self.hf_config, "image_token_id", None),
            Modality.VIDEO: getattr(self.hf_config, "video_token_id", None),
            Modality.AUDIO: getattr(self.hf_config, "audio_token_id", None),
        }
        modality_by_token = {
            token_id: modality for modality, token_id in token_ids.items() if token_id is not None
        }
        item_sizes = {
            modality: [int(np.prod(grid) // (self.spatial_merge_size**2)) for grid in values]
            for modality, values in grids.items()
        }
        item_sizes[Modality.AUDIO] = audio_lengths
        consumed = {modality: 0 for modality in item_sizes}
        input_ids, ranges, modalities = [], [], []

        for token_id in prompt:
            modality = modality_by_token.get(token_id)
            if modality is None:
                input_ids.append(token_id)
                continue

            item_index = consumed[modality]
            if item_index >= len(item_sizes[modality]):
                raise ValueError(f"missing {modality.name} encoder metadata")
            token_count = item_sizes[modality][item_index]
            start = len(input_ids)
            input_ids.extend([token_id] * token_count)
            ranges.append((start, len(input_ids)))
            modalities.append(modality)
            consumed[modality] += 1

        for modality, sizes in item_sizes.items():
            if consumed[modality] != len(sizes):
                raise ValueError(f"unused {modality.name} encoder metadata")
        return input_ids, ranges, modalities

    def get_mm_data(
        self, prompt, embeddings: dict[Modality, ReceivedEmbeddings], **metadata
    ) -> MultimodalInputs:
        """Rebuild native multimodal inputs from encoder-disaggregated outputs."""
        input_ids, ranges, modalities = self.build_input_ids(
            prompt,
            image_grid_thw=metadata.get("image_grid_thw"),
            video_grid_thw=metadata.get("video_grid_thw"),
            audio_seq_lens=metadata.get("audio_feature_lens"),
        )
        consumed = {modality: 0 for modality in Modality.all()}
        consumed_items = {modality: 0 for modality in Modality.all()}
        item_hashes = metadata["item_hashes"]
        mm_items = []
        radix_input_ids = list(input_ids)
        for modality, placeholder_range in zip(modalities, ranges):
            modality_embeddings = embeddings.get(modality)
            if modality_embeddings is None:
                raise ValueError(f"missing {modality.name} encoder embeddings")
            start = consumed[modality]
            end = start + placeholder_range[1] - placeholder_range[0]
            embedding = modality_embeddings.slice_rows(start, end)
            if len(embedding.row_indices) != end - start:
                raise ValueError(f"incomplete {modality.name} encoder embeddings")

            hashes = item_hashes[modality]
            item_index = consumed_items[modality]
            item = MultimodalDataItem(
                modality=modality,
                hash=int(hashes[item_index]),
                placeholder_ranges=[placeholder_range],
                precomputed_embeddings=embedding,
            )
            item.set_pad_value()
            range_start, range_end = placeholder_range
            radix_input_ids[range_start:range_end] = [item.pad_value] * (range_end - range_start)
            mm_items.append(item)
            consumed[modality] = end
            consumed_items[modality] += 1

        for modality, modality_embeddings in embeddings.items():
            length = len(modality_embeddings.row_indices)
            if modality in consumed and consumed[modality] != length:
                raise ValueError(f"unused {modality.name} encoder embeddings")

        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            radix_input_ids=radix_input_ids,
            im_start_id=getattr(self.hf_config, "vision_start_token_id", None),
            im_end_id=getattr(self.hf_config, "vision_end_token_id", None),
            im_token_id=getattr(self.hf_config, "image_token_id", None),
            video_token_id=getattr(self.hf_config, "video_token_id", None),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_id=getattr(self.hf_config, "audio_start_token_id", None),
            audio_end_id=getattr(self.hf_config, "audio_end_token_id", None),
        )
