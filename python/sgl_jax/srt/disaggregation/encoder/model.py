from __future__ import annotations

import logging
import os
from typing import Any

import jax
import numpy as np

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRequest
from sgl_jax.srt.hf_transformers_utils import get_processor
from sgl_jax.srt.model_loader import get_model
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.host_orchestration import (
    precompile_multimodal_encoder,
)
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_vision_inputs,
    plan_encoder_lanes,
)
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.server_args import ServerArgs, apply_multimodal_model_defaults
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

logger = logging.getLogger(__name__)


class EncoderModelRunner:
    """Run the model's native multimodal encoder for EPD requests."""

    _GRID_KEYS = {
        Modality.IMAGE: "image_grid_thw",
        Modality.VIDEO: "video_grid_thw",
    }

    def __init__(self, server_args: ServerArgs) -> None:
        self.model_config = ModelConfig.from_server_args(server_args)
        apply_multimodal_model_defaults(server_args, self.model_config)
        if not self.model_config.is_multimodal:
            raise ValueError("--encoder-only requires an in-model multimodal architecture")

        config = self.model_config.hf_config
        config.vision_encoder_parallel = server_args.vision_encoder_parallel
        config.precompile_vision_patch_paddings = server_args.precompile_vision_patch_paddings
        mesh = create_device_mesh(
            ici_parallelism=[
                server_args.dp_size,
                server_args.tp_size // server_args.dp_size,
            ],
            dcn_parallelism=[1, 1],
            device_indexes=server_args.device_indexes,
        )
        self.model = get_model(
            model_config=self.model_config,
            load_config=LoadConfig(
                load_format=server_args.load_format,
                download_dir=server_args.download_dir,
            ),
            mesh=mesh,
        )
        target = getattr(self.model, "thinker", self.model)
        self.visual = target.visual
        self.mesh = target.mesh
        self.num_lanes = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        self.input_sharding = self.visual.specs.sharding(self.visual.specs.batch_axis)
        self.packed_capacities = ()
        if not server_args.disable_precompile:
            logger.info("Precompiling multimodal encoder")
            self.packed_capacities = precompile_multimodal_encoder(
                self.model,
                num_lanes=self.num_lanes,
                patch_paddings=server_args.precompile_vision_patch_paddings,
            )

        tokenizer_path = server_args.tokenizer_path
        tokenizer_subdir = resolve_tokenizer_subdir(server_args.model_path, tokenizer_path)
        if tokenizer_subdir:
            tokenizer_path = os.path.join(tokenizer_path, tokenizer_subdir)
        processor = get_processor(
            tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=True,
        )
        self.preprocessor = EncoderPreprocessor(config, server_args, processor)

    async def preprocess_request(self, pending: EncoderRequest) -> None:
        """Preprocess one request without committing it to a ViT batch."""
        payload = pending.payload
        modality = pending.modality
        if modality not in self._GRID_KEYS:
            raise ValueError(f"Unsupported encoder modality: {modality.name}")
        mm_items = payload.get("mm_items") or []
        if not mm_items:
            raise ValueError("encoder request contains no multimodal items")
        inputs = await self.preprocessor.process_mm_items(
            mm_items, modality, fps=payload.get("fps"), num_frames=payload.get("num_frames")
        )
        items = [item for item in inputs.mm_items if item.modality == modality]
        if len(items) != len(mm_items):
            raise ValueError(
                f"processor produced {len(items)} {modality.name} items for {len(mm_items)} inputs"
            )
        inputs.mm_items = items
        pending.inputs = inputs
        pending.token_count = sum(
            item.feature.shape[0] // self.visual.spatial_merge_unit for item in items
        )

    def build_batch(
        self,
        inputs: list[MultimodalInputs],
    ) -> tuple[list[list[MultimodalDataItem]], np.ndarray]:
        if not inputs:
            raise ValueError("EncoderModelRunner batches must not be empty")
        items = [item for processed in inputs for item in processed.mm_items]
        lanes, output_indices = plan_encoder_lanes(
            [item.feature.shape[0] for item in items],
            self.num_lanes,
            merge_unit=self.visual.spatial_merge_unit,
        )
        items_by_lane = [[items[index] for index in lane] for lane in lanes]
        return items_by_lane, output_indices

    @property
    def preprocess_concurrency(self) -> int:
        """Keep the processor workers busy with complete preprocessing requests."""
        return self.preprocessor.worker_count

    def encode(
        self,
        items_by_lane: list[list[MultimodalDataItem]],
        *,
        modality: Modality,
    ) -> jax.Array:
        with jax.profiler.TraceAnnotation(f"mm_encode:{modality.name}"):
            visual = self.visual
            input_sharding = self.input_sharding
            patches, output_indices, grid_thw = pack_vision_inputs(
                items_by_lane,
                merge_unit=visual.spatial_merge_unit,
                input_sharding=input_sharding,
            )
            capacity = output_indices.size * visual.spatial_merge_unit // self.num_lanes
            with jax.profiler.TraceAnnotation("encoder_build_vision_metadata"):
                metadata = visual.prepare_metadata(grid_thw, capacity, sharding=input_sharding)
            with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation("encoder_vision_dispatch"):
                embeddings = visual(patches, **metadata)
        return embeddings

    def metadata_for_request(self, request: EncoderRequest) -> dict[str, Any]:
        mm_inputs = request.inputs
        modality = request.modality
        metadata: dict[str, Any] = {}
        item_hashes = [item.hash for item in mm_inputs.mm_items]
        if any(value is None for value in item_hashes):
            raise ValueError("encoder inputs are missing media hashes")
        metadata["item_hashes"] = [int(value) for value in item_hashes]
        grid_key = self._GRID_KEYS.get(modality)
        if grid_key is not None:
            metadata["grid_dim"] = np.concatenate(
                [np.asarray(item.get(grid_key)) for item in mm_inputs.mm_items], axis=0
            ).tolist()
        if modality == Modality.VIDEO:
            timing = [item.get("second_per_grid_ts") for item in mm_inputs.mm_items]
            if all(value is not None for value in timing):
                metadata["second_per_grid_ts"] = np.asarray(timing).ravel().tolist()
        return metadata

    def shutdown(self) -> None:
        self.preprocessor.shutdown()
