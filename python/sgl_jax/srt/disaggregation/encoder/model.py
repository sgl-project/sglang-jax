from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderBatch, EncoderRequest
from sgl_jax.srt.hf_transformers_utils import get_processor, get_tokenizer_from_processor
from sgl_jax.srt.model_loader import get_model
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs
from sgl_jax.srt.multimodal.in_model.host_orchestration import precompile_multimodal_encoder
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_vision_inputs,
    plan_encoder_lanes,
)
from sgl_jax.srt.multimodal.manager.multimodal_processor import get_mm_processor, import_processors
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
    _TOKEN_ID_KEYS = {
        Modality.IMAGE: "image_token_id",
        Modality.VIDEO: "video_token_id",
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
        import_processors("sgl_jax.srt.multimodal.processors")
        self.mm_processor = get_mm_processor(config, server_args, processor)
        self.tokenizer = get_tokenizer_from_processor(processor)

    async def preprocess_request(self, pending: EncoderRequest) -> None:
        """Preprocess one request without committing it to a ViT batch."""
        request = pending.request
        modality = pending.modality
        mm_items = request.get("mm_items") or []
        if not mm_items:
            raise ValueError("encoder request contains no multimodal items")
        request_obj = SimpleNamespace(
            image_data=mm_items if modality == Modality.IMAGE else None,
            video_data=mm_items if modality == Modality.VIDEO else None,
            audio_data=mm_items if modality == Modality.AUDIO else None,
            fps=request.get("fps"),
            num_frames=request.get("num_frames"),
        )
        inputs = await self.mm_processor.process_encoder_mm_data_async(
            image_data=request_obj.image_data,
            input_text=(
                None if modality == Modality.IMAGE else self._placeholder(modality) * len(mm_items)
            ),
            request_obj=request_obj,
        )
        items = [item for item in inputs.mm_items if item.modality == modality]
        if len(items) != len(mm_items):
            raise ValueError(
                f"processor produced {len(items)} {modality.name} items for {len(mm_items)} inputs"
            )
        inputs.mm_items = items
        pending.inputs = inputs
        pending.token_count = sum(
            end - start for item in items for start, end in item.placeholder_ranges or ()
        )


    def build_batch(
        self,
        requests: list[EncoderRequest],
    ) -> EncoderBatch:
        if not requests:
            raise ValueError("EncoderModelRunner batches must not be empty")
        modality = requests[0].modality
        if any(request.modality != modality for request in requests):
            raise ValueError("EncoderModelRunner batches must contain one modality")

        processed = [request.inputs for request in requests]
        lanes, output_indices = plan_encoder_lanes(
            [item.feature.shape[0] for inputs in processed for item in inputs.mm_items],
            self.num_lanes,
            merge_unit=self.visual.spatial_merge_unit,
        )
        return EncoderBatch(requests, lanes, output_indices)

    @property
    def preprocess_concurrency(self) -> int:
        """Keep the processor workers busy with complete preprocessing requests."""
        return self.mm_processor.mm_processor_worker_num

    def encode(
        self,
        batch: EncoderBatch,
    ) -> jax.Array:
        modality = batch.requests[0].modality
        processed = [request.inputs for request in batch.requests]
        with jax.profiler.TraceAnnotation(f"mm_encode:{modality.name}:{len(processed)}"):
            items = [item for mm_inputs in processed for item in mm_inputs.mm_items]
            target = getattr(self.model, "thinker", self.model)
            get_feature = getattr(target, f"get_{modality.name.lower()}_feature", None)
            if get_feature is None:
                raise ValueError(f"model has no {modality.name} encoder")
            visual = self.visual
            num_lanes = self.num_lanes
            input_sharding = self.input_sharding
            patches, output_indices, grid_thw = pack_vision_inputs(
                [[items[index] for index in lane] for lane in batch.lanes],
                merge_unit=visual.spatial_merge_unit,
                input_sharding=input_sharding,
            )
            capacity = output_indices.size * visual.spatial_merge_unit // num_lanes
            with jax.profiler.TraceAnnotation("encoder_build_vision_metadata"):
                metadata = visual.prepare_metadata(grid_thw, capacity, sharding=input_sharding)
            with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation("encoder_vision_dispatch"):
                embeddings = visual(patches, **metadata)
        if sum(request.token_count for request in batch.requests) > embeddings.shape[0]:
            raise ValueError(f"incomplete {modality.name} encoder output")
        return embeddings

    def metadata_for_batch(
        self,
        batch: EncoderBatch,
    ) -> list[dict[str, Any]]:
        return [self._metadata(request.inputs, request.modality) for request in batch.requests]

    def _placeholder(self, modality: Modality) -> str:
        config = self.mm_processor.hf_config
        token_id = getattr(config, self._TOKEN_ID_KEYS.get(modality, ""), None)
        if token_id is None:
            raise ValueError(f"model has no {modality.name} placeholder token")
        return "".join(
            self.tokenizer.convert_ids_to_tokens(
                [config.vision_start_token_id, token_id, config.vision_end_token_id]
            )
        )

    def _metadata(self, mm_inputs: MultimodalInputs, modality: Modality) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        item_hashes = [item.hash for item in mm_inputs.mm_items]
        if any(value is None for value in item_hashes):
            raise ValueError("encoder inputs are missing media hashes")
        metadata["item_hashes"] = [int(value) for value in item_hashes]
        grid_key = self._GRID_KEYS.get(modality)
        if grid_key is not None:
            metadata["grid_dim"] = np.concatenate(
                [np.asarray(item.get(grid_key)) for item in mm_inputs.mm_items], axis=0
            )
        if modality == Modality.VIDEO:
            timing = [item.get("second_per_grid_ts") for item in mm_inputs.mm_items]
            if all(value is not None for value in timing):
                metadata["second_per_grid_ts"] = timing
        return metadata

    def shutdown(self) -> None:
        self.mm_processor.shutdown()
