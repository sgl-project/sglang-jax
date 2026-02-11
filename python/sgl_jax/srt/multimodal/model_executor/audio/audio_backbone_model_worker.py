"""Audio Backbone Model Worker for MiMo Audio."""

from typing import Optional

import jax

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs


class AudioBackboneModelWorker:
    """Simplified worker that forwards calls to the model runner.

    All batch preparation logic has been moved to AudioBackboneScheduler.
    This follows the pattern of other model workers (VitModelWorker, AudioModelWorker).
    """

    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)

    def forward(
        self,
        input_ids: jax.Array,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
        **kwargs
    ):
        """Forward pass through the backbone model."""
        return self.model_runner.forward(input_ids, forward_batch, logits_metadata, **kwargs)

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Decode audio patches from local embeddings."""
        return self.model_runner.patch_decode(local_embeds, key, sampler_config)
