"""Audio Backbone Model Worker for MiMo Audio."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Group size for MiMo Audio backbone
MIMO_GROUP_SIZE = 4


class AudioBackboneModelWorker:
    """Worker for MiMo Audio Backbone model execution."""

    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)

    def forward(
        self,
        batch: Req,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer.

        The input_ids should already be in the correct format [B, 9, seq_len]
        after aggregation in Req._build_backbone_input().

        Args:
            batch: Request batch containing pre-aggregated input_ids
            cache: Optional KV cache

        Returns:
            (text_logits, local_hidden_states, cache), cache_miss_count
        """
        input_ids = batch.input_ids

        if input_ids is None:
            raise ValueError("input_ids must be provided (should be pre-aggregated by Req)")

        logger.info(
            "AudioBackboneModelWorker.forward: input_ids shape=%s, dtype=%s",
            input_ids.shape if input_ids is not None else None,
            input_ids.dtype if input_ids is not None else None,
        )

        # Ensure correct dtype
        if not jnp.issubdtype(input_ids.dtype, jnp.integer):
            input_ids = input_ids.astype(jnp.int32)

        # Ensure batch dimension
        if input_ids.ndim == 2:
            # [9, seq_len] -> [1, 9, seq_len]
            input_ids = input_ids[None, :, :]

        # Ensure seq_len is divisible by group_size
        seq_len = input_ids.shape[2]
        if seq_len % MIMO_GROUP_SIZE != 0:
            pad_len = MIMO_GROUP_SIZE - (seq_len % MIMO_GROUP_SIZE)
            input_ids = jnp.pad(
                input_ids,
                ((0, 0), (0, 0), (0, pad_len)),
                constant_values=0,
            )
            logger.info(
                "Padded input_ids from seq_len=%d to %d",
                seq_len,
                input_ids.shape[2],
            )

        return self.model_runner.forward(input_ids, cache, **kwargs)

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Generate audio tokens for one group.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels], cache_miss_count
        """
        return self.model_runner.patch_decode(local_embeds, key, sampler_config)

    def init_cache(self, batch_size: int) -> list:
        """Initialize KV cache for main transformer."""
        return self.model_runner.init_cache(batch_size)
