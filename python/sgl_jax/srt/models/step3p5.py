"""Inference-only Step 3.5 Flash model skeleton.

Architecture: hybrid sliding/full-attention GQA + sigmoid-gated MoE.
``architectures=["Step3p5ForCausalLM"]``, ``model_type="step3p5"``.

This file registers the class in the sgl-jax model registry via ``EntryClass``.
Decoder layer internals, forward pass, and weight loading are implemented in
follow-up commits; this skeleton is the minimal instantiable stub needed to
wire the registry, config plumbing, and ``patch_model_config``.

Reference: HF modeling_step3p5.py / configuration_step3p5.py
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class Step3p5ForCausalLM(nnx.Module):
    """Step 3.5 Flash causal LM — registry skeleton.

    Only ``embed_tokens``, ``lm_head``, and ``logits_processor`` are
    built here. Decoder layers (``self.model``) are left as ``None``
    until the attention/MoE implementation lands.
    """

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        """Set head_dim=128 on the ModelConfig.

        Step 3.5 Flash always uses 128-dim heads for both full-attention
        and sliding-attention layers. The HF config carries ``head_dim``
        already, but we pin it here so the KV pool and attention backend
        receive the correct value regardless of how the config was loaded.
        """
        mc.head_dim = 128

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        logger.info("Step3p5ForCausalLM dtype=%s", dtype)

        # Embed + untied lm_head (tie_word_embeddings=False in HF config).
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

        # Decoder layers — placeholder until the decoder implementation lands.
        self.model = None  # TODO(step3p5): build Step3p5Model here

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        raise NotImplementedError("TODO(step3p5): forward pass not yet implemented")

    def load_weights(self, model_config: ModelConfig) -> None:
        raise NotImplementedError("TODO(step3p5): weight loading not yet implemented")


EntryClass = [Step3p5ForCausalLM]
