import json
import logging
import os

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
logger = logging.getLogger(__name__)




class MiMoModel(Qwen2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__(config=config, dtype=dtype, rngs=rngs, mesh=mesh)


class MiMoForCausalLM(Qwen2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__(config, dtype, rngs, mesh)
        self.model = MiMoModel(config, dtype=self.dtype, rngs=rngs, mesh=mesh)

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = super()._create_qwen2_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMo weights loaded successfully!")



EntryClass = MiMoForCausalLM
