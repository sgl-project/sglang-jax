# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Inference-only classic LLaMA EAGLE draft model."""

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import LlamaConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class _Identity(nnx.Module):
    def __call__(self, value):
        return value


class LlamaEagleDecoderLayer(LlamaDecoderLayer):
    """Classic EAGLE decoder layer with no first input normalization."""

    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        super().__init__(
            config=config,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        if layer_id == 0:
            self.input_layernorm = _Identity()


class LlamaEagleModel(nnx.Module):
    """Classic EAGLE feature-level autoregressive draft transformer."""

    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.fc = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=config.hidden_size,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                LlamaEagleDecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_id=layer_id,
                    dtype=dtype,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        if forward_batch.spec_info is None or forward_batch.spec_info.hidden_states is None:
            raise ValueError("EAGLE draft model expects speculative hidden states.")

        embeds = self.embed_tokens(forward_batch.input_ids)
        target_hidden = forward_batch.spec_info.hidden_states
        embed_sharding = jax.typeof(embeds).sharding
        if isinstance(embed_sharding, jax.sharding.NamedSharding):
            target_hidden = jax.sharding.reshard(target_hidden, embed_sharding)
        hidden_states, _ = self.fc(jnp.concatenate((embeds, target_hidden), axis=-1))

        residual = None
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flags = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flags)

        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, layers_kv_fused, layers_callback_flag


class LlamaForCausalLMEagle(LlamaForCausalLM):
    """Classic EAGLE checkpoint wrapper using the target vocabulary directly."""

    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = LlamaEagleModel(config, mesh=mesh, dtype=dtype)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
        self.capture_aux_hidden_states = False
        self.hot_token_ids = None

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_eagle_weight_mappings())
        logger.info("Classic LLaMA EAGLE weights loaded successfully.")

    def _create_eagle_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings = {
            "fc.weight": WeightMapping(
                target_path="model.fc.weight",
                sharding=(None, None),
                transpose=True,
            ),
            "fc.bias": WeightMapping(
                target_path="model.fc.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        for layer_idx in range(self.config.num_hidden_layers):
            layer_mappings = super()._create_layer_mappings(layer_idx)
            for source_path, mapping in layer_mappings.items():
                source_path = source_path.removeprefix("model.")
                if layer_idx == 0 and source_path.endswith("input_layernorm.weight"):
                    continue
                mappings[source_path] = mapping
        return mappings

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def set_embed(self, embed_weight: jax.Array | None = None) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch=forward_batch,
            token_to_kv_pool=memory_pools.token_to_kv_pool,
        )
        output = self.logits_processor(
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states=None,
        )
        return output, {"token_to_kv_pool": layers_kv_fused}, layers_callback_flag, None


EntryClass = LlamaForCausalLMEagle
