"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import LlamaConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""


logger = logging.getLogger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        super().__init__(config, layer_id, dtype=dtype, mesh=mesh)

        # override qkv
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        self.self_attn.q_proj = LinearBase(
            input_size=2 * self.hidden_size,
            output_size=self.self_attn.q_head_num * self.self_attn.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.self_attn.k_proj = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=self.self_attn.kv_head_num * self.self_attn.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.self_attn.v_proj = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=config.num_key_value_heads * self.self_attn.head_dim,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        if config.model_type == "llama4_text":
            # inter_size = config.intermediate_size_mlp
            pass
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=inter_size,
            dtype=dtype,
            mesh=mesh,
        )

        self.hidden_norm = RMSNorm(num_features=config.hidden_size, epsilon=config.rms_norm_eps)

    def __call__(
        self,
        positions: jax.Array,
        embeds: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = jnp.concatenate([embeds, hidden_states], axis=-1, dtype=jnp.bfloat16)
        # Self Attention
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        if residual is not None:
            hidden_states = residual + hidden_states
            residual = hidden_states
        hidden_states = self.post_attention_layernorm(x=hidden_states, mask=None)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, kv_fused


class LlamaEagleModel(LlamaModel):
    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        super().__init__(config=config, dtype=dtype, is_draft_model=True, mesh=mesh)
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = nnx.data(
            LinearBase(
                self.hidden_size_in * 3,
                config.hidden_size,
                use_bias=getattr(config, "bias", False),
                params_dtype=dtype,
                kernel_axes=(None, None),
                mesh=mesh,
            )
        )

        self.midlayer = LlamaDecoderLayer(config=config, layer_id=0, dtype=dtype, mesh=mesh)

        self.norm = RMSNorm(num_features=config.hidden_size, epsilon=config.rms_norm_eps)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array], list[jax.Array], list]:
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions
        if positions is None:
            positions = jnp.arange(input_ids.shape[0], dtype=jnp.int32)

        embeds = self.embed_tokens(input_ids)

        if self.is_mrope_enabled:
            # positions = forward_batch.mrope_positions
            pass

        if forward_batch.spec_info is None or forward_batch.spec_info.hidden_states is None:
            raise ValueError("EAGLE3 draft model expects speculative hidden states.")
        hidden_states = forward_batch.spec_info.hidden_states

        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)[0]

        residual = None
        hidden_states, residual, kv_fused = self.midlayer(
            positions=positions,
            embeds=embeds,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            residual=residual,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states.copy()
        hidden_states_to_logits = self.norm(hidden_states)

        layers_kv_fused = [kv_fused] if kv_fused is not None else []
        layers_callback_flag: list = []

        return hidden_states_to_logits, [residual], layers_kv_fused, layers_callback_flag


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.mesh = mesh
        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")
        self.dtype = dtype
        self.model = LlamaEagleModel(config, dtype=dtype, mesh=mesh)
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                dtype=dtype,
            )
        self.logits_processor = LogitsProcessor(vocab_size=config.vocab_size, mesh=self.mesh)
        self.capture_aux_hidden_states = True
        self.hot_token_ids = nnx.Param(jnp.arange(config.draft_vocab_size))

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_llama_ealge3_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("llama EAGLE3 weights loaded successfully!")

    def _create_llama_ealge3_weight_mappings(self):
        # mappings = super()._create_llama_weight_mappings()
        mappings = {}
        mappings["d2t"] = WeightMapping(
            target_path="hot_token_ids",
            sharding=(None,),
            transpose=False,
        )
        mappings["fc.weight"] = WeightMapping(
            target_path="model.fc.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings["lm_head.weight"] = WeightMapping(
            target_path="lm_head.embedding",
            sharding=(None, None),
            transpose=False,
        )
        mappings["norm.weight"] = WeightMapping(
            target_path="model.norm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["midlayer.hidden_norm.weight"] = WeightMapping(
            target_path="model.midlayer.hidden_norm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["midlayer.input_layernorm.weight"] = WeightMapping(
            target_path="model.midlayer.input_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["midlayer.mlp.down_proj.weight"] = WeightMapping(
            target_path="model.midlayer.mlp.down_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings["midlayer.mlp.gate_proj.weight"] = WeightMapping(
            target_path="model.midlayer.mlp.gate_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings["midlayer.mlp.up_proj.weight"] = WeightMapping(
            target_path="model.midlayer.mlp.up_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings["midlayer.post_attention_layernorm.weight"] = WeightMapping(
            target_path="model.midlayer.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["midlayer.self_attn.q_proj.weight"] = WeightMapping(
            target_path="model.midlayer.self_attn.q_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=False,
            is_eagle3=True,
        )
        mappings["midlayer.self_attn.k_proj.weight"] = WeightMapping(
            target_path="model.midlayer.self_attn.k_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=True,
            is_eagle3=True,
        )
        mappings["midlayer.self_attn.v_proj.weight"] = WeightMapping(
            target_path="model.midlayer.self_attn.v_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=True,
            is_eagle3=True,
        )
        mappings["midlayer.self_attn.o_proj.weight"] = WeightMapping(
            target_path="model.midlayer.self_attn.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            head_dim_padding=True,
            kv_head_padding=False,
        )
        if getattr(self.config, "bias", False):
            mappings["model.fc.bias"] = WeightMapping(
                target_path="model.fc.value.bias",
                sharding=(None,),
            )

        return mappings

    def get_hot_token_id(self):
        return self.hot_token_ids


EntryClass = [LlamaForCausalLMEagle3]
