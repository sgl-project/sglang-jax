from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig

from sgl_jax.srt.debug_tracer import global_tracer, trace_function
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.utils.weight_utils import load_hf_weights


# class QWenMLP(nnx.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: int,
#         layer_id: int = 0,
#         rngs: nnx.Rngs = None,
#         dtype: jnp.dtype = jnp.bfloat16,
#     ):
#         self.layer_id = layer_id

#         self.w1 = LinearBase(
#             input_size=hidden_size,
#             output_size=intermediate_size,
#             use_bias=False,
#             kernel_axes=(None, "tensor"),
#             params_dtype=dtype,
#             rngs=rngs,
#         )

#         self.w2 = LinearBase(
#             input_size=hidden_size,
#             output_size=intermediate_size,
#             use_bias=False,
#             kernel_axes=(None, "tensor"),
#             params_dtype=dtype,
#             rngs=rngs,
#         )

#         self.c_proj = LinearBase(
#             input_size=intermediate_size,
#             output_size=hidden_size,
#             use_bias=False,
#             kernel_axes=("tensor", None),
#             params_dtype=dtype,
#             rngs=rngs,
#         )

#         self.act_func = jax.nn.silu

#     @trace_function(stage="MLP", include_args=False, include_output=True)
#     def __call__(self, hidden_states: jnp.ndarray):
#         return _mlp_forward(hidden_states, self.w1.weight.value, self.w2.weight.value, self.c_proj.weight.value)


# @jax.jit
# def _mlp_forward(hidden_states: jax.Array, w1: jax.Array, w2: jax.Array, c_proj: jax.Array):
#     a1 = jnp.dot(hidden_states, w1)
#     a2 = jnp.dot(hidden_states, w2)
#     intermediate_parallel = a1 * jax.nn.silu(a2)
#     intermediate_parallel = jax.lax.with_sharding_constraint(
#         intermediate_parallel, PartitionSpec(None, 'tensor'))
#     output = jnp.dot(intermediate_parallel, c_proj)
#     return output


# class QWenAttention(nnx.Module):
#     def __init__(self,
#                  hidden_size: int,
#                  num_heads: int,
#                  max_position_embeddings: int,
#                  rope_theta: float = 10000,
#                  rope_scaling: Optional[Dict[str, Any]] = None,
#                  layer_id: int = 0,
#                  rngs: nnx.Rngs = None):
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         head_size = hidden_size // num_heads
#         self.scaling = head_size**-0.5

#         self.c_attn = LinearBase(
#             input_size=hidden_size,
#             output_size=3 * hidden_size,
#             use_bias=True,
#             kernel_axes=(None, "tensor"),
#             rngs=rngs,
#         )
#         self.c_proj = LinearBase(
#             input_size=num_heads * head_size,
#             output_size=hidden_size,
#             use_bias=False,
#             kernel_axes=("tensor", None),
#             rngs=rngs,
#         )

#         # Use torch version of RotaryEmbedding directly
#         self.rotary_emb = RotaryEmbedding(
#             head_size=head_size,
#             rotary_dim=head_size,
#             max_position_embeddings=max_position_embeddings,
#             base=rope_theta,
#             is_neox_style=True,
#             dtype=jnp.bfloat16,
#         )

#         self.attn = Attention(
#             num_heads=num_heads,
#             scale=head_size**-0.5,
#             rngs=rngs
#         )

#     @trace_function(stage="ATTENTION", include_args=False, include_output=True)
#     def __call__(
#         self,
#         positions: jax.Array,
#         hidden_states: jax.Array,
#         forward_batch: ForwardBatch,
#         layer_id: int,
#     ) -> jax.Array:
#         qkv, _ = self.c_attn(hidden_states)
#         q, k, v = jnp.split(qkv, 3, axis=-1)
#         q, k = self.rotary_emb(positions, q, k)
#         attn_output = self.attn(
#             q, k, v, forward_batch=forward_batch, layer_id=layer_id, is_causal=True)
#         output, _ = self.c_proj(attn_output)
#         return output


# class QWenBlock(nnx.Module):
#     def __init__(self,
#                  config: PretrainedConfig,
#                  layer_id: int = 0,
#                  rngs: nnx.Rngs = None):
#         self.layer_id = layer_id

#         self.ln_1 = RMSNorm(
#             config.hidden_size,
#             epsilon=config.layer_norm_epsilon,
#             rngs=rngs
#         )

#         rope_theta = getattr(config, "rope_theta", 10000)
#         rope_scaling = getattr(config, "rope_scaling", None)
#         self.attn = QWenAttention(
#             config.hidden_size,
#             config.num_attention_heads,
#             config.max_position_embeddings,
#             rope_theta=rope_theta,
#             rope_scaling=rope_scaling,
#             layer_id=layer_id,
#             rngs=rngs,
#         )

#         self.ln_2 = RMSNorm(
#             config.hidden_size,
#             epsilon=config.layer_norm_epsilon,
#             rngs=rngs
#         )

#         self.mlp = QWenMLP(
#             config.hidden_size,
#             config.intermediate_size // 2,
#             layer_id=layer_id,
#             rngs=rngs,
#         )

#     @trace_function(stage="BLOCK", include_args=False, include_output=True)
#     def __call__(
#         self,
#         positions: jax.Array,
#         hidden_states: jax.Array,
#         forward_batch: ForwardBatch,
#     ) -> jax.Array:
#         residual = hidden_states

#         global_tracer.print(
#             hidden_states, f"RMSNorm_pre_attn_input", f"rmsnorm_layer_id_{self.layer_id}")
#         hidden_states = self.ln_1(hidden_states)
#         global_tracer.print(
#             hidden_states, f"RMSNorm_pre_attn_output", f"rmsnorm_layer_id_{self.layer_id}")

#         hidden_states = self.attn(
#             positions=positions,
#             hidden_states=hidden_states,
#             forward_batch=forward_batch,
#             layer_id=self.layer_id,
#         )
#         hidden_states = residual + hidden_states

#         residual = hidden_states

#         global_tracer.print(hidden_states, f"RMSNorm_pre_mlp_input",
#                             f"rmsnorm_layer_id_{self.layer_id}")
#         hidden_states = self.ln_2(hidden_states)
#         global_tracer.print(
#             hidden_states, f"RMSNorm_pre_mlp_output", f"rmsnorm_layer_id_{self.layer_id}")

#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states
#         return hidden_states


# class QWenModel(nnx.Module):
#     """QWen model"""

#     def __init__(self,
#                  config: PretrainedConfig,
#                  rngs: nnx.Rngs = None):
#         vocab_size = ((config.vocab_size + 63) // 64) * 64

#         self.embed_tokens = Embed(
#             num_embeddings=vocab_size,
#             features=config.hidden_size,
#             rngs=rngs,
#         )

#         self.h = [
#             QWenBlock(
#                 config,
#                 layer_id=i,
#                 rngs=rngs,
#             )
#             for i in range(config.num_hidden_layers)
#         ]

#         self.ln_f = RMSNorm(
#             config.hidden_size,
#             epsilon=config.layer_norm_epsilon,
#             rngs=rngs
#         )

#     @trace_function(stage="TRANSFORMER", include_args=False, include_output=True)
#     def __call__(self,
#                  input_ids: jax.Array,
#                  positions: jax.Array,
#                  forward_batch: ForwardBatch,
#                  ):
#         global_tracer.print(input_ids, "embedding_input", "embedding_all")
#         hidden_states = self.embed_tokens(input_ids)
#         global_tracer.print(hidden_states, "embedding_output", "embedding_all")

#         for layer in self.h:
#             hidden_states = layer(positions, hidden_states, forward_batch)

#         global_tracer.print(
#             hidden_states, "RMSNorm_final_input", "rmsnorm_final")
#         hidden_states = self.ln_f(hidden_states)
#         global_tracer.print(
#             hidden_states, "RMSNorm_final_output", "rmsnorm_final")

#         return hidden_states


class QWenLMHeadJaxModel(nnx.Module):
    """QWen language head model"""

    def __init__(self,
                 config: ModelConfig,
                 rngs: nnx.Rngs = None,
                 mesh: jax.sharding.Mesh = None):
        self.mesh = mesh
        self.config = config
        # self.transformer = QWenModel(config, rngs)
        # vocab_size = ((config.vocab_size + 63) // 64) * 64
        # self.lm_head = ParallelLMHead(
        #     vocab_size, config.hidden_size, rngs=rngs)
        # self.logits_processor = LogitsProcessor(vocab_size)
        # self._setup_debug_tracer()

    def _setup_debug_tracer(self):
        try:
            global_tracer.set_model(self)
        except Exception as e:
            print(f"Warning: Could not setup debug tracer: {str(e)}")

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.embed_tokens": ("embed.embedding", ("model", None)),
            "model.layers.*.input_layernorm":
            ("model.layers.*.input_layernorm.scale", (None, )),
            "model.layers.*.mlp.down_proj":
            ("model.layers.*.mlp.down_proj.kernel", ("model", None)),
            "model.layers.*.mlp.gate_proj":
            ("model.layers.*.mlp.gate_proj.kernel", (None, "model")),
            "model.layers.*.mlp.up_proj": ("model.layers.*.mlp.up_proj.kernel",
                                           (None, "model")),
            "model.layers.*.post_attention_layernorm":
            ("model.layers.*.post_attention_layernorm.scale", (None, )),
            "model.layers.*.self_attn.k_proj":
            ("model.layers.*.self_attn.k_proj.kernel", (None, "model", None)),
            "model.layers.*.self_attn.o_proj":
            ("model.layers.*.self_attn.o_proj.kernel", ("model", None, None)),
            "model.layers.*.self_attn.q_proj":
            ("model.layers.*.self_attn.q_proj.kernel", (None, "model", None)),
            "model.layers.*.self_attn.v_proj":
            ("model.layers.*.self_attn.v_proj.kernel", (None, "model", None)),
            "model.layers.*.self_attn.q_proj.bias":
            ("model.layers.*.self_attn.q_proj.bias", ("model", None)),
            "model.layers.*.self_attn.k_proj.bias":
            ("model.layers.*.self_attn.k_proj.bias", ("model", None)),
            "model.layers.*.self_attn.v_proj.bias":
            ("model.layers.*.self_attn.v_proj.bias", ("model", None)),
            "model.norm": ("model.norm.scale", (None, )),
        }
        if not self.config.hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": ("lm_head", (None, "model")),
            })

        load_hf_weights(model_config=self.config,
                model=self,
                mappings=mappings,
                mesh=self.mesh)

    # def __call__(self,
    #              input_ids: jax.Array,
    #              positions: jax.Array,
    #              forward_batch: ForwardBatch,
    #              ):
    #     hidden_states = self.transformer(input_ids, positions, forward_batch)
    #     result = self.logits_processor(
    #         hidden_states, self.lm_head, forward_batch
    #     )

    #     if global_tracer.is_session_active():
    #         input_data = {
    #             "input_ids": input_ids,
    #             "input_shape": list(input_ids.shape)
    #         }

    #         output_data = {
    #             "output_type": str(type(result).__name__)
    #         }

    #         if hasattr(result, 'next_token_logits') and result.next_token_logits is not None:
    #             output_data.update({
    #                 "logits": result.next_token_logits,
    #                 "logits_shape": list(result.next_token_logits.shape)
    #             })

    #         global_tracer.accumulate_step(input_data, output_data)

    #         if global_tracer.should_auto_save():
    #             global_tracer.end_session()

    #     return result


EntryClass = QWenLMHeadJaxModel
