# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Text-only Llama 4, following Transformers' Llama4TextModel.

Reference: transformers/models/llama4/modeling_llama4.py (Apache-2.0).
The conditional-generation entry point loads only the language_model subtree.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.dtype_config import DtypeConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)


def moe_layer_indices(config):
    layers = getattr(config, "moe_layers", None)
    if layers is not None:
        return set(layers)
    step = config.interleave_moe_layer_step
    if step <= 0:
        raise ValueError("interleave_moe_layer_step must be positive")
    return set(range(step - 1, config.num_hidden_layers, step))


def route_tokens(logits, top_k):
    """Select raw logits before FP32 sigmoid; never normalize selected scores."""
    scores, indices = jax.lax.top_k(logits, top_k)
    return jax.nn.sigmoid(scores.astype(jnp.float32)).astype(logits.dtype), indices


class Llama4MoE(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype, dtype_config):
        self.top_k = config.num_experts_per_tok
        if not 1 <= self.top_k <= config.num_local_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_local_experts]")
        if getattr(config, "moe_backend", "epmoe") != "epmoe":
            raise NotImplementedError("Llama 4 currently requires --moe-backend epmoe")
        self.mesh = mesh
        self.router = LinearBase(
            input_size=config.hidden_size,
            output_size=config.num_local_experts,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype_config.get_dtype("router"),
            mesh=mesh,
        )
        self.experts = EPMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_local_experts,
            # Each selected token/expert pair is one independent, input-scaled token.
            num_experts_per_tok=1,
            ep_size=getattr(config, "ep_size", 1),
            mesh=mesh,
            intermediate_dim=config.intermediate_size,
            weight_dtype=dtype_config.get_dtype("experts"),
            dtype=dtype,
            layer_id=layer_id,
            use_sc_permute=False,
        )
        self.shared_expert = LlamaMLP(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            layer_id=layer_id,
            dtype=dtype,
            dtype_config=dtype_config.get_config("shared_expert"),
        )

    def __call__(self, hidden_states):
        logits, _ = self.router(hidden_states)
        scores, indices = route_tokens(logits, self.top_k)
        routed_inputs = (hidden_states[:, None, :] * scores[:, :, None]).reshape(
            -1, hidden_states.shape[-1]
        )
        routed = self.experts(
            routed_inputs,
            jnp.ones((routed_inputs.shape[0], 1), dtype=scores.dtype),
            indices.reshape(-1, 1),
            out_sharding=NamedSharding(self.mesh, P("data", None)),
        )
        routed = routed.reshape(hidden_states.shape[0], self.top_k, -1).sum(axis=1)
        return routed + self.shared_expert(hidden_states)


class Llama4Attention(LlamaAttention):
    def __init__(self, config, mesh, layer_id, dtype, dtype_config):
        rope = config.rope_parameters
        super().__init__(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            mesh,
            layer_id=layer_id,
            head_dim=config.head_dim,
            rope_theta=rope["rope_theta"],
            rope_scaling=rope,
            rope_is_neox_style=False,
            max_position_embeddings=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            dtype=jnp.float32,
            dtype_config=dtype_config,
        )
        self.use_rope = bool(config.no_rope_layers[layer_id])
        self.use_qk_norm = config.use_qk_norm and self.use_rope
        self.eps = config.rms_norm_eps
        self.temperature_tuning = config.attn_temperature_tuning and not self.use_rope
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        self.mesh = mesh
        layer_types = getattr(config, "layer_types", None)
        is_chunked = layer_types[layer_id] == "chunked_attention" if layer_types else self.use_rope
        self.attn.attention_chunk_size = config.attention_chunk_size if is_chunked else None

    def prepare_qkv(self, positions, hidden_states):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        sharding = NamedSharding(self.mesh, P("data", "tensor", None))
        q = q.reshape(-1, self.q_head_num, self.head_dim, out_sharding=sharding)
        k = k.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=sharding)
        v = v.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=sharding)
        if self.use_rope:
            q_dtype, k_dtype = q.dtype, k.dtype
            q, k = self.rotary_emb(positions, q.astype(jnp.float32), k.astype(jnp.float32))
            q, k = q.astype(q_dtype), k.astype(k_dtype)
        if self.use_qk_norm:

            def norm(x):
                xf = x.astype(jnp.float32)
                return (
                    xf * jax.lax.rsqrt(jnp.mean(xf**2, axis=-1, keepdims=True) + self.eps)
                ).astype(x.dtype)

            q, k = norm(q), norm(k)
        if self.temperature_tuning:
            scale = 1 + self.attn_scale * jnp.log1p(
                jnp.floor((positions.astype(jnp.float32) + 1) / self.floor_scale)
            )
            q = (q * scale[:, None, None]).astype(q.dtype)
        return q, k, v

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool):
        q, k, v = self.prepare_qkv(positions, hidden_states)
        backend = forward_batch.attn_backend
        if self.attn.attention_chunk_size is not None and not getattr(
            backend, "supports_attention_chunk_size", False
        ):
            raise NotImplementedError(
                "Llama 4 chunk-local attention requires the flash attention backend"
            )
        output, kv = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(output)
        return output, kv


class Llama4DecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype, dtype_config):
        self.self_attn = Llama4Attention(
            config, mesh, layer_id, dtype, dtype_config.get_config("self_attn")
        )
        ff_config = dtype_config.get_config("feed_forward")
        if layer_id in moe_layer_indices(config):
            self.feed_forward = Llama4MoE(config, mesh, layer_id, dtype, ff_config)
        else:
            self.feed_forward = LlamaMLP(
                config.hidden_size,
                config.intermediate_size_mlp,
                mesh,
                layer_id=layer_id,
                dtype=dtype,
                dtype_config=ff_config,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype_config.get_dtype("input_layernorm"),
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype_config.get_dtype("post_attention_layernorm"),
            dtype=dtype,
        )

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states, kv = self.self_attn(
            positions, self.input_layernorm(hidden_states), forward_batch, token_to_kv_pool
        )
        residual = hidden_states + residual
        hidden_states = self.feed_forward(self.post_attention_layernorm(residual))
        return hidden_states, residual, kv, []


class Llama4Model(LlamaModel):
    def __init__(self, config, mesh, dtype, dtype_config):
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype_config.get_dtype("embed_tokens"),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Llama4DecoderLayer(config, mesh, i, dtype, dtype_config.get_config("layers"))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype_config.get_dtype("norm"),
            dtype=dtype,
        )
        self.layers_to_capture = []


class Llama4ForCausalLM(LlamaForCausalLM):
    @staticmethod
    def patch_model_config(model_config):
        # The official architecture includes vision, but this entry point does not.
        model_config.is_multimodal = False
        model_config.text_only_model = "Llama 4"

    def __init__(self, config, mesh, dtype=jnp.bfloat16, dtype_config=None):
        self.checkpoint_prefix = ""
        self.mesh, self.config, self.dtype = mesh, config, dtype
        if getattr(config, "quantization_config", None) is not None:
            raise NotImplementedError("Llama 4 quantized checkpoints are not supported")
        if getattr(config, "enable_sequence_parallel", False):
            raise NotImplementedError("Llama 4 sequence parallelism is not supported")
        if config.hidden_act != "silu":
            raise ValueError("Llama 4 requires SwiGLU (hidden_act='silu')")
        if config.head_dim % 2:
            raise ValueError("Llama 4 rotary head_dim must be even")
        if len(config.no_rope_layers) != config.num_hidden_layers:
            raise ValueError("no_rope_layers must specify every layer")
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and (
            len(layer_types) != config.num_hidden_layers
            or any(t not in ("full_attention", "chunked_attention") for t in layer_types)
        ):
            raise ValueError("Llama 4 requires full_attention or chunked_attention per layer")
        if config.attention_chunk_size is not None and config.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive or None")
        if config.attn_temperature_tuning and config.floor_scale <= 0:
            raise ValueError("floor_scale must be positive")
        if not moe_layer_indices(config) <= set(range(config.num_hidden_layers)):
            raise ValueError("moe_layers contains an invalid layer index")
        dtype_config = dtype_config or DtypeConfig(default_dtype=dtype)
        self.model = Llama4Model(config, mesh, dtype, dtype_config.get_config("model"))
        if not config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype_config.get_dtype("lm_head"),
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.capture_aux_hidden_states = False

    def _create_layer_mappings(self, layer_idx):
        mappings = super()._create_layer_mappings(layer_idx)
        mappings = {
            key.replace(".mlp.", ".feed_forward."): replace(
                spec, target_path=spec.target_path.replace(".mlp.", ".feed_forward.")
            )
            for key, spec in mappings.items()
        }
        if layer_idx not in moe_layer_indices(self.config):
            return mappings
        prefix = f"model.layers.{layer_idx}.feed_forward"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.{proj}.weight"
            spec = mappings.pop(key)
            mappings[f"{prefix}.shared_expert.{proj}.weight"] = replace(
                spec, target_path=f"{prefix}.shared_expert.{proj}.weight"
            )
        mappings[f"{prefix}.router.weight"] = WeightSpec(
            f"{prefix}.router.weight", sharding=(None, None), transpose=True
        )

        def split_gate_up(inputs):
            value = inputs[0]
            expected = (self.config.hidden_size, 2 * self.config.intermediate_size)
            if value.ndim != 3 or value.shape[1:] != expected:
                raise ValueError(
                    f"Invalid Llama 4 gate/up shape: {value.shape}, expected [E,{expected}]"
                )
            return tuple(np.split(value, 2, axis=2))

        def load_down(inputs):
            value = inputs[0]
            expected = (self.config.intermediate_size, self.config.hidden_size)
            if value.ndim != 3 or value.shape[1:] != expected:
                raise ValueError(
                    f"Invalid Llama 4 down shape: {value.shape}, expected [E,{expected}]"
                )
            return (value,)

        mappings[f"{prefix}.experts.gate_up_proj"] = WeightSpec(
            [f"{prefix}.experts.wi_0", f"{prefix}.experts.wi_1"],
            sharding=("expert", None, "tensor"),
            sources=(f"{prefix}.experts.gate_up_proj",),
            host_recipe=split_gate_up,
            split_axis=2,
        )
        mappings[f"{prefix}.experts.down_proj"] = WeightSpec(
            f"{prefix}.experts.wo",
            sharding=("expert", "tensor", None),
            sources=(f"{prefix}.experts.down_proj",),
            host_recipe=load_down,
        )
        return mappings

    def weight_mappings(self):
        return {
            self.checkpoint_prefix
            + key: replace(
                spec, sources=tuple(self.checkpoint_prefix + source for source in spec.sources)
            )
            for key, spec in self._create_llama_weight_mappings().items()
        }

    def load_weights(self, model_config):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        mappings = self.weight_mappings()
        if not loader.dummy_mode:
            for name, spec in mappings.items():
                if spec.host_recipe is None or name not in loader.metadata:
                    continue
                config = self.config
                expected = (
                    (config.num_local_experts, config.hidden_size, 2 * config.intermediate_size)
                    if name.endswith("gate_up_proj")
                    else (config.num_local_experts, config.intermediate_size, config.hidden_size)
                )
                infos = loader.metadata[name]
                if len(infos) != 1 or tuple(infos[0]["shape"]) != expected:
                    raise ValueError(
                        f"Invalid Llama 4 expert shape for {name}: expected {expected}"
                    )
        loader.load(mappings)


class Llama4ForConditionalGeneration(Llama4ForCausalLM):
    """Official checkpoint namespace, deliberately text-only (no vision modules)."""

    def __init__(self, config, mesh, dtype=jnp.bfloat16, dtype_config=None):
        text_config = config.text_config
        for key in (
            "moe_backend",
            "ep_size",
            "enable_sequence_parallel",
            "enable_dp_lm_head",
            "quantization_config",
        ):
            if hasattr(config, key):
                setattr(text_config, key, getattr(config, key))
        super().__init__(text_config, mesh, dtype, dtype_config)
        self.checkpoint_prefix = "language_model."


EntryClass = [Llama4ForCausalLM, Llama4ForConditionalGeneration]
