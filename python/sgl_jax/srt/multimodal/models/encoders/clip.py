# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""Minimal implementation of CLIP intended to be used within a vision language model in JAX.

This implements both CLIPVisionModel and CLIPTextModel using flax.nnx and
supports QKV fusion, feature layer extraction, and precise weight loading.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import CLIPTextConfig, CLIPVisionConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.models.encoders.base import BaseEncoderOutput
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

# =============================================================================
# Utilities
# =============================================================================


def quick_gelu(x):
    """CLIP's default fast GELU approximation."""
    return x * jax.nn.sigmoid(1.702 * x)


ACT_FN = {"quick_gelu": quick_gelu, "gelu": jax.nn.gelu, "relu": jax.nn.relu}


def _create_causal_mask(seq_len: int, dtype: jnp.dtype):
    """Creates a causal mask for CLIP Text Model."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return jnp.where(mask, 0.0, jnp.finfo(dtype).min)


# =============================================================================
# Common Modules (Attention, MLP, EncoderLayer)
# =============================================================================


class CLIPAttention(nnx.Module):
    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # CLIP attention HAS biases.
        self.q_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )

        self.out_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )
        self.dropout = nnx.Dropout(config.attention_dropout)

    def __call__(self, x, attention_mask=None, deterministic=True):
        B, L, _ = x.shape
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)

        # [B, L, H, D] -> [B, H, L, D]
        to_heads = lambda t: jnp.transpose(
            t.reshape(B, L, self.num_heads, self.head_dim), (0, 2, 1, 3)
        )
        q_h, k_h, v_h = to_heads(q), to_heads(k), to_heads(v)

        scores = (
            jnp.matmul(
                q_h.astype(jnp.float32), jnp.transpose(k_h.astype(jnp.float32), (0, 1, 3, 2))
            )
            * self.scale
        )

        is_causal = attention_mask is None

        if is_causal:
            # SGLang: is_causal=True, attn_mask=None
            scores = scores + _create_causal_mask(L, scores.dtype)
        else:
            # SGLang: is_causal=False, attn_mask=attention_mask
            # Format Padding Mask (1 -> 0.0, 0 -> -inf) before attn calc
            if attention_mask.ndim == 2:
                # [B, S] -> [B, 1, 1, S]
                attn_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            else:
                attn_mask = attention_mask

            formatted_mask = jnp.where(attn_mask > 0, 0.0, jnp.finfo(scores.dtype).min)
            scores = scores + formatted_mask

        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, deterministic=deterministic)

        out = jnp.matmul(weights, v_h.astype(jnp.float32))
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, L, self.embed_dim)
        out, _ = self.out_proj(out.astype(x.dtype))

        return out


class CLIPMLP(nnx.Module):
    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.act_fn = ACT_FN.get(config.hidden_act, quick_gelu)
        self.fc1 = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.fc2 = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )

    def __call__(self, x):
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class CLIPEncoderLayer(nnx.Module):
    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.self_attn = CLIPAttention(config, mesh, dtype)
        self.layer_norm1 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        self.mlp = CLIPMLP(config, mesh, dtype)
        self.layer_norm2 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x, attention_mask=None, deterministic=True):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x, attention_mask, deterministic)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class CLIPEncoder(nnx.Module):
    def __init__(self, config, mesh, num_hidden_layers_override=None, dtype=jnp.bfloat16):
        num_layers = (
            num_hidden_layers_override
            if num_hidden_layers_override is not None
            else config.num_hidden_layers
        )
        self.layers = nnx.List([CLIPEncoderLayer(config, mesh, dtype) for _ in range(num_layers)])

    def __call__(self, x, attention_mask=None, return_all_hidden_states=False, deterministic=True):
        hidden_states_pool = [x]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, deterministic=deterministic)
            if return_all_hidden_states:
                hidden_states_pool.append(x)

        if return_all_hidden_states:
            return hidden_states_pool
        return [x]


def _layer_mappings(config, idx, src_prefix, tgt_prefix):
    """Generate weight mappings for an encoder layer."""
    s, t = f"{src_prefix}.{idx}", f"{tgt_prefix}.{idx}"
    return {
        f"{s}.layer_norm1.weight": WeightMapping(f"{t}.layer_norm1.scale", (None,), False),
        f"{s}.layer_norm1.bias": WeightMapping(f"{t}.layer_norm1.bias", (None,), False),
        f"{s}.layer_norm2.weight": WeightMapping(f"{t}.layer_norm2.scale", (None,), False),
        f"{s}.layer_norm2.bias": WeightMapping(f"{t}.layer_norm2.bias", (None,), False),
        f"{s}.self_attn.q_proj.weight": WeightMapping(
            f"{t}.self_attn.q_proj.weight", (None, "tensor"), True
        ),
        f"{s}.self_attn.q_proj.bias": WeightMapping(f"{t}.self_attn.q_proj.bias", (None,), False),
        f"{s}.self_attn.k_proj.weight": WeightMapping(
            f"{t}.self_attn.k_proj.weight", (None, "tensor"), True
        ),
        f"{s}.self_attn.k_proj.bias": WeightMapping(f"{t}.self_attn.k_proj.bias", (None,), False),
        f"{s}.self_attn.v_proj.weight": WeightMapping(
            f"{t}.self_attn.v_proj.weight", (None, "tensor"), True
        ),
        f"{s}.self_attn.v_proj.bias": WeightMapping(f"{t}.self_attn.v_proj.bias", (None,), False),
        f"{s}.self_attn.out_proj.weight": WeightMapping(
            f"{t}.self_attn.out_proj.weight", ("tensor", None), False
        ),
        f"{s}.self_attn.out_proj.bias": WeightMapping(
            f"{t}.self_attn.out_proj.bias", (None,), True
        ),
        f"{s}.mlp.fc1.weight": WeightMapping(f"{t}.mlp.fc1.weight", (None, "tensor"), True),
        f"{s}.mlp.fc1.bias": WeightMapping(f"{t}.mlp.fc1.bias", (None,), False),
        f"{s}.mlp.fc2.weight": WeightMapping(f"{t}.mlp.fc2.weight", ("tensor", None), True),
        f"{s}.mlp.fc2.bias": WeightMapping(f"{t}.mlp.fc2.bias", (None,), False),
    }


# =============================================================================
# Vision Model specific Modules
# =============================================================================


class CLIPVisionEmbeddings(nnx.Module):
    def __init__(self, config: CLIPVisionConfig, mesh, dtype=jnp.bfloat16):
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.class_embedding = nnx.Param(jnp.zeros((self.embed_dim,), dtype=dtype))

        # Note: input is NCHW in PT, JAX Conv takes NHWC. We will transpose in __call__
        self.patch_embedding = nnx.Conv(
            in_features=config.num_channels,
            out_features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )

        num_patches = (config.image_size // self.patch_size) ** 2
        self.num_positions = num_patches + 1
        self.position_embedding = Embed(
            self.num_positions,
            self.embed_dim,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )

    def __call__(self, pixel_values: jax.Array):
        # pixel_values from standard PT pipelines are [B, C, H, W]
        # JAX Conv expects [B, H, W, C]
        x = jnp.transpose(pixel_values, (0, 2, 3, 1))

        patch_embeds = self.patch_embedding(x)
        B = patch_embeds.shape[0]
        patch_embeds = patch_embeds.reshape(B, -1, self.embed_dim)

        class_embeds = jnp.broadcast_to(self.class_embedding, (B, 1, self.embed_dim))
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding.embedding
        return embeddings


class CLIPVisionTransformer(nnx.Module):
    def __init__(self, config: CLIPVisionConfig, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.embeddings = CLIPVisionEmbeddings(config, mesh, dtype)
        self.pre_layrnorm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        self.encoder = CLIPEncoder(
            config,
            mesh,
            num_hidden_layers_override=getattr(config, "num_hidden_layers_override", None),
            dtype=dtype,
        )

        num_layers = config.num_hidden_layers
        require_post_norm = getattr(
            config, "require_post_norm", len(self.encoder.layers) == num_layers
        )
        self.post_layernorm = (
            nnx.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_eps,
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                rngs=nnx.Rngs(0),
            )
            if require_post_norm
            else None
        )

    def __call__(
        self,
        pixel_values: jax.Array,
        feature_sample_layers: list[int] | None = None,
        output_hidden_states: bool | None = None,
        deterministic=True,
    ) -> BaseEncoderOutput:
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)

        return_all = output_hidden_states or (feature_sample_layers is not None)
        encoder_outputs = self.encoder(
            x, return_all_hidden_states=return_all, deterministic=deterministic
        )

        if not return_all:
            out = encoder_outputs[0]
            if self.post_layernorm is not None:
                out = self.post_layernorm(out)
            return BaseEncoderOutput(last_hidden_state=out)

        if feature_sample_layers is not None:
            selected_outputs = []
            for idx in feature_sample_layers:
                layer_out = encoder_outputs[idx]
                if self.post_layernorm is not None:
                    layer_out = self.post_layernorm(layer_out)
                selected_outputs.append(layer_out)
            return BaseEncoderOutput(
                last_hidden_state=selected_outputs[-1], hidden_states=selected_outputs
            )

        last_hidden_state = encoder_outputs[-1]
        if self.post_layernorm is not None:
            last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=encoder_outputs)


class CLIPVisionModel(nnx.Module):
    def __init__(self, config: CLIPVisionConfig, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.vision_model = CLIPVisionTransformer(config, mesh, dtype)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

        # =====================================================================
        # POST-LOAD FIX: PyTorch to JAX Conv2D Kernel Transposition
        # PyTorch Conv2d weight shape: (Out, In, H, W)
        # JAX/Flax nnx.Conv weight shape: (H, W, In, Out)
        # =====================================================================
        kernel = self.vision_model.embeddings.patch_embedding.kernel.value
        if kernel.ndim == 4 and kernel.shape[0] == self.config.hidden_size:
            transposed_kernel = jnp.transpose(kernel, (2, 3, 1, 0))
            self.vision_model.embeddings.patch_embedding.kernel.value = transposed_kernel

    def _weight_mappings(self):
        # We explicitly map HF's vision_model.* to our self.vision_model.* structure
        m = {
            "vision_model.embeddings.class_embedding": WeightMapping(
                "vision_model.embeddings.class_embedding", (None,), False
            ),
            "vision_model.embeddings.patch_embedding.weight": WeightMapping(
                "vision_model.embeddings.patch_embedding.kernel", (None,), False
            ),
            "vision_model.embeddings.position_embedding.weight": WeightMapping(
                "vision_model.embeddings.position_embedding.embedding", ("tensor", None), False
            ),
            "vision_model.pre_layrnorm.weight": WeightMapping(
                "vision_model.pre_layrnorm.scale", (None,), False
            ),
            "vision_model.pre_layrnorm.bias": WeightMapping(
                "vision_model.pre_layrnorm.bias", (None,), False
            ),
        }

        layer_count = len(self.vision_model.encoder.layers)
        for i in range(layer_count):
            m.update(
                _layer_mappings(
                    self.config, i, "vision_model.encoder.layers", "vision_model.encoder.layers"
                )
            )

        if self.vision_model.post_layernorm is not None:
            m["vision_model.post_layernorm.weight"] = WeightMapping(
                "vision_model.post_layernorm.scale", (None,), False
            )
            m["vision_model.post_layernorm.bias"] = WeightMapping(
                "vision_model.post_layernorm.bias", (None,), False
            )

        return m

    def __call__(
        self,
        pixel_values: jax.Array,
        feature_sample_layers: list[int] | None = None,
        output_hidden_states: bool | None = None,
        deterministic=True,
    ) -> BaseEncoderOutput:
        return self.vision_model(
            pixel_values, feature_sample_layers, output_hidden_states, deterministic
        )


# =============================================================================
# Text Model specific Modules
# =============================================================================


class CLIPTextEmbeddings(nnx.Module):
    def __init__(self, config: CLIPTextConfig, mesh, dtype=jnp.bfloat16):
        self.token_embedding = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.position_embedding = Embed(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )

    def __call__(self, input_ids: jax.Array, position_ids: jax.Array | None = None):
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype=jnp.int32).reshape(1, -1)

        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        return inputs_embeds + position_embeddings


class CLIPTextTransformer(nnx.Module):
    def __init__(self, config: CLIPTextConfig, mesh, dtype=jnp.bfloat16):
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config, mesh, dtype)
        self.encoder = CLIPEncoder(config, mesh, dtype=dtype)
        self.final_layer_norm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        self.eos_token_id = getattr(config, "eos_token_id", 2)

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        deterministic=True,
    ) -> BaseEncoderOutput:
        hidden_states = self.embeddings(input_ids, position_ids)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            return_all_hidden_states=output_hidden_states,
            deterministic=deterministic,
        )

        last_hidden_state = encoder_outputs[-1] if output_hidden_states else encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Compute `pooler_output` using EOS token representation (Aligned with SGLang / HF logic)
        B = last_hidden_state.shape[0]
        if self.eos_token_id == 2:
            idx = jnp.argmax(input_ids, axis=-1)
        else:
            idx = jnp.argmax((input_ids == self.eos_token_id).astype(jnp.int32), axis=-1)

        batch_indices = jnp.arange(B)
        pooled_output = last_hidden_state[batch_indices, idx]

        return BaseEncoderOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs if output_hidden_states else None,
            attention_mask=attention_mask,
        )


class CLIPTextModel(nnx.Module):
    def __init__(self, config: CLIPTextConfig, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.text_model = CLIPTextTransformer(config, mesh, dtype)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "text_model.embeddings.token_embedding.weight": WeightMapping(
                "text_model.embeddings.token_embedding.embedding", ("tensor", None), False
            ),
            "text_model.embeddings.position_embedding.weight": WeightMapping(
                "text_model.embeddings.position_embedding.embedding", (None, "tensor"), False
            ),
            "text_model.final_layer_norm.weight": WeightMapping(
                "text_model.final_layer_norm.scale", (None,), False
            ),
            "text_model.final_layer_norm.bias": WeightMapping(
                "text_model.final_layer_norm.bias", (None,), False
            ),
        }
        for i in range(self.config.num_hidden_layers):
            m.update(
                _layer_mappings(
                    self.config, i, "text_model.encoder.layers", "text_model.encoder.layers"
                )
            )
        return m

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        deterministic=True,
    ) -> BaseEncoderOutput:
        return self.text_model(
            input_ids, position_ids, attention_mask, output_hidden_states, deterministic
        )


EntryClass = [CLIPTextModel, CLIPVisionModel]
