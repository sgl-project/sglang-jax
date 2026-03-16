# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""T5 & UMT5 Encoder model for Multimodal implementations.

This implementation provides a clean, dense-batch forward pass for the T5
encoder, optimized for extracting text embeddings in multimodal pipelines
(e.g., SD3, Flux).
"""

import glob
import math
import os

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import T5Config

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.models.encoders.base import BaseEncoderOutput
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

# =============================================================================
# Utilities
# =============================================================================


def fp16_clamp(x):
    if x.dtype == jnp.float16 and jnp.isinf(x).any():
        clamp = jnp.finfo(x.dtype).max - 1000
        return jax.lax.clamp(x=x, min=-clamp, max=clamp)
    return x


def gelu_new(x):
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


ACT_FN = {"gelu": jax.nn.gelu, "gelu_new": gelu_new, "relu": jax.nn.relu}


def relative_position_bucket(rel_pos, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -rel_pos
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).astype(jnp.int32) * num_buckets
        n = jnp.abs(n)
    else:
        n = jnp.maximum(n, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_large = jnp.minimum(val_large, num_buckets - 1)
    return jnp.where(is_small, n, val_large) + ret


# =============================================================================
# FFN Modules
# =============================================================================


class T5FFN(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16):
        mk_linear = lambda in_sz, out_sz, axes: LinearBase(
            in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype
        )

        ff_proj = getattr(config, "feed_forward_proj", "relu")
        self.is_gated = getattr(config, "is_gated_act", ff_proj.startswith("gated"))
        act_str = getattr(
            config,
            "dense_act_fn",
            "gelu_new" if ff_proj == "gated-gelu" else ff_proj.split("-")[-1],
        )

        if self.is_gated:
            self.wi_0 = mk_linear(config.d_model, config.d_ff, (None, "tensor"))
            self.wi_1 = mk_linear(config.d_model, config.d_ff, (None, "tensor"))
        else:
            self.wi = mk_linear(config.d_model, config.d_ff, (None, "tensor"))

        self.wo = mk_linear(config.d_ff, config.d_model, ("tensor", None))
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(act_str, jax.nn.relu)

    def __call__(self, x, deterministic=True):
        dtype = x.dtype
        if self.is_gated:
            h0, _ = self.wi_0(x)
            h1, _ = self.wi_1(x)
            h = self.dropout((self.act(h0) * h1).astype(dtype), deterministic=deterministic)
        else:
            h, _ = self.wi(x)
            h = self.dropout(self.act(h).astype(dtype), deterministic=deterministic)
        out, _ = self.wo(h)
        return out


# =============================================================================
# Attention Module
# =============================================================================


class T5Attention(nnx.Module):
    def __init__(
        self, config: T5Config, mesh, dtype=jnp.bfloat16, has_relative_attention_bias=False
    ):
        self.mesh = mesh
        self.d_model = config.d_model
        self.d_kv = getattr(config, "d_kv", getattr(config, "head_dim", 64))
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        mk_linear = lambda in_sz, out_sz, axes: LinearBase(
            in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype
        )
        self.q = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.k = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.v = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.o = mk_linear(self.inner_dim, self.d_model, ("tensor", None))

        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.num_buckets = config.relative_attention_num_buckets
            self.max_distance = getattr(config, "relative_attention_max_distance", 128)
            self.rel_bias = Embed(
                self.num_buckets,
                self.n_heads,
                dtype=dtype,
                param_dtype=dtype,
                mesh=mesh,
                kernel_axes=(None, "tensor"),
            )

        self.dropout = nnx.Dropout(config.dropout_rate)

    def _compute_position_bias(self, seq_len):
        q_pos = jnp.arange(seq_len)
        k_pos = jnp.arange(seq_len)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        buckets = relative_position_bucket(
            rel_pos,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        return jnp.transpose(self.rel_bias(buckets), (2, 0, 1))

    def __call__(self, x, attention_mask=None, position_bias=None, deterministic=True):
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype

        q, _ = self.q(x)
        k, _ = self.k(x)
        v, _ = self.v(x)

        to_heads = lambda t: jnp.transpose(
            t.reshape(batch_size, seq_len, self.n_heads, self.d_kv), (0, 2, 1, 3)
        )
        q_h, k_h, v_h = to_heads(q), to_heads(k), to_heads(v)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q_h.astype(jnp.float32), k_h.astype(jnp.float32))

        # Handle relative position bias
        if self.has_relative_attention_bias:
            position_bias = self._compute_position_bias(seq_len)

        if position_bias is not None:
            scores = scores + position_bias[None, ...]

        # Apply standard causal/padding mask (e.g. from tokenizer)
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_h.astype(jnp.float32))

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.inner_dim)
        out = self.dropout(out.astype(dtype), deterministic=deterministic)
        out, _ = self.o(out)

        return out, position_bias


# =============================================================================
# Transformer Block & Stack
# =============================================================================


class T5Block(nnx.Module):
    def __init__(
        self, config: T5Config, mesh, dtype=jnp.bfloat16, has_relative_attention_bias=False
    ):
        self.ln1 = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.self_attn = T5Attention(config, mesh, dtype, has_relative_attention_bias)
        self.drop1 = nnx.Dropout(config.dropout_rate)

        self.ln2 = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.mlp = T5FFN(config, mesh, dtype)
        self.drop2 = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, attention_mask=None, position_bias=None, deterministic=True):
        # Self Attention
        h, new_position_bias = self.self_attn(
            self.ln1(x), attention_mask, position_bias, deterministic=deterministic
        )
        x = fp16_clamp(x + self.drop1(h, deterministic=deterministic))

        # FFN
        h = self.mlp(self.ln2(x), deterministic=deterministic)
        x = fp16_clamp(x + self.drop2(h, deterministic=deterministic))

        return x, new_position_bias


class T5Stack(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, is_umt5=False):
        self.blocks = nnx.List(
            [
                T5Block(
                    config, mesh, dtype, has_relative_attention_bias=True if is_umt5 else (i == 0)
                )
                for i in range(config.num_layers)
            ]
        )
        self.final_ln = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, attention_mask=None, deterministic=True):
        x = self.dropout(x, deterministic=deterministic)
        position_bias = None
        for block in self.blocks:
            x, pb = block(x, attention_mask, position_bias, deterministic)
            if (
                position_bias is None
            ):  # In standard T5, only the first layer returns PB to be shared.
                position_bias = pb

        return self.dropout(fp16_clamp(self.final_ln(x)), deterministic=deterministic)


# =============================================================================
# Weight Mapping Helper
# =============================================================================


def _block_mappings(config, idx, src_prefix, tgt_prefix, has_rel_bias=False):
    """Simplified block mapping logic for pure Encoder blocks."""
    s, t = f"{src_prefix}.{idx}", f"{tgt_prefix}.{idx}"
    m = {
        f"{s}.layer.0.layer_norm.weight": WeightMapping(f"{t}.ln1.scale", (None,), False),
        f"{s}.layer.0.SelfAttention.q.weight": WeightMapping(
            f"{t}.self_attn.q.weight", (None, "tensor"), True
        ),
        f"{s}.layer.0.SelfAttention.k.weight": WeightMapping(
            f"{t}.self_attn.k.weight", (None, "tensor"), True
        ),
        f"{s}.layer.0.SelfAttention.v.weight": WeightMapping(
            f"{t}.self_attn.v.weight", (None, "tensor"), True
        ),
        f"{s}.layer.0.SelfAttention.o.weight": WeightMapping(
            f"{t}.self_attn.o.weight", ("tensor", None), True
        ),
    }

    if has_rel_bias:
        m[f"{s}.layer.0.SelfAttention.relative_attention_bias.weight"] = WeightMapping(
            f"{t}.self_attn.rel_bias.embedding", (None, "tensor"), False
        )

    # In a T5 Encoder block, the FFN is layer.1 (unlike Decoder where it's layer.2)
    ffn_idx = 1

    m[f"{s}.layer.{ffn_idx}.layer_norm.weight"] = WeightMapping(f"{t}.ln2.scale", (None,), False)

    ff_proj = getattr(config, "feed_forward_proj", "relu")
    is_gated = getattr(config, "is_gated_act", ff_proj.startswith("gated"))
    prefix_dense = "DenseReluDense"

    if is_gated:
        m.update(
            {
                f"{s}.layer.{ffn_idx}.{prefix_dense}.wi_0.weight": WeightMapping(
                    f"{t}.mlp.wi_0.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.{prefix_dense}.wi_1.weight": WeightMapping(
                    f"{t}.mlp.wi_1.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.{prefix_dense}.wo.weight": WeightMapping(
                    f"{t}.mlp.wo.weight", ("tensor", None), True
                ),
            }
        )
    else:
        m.update(
            {
                f"{s}.layer.{ffn_idx}.{prefix_dense}.wi.weight": WeightMapping(
                    f"{t}.mlp.wi.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.{prefix_dense}.wo.weight": WeightMapping(
                    f"{t}.mlp.wo.weight", ("tensor", None), True
                ),
            }
        )
    return m


# =============================================================================
# Model Classes
# =============================================================================


class T5EncoderModel(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, is_umt5=False):
        self.config, self.mesh, self.dtype, self.is_umt5 = config, mesh, dtype, is_umt5
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.encoder = T5Stack(config, mesh, dtype, is_umt5=is_umt5)

    def load_weights(self, model_config: ModelConfig):
        path = model_config.model_path
        if (
            os.path.isabs(path)
            and os.path.exists(path)
            and not glob.glob(os.path.join(path, "*.safetensors"))
            and os.path.exists(os.path.join(path, "text_encoder"))
        ):
            model_config.model_path = os.path.join(path, "text_encoder")
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "shared.weight": WeightMapping("shared.embedding", ("tensor", None), False),
            "encoder.final_layer_norm.weight": WeightMapping(
                "encoder.final_ln.scale", (None,), False
            ),
        }
        for i in range(self.config.num_layers):
            m.update(
                _block_mappings(
                    self.config,
                    i,
                    "encoder.block",
                    "encoder.blocks",
                    has_rel_bias=True if self.is_umt5 else (i == 0),
                )
            )
        return m

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        deterministic: bool = True,
    ) -> BaseEncoderOutput:

        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(1, -1)

        # Prepare attention_mask for dense batch sequence (HuggingFace style broadcast)
        extended_attention_mask = None
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * jnp.finfo(
                jnp.float32
            ).min

        x = self.shared(input_ids)
        hidden_states = self.encoder(x, extended_attention_mask, deterministic=deterministic)

        return BaseEncoderOutput(last_hidden_state=hidden_states)


class UMT5EncoderModel(T5EncoderModel):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16):
        super().__init__(config, mesh, dtype, is_umt5=True)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        deterministic: bool = True,
    ) -> BaseEncoderOutput:

        base_output = super().__call__(
            input_ids, attention_mask, output_hidden_states, deterministic
        )

        return BaseEncoderOutput(
            last_hidden_state=base_output.last_hidden_state,
            attention_mask=attention_mask,
        )


EntryClass = [T5EncoderModel, UMT5EncoderModel]
