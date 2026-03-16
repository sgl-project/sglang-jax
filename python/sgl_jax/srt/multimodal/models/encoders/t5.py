# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""T5 & UMT5 model using RadixAttention for efficient attention computation.

This implementation uses RadixAttention for decoder self-attention and native
attention for encoder and cross-attention, providing optimal balance between
performance and compatibility with T5's position bias mechanism.
"""

import copy
import math
import os
import glob
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import T5Config

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
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


def update_decoder_seq_lens(forward_batch: ForwardBatch, dec_ids):
    if hasattr(forward_batch, "decoder_seq_lens"):
        forward_batch.seq_lens = forward_batch.decoder_seq_lens
        forward_batch.extend_seq_lens = forward_batch.decoder_seq_lens
    else:
        dec_len = len(dec_ids)
        forward_batch.seq_lens = jnp.array([dec_len], dtype=jnp.int32)
        forward_batch.extend_seq_alens = jnp.array([dec_len], dtype=jnp.int32)
    forward_batch.positions = jnp.arange(len(dec_ids), dtype=jnp.int32)


ACT_FN = {"gelu": jax.nn.gelu, "gelu_new": gelu_new, "relu": jax.nn.relu}


def _apply_block_diagonal_mask(scores, q_lens, k_lens, is_causal=False):
    _, q_len, k_len = scores.shape
    q_total, k_total = jnp.sum(q_lens), jnp.sum(k_lens)

    q_valid = jnp.arange(q_len) < q_total
    k_valid = jnp.arange(k_len) < k_total

    q_starts = jnp.cumsum(q_lens, dtype=jnp.int32) - q_lens
    k_starts = jnp.cumsum(k_lens, dtype=jnp.int32) - k_lens
    q_indicators = jnp.zeros(q_len, dtype=jnp.int32).at[q_starts].set(1)
    k_indicators = jnp.zeros(k_len, dtype=jnp.int32).at[k_starts].set(1)
    q_ids = jnp.cumsum(q_indicators, dtype=jnp.int32) - 1
    k_ids = jnp.cumsum(k_indicators, dtype=jnp.int32) - 1

    mask = (q_ids[:, None] == k_ids[None, :]) & q_valid[:, None] & k_valid[None, :]

    if is_causal:
        q_pos_in_seq = jnp.arange(q_len) - q_starts[q_ids]
        k_pos_in_seq = jnp.arange(k_len) - k_starts[k_ids]
        causal_mask = k_pos_in_seq[None, :] <= q_pos_in_seq[:, None]
        mask = mask & causal_mask

    return jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)


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
            jnp.log(n.astype(jnp.float32) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_large = jnp.minimum(val_large, num_buckets - 1)
    return jnp.where(is_small, n, val_large) + ret

# =============================================================================
# FFN Modules
# =============================================================================


class T5FFN(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16):
        mk_linear = lambda in_sz, out_sz, axes: LinearBase(in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype)

        ff_proj = getattr(config, "feed_forward_proj", "relu")
        self.is_gated = getattr(config, "is_gated_act", ff_proj.startswith("gated"))
        act_str = getattr(config, "dense_act_fn", "gelu_new" if ff_proj == "gated-gelu" else ff_proj.split("-")[-1])

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
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, layer_idx=0, is_cross_attention=False, is_decoder=False, has_relative_attention_bias=False):
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.mesh = mesh

        self.d_model = config.d_model
        self.d_kv = getattr(config, "d_kv", getattr(config, "head_dim", 64))
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        mk_linear = lambda in_sz, out_sz, axes: LinearBase(in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype)
        self.q = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.k = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.v = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.o = mk_linear(self.inner_dim, self.d_model, ("tensor", None))

        # 【核心修改 1】按条件初始化 Embed
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias and not is_cross_attention:
            self.num_buckets = config.relative_attention_num_buckets
            self.max_distance = getattr(config, "relative_attention_max_distance", 128)
            self.rel_bias = Embed(self.num_buckets, self.n_heads, dtype=dtype, param_dtype=dtype, mesh=mesh, kernel_axes=(None, "tensor"))

        self.dropout = nnx.Dropout(config.dropout_rate)

        if is_decoder and not is_cross_attention:
            self.radix_attn = RadixAttention(
                num_heads=self.n_heads, head_dim=self.d_kv, scaling=1.0,
                num_kv_heads=self.n_heads, layer_id=layer_idx, attn_type=AttentionType.DECODER
            )

    def __call__(self, x, forward_batch: ForwardBatch, encoder_hidden_states=None, token_to_kv_pool: KVCache = None, deterministic=True, cached_attn_bias=None):
        dtype = x.dtype
        q, _ = self.q(x)
        if self.is_cross_attention:
            k, _ = self.k(encoder_hidden_states)
            v, _ = self.v(encoder_hidden_states)
        else:
            k, _ = self.k(x)
            v, _ = self.v(x)

        new_attn_bias = cached_attn_bias

        if self.is_decoder and not self.is_cross_attention and token_to_kv_pool is not None:
            q_3d, k_3d, v_3d = q.reshape(-1, self.n_heads, self.d_kv), k.reshape(-1, self.n_heads, self.d_kv), v.reshape(-1, self.n_heads, self.d_kv)
            out, _ = self.radix_attn(q_3d, k_3d, v_3d, forward_batch, token_to_kv_pool)
            out = out.reshape(q.shape[0], self.inner_dim)
        else:
            out, new_attn_bias = self._native_attention(q, k, v, forward_batch, cached_attn_bias)

        out = self.dropout(out.astype(dtype), deterministic=deterministic)
        out, _ = self.o(out)
        return out, new_attn_bias

    def _native_attention(self, q, k, v, forward_batch: ForwardBatch, cached_attn_bias=None):
        num_tokens, hidden = q.shape[0], q.shape[-1]
        head_dim = hidden // self.n_heads
        to_heads = lambda x: jnp.transpose(x.reshape(x.shape[0], self.n_heads, head_dim), (1, 0, 2))
        q_h, k_h, v_h = to_heads(q), to_heads(k), to_heads(v)

        scores = jnp.einsum("hqd,hkd->hqk", q_h.astype(jnp.float32), k_h.astype(jnp.float32))

        q_lens = getattr(forward_batch, "extend_seq_lens", forward_batch.seq_lens)
        if q_lens is None: q_lens = jnp.array([q.shape[0]], dtype=jnp.int32)

        new_attn_bias = cached_attn_bias
        if not self.is_cross_attention:
            if self.has_relative_attention_bias:
                new_attn_bias = self._compute_position_bias(q_lens, q.shape[0], k.shape[0])
            if new_attn_bias is not None:
                scores = scores + new_attn_bias.astype(jnp.float32)

        kv_lens = getattr(forward_batch, "encoder_seq_lens", q_lens) if self.is_cross_attention else q_lens
        is_causal = self.is_decoder and not self.is_cross_attention

        scores = _apply_block_diagonal_mask(scores, q_lens, kv_lens, is_causal=is_causal)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", weights, v_h.astype(jnp.float32))
        return jnp.transpose(out, (1, 0, 2)).reshape(num_tokens, hidden), new_attn_bias

    def _compute_position_bias(self, seq_lens, q_len, k_len):
        starts = jnp.cumsum(seq_lens) - seq_lens
        indicators = jnp.zeros(q_len, dtype=jnp.int32).at[starts].set(1)
        batch_ids = jnp.cumsum(indicators) - 1
        q_pos = jnp.arange(q_len) - starts[batch_ids]
        k_pos = jnp.arange(k_len) - starts[batch_ids]
        rel_pos = k_pos[None, :] - q_pos[:, None]
        buckets = relative_position_bucket(rel_pos, bidirectional=(not self.is_decoder), num_buckets=self.num_buckets, max_distance=self.max_distance)
        return jnp.transpose(self.rel_bias(buckets), (2, 0, 1))

# =============================================================================
# Transformer Block & Stack
# =============================================================================


class T5Block(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, layer_idx=0, is_decoder=False, has_relative_attention_bias=False):
        self.is_decoder = is_decoder
        self.ln1 = RMSNorm(config.d_model, epsilon=config.layer_norm_epsilon, dtype=dtype, param_dtype=dtype, use_scale=True)
        self.self_attn = T5Attention(config, mesh, dtype, layer_idx, False, is_decoder, has_relative_attention_bias)
        self.drop1 = nnx.Dropout(config.dropout_rate)

        if is_decoder:
            self.ln_cross = RMSNorm(config.d_model, epsilon=config.layer_norm_epsilon, dtype=dtype, param_dtype=dtype, use_scale=True)
            self.cross_attn = T5Attention(config, mesh, dtype, layer_idx, True, is_decoder, False)
            self.drop_cross = nnx.Dropout(config.dropout_rate)

        self.ln2 = RMSNorm(config.d_model, epsilon=config.layer_norm_epsilon, dtype=dtype, param_dtype=dtype, use_scale=True)
        self.mlp = T5FFN(config, mesh, dtype)
        self.drop2 = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, forward_batch: ForwardBatch, token_to_kv_pool=None, deterministic=True, cached_attn_bias=None):
        h, new_attn_bias = self.self_attn(self.ln1(x), forward_batch, token_to_kv_pool=token_to_kv_pool, deterministic=deterministic, cached_attn_bias=cached_attn_bias)
        x = fp16_clamp(x + self.drop1(h, deterministic=deterministic))

        if self.is_decoder and (enc_h := getattr(forward_batch, "encoder_hidden_states", None)) is not None:
            h, _ = self.cross_attn(self.ln_cross(x), forward_batch, enc_h, deterministic=deterministic)
            x = fp16_clamp(x + self.drop_cross(h, deterministic=deterministic))

        h = self.mlp(self.ln2(x), deterministic=deterministic)
        return fp16_clamp(x + self.drop2(h, deterministic=deterministic)), new_attn_bias


class T5Stack(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, is_decoder=False, is_umt5=False):
        self.is_decoder = is_decoder
        num_layers = getattr(config, "num_decoder_layers", config.num_layers) if is_decoder else config.num_layers

        self.blocks = nnx.List([
            T5Block(config, mesh, dtype, i, is_decoder, has_relative_attention_bias=True if is_umt5 else (i == 0))
            for i in range(num_layers)
        ])
        self.final_ln = RMSNorm(config.d_model, epsilon=config.layer_norm_epsilon, dtype=dtype, param_dtype=dtype, use_scale=True)
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, forward_batch: ForwardBatch, token_to_kv_pool=None, deterministic=True):
        x = self.dropout(x, deterministic=deterministic)
        attn_bias = None
        for block in self.blocks:
            x, attn_bias = block(x, forward_batch, token_to_kv_pool, deterministic, cached_attn_bias=attn_bias)
        return self.dropout(fp16_clamp(self.final_ln(x)), deterministic=deterministic)

# =============================================================================
# Weight Mapping Helper
# =============================================================================


def _block_mappings(config, idx, is_decoder, src_prefix, tgt_prefix, has_rel_bias=False):
    s, t = f"{src_prefix}.{idx}", f"{tgt_prefix}.{idx}"
    m = {
        f"{s}.layer.0.layer_norm.weight": WeightMapping(f"{t}.ln1.scale", (None,), False),
        f"{s}.layer.0.SelfAttention.q.weight": WeightMapping(f"{t}.self_attn.q.weight", (None, "tensor"), True),
        f"{s}.layer.0.SelfAttention.k.weight": WeightMapping(f"{t}.self_attn.k.weight", (None, "tensor"), True),
        f"{s}.layer.0.SelfAttention.v.weight": WeightMapping(f"{t}.self_attn.v.weight", (None, "tensor"), True),
        f"{s}.layer.0.SelfAttention.o.weight": WeightMapping(f"{t}.self_attn.o.weight", ("tensor", None), True),
    }

    if has_rel_bias:
        m[f"{s}.layer.0.SelfAttention.relative_attention_bias.weight"] = WeightMapping(
            f"{t}.self_attn.rel_bias.embedding", (None, "tensor"), False
        )

    if is_decoder:
        m.update({
            f"{s}.layer.1.layer_norm.weight": WeightMapping(f"{t}.ln_cross.scale", (None,), False),
            f"{s}.layer.1.EncDecAttention.q.weight": WeightMapping(f"{t}.cross_attn.q.weight", (None, "tensor"), True),
            f"{s}.layer.1.EncDecAttention.k.weight": WeightMapping(f"{t}.cross_attn.k.weight", (None, "tensor"), True),
            f"{s}.layer.1.EncDecAttention.v.weight": WeightMapping(f"{t}.cross_attn.v.weight", (None, "tensor"), True),
            f"{s}.layer.1.EncDecAttention.o.weight": WeightMapping(f"{t}.cross_attn.o.weight", ("tensor", None), True),
        })
        ffn_idx = 2
    else:
        ffn_idx = 1

    m[f"{s}.layer.{ffn_idx}.layer_norm.weight"] = WeightMapping(f"{t}.ln2.scale", (None,), False)

    ff_proj = getattr(config, "feed_forward_proj", "relu")
    is_gated = getattr(config, "is_gated_act", ff_proj.startswith("gated"))

    prefix_dense = "DenseReluDense"
    if is_gated:
        m.update({
            f"{s}.layer.{ffn_idx}.{prefix_dense}.wi_0.weight": WeightMapping(f"{t}.mlp.wi_0.weight", (None, "tensor"), True),
            f"{s}.layer.{ffn_idx}.{prefix_dense}.wi_1.weight": WeightMapping(f"{t}.mlp.wi_1.weight", (None, "tensor"), True),
            f"{s}.layer.{ffn_idx}.{prefix_dense}.wo.weight": WeightMapping(f"{t}.mlp.wo.weight", ("tensor", None), True),
        })
    else:
        m.update({
            f"{s}.layer.{ffn_idx}.{prefix_dense}.wi.weight": WeightMapping(f"{t}.mlp.wi.weight", (None, "tensor"), True),
            f"{s}.layer.{ffn_idx}.{prefix_dense}.wo.weight": WeightMapping(f"{t}.mlp.wo.weight", ("tensor", None), True),
        })
    return m


# =============================================================================
# Model Classes
# =============================================================================

class T5EncoderModel(nnx.Module):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16, is_umt5=False):
        self.config, self.mesh, self.dtype, self.is_umt5 = config, mesh, dtype, is_umt5
        self.shared = Embed(config.vocab_size, config.d_model, dtype=dtype, param_dtype=dtype, mesh=mesh, kernel_axes=("tensor", None))
        self.encoder = T5Stack(config, mesh, dtype, is_decoder=False, is_umt5=is_umt5)

    def load_weights(self, model_config: ModelConfig):
        path = model_config.model_path
        if os.path.isabs(path) and os.path.exists(path) and not glob.glob(os.path.join(path, "*.safetensors")) and os.path.exists(os.path.join(path, "text_encoder")):
            model_config.model_path = os.path.join(path, "text_encoder")
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "shared.weight": WeightMapping("shared.embedding", ("tensor", None), False),
            "encoder.final_layer_norm.weight": WeightMapping("encoder.final_ln.scale", (None,), False),
        }
        for i in range(self.config.num_layers):
            m.update(_block_mappings(self.config, i, False, "encoder.block", "encoder.blocks", has_rel_bias=True if self.is_umt5 else (i == 0)))
        return m

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: Optional[jax.Array] = None,
        output_hidden_states: Optional[bool] = None,
        deterministic: bool = True
    ):
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
            if position_ids is not None:
                position_ids = position_ids.reshape(1, -1)

        batch_size, seq_len = input_ids.shape

        flat_input_ids = input_ids.reshape(-1)
        seq_lens_array = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

        forward_batch = ForwardBatch(
            bid=0,
            forward_mode=0,
            batch_size=batch_size,
            input_ids=flat_input_ids,
            seq_lens=seq_lens_array,
            extend_seq_lens=seq_lens_array,
            positions=position_ids.reshape(-1) if position_ids is not None else jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), batch_size),
            req_pool_indices=jnp.arange(batch_size, dtype=jnp.int32),
            out_cache_loc=jnp.zeros(flat_input_ids.shape[0], dtype=jnp.int32),
        )

        x = self.shared(flat_input_ids)
        flat_hidden = self.encoder(x, forward_batch, token_to_kv_pool=None, deterministic=deterministic)

        hidden_states = flat_hidden.reshape(batch_size, seq_len, self.config.d_model)

        if output_hidden_states:
            return hidden_states, [hidden_states]

        return hidden_states


class UMT5EncoderModel(T5EncoderModel):
    def __init__(self, config: T5Config, mesh, dtype=jnp.bfloat16):
        super().__init__(config, mesh, dtype, is_umt5=True)


EntryClass = [T5EncoderModel, UMT5EncoderModel]