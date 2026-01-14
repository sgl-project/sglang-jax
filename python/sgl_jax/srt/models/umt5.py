# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""UMT5 model using RadixAttention for efficient attention computation.

This implementation uses RadixAttention for decoder self-attention and native
attention for encoder and cross-attention, providing optimal balance between
performance and compatibility with T5's position bias mechanism.
"""

import copy
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import UMT5Config

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

# =============================================================================
# Utilities
# =============================================================================


def fp16_clamp(x):
    """Clamp to prevent float16 overflow."""
    if x.dtype == jnp.float16 and jnp.isinf(x).any():
        clamp = jnp.finfo(x.dtype).max - 1000
        return jax.lax.clamp(x=x, min=-clamp, max=clamp)
    return x


def gelu_new(x):
    """GELU with tanh approximation (HF T5/UMT5 style)."""
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def update_decoder_seq_lens(forward_batch: ForwardBatch, dec_ids):
    """Update forward_batch.seq_lens for decoder using decoder_seq_lens if available."""
    if hasattr(forward_batch, "decoder_seq_lens"):
        forward_batch.seq_lens = forward_batch.decoder_seq_lens
        forward_batch.extend_seq_lens = forward_batch.decoder_seq_lens
    else:
        dec_len = len(dec_ids)
        forward_batch.seq_lens = jnp.array([dec_len], dtype=jnp.int32)
        forward_batch.extend_seq_lens = jnp.array([dec_len], dtype=jnp.int32)
    forward_batch.positions = jnp.arange(len(dec_ids), dtype=jnp.int32)


ACT_FN = {"gelu": jax.nn.gelu, "gelu_new": gelu_new, "relu": jax.nn.relu}


def _apply_block_diagonal_mask(scores, q_lens, k_lens, is_causal=False):
    """Block-diagonal mask: each sequence attends only to itself.

    For cross-attention: q_lens (decoder) and k_lens (encoder) can differ.
    For self-attention: pass same seq_lens for both.
    For decoder self-attention: set is_causal=True to add causal mask.
    """
    _, q_len, k_len = scores.shape
    q_total, k_total = jnp.sum(q_lens), jnp.sum(k_lens)

    # Validity masks
    q_valid = jnp.arange(q_len) < q_total
    k_valid = jnp.arange(k_len) < k_total

    # Batch IDs via cumsum
    q_starts = jnp.cumsum(q_lens, dtype=jnp.int32) - q_lens
    k_starts = jnp.cumsum(k_lens, dtype=jnp.int32) - k_lens
    q_indicators = jnp.zeros(q_len, dtype=jnp.int32).at[q_starts].set(1)
    k_indicators = jnp.zeros(k_len, dtype=jnp.int32).at[k_starts].set(1)
    q_ids = jnp.cumsum(q_indicators, dtype=jnp.int32) - 1
    k_ids = jnp.cumsum(k_indicators, dtype=jnp.int32) - 1

    # Block-diagonal: same batch ID
    mask = (q_ids[:, None] == k_ids[None, :]) & q_valid[:, None] & k_valid[None, :]

    # Add causal mask for decoder self-attention
    if is_causal:
        q_pos_in_seq = jnp.arange(q_len) - q_starts[q_ids]
        k_pos_in_seq = jnp.arange(k_len) - k_starts[k_ids]
        causal_mask = k_pos_in_seq[None, :] <= q_pos_in_seq[:, None]
        mask = mask & causal_mask

    return jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)


def relative_position_bucket(rel_pos, bidirectional=True, num_buckets=32, max_distance=128):
    """T5-style relative position bucketing."""
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


class UMT5FFN(nnx.Module):
    """UMT5 Feed-Forward Network (gated or standard)."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        mk_linear = lambda in_sz, out_sz, axes: LinearBase(
            in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype
        )

        if config.is_gated_act:
            # Gated: (GeLU(x @ wi_0) * (x @ wi_1)) @ wo
            self.wi_0 = mk_linear(config.d_model, config.d_ff, (None, "tensor"))
            self.wi_1 = mk_linear(config.d_model, config.d_ff, (None, "tensor"))
        else:
            # Standard: GeLU(x @ wi) @ wo
            self.wi = mk_linear(config.d_model, config.d_ff, (None, "tensor"))

        self.wo = mk_linear(config.d_ff, config.d_model, ("tensor", None))
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(config.dense_act_fn, jax.nn.gelu)
        self.is_gated = config.is_gated_act

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


class UMT5Attention(nnx.Module):
    """UMT5 attention: RadixAttention for decoder self-attn, custom for others."""

    def __init__(
        self,
        config: UMT5Config,
        mesh,
        dtype=jnp.bfloat16,
        layer_idx=0,
        is_cross_attention=False,
        is_decoder=False,
    ):
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.mesh = mesh

        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        # QKV + O projections
        mk_linear = lambda in_sz, out_sz, axes: LinearBase(
            in_sz, out_sz, mesh, use_bias=False, kernel_axes=axes, params_dtype=dtype
        )
        self.q = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.k = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.v = mk_linear(self.d_model, self.inner_dim, (None, "tensor"))
        self.o = mk_linear(self.inner_dim, self.d_model, ("tensor", None))

        # T5 relative position bias (self-attention only)
        if not is_cross_attention:
            num_buckets = config.relative_attention_num_buckets
            self.rel_bias = Embed(
                num_buckets,
                self.n_heads,
                dtype=dtype,
                param_dtype=dtype,
                mesh=mesh,
                kernel_axes=(None, "tensor"),
            )
            self.num_buckets = num_buckets
            self.max_distance = getattr(config, "relative_attention_max_distance", 128)

        self.dropout = nnx.Dropout(config.dropout_rate)

        # RadixAttention for decoder self-attention (needs KV cache)
        if is_decoder and not is_cross_attention:
            self.radix_attn = RadixAttention(
                num_heads=self.n_heads,
                head_dim=self.d_kv,
                scaling=1.0,  # T5 uses scale=1.0
                num_kv_heads=self.n_heads,
                layer_id=layer_idx,
                attn_type=AttentionType.DECODER,
            )

    def __call__(
        self,
        x,
        forward_batch: ForwardBatch,
        encoder_hidden_states=None,
        token_to_kv_pool: KVCache = None,
        deterministic=True,
    ):
        dtype = x.dtype

        # Q/K/V projections
        q, _ = self.q(x)
        if self.is_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states required for cross-attention")
            k, _ = self.k(encoder_hidden_states)
            v, _ = self.v(encoder_hidden_states)
        else:
            k, _ = self.k(x)
            v, _ = self.v(x)

        # Decoder self-attention with KV cache: use RadixAttention
        if self.is_decoder and not self.is_cross_attention and token_to_kv_pool is not None:
            q_3d = q.reshape(-1, self.n_heads, self.d_kv)
            k_3d = k.reshape(-1, self.n_heads, self.d_kv)
            v_3d = v.reshape(-1, self.n_heads, self.d_kv)
            out, _ = self.radix_attn(q_3d, k_3d, v_3d, forward_batch, token_to_kv_pool)
            out = out.reshape(q.shape[0], self.inner_dim)
        else:
            # Encoder self-attention and cross-attention: use native attention
            out = self._native_attention(q, k, v, forward_batch)

        out = self.dropout(out.astype(dtype), deterministic=deterministic)
        out, _ = self.o(out)
        return out

    def _native_attention(self, q, k, v, forward_batch: ForwardBatch):
        """Native attention for encoder/cross-attention with T5 position bias."""
        num_tokens, hidden = q.shape[0], q.shape[-1]
        head_dim = hidden // self.n_heads

        # Reshape to [heads, tokens, head_dim]
        def to_heads(x):
            n_tok = x.shape[0]
            return jnp.transpose(x.reshape(n_tok, self.n_heads, head_dim), (1, 0, 2))

        q_h, k_h, v_h = to_heads(q), to_heads(k), to_heads(v)

        # Compute scores in float32
        scores = jnp.einsum("hqd,hkd->hqk", q_h.astype(jnp.float32), k_h.astype(jnp.float32))

        # Get sequence lengths
        q_lens = getattr(forward_batch, "extend_seq_lens", forward_batch.seq_lens)
        # Fallback if seq_lens is None: assume single sequence
        if q_lens is None:
            q_lens = jnp.array([q.shape[0]], dtype=jnp.int32)

        # Add position bias for self-attention (T5-specific)
        if not self.is_cross_attention and hasattr(self, "rel_bias"):
            pos_bias = self._compute_position_bias(q_lens, q.shape[0], k.shape[0])
            scores = scores + pos_bias.astype(jnp.float32)

        # Apply masking
        kv_lens = (
            getattr(forward_batch, "encoder_seq_lens", q_lens)
            if self.is_cross_attention
            else q_lens
        )
        is_causal = self.is_decoder and not self.is_cross_attention

        # Apply block_diagonal_mask
        scores = _apply_block_diagonal_mask(scores, q_lens, kv_lens, is_causal=is_causal)

        # Softmax and weighted sum
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", weights, v_h.astype(jnp.float32))

        return jnp.transpose(out, (1, 0, 2)).reshape(num_tokens, hidden)

    def _compute_position_bias(self, seq_lens, q_len, k_len):
        """Compute T5 position bias [heads, q_len, k_len]."""
        starts = jnp.cumsum(seq_lens) - seq_lens
        indicators = jnp.zeros(q_len, dtype=jnp.int32).at[starts].set(1)
        batch_ids = jnp.cumsum(indicators) - 1

        q_pos = jnp.arange(q_len) - starts[batch_ids]
        k_pos = jnp.arange(k_len) - starts[batch_ids]
        rel_pos = k_pos[None, :] - q_pos[:, None]

        buckets = relative_position_bucket(
            rel_pos,
            bidirectional=(not self.is_decoder),
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        return jnp.transpose(self.rel_bias(buckets), (2, 0, 1))


# =============================================================================
# Transformer Block & Stack
# =============================================================================


class UMT5Block(nnx.Module):
    """UMT5 transformer block."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16, layer_idx=0, is_decoder=False):
        self.is_decoder = is_decoder

        # Self attention
        self.ln1 = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.self_attn = UMT5Attention(config, mesh, dtype, layer_idx, False, is_decoder)
        self.drop1 = nnx.Dropout(config.dropout_rate)

        # Cross attention (decoder only)
        if is_decoder:
            self.ln_cross = RMSNorm(
                config.d_model,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=dtype,
                use_scale=True,
            )
            self.cross_attn = UMT5Attention(config, mesh, dtype, layer_idx, True, is_decoder)
            self.drop_cross = nnx.Dropout(config.dropout_rate)

        # FFN
        self.ln2 = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.mlp = UMT5FFN(config, mesh, dtype)
        self.drop2 = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, forward_batch: ForwardBatch, token_to_kv_pool=None, deterministic=True):
        # Self attention
        h = self.self_attn(
            self.ln1(x),
            forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            deterministic=deterministic,
        )
        x = fp16_clamp(x + self.drop1(h, deterministic=deterministic))

        # Cross attention
        if (
            self.is_decoder
            and (enc_h := getattr(forward_batch, "encoder_hidden_states", None)) is not None
        ):
            h = self.cross_attn(self.ln_cross(x), forward_batch, enc_h, deterministic=deterministic)
            x = fp16_clamp(x + self.drop_cross(h, deterministic=deterministic))

        # FFN
        h = self.mlp(self.ln2(x), deterministic=deterministic)
        return fp16_clamp(x + self.drop2(h, deterministic=deterministic))


class UMT5Stack(nnx.Module):
    """Stack of UMT5 transformer blocks."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        self.is_decoder = config.is_decoder
        self.blocks = nnx.List(
            [UMT5Block(config, mesh, dtype, i, config.is_decoder) for i in range(config.num_layers)]
        )
        self.final_ln = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(self, x, forward_batch: ForwardBatch, token_to_kv_pool=None, deterministic=True):
        x = self.dropout(x, deterministic=deterministic)
        for block in self.blocks:
            x = block(x, forward_batch, token_to_kv_pool, deterministic)
        return self.dropout(fp16_clamp(self.final_ln(x)), deterministic=deterministic)


# =============================================================================
# Model Classes
# =============================================================================


class UMT5EncoderModel(nnx.Module):
    """UMT5 encoder-only model."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.encoder = UMT5Stack(config, mesh, dtype)

    def load_weights(self, model_config: ModelConfig):
        import glob
        import os

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
            m.update(_block_mappings(self.config, i, False, "encoder.block", "encoder.blocks"))
        return m

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool=None, logits_metadata=None):
        x = self.shared(forward_batch.input_ids)
        deterministic = getattr(forward_batch, "deterministic", True)
        hidden = self.encoder(x, forward_batch, token_to_kv_pool, deterministic)

        # Dummy logits for interface compatibility
        bs = forward_batch.seq_lens.shape[0]
        dummy = jnp.zeros((bs, self.config.vocab_size), dtype=self.dtype)
        dummy = jax.sharding.reshard(dummy, NamedSharding(self.mesh, P(None, "tensor")))
        return LogitsProcessorOutput(next_token_logits=dummy, hidden_states=hidden), [], []


class UMT5DecoderModel(nnx.Module):
    """UMT5 decoder-only model."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        dec_cfg = copy.deepcopy(config)
        dec_cfg.is_decoder = True
        dec_cfg.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(dec_cfg, mesh, dtype)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "shared.weight": WeightMapping("shared.embedding", ("tensor", None), False),
            "decoder.final_layer_norm.weight": WeightMapping(
                "decoder.final_ln.scale", (None,), False
            ),
        }
        for i in range(self.config.num_decoder_layers):
            m.update(_block_mappings(self.config, i, True, "decoder.block", "decoder.blocks"))
        return m

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool=None):
        x = self.shared(forward_batch.input_ids)
        deterministic = getattr(forward_batch, "deterministic", True)
        return self.decoder(x, forward_batch, token_to_kv_pool, deterministic)


class UMT5Model(nnx.Module):
    """UMT5 encoder-decoder model."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

        enc_cfg = copy.deepcopy(config)
        enc_cfg.is_decoder = False
        self.encoder = UMT5Stack(enc_cfg, mesh, dtype)

        dec_cfg = copy.deepcopy(config)
        dec_cfg.is_decoder = True
        dec_cfg.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(dec_cfg, mesh, dtype)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "shared.weight": WeightMapping("shared.embedding", ("tensor", None), False),
            "encoder.final_layer_norm.weight": WeightMapping(
                "encoder.final_ln.scale", (None,), False
            ),
            "decoder.final_layer_norm.weight": WeightMapping(
                "decoder.final_ln.scale", (None,), False
            ),
        }
        for i in range(self.config.num_layers):
            m.update(_block_mappings(self.config, i, False, "encoder.block", "encoder.blocks"))
        for i in range(self.config.num_decoder_layers):
            m.update(_block_mappings(self.config, i, True, "decoder.block", "decoder.blocks"))
        return m

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool=None):
        deterministic = getattr(forward_batch, "deterministic", True)

        # Encoder pass - save state immediately
        enc_h = self.encoder(
            self.shared(forward_batch.input_ids), forward_batch, None, deterministic
        )
        forward_batch.encoder_seq_lens = getattr(
            forward_batch, "extend_seq_lens", forward_batch.seq_lens
        )
        forward_batch.encoder_hidden_states = enc_h

        # Decoder pass - update seq_lens if needed
        dec_ids = getattr(forward_batch, "decoder_input_ids", forward_batch.input_ids)
        if (
            hasattr(forward_batch, "decoder_input_ids")
            and forward_batch.decoder_input_ids is not forward_batch.input_ids
        ):
            update_decoder_seq_lens(forward_batch, dec_ids)

        return self.decoder(self.shared(dec_ids), forward_batch, token_to_kv_pool, deterministic)


class UMT5ForConditionalGeneration(nnx.Module):
    """UMT5 for conditional generation with LM head."""

    def __init__(self, config: UMT5Config, mesh, dtype=jnp.bfloat16):
        self.config, self.mesh, self.dtype = config, mesh, dtype
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

        enc_cfg = copy.deepcopy(config)
        enc_cfg.is_decoder = False
        self.encoder = UMT5Stack(enc_cfg, mesh, dtype)

        dec_cfg = copy.deepcopy(config)
        dec_cfg.is_decoder = True
        dec_cfg.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(dec_cfg, mesh, dtype)

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(self._weight_mappings())

    def _weight_mappings(self):
        m = {
            "shared.weight": WeightMapping("shared.embedding", ("tensor", None), False),
            "encoder.final_layer_norm.weight": WeightMapping(
                "encoder.final_ln.scale", (None,), False
            ),
            "decoder.final_layer_norm.weight": WeightMapping(
                "decoder.final_ln.scale", (None,), False
            ),
            "lm_head.weight": WeightMapping("lm_head.embedding", ("tensor", None), False),
        }
        for i in range(self.config.num_layers):
            m.update(_block_mappings(self.config, i, False, "encoder.block", "encoder.blocks"))
        for i in range(self.config.num_decoder_layers):
            m.update(_block_mappings(self.config, i, True, "decoder.block", "decoder.blocks"))
        return m

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool=None, logits_metadata=None):
        deterministic = getattr(forward_batch, "deterministic", True)

        # Encoder pass (cache if needed)
        if (
            not hasattr(forward_batch, "encoder_hidden_states")
            or forward_batch.encoder_hidden_states is None
        ):
            enc_h = self.shared(forward_batch.input_ids)
            enc_h = self.encoder(enc_h, forward_batch, None, deterministic)
            forward_batch.encoder_seq_lens = getattr(
                forward_batch, "extend_seq_lens", forward_batch.seq_lens
            )
            forward_batch.encoder_hidden_states = enc_h

        # Decoder pass
        dec_ids = getattr(forward_batch, "decoder_input_ids", forward_batch.input_ids)

        if (
            hasattr(forward_batch, "decoder_input_ids")
            and forward_batch.decoder_input_ids is not forward_batch.input_ids
        ):
            update_decoder_seq_lens(forward_batch, dec_ids)

        dec_h = self.decoder(self.shared(dec_ids), forward_batch, token_to_kv_pool, deterministic)

        if logits_metadata is not None:
            logits = self.logits_processor(dec_h, self.lm_head, logits_metadata)
        else:
            logits = jnp.matmul(dec_h, self.lm_head.embedding[...].T)
        return logits, [], []


# =============================================================================
# Weight Mapping Helper
# =============================================================================


def _block_mappings(config, idx, is_decoder, src_prefix, tgt_prefix):
    """Generate weight mappings for a transformer block."""
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
        f"{s}.layer.0.SelfAttention.relative_attention_bias.weight": WeightMapping(
            f"{t}.self_attn.rel_bias.embedding", (None, "tensor"), False
        ),
    }

    if is_decoder:
        m.update(
            {
                f"{s}.layer.1.layer_norm.weight": WeightMapping(
                    f"{t}.ln_cross.scale", (None,), False
                ),
                f"{s}.layer.1.EncDecAttention.q.weight": WeightMapping(
                    f"{t}.cross_attn.q.weight", (None, "tensor"), True
                ),
                f"{s}.layer.1.EncDecAttention.k.weight": WeightMapping(
                    f"{t}.cross_attn.k.weight", (None, "tensor"), True
                ),
                f"{s}.layer.1.EncDecAttention.v.weight": WeightMapping(
                    f"{t}.cross_attn.v.weight", (None, "tensor"), True
                ),
                f"{s}.layer.1.EncDecAttention.o.weight": WeightMapping(
                    f"{t}.cross_attn.o.weight", ("tensor", None), True
                ),
            }
        )
        ffn_idx = 2
    else:
        ffn_idx = 1

    m[f"{s}.layer.{ffn_idx}.layer_norm.weight"] = WeightMapping(f"{t}.ln2.scale", (None,), False)

    if config.is_gated_act:
        m.update(
            {
                f"{s}.layer.{ffn_idx}.DenseReluDense.wi_0.weight": WeightMapping(
                    f"{t}.mlp.wi_0.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.DenseReluDense.wi_1.weight": WeightMapping(
                    f"{t}.mlp.wi_1.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.DenseReluDense.wo.weight": WeightMapping(
                    f"{t}.mlp.wo.weight", ("tensor", None), True
                ),
            }
        )
    else:
        m.update(
            {
                f"{s}.layer.{ffn_idx}.DenseReluDense.wi.weight": WeightMapping(
                    f"{t}.mlp.wi.weight", (None, "tensor"), True
                ),
                f"{s}.layer.{ffn_idx}.DenseReluDense.wo.weight": WeightMapping(
                    f"{t}.mlp.wo.weight", ("tensor", None), True
                ),
            }
        )

    return m


EntryClass = [UMT5EncoderModel, UMT5DecoderModel, UMT5Model, UMT5ForConditionalGeneration]
