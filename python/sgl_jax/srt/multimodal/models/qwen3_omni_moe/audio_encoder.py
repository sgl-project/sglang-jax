import math

import jax
from flax import nnx
from jax import numpy as jnp
from jax.lax import Precision
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.layers.attention.layer import simple_attention


class Qwen3OmniMoeAudioAttention(nnx.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.mesh = mesh
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.v_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.q_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.out_proj = LinearBase(
            self.embed_dim,
            self.embed_dim,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        seq_length, _ = hidden_states.shape

        # Attention spans the packed sequence, so gather tokens before Q/K/V
        # reshaping; the output projection restores the data layout.
        replicated = NamedSharding(self.mesh, P(None, None))
        query_states = self.q_proj(hidden_states, out_sharding=replicated)[0].reshape(
            1, seq_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states, out_sharding=replicated)[0].reshape(
            1, seq_length, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states, out_sharding=replicated)[0].reshape(
            1, seq_length, self.num_heads, self.head_dim
        )

        attn_output = simple_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            causal=False,
            mask=attention_mask,
        )

        attn_output = attn_output.reshape(seq_length, -1)
        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class Qwen3OmniMoeAudioEncoderLayer(nnx.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(config, mesh=mesh, dtype=dtype, rngs=rngs)
        self.self_attn_layer_norm = nnx.LayerNorm(self.embed_dim, param_dtype=dtype, rngs=rngs)
        self.dropout = config.dropout
        self.activation_fn = (
            jax.nn.gelu
        )  # todo(Garrybest): should extract from config.activation_function
        self.activation_dropout = config.activation_dropout
        self.fc1 = LinearBase(
            self.embed_dim,
            config.encoder_ffn_dim,
            mesh=mesh,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.fc2 = LinearBase(
            config.encoder_ffn_dim,
            self.embed_dim,
            mesh=mesh,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.final_layer_norm = nnx.LayerNorm(self.embed_dim, param_dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states, approximate=False)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == jnp.float16:
            clamp_value = jnp.finfo(hidden_states.dtype).max - 1000
            hidden_states = jnp.clip(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class Qwen3OmniMoeAudioEncoder(nnx.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.mesh = mesh
        self.dtype = dtype
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nnx.List(
            [
                Qwen3OmniMoeAudioEncoderLayer(config, mesh=mesh, dtype=dtype, rngs=rngs)
                for _ in range(config.encoder_layers)
            ]
        )
        self.ln_post = nnx.LayerNorm(config.d_model, dtype=dtype, rngs=rngs)
        self.gradient_checkpointing = False
        self.conv2d1 = nnx.Conv(
            in_features=1,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            param_dtype=dtype,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.conv2d2 = nnx.Conv(
            in_features=config.downsample_hidden_size,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            param_dtype=dtype,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.conv2d3 = nnx.Conv(
            in_features=config.downsample_hidden_size,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            param_dtype=dtype,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.conv_out = LinearBase(
            input_size=config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            output_size=config.d_model,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.proj1 = LinearBase(
            config.d_model, config.d_model, mesh=mesh, params_dtype=dtype, kernel_axes=(None, None)
        )
        self.act = jax.nn.gelu  # todo(Garrybest): should extract from config.activation_function
        self.proj2 = LinearBase(
            config.d_model,
            config.output_dim,
            mesh=mesh,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

    def __call__(self, input_features: jax.Array, feature_lens=None):
        r"""
        input_features: [f, t]
        feature_lens: mel length
        """
        chunk_num = (feature_lens + self.n_window * 2 - 1) // (self.n_window * 2)
        chunk_lengths = jnp.full(chunk_num.sum(), self.n_window * 2, dtype=jnp.int32)

        tail_chunk_index = jnp.pad(chunk_num, (1, 0), constant_values=-1).cumsum(0)[1:]
        chunk_lengths = chunk_lengths.at[tail_chunk_index].set(feature_lens % (self.n_window * 2))
        chunk_lengths = chunk_lengths.at[chunk_lengths == 0].set(self.n_window * 2)

        split_indices = jnp.cumsum(chunk_lengths)[:-1]
        chunk_list = jnp.split(
            input_features.T, split_indices.tolist(), axis=0
        )  # list of [chunk_len, mel_freq]
        max_chunk_length = int(chunk_lengths.max())
        padded_feature = jnp.stack(
            [jnp.pad(x, ((0, max_chunk_length - x.shape[0]), (0, 0))) for x in chunk_list]
        ).swapaxes(
            1, 2
        )  # [b, f, t]

        # Each chunk passes through three stride-2 convolutions independently.
        feature_lens_after_cnn = (chunk_lengths + 7) // 8
        padded_feature = jnp.expand_dims(padded_feature, axis=3)  # [b, f, t, c]
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        conv_chunk_indices = jnp.arange(
            self.conv_chunksize, padded_feature.shape[0], self.conv_chunksize
        )
        for chunk in jnp.split(padded_feature, conv_chunk_indices, axis=0):
            # Now chunk shape is [b, f, t, c]
            padded_embed = jax.nn.gelu(self.conv2d1(chunk), approximate=False)
            padded_embed = jax.nn.gelu(self.conv2d2(padded_embed), approximate=False)
            padded_embed = jax.nn.gelu(self.conv2d3(padded_embed), approximate=False)
            padded_embeds.append(padded_embed)
        padded_embed = jnp.concatenate(padded_embeds, axis=0)
        b, f, t, c = padded_embed.shape
        padded_embed, _ = self.conv_out(padded_embed.transpose(0, 2, 3, 1).reshape(b, t, c * f))
        pos_embed_slice = self.positional_embedding(padded_embed.shape[1])
        positional_embedding = jnp.expand_dims(pos_embed_slice, axis=0).astype(padded_embed.dtype)
        padded_embed = padded_embed + positional_embedding
        padded_mask_after_cnn = jnp.arange(t)[None, :] < feature_lens_after_cnn[:, None]
        hidden_states = padded_embed.at[padded_mask_after_cnn].get(
            out_sharding=NamedSharding(self.mesh, P(None, None))
        )

        # Packed samples and inference windows must not attend to one another.
        sample_lengths = jnp.split(feature_lens_after_cnn, jnp.cumsum(chunk_num)[:-1].tolist())
        window_size = t * (self.n_window_infer // (self.n_window * 2))
        attention_mask = jnp.zeros((hidden_states.shape[0], hidden_states.shape[0]), dtype=bool)
        offset = 0
        for lengths in sample_lengths:
            sample_length = int(lengths.sum())
            for start in range(0, sample_length, window_size):
                end = offset + min(start + window_size, sample_length)
                attention_mask = attention_mask.at[offset + start : end, offset + start : end].set(
                    True
                )
            offset += sample_length

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)

        hidden_states = self.ln_post(hidden_states)
        hidden_states, _ = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states, approximate=False)
        hidden_states, _ = self.proj2(hidden_states)
        return hidden_states


class SinusoidsPositionEmbedding(nnx.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

    def __call__(self, seqlen: int):
        log_timescale_increment = jnp.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = jnp.exp(
            -log_timescale_increment * jnp.arange(self.channels // 2).astype(jnp.float32)
        )
        scaled_time = jnp.arange(self.length)[:, jnp.newaxis] * inv_timescales[jnp.newaxis, :]
        embedding = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
        return embedding[:seqlen, :]
