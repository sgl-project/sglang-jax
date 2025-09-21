# Adapted from
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/siglip/modeling_siglip.py

from functools import partial
from typing import Optional, Type, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import SiglipVisionConfig

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

class SiglipVisionEmbeddings(nnx.Module):
    def __init__(self, config: SiglipVisionConfig, dtype: jnp.dtype):
        super().__init__()
        self.config = config
        self.embed_dim = config.embedding_length
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.dtype_mm = dtype

        self.patch_embedding = nnx.Conv(
            self.embed_dim,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype_mm,
        )

        self.position_embedding = self.Embed(
            num_embeddings=(self.image_size // self.patch_size) ** 2,
            features=self.embed_dim,
            dtype=self.dtype_mm,
        )

    def forward(self, pixel_values: jax.Array) -> jax.Array:
        x = self.patch_embedding(pixel_values)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        # interpolate_pos_encoding is never used in sglang
        x = x + self.position_embedding(x)

        return x


# Copied from sglang.srt.models.clip.CLIPMLP
class SiglipMLP(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype,
    ):
        self.dtype_mm = dtype 
        self.mlp_dim = config.embedding_length
        self.hidden_size = config.feed_forward_length

        self.fc1 = nnx.Linear(in_features=self.mlp_dim, out_features=self.hidden_size, dtype=self.dtype_mm)
        self.act = nnx.gelu(approximate=True)
        self.fc2 = nnx.Linear(in_features=self.hidden_size, out_features=self.mlp_dim, dtype=self.dtype_mm)

    def forward(self, x: jax.Array) -> jax.Array:
        x_parallel = self.fc1(x)
        x_parallel = self.act(x_parallel)
        return self.fc2(x_parallel)


# Copied from sglang.srt.models.clip.CLIPEncoderLayer
class SiglipEncoderLayer(nnx.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype,
    ) -> None:
        self.dtype_mm = dtype
        self.mlp = SiglipMLP(config, dtype)
        self.self_attn = nnx.MultiHeadDotProductAttention(
            num_heads=config.num_attention_heads,
            dtype=self.dtype_mm,
        )
        self.layer_norm1 = nnx.LayerNorm(num_features=config.embedding_length, epsilon=config.layer_norm_epsilon, dtype=self.dtype_mm)
        self.layer_norm2 = nnx.LayerNorm(num_features=config.embedding_length, epsilon=config.layer_norm_epsilon, dtype=self.dtype_mm)

    def __call__(
        self,
        hidden_states: jax.Array,
    ) -> jax.Array:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
    
        hidden_states = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nnx.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype,
    ) -> None:
        self.config = config
        self.dtype_mm = dtype

        self.layers = [
            SiglipEncoderLayer(config=config,dtype=dtype)
            for i in range(config.num_hidden_layers)
        ]
        self.layer_norm = nnx.LayerNorm(num_features=config.embedding_length, epsilon=config.layer_norm_epsilon, dtype=self.dtype_mm)

    def __call__(
        self,
        inputs_embeds: jax.Array,
    ) -> jax.Array:
        hidden_states = inputs_embeds

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry))
        def forward(hidden_states, layer):
            hidden_states = layer(hidden_states)
            return hidden_states
        
        return self.layer_norm(forward(hidden_states, self.layers))


class SiglipVisionTransformer(nnx.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config, dtype)
        self.encoder = SiglipEncoder(
            config=config,
            dtype=dtype,
        )

        # VisionAttention in SiglipEncoderLayer is multihead attention
        self.post_layernorm = nnx.LayerNorm(num_features=config.embedding_length, epsilon=config.layer_norm_epsilon, dtype=dtype)

    def __call__(
        self,
        forward_batch: jax.Array,
    ):
        hidden_states = forward_batch + self.embeddings()

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nnx.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.vision_model = SiglipVisionTransformer(
            config, dtype,
        )

    def __call__(self, forward_batch: jax.Array):
        return self.vision_model(forward_batch)

class SiglipVisionConfig:
    def __init__(self):
        self.embedding_length = 1152
        self.feed_forward_length = 4304
        self.layer_norm_epsilon = 1e-6
        self.num_attention_heads = 16
        self.num_hidden_layers = 27
        self.patch_size = 14
        self.image_size = 896

if __name__ == "__main__":
    config = SiglipVisionConfig()
    model = SiglipVisionModel(config)

    # batch_size, num_frames, num_patchs, patch ** 2 * channels
    jnp.ones((4, 1, 4096, 588), dtype=jnp.bfloat16)
    jax.jit(model)(jnp.ones((4, 1, 4096, 588), dtype=jnp.bfloat16))