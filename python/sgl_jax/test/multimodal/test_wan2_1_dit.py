import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    FlashAttentionMetadata,
)
from sgl_jax.srt.multimodal.models.wan2_1.diffusion.wan2_1_dit import (
    WanTransformer3DModel,
)

# Ensure the python directory is in the path


class MockConfig:
    def __init__(self):
        self.patch_size = (1, 2, 2)
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_attention_heads = 4
        self.attention_head_dim = 16
        self.in_channels = 4
        self.out_channels = 4  # Output channels (same as input for diffusion models)
        self.freq_dim = 16
        self.text_dim = 32
        self.image_dim = 32
        self.ffn_dim = 128
        self.qk_norm = "rms_norm"
        self.cross_attn_norm = True
        self.epsilon = 1e-6
        self.added_kv_proj_dim = None
        self.num_layers = 30


class MockRequest:
    def __init__(self, config, batch_size=1, seq_len=256):
        self.attention_backend = FlashAttention(
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
        )
        # Mock attention metadata
        metadata = FlashAttentionMetadata()
        metadata.num_seqs = jnp.array([batch_size], dtype=jnp.int32)
        metadata.cu_q_lens = jnp.array([0, seq_len], dtype=jnp.int32)
        metadata.cu_kv_lens = jnp.array([0, seq_len], dtype=jnp.int32)
        metadata.page_indices = jnp.arange(seq_len, dtype=jnp.int32)
        metadata.seq_lens = jnp.array([seq_len], dtype=jnp.int32)
        metadata.distribution = jnp.array([0, batch_size, batch_size], dtype=jnp.int32)
        metadata.custom_mask = None
        self.attention_backend.forward_metadata = metadata


class TestWanTransformer3DModel(unittest.TestCase):
    def test_forward(self):
        config = MockConfig()
        rngs = nnx.Rngs(params=jax.random.key(0))

        model = WanTransformer3DModel(config, rngs=rngs)

        # (batch_size, num_channels, num_frames, height, width) - channel-first format
        input_shape = (1, 4, 1, 32, 32)
        hidden_states = jax.random.normal(jax.random.key(0), input_shape)

        # text embeddings
        encoder_hidden_states = jax.random.normal(jax.random.key(1), (1, 8, 32))  # B, L, D

        # timesteps
        timesteps = jax.numpy.array([1.0])

        # image embeddings
        encoder_hidden_states_image = jax.random.normal(jax.random.key(2), (1, 1, 32))  # B, L, D

        # Run forward (no req needed for diffusion - uses simple attention)
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        # Expected output shape: same as input (B, C, F, H, W) = (1, 4, 1, 32, 32)
        self.assertEqual(output.shape, (1, 4, 1, 32, 32))


if __name__ == "__main__":
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    with jax.set_mesh(mesh):
        unittest.main()
