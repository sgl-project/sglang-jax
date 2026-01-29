# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Vision layers for Qwen3OmniMoe Vision Encoder."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase


class Vision3DPatchEmbed(nnx.Module):
    """
    3D Convolutional Patch Embedding for video/image input.

    Converts input (B, T, H, W, C) into patches (N_patches, embed_dim).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        # 3D Conv: (T, H, W, C_in, C_out)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, T, H, W, C) video/image tensor

        Returns:
            patches: (total_patches, embed_dim)
        """
        # x shape: (B, T, H, W, C)
        # Need to merge batch and temporal for conv
        B, T, H, W, C = x.shape

        # Reshape for 3D conv: combine examples into batch
        # JAX Conv expects: (batch, spatial_dims..., channels)
        x = x.reshape(-1, T, H, W, C)  # (B, T, H, W, C)

        # Apply 3D convolution
        patches = self.proj(x)  # (B, T', H', W', embed_dim)

        # Flatten all spatial dimensions
        patches = patches.reshape(-1, self.embed_dim)  # (B*T'*H'*W', embed_dim)

        return patches


class Vision2DRotaryEmbedding(nnx.Module):
    """2D Rotary Position Embedding for vision transformers."""

    def __init__(
        self,
        dim: int,  # head_dim // 2
        theta: float = 10000.0,
    ):
        self.dim = dim  # head_dim // 2
        self.theta = theta

        # Compute inverse frequencies for dim // 2 positions
        # inv_freq shape: (dim // 2,) = (head_dim // 4,)
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(
        self,
        position_ids: jax.Array,  # (seq_len, 2) - [h_pos, w_pos]
    ) -> jax.Array:
        """
        Compute position embeddings for 2D positions.

        Returns freqs that should be duplicated before computing cos/sin.

        Args:
            position_ids: (seq_len, 2) containing [height_pos, width_pos]

        Returns:
            freqs: (seq_len, dim) with CONCATENATED [h..., w...] structure
                   where dim = head_dim // 2
        """
        h_pos = position_ids[:, 0]
        w_pos = position_ids[:, 1]
        h_freqs = jnp.outer(h_pos, self.inv_freq)
        w_freqs = jnp.outer(w_pos, self.inv_freq)
        freqs = jnp.concatenate([h_freqs, w_freqs], axis=-1)

        return freqs


def apply_rotary_pos_emb_vision(
    q: jax.Array,  # (seq_len, num_heads, head_dim)
    k: jax.Array,  # (seq_len, num_heads, head_dim)
    cos: jax.Array,  # (seq_len, head_dim)
    sin: jax.Array,  # (seq_len, head_dim)
) -> tuple[jax.Array, jax.Array]:
    """Apply 2D rotary position embedding to query and key tensors."""
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class VisionAttention(nnx.Module):
    """
    Multi-head self-attention for vision transformer with 2D RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size * 3,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.o_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.rotary_emb = Vision2DRotaryEmbedding(
            dim=self.head_dim // 2,
            theta=rope_theta,
        )

    def __call__(
        self,
        hidden_states: jax.Array,  # (seq_len, hidden_size)
        position_ids: jax.Array,  # (seq_len, 2)
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Forward pass of vision attention.

        Args:
            hidden_states: (seq_len, hidden_size)
            position_ids: (seq_len, 2) - 2D spatial positions
            attention_mask: Optional attention mask

        Returns:
            output: (seq_len, hidden_size)
        """
        seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)

        # Reshape with explicit sharding for TP compatibility
        qkv = jax.lax.reshape(
            qkv,
            (seq_len, 3, self.num_heads, self.head_dim),
            out_sharding=NamedSharding(self.mesh, P(None, None, "tensor", None)),
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Apply 2D RoPE
        freqs = self.rotary_emb(position_ids)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos, sin = jnp.cos(emb), jnp.sin(emb)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = jnp.einsum("qhd,khd->hqk", q * self.scaling, k)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("hqk,khd->qhd", attn_weights, v)

        # Reshape with explicit sharding for TP compatibility
        attn_output = jax.lax.reshape(
            attn_output,
            (seq_len, self.hidden_size),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        output, _ = self.o_proj(attn_output)

        return output


class VisionMLP(nnx.Module):
    """
    Feed-forward network for vision transformer.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.fc1 = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.fc2 = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        act_fns = {
            "gelu": jax.nn.gelu,
            "gelu_pytorch_tanh": lambda x: jax.nn.gelu(x, approximate=True),
            "relu": jax.nn.relu,
            "silu": jax.nn.silu,
        }
        if hidden_act not in act_fns:
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act = act_fns[hidden_act]

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """
        Args:
            hidden_states: (seq_len, hidden_size)

        Returns:
            output: (seq_len, hidden_size)
        """
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class VisionTransformerBlock(nnx.Module):
    """
    Vision Transformer block with pre-normalization.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
        )

        self.attn = VisionAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            rope_theta=config.rope_theta,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

        self.mlp = VisionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Forward pass of transformer block.

        Args:
            hidden_states: (seq_len, hidden_size)
            position_ids: (seq_len, 2)
            attention_mask: Optional mask

        Returns:
            hidden_states: (seq_len, hidden_size)
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionPatchMerger(nnx.Module):
    """
    Patch merger that projects vision features to language model dimension.

    Projects from hidden_size * (spatial_merge_size^2) to out_hidden_size.

    TP Strategy:
        - mlp_fc1: Column-wise sharding (None, "tensor") - split output dimension
        - mlp_fc2: Row-wise sharding ("tensor", None) - split input dimension, all-reduce output
    """

    def __init__(
        self,
        config,
        use_postshuffle_norm: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.out_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm

        merged_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        norm_size = merged_hidden_size if use_postshuffle_norm else config.hidden_size
        self.ln_q = nnx.LayerNorm(
            num_features=norm_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
        )

        # TP: column-wise sharding
        self.mlp_fc1 = LinearBase(
            input_size=merged_hidden_size,
            output_size=merged_hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),  # TP: column-wise sharding
            params_dtype=dtype,
            mesh=mesh,
        )

        # TP: row-wise sharding
        self.mlp_fc2 = LinearBase(
            input_size=merged_hidden_size,
            output_size=config.out_hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),  # TP: row-wise sharding
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """
        Args:
            hidden_states: (seq_len, hidden_size)

        Returns:
            output: (seq_len, out_hidden_size)
        """
        merged_hidden_size = self.hidden_size * (self.spatial_merge_size**2)

        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)
            hidden_states = self.ln_q(hidden_states)
        else:
            hidden_states = self.ln_q(hidden_states)
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)

        hidden_states, _ = self.mlp_fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states, _ = self.mlp_fc2(hidden_states)

        return hidden_states
