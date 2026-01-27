"""
Vision layers for Qwen3OmniMoe Vision Encoder.

Includes:
- Vision3DPatchEmbed: 3D convolution for patch embedding
- Vision2DRotaryEmbedding: 2D rotary position embedding
- VisionAttention: Multi-head self-attention with 2D RoPE
- VisionMLP: Feed-forward network
- VisionTransformerBlock: Complete transformer block
- VisionPatchMerger: Patch merging layer
"""

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
    """
    2D Rotary Position Embedding for vision transformers.

    PyTorch compatible implementation:
    - Initialized with dim = head_dim // 2
    - Returns freqs of shape (seq_len, head_dim // 2) with CONCATENATED [h..., w...] structure
    - Caller duplicates to get (seq_len, head_dim) before computing cos/sin
    """

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
        # position_ids: (seq_len, 2)
        h_pos = position_ids[:, 0]  # (seq_len,)
        w_pos = position_ids[:, 1]  # (seq_len,)

        # Compute frequencies for height and width
        # (seq_len,) x (dim // 2,) -> (seq_len, dim // 2)
        h_freqs = jnp.outer(h_pos, self.inv_freq)  # (seq_len, head_dim // 4)
        w_freqs = jnp.outer(w_pos, self.inv_freq)  # (seq_len, head_dim // 4)

        # Concatenate height and width frequencies: [h0, h1, ..., w0, w1, ...]
        # This matches PyTorch's flatten(1) behavior
        freqs = jnp.concatenate([h_freqs, w_freqs], axis=-1)  # (seq_len, head_dim // 2)

        return freqs


def apply_rotary_pos_emb_vision(
    q: jax.Array,  # (seq_len, num_heads, head_dim)
    k: jax.Array,  # (seq_len, num_heads, head_dim)
    cos: jax.Array,  # (seq_len, head_dim)
    sin: jax.Array,  # (seq_len, head_dim)
) -> tuple[jax.Array, jax.Array]:
    """
    Apply 2D rotary position embedding to query and key tensors.

    PyTorch compatible implementation:
    - cos/sin have shape (seq_len, head_dim)
    - Apply standard RoPE formula: q * cos + rotate_half(q) * sin

    Args:
        q: Query tensor (seq_len, num_heads, head_dim)
        k: Key tensor (seq_len, num_heads, head_dim)
        cos: Cosine embeddings (seq_len, head_dim)
        sin: Sine embeddings (seq_len, head_dim)

    Returns:
        Rotated q and k
    """
    # Expand cos/sin for broadcasting with num_heads dimension
    cos = cos[:, None, :]  # (seq_len, 1, head_dim)
    sin = sin[:, None, :]  # (seq_len, 1, head_dim)

    # Helper function: rotate half the hidden dims
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    # Apply rotation using standard RoPE formula
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
        self.mesh = mesh  # Store mesh for TP sharding constraints

        # QKV projection (combined)
        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size * 3,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        # Output projection
        self.o_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        # Rotary embedding (PyTorch compatible: pass head_dim // 2)
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

        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)  # (seq_len, hidden_size * 3)

        # Reshape and split with explicit sharding for TP compatibility
        # QKV output is sharded on last dim (hidden_size * 3), after reshape
        # the num_heads dimension should be sharded
        # Use jax.lax.reshape with out_sharding to handle sharded tensor reshape
        qkv = jax.lax.reshape(
            qkv,
            (seq_len, 3, self.num_heads, self.head_dim),
            out_sharding=NamedSharding(self.mesh, P(None, None, "tensor", None)),
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Each: (seq_len, num_heads, head_dim) with num_heads sharded

        # Apply 2D RoPE (PyTorch compatible)
        freqs = self.rotary_emb(position_ids)  # (seq_len, head_dim // 2)
        # Duplicate freqs to get full head_dim
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # (seq_len, head_dim)
        # Compute cos and sin
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Scaled dot-product attention
        # q: (seq_len, num_heads, head_dim)
        q = q * self.scaling

        # Compute attention scores
        # (seq_len, num_heads, head_dim) @ (seq_len, num_heads, head_dim).T
        # -> (num_heads, seq_len, seq_len)
        attn_weights = jnp.einsum("qhd,khd->hqk", q, k)

        # Apply attention mask if provided
        # attention_mask: (seq_len, seq_len) with 0.0 for valid, -inf for masked
        # Will be broadcast to (num_heads, seq_len, seq_len)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # Broadcasting applies

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        # (num_heads, seq_len, seq_len) @ (seq_len, num_heads, head_dim)
        # -> (seq_len, num_heads, head_dim)
        attn_output = jnp.einsum("hqk,khd->qhd", attn_weights, v)

        # Reshape and project with explicit sharding for TP compatibility
        # attn_output has num_heads sharded, after reshape the last dim should be sharded
        # Use jax.lax.reshape with out_sharding to handle sharded tensor reshape
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

        # Activation function
        if hidden_act == "gelu":
            self.act = jax.nn.gelu
        elif hidden_act == "gelu_pytorch_tanh":
            # Approximate GELU (PyTorch tanh version)
            self.act = lambda x: jax.nn.gelu(x, approximate=True)
        elif hidden_act == "relu":
            self.act = jax.nn.relu
        elif hidden_act == "silu":
            self.act = jax.nn.silu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

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
        # Pre-normalization
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

        # Attention
        self.attn = VisionAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            rope_theta=config.rope_theta,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

        # MLP
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
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
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

        # Input dimension after spatial merging
        merged_hidden_size = config.hidden_size * (config.spatial_merge_size**2)

        # Layer norm
        norm_size = merged_hidden_size if use_postshuffle_norm else config.hidden_size
        self.ln_q = nnx.LayerNorm(
            num_features=norm_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
        )

        # MLP: Linear -> GELU -> Linear
        # TP: Column-wise sharding for fc1 (split output dimension)
        self.mlp_fc1 = LinearBase(
            input_size=merged_hidden_size,
            output_size=merged_hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),  # TP: column-wise sharding
            params_dtype=dtype,
            mesh=mesh,
        )

        # TP: Row-wise sharding for fc2 (split input dimension, all-reduce output)
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
        # Reshape for spatial merging
        # (seq_len, hidden_size) -> (seq_len, hidden_size * spatial_merge_size^2)
        merged_hidden_size = self.hidden_size * (self.spatial_merge_size**2)

        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)
            hidden_states = self.ln_q(hidden_states)
        else:
            hidden_states = self.ln_q(hidden_states)
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)

        # MLP
        hidden_states, _ = self.mlp_fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states, _ = self.mlp_fc2(hidden_states)

        return hidden_states
