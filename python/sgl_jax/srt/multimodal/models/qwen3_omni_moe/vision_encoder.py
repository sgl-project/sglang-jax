import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase


class Vision3DPatchEmbed(nnx.Module):
    """
    3D Convolutional Patch Embedding for video/image input.

    Converts flattened input (B, C*T*H*W) into patches (N_patches, embed_dim).
    Input is assumed to be flattened in C-first order: (B, C, T, H, W) -> (B, C*T*H*W).
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
            x: (B, C*T*H*W) flattened video/image tensor in C-first order

        Returns:
            patches: (total_patches, embed_dim)
        """
        # x shape: (B, C*T*H*W)
        # Reshape to patch blocks: (B*N_patches, C, t_patch, p, p)
        # where N_patches = (T/t_patch) * (H/p) * (W/p)
        x = x.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )

        # JAX Conv expects: (batch, spatial_dims..., channels)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # (B*N_patches, t_patch, p, p, C)

        # Apply 3D convolution (outputs 1x1x1 per patch)
        patches = self.proj(x)  # (B*N_patches, 1, 1, 1, embed_dim)

        # Flatten all spatial dimensions
        patches = patches.reshape(-1, self.embed_dim)  # (B*N_patches, embed_dim)

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
        config: Qwen3OmniMoeVisionEncoderConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=1e-6,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=1e-6,
            rngs=rngs,
        )

        self.attn = VisionAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
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
        config: Qwen3OmniMoeVisionEncoderConfig,
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
            epsilon=1e-6,
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


class Qwen3OmniMoeVisionEncoder(nnx.Module):
    """
    Qwen3OmniMoe Vision Encoder for processing images and videos.

    Architecture:
        1. 3D Conv Patch Embedding
        2. Learnable Position Embeddings (interpolated)
        3. 27 Vision Transformer Blocks with 2D RoPE
        4. Deepstack feature extraction at layers 8, 16, 24
        5. Final Patch Merger to language model dimension
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Patch embedding
        self.patch_embed = Vision3DPatchEmbed(
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            rngs=rngs,
        )

        # Learnable position embeddings
        self.pos_embed = Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=(None, None),  # Position embeddings are replicated
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        # Transformer blocks - use nnx.List for Flax NNX 0.12.0+ compatibility
        self.blocks = nnx.List(
            [
                VisionTransformerBlock(
                    config=config,
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.depth)
            ]
        )

        # Deepstack mergers (for intermediate features) - use nnx.List for Flax NNX 0.12.0+ compatibility
        self.deepstack_mergers = nnx.List(
            [
                VisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,  # PyTorch uses True for deepstack mergers
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )

        # Final merger
        self.merger = VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,  # PyTorch uses False for final merger
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

        self.spatial_merge_size = config.spatial_merge_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def interpolate_pos_embed(
        self,
        grid_thw: jax.Array,
    ) -> jax.Array:
        """
        Interpolate position embeddings for variable resolution input.
        Matches PyTorch's fast_pos_embed_interpolate logic.

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            pos_embeds: (total_patches, hidden_size)
        """
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]

        all_pos_embeds = []

        # Process each image/video separately
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # Convert traced values to Python ints for use in shape operations
            t_val = int(t)
            h_val = int(h)
            w_val = int(w)

            # Create interpolation indices
            h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, h_val)
            w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, w_val)

            h_idxs_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_idxs_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_idxs_ceil = jnp.minimum(h_idxs_floor + 1, self.num_grid_per_side - 1)
            w_idxs_ceil = jnp.minimum(w_idxs_floor + 1, self.num_grid_per_side - 1)

            # Compute interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Compute base indices for 2D grid
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Four corners for bilinear interpolation
            idx_tl = base_h[:, None] + w_idxs_floor[None, :]  # Top-left
            idx_tr = base_h[:, None] + w_idxs_ceil[None, :]  # Top-right
            idx_bl = base_h_ceil[:, None] + w_idxs_floor[None, :]  # Bottom-left
            idx_br = base_h_ceil[:, None] + w_idxs_ceil[None, :]  # Bottom-right

            # Flatten indices
            idx_tl = idx_tl.reshape(-1)
            idx_tr = idx_tr.reshape(-1)
            idx_bl = idx_bl.reshape(-1)
            idx_br = idx_br.reshape(-1)

            # Fetch embeddings
            emb_tl = self.pos_embed(idx_tl)  # (h*w, hidden_size)
            emb_tr = self.pos_embed(idx_tr)
            emb_bl = self.pos_embed(idx_bl)
            emb_br = self.pos_embed(idx_br)

            # Bilinear interpolation weights
            dh_2d = dh[:, None]
            dw_2d = dw[None, :]
            w_tl = ((1 - dh_2d) * (1 - dw_2d)).reshape(-1, 1)
            w_tr = ((1 - dh_2d) * dw_2d).reshape(-1, 1)
            w_bl = (dh_2d * (1 - dw_2d)).reshape(-1, 1)
            w_br = (dh_2d * dw_2d).reshape(-1, 1)

            pos_embed = emb_tl * w_tl + emb_tr * w_tr + emb_bl * w_bl + emb_br * w_br

            # Repeat for temporal and apply spatial-merge permutation
            pos_embed = jnp.broadcast_to(
                pos_embed[None, :, :], (t_val, h_val * w_val, pos_embed.shape[-1])
            )
            merge_size = self.config.spatial_merge_size
            pos_embed = pos_embed.reshape(
                t_val,
                h_val // merge_size,
                merge_size,
                w_val // merge_size,
                merge_size,
                self.config.hidden_size,
            )
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, self.config.hidden_size)

            all_pos_embeds.append(pos_embed)

        # Concatenate all position embeddings
        return jnp.concatenate(all_pos_embeds, axis=0)

    def compute_2d_position_ids(
        self,
        grid_thw: jax.Array,
    ) -> jax.Array:
        """
        Compute 2D spatial position IDs for RoPE in spatial-merge order.

        This matches PyTorch's rot_pos_emb which generates positions in spatial-merge order:
        For merge_size=2 and 8x8 grid: (0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3), ...

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            position_ids: (total_patches, 2) - [h_pos, w_pos] in spatial-merge order
        """
        all_pos_ids = []
        merge_size = self.config.spatial_merge_size

        for num_frames, height, width in grid_thw:
            # Convert traced values to Python ints
            num_frames_val = int(num_frames)
            height_val = int(height)
            width_val = int(width)

            merged_h = height_val // merge_size
            merged_w = width_val // merge_size

            # Create spatial-merge order position IDs
            # block_rows: [0, 1, 2, ...] for blocks
            # intra_row: [0, 1] for positions within block
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            # Compute full-resolution positions
            # row_idx: block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            # This creates the pattern: for each (block_h, block_w), iterate (intra_h, intra_w)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            # Expand and flatten
            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size))
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size))
            row_idx = row_idx.reshape(-1)
            col_idx = col_idx.reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)  # (h*w, 2)

            # Repeat for temporal dimension
            if num_frames_val > 1:
                coords = jnp.tile(coords, (num_frames_val, 1))

            all_pos_ids.append(coords)

        return jnp.concatenate(all_pos_ids, axis=0)

    def _apply_spatial_merge_permutation(
        self,
        hidden_states: jax.Array,  # (total_patches, hidden_size)
        grid_thw: jax.Array,  # (num_images, 3) - [T, H_patches, W_patches]
    ) -> jax.Array:
        """
        Apply spatial merge permutation to hidden_states to match pos_embeds order.

        The patch_embed outputs patches in row-major order (T, H, W flattened),
        but position embeddings and position IDs use spatial-merge-permuted order.
        This function reorders hidden_states to match.

        Args:
            hidden_states: (total_patches, hidden_size) in row-major order
            grid_thw: (num_images, 3) containing [T, H_patches, W_patches]

        Returns:
            hidden_states: (total_patches, hidden_size) in spatial-merge-permuted order
        """
        merge_size = self.spatial_merge_size
        all_permuted = []
        offset = 0

        for num_frames, height, width in grid_thw:
            # Convert traced values to Python ints
            num_frames_val = int(num_frames)
            height_val = int(height)
            width_val = int(width)

            num_patches = num_frames_val * height_val * width_val
            patches = hidden_states[offset : offset + num_patches]  # (t*h*w, hidden_size)

            # Reshape to (t, h, w, hidden_size)
            patches = patches.reshape(num_frames_val, height_val, width_val, -1)

            # Apply spatial merge permutation
            # (t, h, w, hidden_size) -> (t, h//merge_size, merge_size, w//merge_size, merge_size, hidden_size)
            patches = patches.reshape(
                num_frames_val,
                height_val // merge_size,
                merge_size,
                width_val // merge_size,
                merge_size,
                -1,
            )

            # Transpose to (t, h//merge_size, w//merge_size, merge_size, merge_size, hidden_size)
            patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))

            # Flatten back to (t*h*w, hidden_size)
            patches = patches.reshape(-1, self.config.hidden_size)

            all_permuted.append(patches)
            offset += num_patches

        return jnp.concatenate(all_permuted, axis=0)

    def _create_attention_mask(self, grid_thw: jax.Array) -> jax.Array:
        """
        Create attention mask to prevent cross-sequence attention.

        IMPORTANT: This matches PyTorch's cu_seqlens behavior where attention is computed
        WITHIN each temporal frame separately. Each (h*w) block of patches forms an
        independent attention group.

        For video with grid_thw = [[2, 8, 8]] (2 temporal patches, 8x8 spatial):
        - Patches 0-63 (t=0) attend only to each other
        - Patches 64-127 (t=1) attend only to each other

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            attention_mask: (total_patches, total_patches) or None if single sequence
                           - 0.0 for valid positions (can attend)
                           - -inf for masked positions (cannot attend)
        """
        # Calculate sequence lengths for each temporal frame
        # This matches PyTorch: cu_seqlens = repeat_interleave(h*w, t).cumsum()
        all_seq_lens = []
        for t, h, w in grid_thw:
            t_val = int(t)
            h_val = int(h)
            w_val = int(w)
            # Each temporal frame has h*w patches as one sequence
            seq_len = h_val * w_val
            for _ in range(t_val):
                all_seq_lens.append(seq_len)

        total_patches = sum(all_seq_lens)

        # Check if we can skip masking (single sequence)
        if len(all_seq_lens) == 1:
            return None

        # Create block diagonal mask matching PyTorch cu_seqlens behavior
        # Initialize with -inf (mask everything by default)
        mask = jnp.full((total_patches, total_patches), -1e10, dtype=jnp.float32)

        offset = 0
        for seq_len in all_seq_lens:
            # Allow attention within the same sequence (set to 0.0)
            mask = mask.at[offset : offset + seq_len, offset : offset + seq_len].set(0.0)
            offset += seq_len

        return mask

    def _validate_input_shapes(self, grid_thw: jax.Array):
        """
        Validate that input dimensions are divisible by spatial_merge_size.

        Args:
            grid_thw: (num_images, 3)
        """
        merge_size = self.spatial_merge_size

        # Check height divisibility
        h_rem = grid_thw[:, 1] % merge_size
        if jnp.any(h_rem != 0):
            raise ValueError(f"Input height must be divisible by spatial_merge_size {merge_size}")

        # Check width divisibility
        w_rem = grid_thw[:, 2] % merge_size
        if jnp.any(w_rem != 0):
            raise ValueError(f"Input width must be divisible by spatial_merge_size {merge_size}")

    def __call__(
        self,
        pixel_values: jax.Array,  # (B, C*T*H*W)
        grid_thw: jax.Array,  # (B, 3) - [T_out, H_patches, W_patches]
    ) -> dict[str, jax.Array]:
        """
        Forward pass of vision encoder.

        Args:
            pixel_values: (B, C*T*H*W) flattened video/image tensor in C-first order
            grid_thw: (B, 3) containing [T_out, H_patches, W_patches] for each input
                where T_out = T / temporal_patch_size

        Returns:
            Dictionary containing:
                - last_hidden_state: (total_patches, hidden_size)
                - pooler_output: (total_patches, out_hidden_size)
                - deepstack_features: List of (total_patches, out_hidden_size)

        Raises:
            ValueError: If input dimensions are not compatible with spatial_merge_size
        """
        # 0. Validate input shapes
        self._validate_input_shapes(grid_thw)

        # 1. Patch Embedding (handles flattened input internally)
        hidden_states = self.patch_embed(pixel_values)  # (total_patches, hidden_size)

        # 2. Apply spatial merge permutation to match PyTorch order
        hidden_states = self._apply_spatial_merge_permutation(hidden_states, grid_thw)

        # 3. Add Position Embeddings (both in spatial-merge order)
        pos_embeds = self.interpolate_pos_embed(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 4. Compute 2D position IDs for RoPE
        position_ids = self.compute_2d_position_ids(grid_thw)

        # 5. Create attention mask to match PyTorch's cu_seqlens behavior
        # Mask is needed when:
        # - batch > 1 (multiple images) OR
        # - any image has T > 1 (video with multiple temporal patches)
        # This ensures each temporal frame attends only within itself
        attention_mask = self._create_attention_mask(grid_thw)

        # 6. Through Transformer Blocks
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # Extract deepstack features
            if layer_idx in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_idx)
                feature = self.deepstack_mergers[idx](hidden_states)
                deepstack_features.append(feature)

        # 7. Final projection (merger)
        merged_hidden_states = self.merger(hidden_states)

        return {
            "last_hidden_state": hidden_states,  # Pre-merger: (total_patches, hidden_size)
            "pooler_output": merged_hidden_states,  # Post-merger: (merged_patches, out_hidden_size)
            "deepstack_features": deepstack_features,
            "encoder_hidden_state": hidden_states,  # Alias for pre-merger output
        }
