"""Qwen3-VL Vision Transformer for SGLang-JAX.

This module implements the vision encoder for Qwen3-VL with DeepStack feature extraction.
Key differences from Qwen2.5-VL:
- DeepStack: Extracts features at intermediate layers for early LLM fusion
- Full attention: No windowed attention in vision encoder
- Simpler position embeddings: Bilinear interpolation + 2D RoPE
"""

import logging
import math
from functools import partial
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen3_vl_config import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

_FLASH_MHA = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fn = nnx.initializers.uniform()


def _get_flash_mha():
    global _FLASH_MHA
    if _FLASH_MHA is None:
        from flash_attn_jax import flash_mha as _FLASH_MHA
    return _FLASH_MHA


class Qwen3_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: tuple[tuple[int, int, int], ...]


class Qwen3_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: jax.Array
    image_grid_thw: jax.Array


Qwen3_VLImageInputs = Qwen3_VLImagePixelInputs | Qwen3_VLImageEmbeddingInputs


def apply_rotary_pos_emb_vision(
    q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply rotary position embeddings to query and key for vision.

    Uses the rotate_half formulation: x * cos + rotate_half(x) * sin
    """
    # q, k: (seq, heads, head_dim)
    # cos, sin: (seq, head_dim)
    half_dim = q.shape[-1] // 2

    # Split into real and imaginary parts
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    # cos, sin need to be (seq, 1, half_dim) for broadcasting
    cos = cos[:, None, :]  # (seq, 1, head_dim)
    sin = sin[:, None, :]
    cos1, cos2 = cos[..., :half_dim], cos[..., half_dim:]
    sin1, sin2 = sin[..., :half_dim], sin[..., half_dim:]

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    q_rot = jnp.concatenate([q1 * cos1 - q2 * sin1, q2 * cos2 + q1 * sin2], axis=-1)
    k_rot = jnp.concatenate([k1 * cos1 - k2 * sin1, k2 * cos2 + k1 * sin2], axis=-1)

    return q_rot, k_rot


def vision_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
) -> jax.Array:
    """Compute vision attention.

    Full attention (no windowing) for Qwen3-VL.

    Args:
        q, k, v: Input tensors of shape [seq_len, num_heads, head_dim]
        scale: Attention scale factor (1/sqrt(head_dim))

    Returns:
        Output tensor of shape [seq_len, num_heads, head_dim]
    """
    if not is_tpu_runtime():
        # GPU: use flash_mha with batch dim
        flash_mha = _get_flash_mha()
        original_dtype = q.dtype
        if q.dtype not in [jnp.bfloat16, jnp.float16]:
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        # Add batch dimension for flash_mha: (1, seq, heads, dim)
        q = q[None, :, :, :]
        k = k[None, :, :, :]
        v = v[None, :, :, :]

        output = flash_mha(q, k, v, softmax_scale=scale, is_causal=False)
        output = output[0]  # Remove batch dim

        if output.dtype != original_dtype:
            output = output.astype(original_dtype)
        return output
    else:
        # TPU: native attention
        # Transpose to (heads, seq, dim) for matmul
        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * scale
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        output = jnp.matmul(attn_weights, v)

        # Transpose back to (seq, heads, dim)
        return jnp.transpose(output, (1, 0, 2))


class Qwen3_VLVisionPatchEmbed(nnx.Module):
    """3D Convolutional patch embedding for vision input."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ) -> None:
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)

        self.proj = nnx.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (num_patches, in_channels * temporal_patch_size * patch_size * patch_size)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size * self.patch_size)

        # Reshape to (L, C, T, H, W) then transpose to (L, T, H, W, C) for Conv
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Apply conv: (L, T, H, W, C) -> (L, 1, 1, 1, hidden_size)
        x = self.proj(x)
        return x.reshape(L, self.hidden_size)


class Qwen3_VLVisionRotaryEmbedding(nnx.Module):
    """Rotary position embedding for vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seq_len: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs


class Qwen3_VLVisionMLP(nnx.Module):
    """Vision encoder MLP with GELU activation."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.linear_fc1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.linear_fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear_fc1(x)
        x = nnx.gelu(x, approximate=True)
        x = self.linear_fc2(x)
        return x


class Qwen3_VLVisionAttention(nnx.Module):
    """Vision encoder multi-head attention with RoPE."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        _rngs = rngs or nnx.Rngs(0)
        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        seq_len = hidden_states.shape[0]
        cos, sin = position_embeddings

        # QKV projection: (seq, hidden) -> (seq, 3, heads, head_dim)
        qkv = self.qkv_proj(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (seq, heads, head_dim)

        # Apply RoPE
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Compute attention
        out = vision_attention(q, k, v, self.scale)

        # Reshape and project
        out = out.reshape(seq_len, -1)
        return self.proj(out)


class Qwen3_VLVisionBlock(nnx.Module):
    """Single transformer block for vision encoder."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.norm1 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=_rngs,
        )
        self.norm2 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=_rngs,
        )
        self.attn = Qwen3_VLVisionAttention(config, dtype=dtype, rngs=rngs)
        self.mlp = Qwen3_VLVisionMLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_VLVisionPatchMerger(nnx.Module):
    """Merge spatial patches after vision encoding.

    Two modes:
    - use_postshuffle_norm=False: For main merger (norm before shuffle)
    - use_postshuffle_norm=True: For DeepStack mergers (norm after shuffle)
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        use_postshuffle_norm: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.spatial_merge_size = config.spatial_merge_size
        merge_factor = config.spatial_merge_size**2
        self.hidden_merged = config.hidden_size * merge_factor
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.hidden_merged if use_postshuffle_norm else config.hidden_size

        _rngs = rngs or nnx.Rngs(0)
        self.norm = nnx.LayerNorm(
            norm_dim,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=_rngs,
        )
        self.linear_fc1 = nnx.Linear(
            self.hidden_merged,
            self.hidden_merged,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.linear_fc2 = nnx.Linear(
            self.hidden_merged,
            config.out_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.use_postshuffle_norm:
            x = self.norm(x)

        # Reshape to merge spatial patches
        merge_factor = self.spatial_merge_size**2
        n_patches = x.shape[0] // merge_factor
        x = x.reshape(n_patches, -1)

        if self.use_postshuffle_norm:
            x = self.norm(x)

        x = self.linear_fc1(x)
        x = nnx.gelu(x)
        x = self.linear_fc2(x)
        return x


class Qwen3_VL_VisionTransformer(nnx.Module):
    """Complete vision transformer with DeepStack feature extraction."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.config = config
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2

        self.patch_embed = Qwen3_VLVisionPatchEmbed(config, dtype=dtype, rngs=rngs)

        # Position embedding (learnable, for interpolation)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        _rngs = rngs or nnx.Rngs(0)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            param_dtype=dtype,
            rngs=_rngs,
        )

        # Rotary embeddings
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_VLVisionRotaryEmbedding(head_dim // 2, theta=config.rope_theta)

        # Transformer blocks
        self.blocks = nnx.List(
            [Qwen3_VLVisionBlock(config, dtype=dtype, rngs=rngs) for _ in range(config.depth)]
        )

        # Main merger
        self.merger = Qwen3_VLVisionPatchMerger(
            config,
            use_postshuffle_norm=False,
            dtype=dtype,
            rngs=rngs,
        )

        # DeepStack mergers (extract at intermediate layers)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nnx.List(
            [
                Qwen3_VLVisionPatchMerger(
                    config,
                    use_postshuffle_norm=True,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    def _fast_pos_embed_interpolate(self, grid_thw: tuple[tuple[int, int, int], ...]) -> jax.Array:
        """Bilinear interpolation for position embeddings."""
        all_pos_embeds = []

        for t, h, w in grid_thw:
            # Create interpolation indices
            h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, w)

            h_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_ceil = jnp.clip(h_floor + 1, 0, self.num_grid_per_side - 1)
            w_ceil = jnp.clip(w_floor + 1, 0, self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # 2D grid indices for 4 corners
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            idx00 = (base_h[:, None] + w_floor[None, :]).flatten()
            idx01 = (base_h[:, None] + w_ceil[None, :]).flatten()
            idx10 = (base_h_ceil[:, None] + w_floor[None, :]).flatten()
            idx11 = (base_h_ceil[:, None] + w_ceil[None, :]).flatten()

            # Weights for bilinear interpolation
            w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
            w01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
            w10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
            w11 = (dh[:, None] * dw[None, :]).flatten()

            # Lookup and interpolate
            pos_embeds = (
                self.pos_embed(idx00) * w00[:, None]
                + self.pos_embed(idx01) * w01[:, None]
                + self.pos_embed(idx10) * w10[:, None]
                + self.pos_embed(idx11) * w11[:, None]
            )

            # Repeat for temporal dimension and apply spatial merge permutation
            pos_embeds = pos_embeds.reshape(h, w, -1)
            if t > 1:
                pos_embeds = jnp.tile(pos_embeds[None], (t, 1, 1, 1))
            else:
                pos_embeds = pos_embeds[None]

            # Permute for spatial merge
            merge_size = self.spatial_merge_size
            merged_h, merged_w = h // merge_size, w // merge_size
            pos_embeds = pos_embeds.reshape(t, merged_h, merge_size, merged_w, merge_size, -1)
            pos_embeds = pos_embeds.transpose(0, 1, 3, 2, 4, 5)
            pos_embeds = pos_embeds.reshape(-1, pos_embeds.shape[-1])

            all_pos_embeds.append(pos_embeds)

        return jnp.concatenate(all_pos_embeds, axis=0)

    def _rot_pos_emb(
        self, grid_thw: tuple[tuple[int, int, int], ...]
    ) -> tuple[jax.Array, jax.Array]:
        """Compute rotary position embeddings."""
        merge_size = self.spatial_merge_size
        all_embeddings = []

        for grid_t, grid_h, grid_w in grid_thw:
            merged_h, merged_w = grid_h // merge_size, grid_w // merge_size

            # Compute position indices for each patch
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            # Full resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = jnp.broadcast_to(
                row_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)
            col_idx = jnp.broadcast_to(
                col_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)

            # Repeat for temporal dimension
            if grid_t > 1:
                row_idx = jnp.tile(row_idx, grid_t)
                col_idx = jnp.tile(col_idx, grid_t)

            # Create frequency table
            max_hw = max(grid_h, grid_w)
            head_dim = self.config.hidden_size // self.config.num_heads
            freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, rotary_dim//2)

            # Lookup embeddings
            row_emb = freq_table[row_idx]
            col_emb = freq_table[col_idx]

            # Concatenate and double
            emb = jnp.concatenate([row_emb, col_emb], axis=-1)
            emb = jnp.concatenate([emb, emb], axis=-1)

            all_embeddings.append(emb)

        all_emb = jnp.concatenate(all_embeddings, axis=0)
        cos = jnp.cos(all_emb)
        sin = jnp.sin(all_emb)
        return cos, sin

    def __call__(
        self,
        hidden_states: jax.Array,
        grid_thw: tuple[tuple[int, int, int], ...],
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Forward pass through vision transformer.

        Args:
            hidden_states: Flattened pixel values (num_patches, patch_dim)
            grid_thw: Grid dimensions for each image/video

        Returns:
            Tuple of (merged_features, deepstack_features)
        """
        hidden_states = self.patch_embed(hidden_states)
        seq_len = hidden_states.shape[0]

        # Position embeddings with bilinear interpolation
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds[:seq_len]

        # RoPE embeddings
        cos, sin = self._rot_pos_emb(grid_thw)
        position_embeddings = (cos[:seq_len], sin[:seq_len])

        # Process through transformer blocks
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings)

            # Extract DeepStack features at specified layers
            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = list(self.deepstack_visual_indexes).index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))

        # Final merger
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_features


class Qwen3_VL_VisionModel(nnx.Module):
    """Qwen3-VL Vision Model with weight loading support."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.mesh = mesh
        self.visual = Qwen3_VL_VisionTransformer(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        logger.info("Qwen3_VL_VisionModel initialized with dtype %s", dtype)

    def load_weights(self, model_config) -> None:
        """Load model weights from safetensors."""
        if not hasattr(self, "text_embed"):
            self.text_embed = Embed(
                num_embeddings=model_config.vocab_size,
                features=model_config.text_hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=(None, None),
                mesh=self.mesh,
            )

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_vision_weight_mappings()

        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)

        logger.info("Qwen3-VL Vision weights loaded successfully!")

    def _create_vision_weight_mappings(self) -> dict:
        """Create weight mappings for vision encoder."""
        mappings = {}

        # Text embedding (for multimodal fusion later)
        mappings["model.embed_tokens.weight"] = WeightMapping(
            target_path="text_embed.embedding",
            sharding=(None, None),
            transpose=False,
        )

        # Patch embedding (Conv3D)
        mappings["model.visual.patch_embed.proj.weight"] = WeightMapping(
            target_path="visual.patch_embed.proj.kernel",
            sharding=(None, None, None, None, None),
            transpose_axes=(2, 3, 4, 1, 0),
        )
        mappings["model.visual.patch_embed.proj.bias"] = WeightMapping(
            target_path="visual.patch_embed.proj.bias",
            sharding=(None,),
            transpose=False,
        )

        # Position embedding
        mappings["model.visual.pos_embed.weight"] = WeightMapping(
            target_path="visual.pos_embed.embedding",
            sharding=(None, None),
            transpose=False,
        )

        # Main merger
        mappings["model.visual.merger.norm.weight"] = WeightMapping(
            target_path="visual.merger.norm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["model.visual.merger.norm.bias"] = WeightMapping(
            target_path="visual.merger.norm.bias",
            sharding=(None,),
            transpose=False,
        )
        mappings["model.visual.merger.linear_fc1.weight"] = WeightMapping(
            target_path="visual.merger.linear_fc1.kernel",
            sharding=(None, None),
            transpose=True,
        )
        mappings["model.visual.merger.linear_fc1.bias"] = WeightMapping(
            target_path="visual.merger.linear_fc1.bias",
            sharding=(None,),
            transpose=False,
        )
        mappings["model.visual.merger.linear_fc2.weight"] = WeightMapping(
            target_path="visual.merger.linear_fc2.kernel",
            sharding=(None, None),
            transpose=True,
        )
        mappings["model.visual.merger.linear_fc2.bias"] = WeightMapping(
            target_path="visual.merger.linear_fc2.bias",
            sharding=(None,),
            transpose=False,
        )

        # DeepStack mergers
        for i in range(len(self.visual.deepstack_visual_indexes)):
            prefix = f"model.visual.deepstack_merger_list.{i}"
            target_prefix = f"visual.deepstack_merger_list.{i}"

            mappings[f"{prefix}.norm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.norm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.norm.bias"] = WeightMapping(
                target_path=f"{target_prefix}.norm.bias",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.linear_fc1.weight"] = WeightMapping(
                target_path=f"{target_prefix}.linear_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.linear_fc1.bias"] = WeightMapping(
                target_path=f"{target_prefix}.linear_fc1.bias",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.linear_fc2.weight"] = WeightMapping(
                target_path=f"{target_prefix}.linear_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.linear_fc2.bias"] = WeightMapping(
                target_path=f"{target_prefix}.linear_fc2.bias",
                sharding=(None,),
                transpose=False,
            )

        # Vision blocks
        num_layers = self.config.depth
        for layer_idx in range(num_layers):
            mappings.update(self._create_vision_block_mappings(layer_idx))

        return mappings

    def _create_vision_block_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single vision block."""
        prefix = f"model.visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"

        return {
            # Layer norms
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm1.bias": WeightMapping(
                target_path=f"{target_prefix}.norm1.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm2.bias": WeightMapping(
                target_path=f"{target_prefix}.norm2.bias",
                sharding=(None,),
                transpose=False,
            ),
            # QKV projection
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # Output projection
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # MLP
            f"{prefix}.mlp.linear_fc1.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.linear_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.linear_fc1.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.linear_fc1.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.linear_fc2.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.linear_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.linear_fc2.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.linear_fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> jax.Array:
        if isinstance(mm_input, list):
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        if hasattr(mm_input, "ndim"):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

    def _parse_and_validate_image_input(
        self,
        image_grid_thw: tuple[tuple[int, int, int], ...],
        **kwargs: object,
    ) -> Qwen3_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
            return Qwen3_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        return None

    def _parse_and_validate_multimodal_inputs(
        self,
        image_grid_thw: tuple[tuple[int, int, int], ...],
        **kwargs: object,
    ) -> dict:
        mm_input_by_modality = {}

        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    image_grid_thw, **kwargs
                )
        return mm_input_by_modality

    def get_single_image_embedding(
        self,
        image_pixel_values: jax.Array,
        image_grid_thw: tuple[int, int, int],
    ) -> jax.Array:
        hidden_states, _ = self.visual(image_pixel_values, (image_grid_thw,))
        return hidden_states

    def _process_image_input(
        self,
        image_input: Qwen3_VLImageInputs,
    ) -> tuple[jax.Array, ...]:
        grid_thw = image_input["image_grid_thw"]

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = []
            current_idx = 0
            for image_thw in grid_thw:
                t, h, w = image_thw
                image_size = t * h * w
                end_idx = current_idx + image_size
                image_pixel_values = pixel_values[current_idx:end_idx, :]
                image_embeds.append(self.get_single_image_embedding(image_pixel_values, image_thw))
                current_idx = end_idx
            image_embeds = jnp.concatenate(image_embeds, axis=0)

        merge_size = self.visual.config.spatial_merge_size
        sizes = np.prod(np.array(grid_thw, dtype=np.int64), axis=-1) // merge_size // merge_size

        if sizes.size == 0:
            return ()
        if sizes.size == 1:
            return (image_embeds,)

        split_indices = np.cumsum(sizes)[:-1]
        return tuple(jnp.split(image_embeds, split_indices))

    def get_multimodal_embeddings(
        self,
        image_grid_thw: tuple[tuple[int, int, int], ...],
        **kwargs: object,
    ) -> list[jax.Array]:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[jax.Array, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings

        return list(multimodal_embeddings)

    def __call__(
        self,
        pixel_values: jax.Array,
        image_grid_thw: tuple[tuple[int, int, int], ...] = None,
        video_grid_thw: tuple[tuple[int, int, int], ...] = None,
    ) -> jax.Array:
        """Encode vision inputs to embeddings.

        Args:
            pixel_values: Pixel values [num_patches, channels * patch_size^2]
            image_grid_thw: Grid dimensions for each image
            video_grid_thw: Grid dimensions for each video

        Returns:
            Vision embeddings [total_patches, hidden_dim]
        """
        combined_grid_thw = []
        if image_grid_thw:
            combined_grid_thw.extend(image_grid_thw)
        if video_grid_thw:
            combined_grid_thw.extend(video_grid_thw)

        if not combined_grid_thw:
            return jnp.zeros((0, self.config.hidden_size), dtype=pixel_values.dtype)

        combined_grid_thw = tuple(combined_grid_thw)
        vision_embeds_list = self.get_multimodal_embeddings(
            image_grid_thw=combined_grid_thw,
            pixel_values=pixel_values,
        )
        return jnp.concatenate(vision_embeds_list, axis=0)
