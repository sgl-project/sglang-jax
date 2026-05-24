# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_vl.py
"""Inference-only Qwen3-VL model compatible with HuggingFace weights (Dense variant).

Pilot scope (#256): Vision Transformer only.
- Qwen3VLVisionPatchEmbed: Conv3D patch embedding with bias
- Qwen3VLVisionBlock: pre-norm LayerNorm + cacheless attention + MLP (single FC1/FC2)
- Qwen3VLVisionPatchMerger: spatial-merge + 2-linear + optional post-shuffle norm
- Qwen3VLVisionModel: 27 blocks + main merger + N deepstack mergers
  Output: [N_merged_patches, out_hidden_size * (1 + num_deepstack)]
"""

from __future__ import annotations

import logging
import math
from functools import partial

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import MRotaryEmbedding, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import QWen3Model
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

_FLASH_MHA = None


def _get_flash_mha():
    global _FLASH_MHA
    if _FLASH_MHA is None:
        from flash_attn_jax import flash_mha as _FLASH_MHA
    return _FLASH_MHA


init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _apply_rotary_pos_emb_vision(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply RoPE to vision attention Q/K.

    Args:
        x: shape [B, T, N, H] (H == head_dim == 2 * rotary_dim)
        cos, sin: shape [T, head_dim] (we use the first half == rotary_dim)
    """
    # Match upstream apply_rotary_pos_emb_native: compute in float32 to avoid
    # bfloat16 accumulation error that destroys 2D spatial RoPE discrimination.
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    half_dim = x.shape[-1] // 2
    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]
    # cos/sin produced by _build_rotary_pos_emb have shape [T, head_dim] but only
    # the first half (rotary_dim) is meaningful; trim to match x_real/x_imag.
    cos = cos[:, :half_dim][None, :, None, :].astype(jnp.float32)
    sin = sin[:, :half_dim][None, :, None, :].astype(jnp.float32)
    out = jnp.concatenate([x_real * cos - x_imag * sin, x_real * sin + x_imag * cos], axis=-1)
    return out.astype(orig_dtype)


_VIT_FLASH_BLOCK_Q = 256


def _vit_flash_attention_tpu(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    segment_ids_1d: jax.Array,
) -> jax.Array:
    """Pallas TPU flash-attention path for the vision tower.

    Layout: q/k/v are [B=1, T, N, H]. Returns the same shape.

    Pads T to a multiple of `_VIT_FLASH_BLOCK_Q` (block_q from the tuned table)
    so the kernel's `q_seq_len % block_q == 0` invariant holds. Padded positions
    get a sentinel segment id that matches no real id; their outputs are sliced
    off after the kernel call. NUM_LANES=128 is automatically satisfied because
    256 is a multiple of 128.

    q/k stay fp32, v is cast to bf16 — matches tuned-table dtype config and
    avoids the bf16-softmax spatial-signal regression that the einsum fallback
    explicitly avoids (see comment in `_vision_attention`).
    """
    from sgl_jax.srt.multimodal.kernels.flash_attention import (
        SegmentIds,
        flash_attention,
    )

    B, T, N, H = q.shape
    assert B == 1, "ViT flash path expects batch size 1"
    orig_dtype = q.dtype

    pad_to = _VIT_FLASH_BLOCK_Q
    T_pad = ((T + pad_to - 1) // pad_to) * pad_to
    pad_len = T_pad - T

    # [B, T, N, H] -> [B, N, T, H] (kernel layout)
    q_t = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)
    k_t = jnp.transpose(k, (0, 2, 1, 3)).astype(jnp.float32)
    v_t = jnp.transpose(v, (0, 2, 1, 3)).astype(jnp.bfloat16)

    if pad_len > 0:
        zeros_qk = jnp.zeros((B, N, pad_len, H), dtype=q_t.dtype)
        zeros_v = jnp.zeros((B, N, pad_len, H), dtype=v_t.dtype)
        q_t = jnp.concatenate([q_t, zeros_qk], axis=2)
        k_t = jnp.concatenate([k_t, zeros_qk], axis=2)
        v_t = jnp.concatenate([v_t, zeros_v], axis=2)

    # Sentinel = -1: never matches any non-negative real segment id, so padded
    # rows are isolated (and their outputs are sliced off below).
    seg = segment_ids_1d.astype(jnp.int32)
    if pad_len > 0:
        seg = jnp.concatenate([seg, jnp.full((pad_len,), -1, dtype=jnp.int32)])
    seg_2d = seg[None, :]  # [B, T_pad]

    out = flash_attention(
        q_t,
        k_t,
        v_t,
        segment_ids=SegmentIds(q=seg_2d, kv=seg_2d),
        sm_scale=scale,
        causal=False,
    )
    if pad_len > 0:
        out = out[:, :, :T, :]
    return jnp.transpose(out, (0, 2, 1, 3)).astype(orig_dtype)


def _vision_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    attn_mask: jax.Array | None = None,
    segment_ids_1d: jax.Array | None = None,
) -> jax.Array:
    """Cacheless full attention for vision tower.

    Args:
        q, k, v: shape [B, T, N, H]
        attn_mask: optional [T, T] bool array. True (i, j) means query i may
            attend to key j. False positions get a large-negative bias added.
            Used to express block-diagonal masking (one image per block) plus
            padding suppression (padded patches belong to no segment).
        segment_ids_1d: optional [T] int32 segment ids (one per token). When
            provided on TPU, routed to the Pallas flash-attention kernel; the
            same-segment mask replaces the dense [T,T] block-diagonal mask and
            keeps memory at O(T * block_q) instead of O(T^2).
    Returns:
        [B, T, N, H]
    """
    if not is_tpu_runtime():
        flash_mha = _get_flash_mha()
        original_dtype = q.dtype
        if q.dtype not in [jnp.bfloat16, jnp.float16]:
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)
        # flash_mha doesn't expose an additive 2D mask; fall back to native for masked path.
        if attn_mask is None:
            output = flash_mha(q, k, v, softmax_scale=scale, is_causal=False)
            if output.dtype != original_dtype:
                output = output.astype(original_dtype)
            return output
        # fall through to native einsum path below

    if is_tpu_runtime() and segment_ids_1d is not None:
        return _vit_flash_attention_tpu(q, k, v, scale, segment_ids_1d)

    B, T, N, H = q.shape
    # Compute attention in float32 to match upstream / HF semantics.
    # Bfloat16 attention softmax destroys spatial signal in multi-row layouts.
    orig_dtype = q.dtype
    q = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)  # [B, N, T, H]
    k = jnp.transpose(k, (0, 2, 1, 3)).astype(jnp.float32)
    v_f32 = jnp.transpose(v, (0, 2, 1, 3)).astype(jnp.float32)
    attn_weights = jnp.einsum("bnth,bnsh->bnts", q, k) * scale
    if attn_mask is not None:
        # Broadcast [T, T] -> [1, 1, T, T]
        neg = jnp.finfo(attn_weights.dtype).min
        bias = jnp.where(attn_mask, 0.0, neg).astype(attn_weights.dtype)
        attn_weights = attn_weights + bias[None, None, :, :]
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    out = jnp.einsum("bnts,bnsh->bnth", attn_weights, v_f32)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(orig_dtype)


class Qwen3VLVisionPatchEmbed(nnx.Module):
    """Conv3D patch embedding (kernel = stride = [Tp, P, P]). bias=True for Qwen3-VL."""

    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [L, C*Tp*P*P]
        L = x.shape[0]
        x = x.reshape(
            L, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # channels-last for nnx.Conv: [L, Tp, P, P, C]
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)  # [L, 1, 1, 1, hidden]
        return x.reshape(L, self.hidden_size)


class Qwen3VLVisionMLP(nnx.Module):
    """Single-path FC1 -> act -> FC2 (no SwiGLU gate; Qwen3-VL uses gelu_pytorch_tanh)."""

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        hidden_act: str,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.linear_fc1 = nnx.Linear(
            in_features, intermediate_size, use_bias=True, param_dtype=dtype, rngs=_rngs
        )
        self.linear_fc2 = nnx.Linear(
            intermediate_size, in_features, use_bias=True, param_dtype=dtype, rngs=_rngs
        )
        self.act_fn = modeling_flax_utils.ACT2FN[hidden_act]

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionAttention(nnx.Module):
    """Fused QKV attention with 2D RoPE; cacheless full attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        _rngs = rngs or nnx.Rngs(0)
        self.qkv = nnx.Linear(
            hidden_size, 3 * hidden_size, use_bias=True, param_dtype=dtype, rngs=_rngs
        )
        self.proj = nnx.Linear(
            hidden_size, hidden_size, use_bias=True, param_dtype=dtype, rngs=_rngs
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attn_mask: jax.Array | None = None,
        segment_ids_1d: jax.Array | None = None,
    ) -> jax.Array:
        # x: [T, B, D] (matches Qwen2.5-VL convention)
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # [T, B, D] -> [B, T, N, H]
        q = q.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        k = k.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        v = v.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        # cos/sin: [T, head_dim] after concat half+half from rotary_dim=head_dim//2 (see top-level builder)
        q = _apply_rotary_pos_emb_vision(q, cos, sin)
        k = _apply_rotary_pos_emb_vision(k, cos, sin)
        out = _vision_attention(
            q, k, v, self.scale, attn_mask=attn_mask, segment_ids_1d=segment_ids_1d
        )
        # [B, T, N, H] -> [T, B, D]
        out = out.transpose(1, 0, 2, 3).reshape(T, B, D)
        return self.proj(out)


class Qwen3VLVisionBlock(nnx.Module):
    """Pre-norm LayerNorm + attention + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_act: str,
        norm_eps: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        # Qwen3-VL ViT uses nn.LayerNorm (NOT RMSNorm)
        norm_layer = partial(
            nnx.LayerNorm,
            epsilon=norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
        )
        self.norm1 = norm_layer(hidden_size, rngs=_rngs)
        self.norm2 = norm_layer(hidden_size, rngs=_rngs)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads, dtype, rngs=rngs)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size, hidden_act, dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attn_mask: jax.Array | None = None,
        segment_ids_1d: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(
            self.norm1(x), cos, sin, attn_mask=attn_mask, segment_ids_1d=segment_ids_1d
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nnx.Module):
    """Spatial-merge + 2-linear projection. Supports post-shuffle norm for deepstack mergers."""

    def __init__(
        self,
        out_dim: int,
        context_dim: int,
        spatial_merge_size: int,
        norm_eps: float,
        dtype: jnp.dtype,
        use_postshuffle_norm: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.merged_dim = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        _rngs = rngs or nnx.Rngs(0)
        norm_features = self.merged_dim if use_postshuffle_norm else context_dim
        self.norm = nnx.LayerNorm(
            norm_features,
            epsilon=norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
            rngs=_rngs,
        )
        self.linear_fc1 = nnx.Linear(
            self.merged_dim, self.merged_dim, use_bias=True, param_dtype=dtype, rngs=_rngs
        )
        self.act_fn = modeling_flax_utils.ACT2FN["gelu"]
        self.linear_fc2 = nnx.Linear(
            self.merged_dim, out_dim, use_bias=True, param_dtype=dtype, rngs=_rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [seq_len, B, context_dim] OR [seq_len, context_dim]; squeeze B if present
        if x.ndim == 3:
            # [T, 1, D] -> [T, D]
            x = x.squeeze(axis=1)
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.merged_dim))
        else:
            x = self.norm(x).reshape(-1, self.merged_dim)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionModel(nnx.Module):
    """Qwen3-VL Vision Transformer with deepstack mergers.

    Forward output shape: [N_merged_patches, out_hidden_size * (1 + num_deepstack)]
    where main features occupy [:, :out_hidden_size] and deepstack features
    occupy [:, out_hidden_size:].
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = config.temporal_patch_size
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.num_position_embeddings = config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        head_dim = self.hidden_size // self.num_heads
        # RoPE applies on half of head_dim (matches upstream rotary_dim=head_dim//2)
        self.head_dim = head_dim
        self.rotary_dim = head_dim // 2
        self.rope_theta = 10000.0

        _rngs = rngs or nnx.Rngs(0)

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            rngs=rngs,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
        )

        # Learned absolute position embedding table (interpolated at runtime).
        # Shape: [num_position_embeddings, hidden_size]
        self.pos_embed = nnx.Embed(
            num_embeddings=self.num_position_embeddings,
            features=self.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.initializer_range), (None, None)
            ),
            rngs=_rngs,
        )

        self.blocks = nnx.List(
            [
                Qwen3VLVisionBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    norm_eps=norm_eps,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            out_dim=config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            norm_eps=norm_eps,
            dtype=dtype,
            use_postshuffle_norm=False,
            rngs=rngs,
        )

        self.deepstack_merger_list = nnx.List(
            [
                Qwen3VLVisionPatchMerger(
                    out_dim=config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    norm_eps=norm_eps,
                    dtype=dtype,
                    use_postshuffle_norm=True,
                    rngs=rngs,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    # ---- rotary position embedding (2D, applied per-(h,w) patch position) ----
    def _rope_inv_freq(self) -> jax.Array:
        # [rotary_dim // 2] -- matches `1 / (theta ** (arange(0, rotary_dim, 2) / rotary_dim))`
        return 1.0 / (
            self.rope_theta
            ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )

    def _rope_cos_sin(self, max_grid: int) -> tuple[jax.Array, jax.Array]:
        inv_freq = self._rope_inv_freq()
        positions = jnp.arange(max_grid, dtype=jnp.float32)
        freqs = jnp.outer(positions, inv_freq)  # [max_grid, rotary_dim//2]
        return jnp.cos(freqs), jnp.sin(freqs)

    def _rot_pos_ids_2d(self, h: int, w: int) -> jax.Array:
        """Compute 2D (h, w) position ids reordered by spatial_merge_size.

        Returns ids of shape [h*w, 2].
        """
        m = self.spatial_merge_size
        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = hpos_ids.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()
        return jnp.stack([hpos_ids, wpos_ids], axis=-1)

    def _build_rotary_pos_emb(
        self, grid_thw: tuple[tuple[int, int, int], ...]
    ) -> tuple[jax.Array, jax.Array]:
        """Build per-token RoPE cos/sin for the full concatenated sequence.

        Returns:
            cos, sin: shape [total_seq_len, head_dim]
              -- formed by interleaving two halves of rotary_dim (split rule:
                 first half = h-axis RoPE, second half = w-axis RoPE).
        """
        max_grid = 0
        for _, h, w in grid_thw:
            max_grid = max(max_grid, h, w)
        cos_base, sin_base = self._rope_cos_sin(max_grid)  # [max_grid, rotary_dim//2]

        cos_list, sin_list = [], []
        for t, h, w in grid_thw:
            pos2d = self._rot_pos_ids_2d(h, w)  # [h*w, 2]
            # Index cos/sin per axis: [h*w, rotary_dim//2] each
            cos_h = cos_base[pos2d[:, 0]]
            cos_w = cos_base[pos2d[:, 1]]
            sin_h = sin_base[pos2d[:, 0]]
            sin_w = sin_base[pos2d[:, 1]]
            # Concatenate h-axis and w-axis halves; final dim = rotary_dim = head_dim
            cos = jnp.concatenate([cos_h, cos_w], axis=-1)
            sin = jnp.concatenate([sin_h, sin_w], axis=-1)
            # Repeat over temporal dim
            if t != 1:
                cos = jnp.tile(cos, (t, 1))
                sin = jnp.tile(sin, (t, 1))
            cos_list.append(cos)
            sin_list.append(sin)
        cos_full = jnp.concatenate(cos_list, axis=0)
        sin_full = jnp.concatenate(sin_list, axis=0)
        # Match flash RoPE convention: head_dim halves are (real, imag). Our concat
        # produces [h_freqs | w_freqs] which serves as the half-split for the
        # _apply_rotary_pos_emb_vision splitter. Pad to head_dim if needed.
        if cos_full.shape[-1] < self.head_dim:
            pad = self.head_dim - cos_full.shape[-1]
            cos_full = jnp.pad(cos_full, ((0, 0), (0, pad)))
            sin_full = jnp.pad(sin_full, ((0, 0), (0, pad)))
        return cos_full.astype(self.dtype), sin_full.astype(self.dtype)

    # ---- learned position embedding with bilinear interpolation ----
    def _interpolate_pos_embed(self, grid_thw: tuple[tuple[int, int, int], ...]) -> jax.Array:
        """Bilinear-interpolate the learned `pos_embed` table to each grid size.

        Returns concatenated tensor of shape [total_seq_len, hidden_size] with the
        spatial_merge_size reordering applied so it aligns with `patch_embed` output
        AFTER its own merge-size reshape.
        """
        side = self.num_grid_per_side
        m = self.spatial_merge_size
        hidden = self.hidden_size

        out_parts = []
        for t, h, w in grid_thw:
            # Compute bilinear-interpolation indices/weights on host (numpy)
            # to keep this static-shaped for JIT-trace safety.
            h_idxs = np.linspace(0, side - 1, h, dtype=np.float32)
            w_idxs = np.linspace(0, side - 1, w, dtype=np.float32)
            h_f = np.floor(h_idxs).astype(np.int64)
            h_c = np.clip(h_f + 1, 0, side - 1)
            dh = (h_idxs - h_f).astype(np.float32)
            w_f = np.floor(w_idxs).astype(np.int64)
            w_c = np.clip(w_f + 1, 0, side - 1)
            dw = (w_idxs - w_f).astype(np.float32)

            # 4 corner indices for each (h, w) pair, flattened
            idx00 = (h_f[:, None] * side + w_f[None, :]).reshape(-1)
            idx01 = (h_f[:, None] * side + w_c[None, :]).reshape(-1)
            idx10 = (h_c[:, None] * side + w_f[None, :]).reshape(-1)
            idx11 = (h_c[:, None] * side + w_c[None, :]).reshape(-1)
            w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
            w01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
            w10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
            w11 = (dh[:, None] * dw[None, :]).reshape(-1)

            idx_all = jnp.asarray(np.stack([idx00, idx01, idx10, idx11], axis=0))
            w_all = jnp.asarray(np.stack([w00, w01, w10, w11], axis=0).astype(np.float32)).astype(
                self.dtype
            )

            # Lookup pos_embed for all 4 corners at once: [4, h*w, hidden]
            embeds = self.pos_embed(idx_all)
            combined = (embeds * w_all[:, :, None]).sum(axis=0)  # [h*w, hidden]

            # Reorder by spatial_merge_size to align with patch_embed's merge-reshape
            combined = combined.reshape(h // m, m, w // m, m, hidden)
            combined = combined.transpose(0, 2, 1, 3, 4).reshape(-1, hidden)

            if t != 1:
                combined = jnp.tile(combined, (t, 1))
            out_parts.append(combined)
        return jnp.concatenate(out_parts, axis=0)

    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thw: tuple[tuple[int, int, int], ...],
        cu_seqlens: jax.Array | None = None,
        n_real_images: int = 0,
    ) -> jax.Array:
        """Run the vision encoder over a (possibly padded) batch of images.

        Args:
            pixel_values:  [N_padded_patches, C*Tp*P*P]  -- bucket-padded; padded
                           rows are zero. Replicated across the mesh.
            grid_thw:      Python tuple of (t, h, w) for ONLY the real images.
                           Used to build RoPE / pos_embed; jit specializes per
                           unique tuple. May be empty if no real image.
            cu_seqlens:    [N_padded_images + 1] int32, block-diagonal bounds in
                           patch units. Entries beyond `n_real_images` repeat the
                           last real cumsum so padded segments have zero length.
            n_real_images: number of real images in the batch (Python int).

        Returns:
            [N_padded_merged_patches, out_hidden_size * (1 + num_deepstack)]
            where the first n_real_merged_patches entries are real and the rest
            are masked-to-zero outputs (and will be scattered into a sink slot).
        """
        x = self.patch_embed(pixel_values)  # [N_padded_patches, hidden]

        # ------------- block-diagonal attention mask -------------
        # Derive a per-token segment id from cu_seqlens via searchsorted on the
        # right boundaries cu_seqlens[1:]. Real-image tokens get id 0..n_real-1;
        # tokens in the padded tail land at n_real_images or above.
        #
        # We materialize TWO views and pick one depending on backend:
        # * `attn_mask` [T, T]: dense bool mask used by the einsum fallback (and
        #   the GPU non-flash path). Padded-key columns are masked off via
        #   `valid[None, :]` so padded tokens contribute zero to real outputs.
        # * `seg_ids_for_kernel` [T]: replaces padded segment ids with -1
        #   (sentinel that never matches any real id) so the Pallas flash
        #   kernel's segment_ids mask reproduces exactly the same masking
        #   semantics as the dense mask above without O(T^2) memory.
        attn_mask = None
        seg_ids_for_kernel = None
        if cu_seqlens is not None:
            T = x.shape[0]
            token_idx = jnp.arange(T, dtype=jnp.int32)
            seg_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
            n_real = jnp.asarray(n_real_images, dtype=jnp.int32)
            valid = seg_ids < n_real  # [T]
            seg_ids_for_kernel = jnp.where(valid, seg_ids, jnp.int32(-1))
            if not is_tpu_runtime():
                attn_mask = (seg_ids[:, None] == seg_ids[None, :]) & valid[None, :]

        # ------------- pos_embed + RoPE (built from static grid_thw) -------------
        # If no real images, skip the static-grid path (would produce empty
        # tensors and break concat). Pure-text branch never hits this code.
        if grid_thw is None or len(grid_thw) == 0:
            # No mm content: ViT outputs zeros at the same shape so the scatter
            # to the sink slot is a no-op. Use the merger to get the right
            # final feature dimensionality.
            zeros_main = jnp.zeros(
                (
                    x.shape[0] // self.spatial_merge_unit,
                    self.merger.linear_fc2.kernel.value.shape[-1],
                ),
                dtype=self.dtype,
            )
            n_ds = len(self.deepstack_visual_indexes)
            zeros_ds = jnp.zeros(
                (zeros_main.shape[0], zeros_main.shape[1] * n_ds), dtype=self.dtype
            )
            return jnp.concatenate([zeros_main, zeros_ds], axis=1)

        pos_embeds_real = self._interpolate_pos_embed(grid_thw)  # [N_real_patches, hidden]
        cos_real, sin_real = self._build_rotary_pos_emb(grid_thw)  # [N_real_patches, head_dim]
        # Pad pos_embed / cos / sin to the full padded length so all 27 blocks
        # see fixed-shape tensors. Padded rows get zero pos_embed and identity
        # RoPE (cos=1, sin=0); the block-diagonal mask already prevents these
        # rows from contributing to any real token's attention output.
        N_padded = x.shape[0]
        N_real = pos_embeds_real.shape[0]
        pad_len = N_padded - N_real
        if pad_len > 0:
            pos_embeds = jnp.concatenate(
                [pos_embeds_real, jnp.zeros((pad_len, self.hidden_size), dtype=self.dtype)],
                axis=0,
            )
            cos = jnp.concatenate(
                [cos_real, jnp.ones((pad_len, self.head_dim), dtype=self.dtype)], axis=0
            )
            sin = jnp.concatenate(
                [sin_real, jnp.zeros((pad_len, self.head_dim), dtype=self.dtype)], axis=0
            )
        else:
            pos_embeds = pos_embeds_real
            cos, sin = cos_real, sin_real

        x = x + pos_embeds

        # Add batch dim for attention path: [T, B=1, D]
        x = jnp.expand_dims(x, axis=1)

        deepstack_features = []
        num_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, attn_mask=attn_mask, segment_ids_1d=seg_ids_for_kernel)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_features.append(self.deepstack_merger_list[num_captured](x))
                num_captured += 1

        main_features = self.merger(x)
        # main + deepstack concatenated along feature dim
        return jnp.concatenate([main_features] + deepstack_features, axis=1)


# ============================================================================
# LLM extension + wrapper
# ============================================================================


class Qwen3VLLanguageModel(QWen3Model):
    """Qwen3 LLM with MRoPE, input_embeds bypass, and deepstack injection."""

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._mrope_interleaved = rope_scaling.get("mrope_interleaved", False)
        if self._mrope_section:
            rope_theta = getattr(config, "rope_theta", 5000000.0)
            max_position_embeddings = getattr(config, "max_position_embeddings", 128000)
            for layer in self.layers:
                head_dim = layer.self_attn.head_dim
                layer.self_attn.rotary_emb = MRotaryEmbedding(
                    head_size=head_dim,
                    rotary_dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                    is_neox_style=True,
                    dtype=dtype,
                    mrope_section=self._mrope_section,
                    mrope_interleaved=self._mrope_interleaved,
                )

        # Which decoder layers receive deepstack injection.
        # Filled by the wrapper after instantiation based on vision_config.
        self.deepstack_embed_to_decoder_layer: list[int] | None = None
        self.deepstack_hidden_size: int | None = None

    def _get_deepstack_slice(
        self,
        layer_idx: int,
        input_deepstack_embeds: jax.Array | None,
    ) -> jax.Array | None:
        """Return the deepstack tensor for the given LLM layer index (or None)."""
        if input_deepstack_embeds is None or self.deepstack_embed_to_decoder_layer is None:
            return None
        if layer_idx not in self.deepstack_embed_to_decoder_layer:
            return None
        # Position in the per-layer list determines the slice offset
        idx = self.deepstack_embed_to_decoder_layer.index(layer_idx)
        h = self.deepstack_hidden_size
        return input_deepstack_embeds[:, idx * h : (idx + 1) * h]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        input_deepstack_embeds: jax.Array | None = None,
    ):
        residual = None
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        rope_positions = (
            forward_batch.mrope_positions
            if self._mrope_section and forward_batch.mrope_positions is not None
            else forward_batch.positions
        )
        layers_kv_fused = []
        layers_callback_flag = []
        for layer_id, layer in enumerate(self.layers):
            # Deepstack injection happens BEFORE the next layer's residual path.
            # Upstream injects deepstack of `layer_idx - 1` into layer `layer_idx`;
            # we mirror that contract by querying the slice for the *previous* layer.
            ds = self._get_deepstack_slice(layer_id - 1, input_deepstack_embeds)
            hidden_states, residual, kv_fused, callback_flag = layer(
                rope_positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                post_residual_addition=ds,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag


class Qwen3VLForConditionalGeneration(nnx.Module):
    """Qwen3-VL Dense for conditional generation.

    Monolithic VLM: vision encoder + LLM live as submodules and run in the same
    forward jit. Plugs into the standard sgl-jax Scheduler/TpWorker/ModelRunner
    pipeline (no stage scheduler).

    Image flow:
        ViT(pixel_values, image_grid_thw) -> [N, out_hidden * (1 + N_ds)]
        split -> main_embeds [N, hidden] + deepstack [N, hidden*N_ds]
        splice main_embeds into input_embeds at placeholder positions
        LLM(input_embeds, ..., input_deepstack_embeds=deepstack) -> logits
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        # `text_config` is the LLM-only config; we wire to Qwen3 LLM with it.
        self.text_config = get_hf_text_config(config) or config.text_config
        self.vision_config = config.vision_config

        # Build LLM (with MRoPE + input_embeds + deepstack hook).
        self.model = Qwen3VLLanguageModel(self.text_config, mesh=mesh, dtype=self.dtype)

        # Wire deepstack metadata into the LLM after construction.
        # Default policy (matches upstream): inject into the *first*
        # `num_deepstack` LLM layers (0, 1, ..., N-1).
        num_deepstack = len(self.vision_config.deepstack_visual_indexes)
        self.model.deepstack_embed_to_decoder_layer = list(range(num_deepstack))
        self.model.deepstack_hidden_size = self.text_config.hidden_size

        # Vision encoder.
        self.visual = Qwen3VLVisionModel(
            config=self.vision_config,
            dtype=self.dtype,
            mesh=mesh,
        )

        # LM head.
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)

        self.image_token_id = getattr(self.config, "image_token_id", 151655)
        self.video_token_id = getattr(self.config, "video_token_id", 151656)

    def separate_deepstack_embeds(self, vision_features: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Split ViT output into (main_embeds, deepstack_embeds)."""
        hidden = self.text_config.hidden_size
        main = vision_features[:, :hidden]
        deepstack = vision_features[:, hidden:]
        return main, deepstack

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        token_to_kv_pool = memory_pools.token_to_kv_pool

        # --- Multimodal splice (extend/prefill only) ---
        # ForwardBatch carries bucket-padded vision tensors as pytree children
        # (pixel_values / placeholder_positions / cu_seqlens) plus a static
        # image_grid_thw (Python tuple) in aux_data. In decode mode these are
        # all None and we fall straight through to the text-only LLM path.
        input_deepstack_embeds = None
        if (
            forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            and forward_batch.pixel_values is not None
        ):
            # 1) ViT: pixel_values (padded) -> [N_padded_image_tokens, hidden*(1+N_ds)]
            vision_features = self.visual(
                forward_batch.pixel_values,
                forward_batch.image_grid_thw,
                cu_seqlens=forward_batch.cu_seqlens,
                n_real_images=forward_batch.n_real_images,
            )
            vision_main, vision_deepstack = self.separate_deepstack_embeds(vision_features)

            # 2) Splice main features into input_embeds at placeholder positions.
            #    placeholder_positions is already replicated and bucket-padded:
            #    real entries point at real image tokens, padded entries point
            #    at the sink slot (last token), where the ViT output is zero so
            #    the overwrite is observationally a no-op.
            input_embeds = self.model.embed_tokens(forward_batch.input_ids)
            # Reshard to replicated for scatter — placeholder_positions and the
            # vision feature tensors are all replicated, and the scatter is
            # simplest on a replicated input_embeds.
            repl = NamedSharding(self.mesh, P())
            input_embeds_repl = jax.sharding.reshard(input_embeds, repl)
            positions = forward_batch.placeholder_positions
            input_embeds_out = input_embeds_repl.at[positions].set(
                vision_main.astype(input_embeds_repl.dtype)
            )

            # 3) Scatter deepstack into a padded zero tensor matching input_embeds
            #    so the LLM's post-residual addition broadcasts cleanly.
            full_deepstack = jnp.zeros(
                (input_embeds_out.shape[0], vision_deepstack.shape[1]),
                dtype=input_embeds_out.dtype,
            )
            full_deepstack = full_deepstack.at[positions].set(
                vision_deepstack.astype(input_embeds_out.dtype)
            )

            forward_batch.input_embedding = input_embeds_out
            input_deepstack_embeds = full_deepstack

        # Fall back to whatever ForwardBatch already carries (text-only or
        # externally-spliced deepstack).
        if input_deepstack_embeds is None:
            input_deepstack_embeds = getattr(forward_batch, "deepstack_visual_embedding", None)

        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch,
            token_to_kv_pool,
            input_deepstack_embeds=input_deepstack_embeds,
        )

        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None

    # ---- weight loading ----
    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3-VL weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        """Build HF Qwen3-VL safetensors -> sgl-jax parameter path mapping.

        HF key layout (Qwen3-VL-8B-Instruct):
          model.language_model.layers.{i}.*   -> sgl-jax model.layers.{i}.*
          model.visual.blocks.{i}.*           -> sgl-jax visual.blocks.{i}.*
          model.visual.merger.*               -> sgl-jax visual.merger.*
          model.visual.deepstack_merger_list.{i}.* -> sgl-jax visual.deepstack_merger_list.{i}.*
          model.visual.patch_embed.proj.*     -> sgl-jax visual.patch_embed.proj.*
          model.visual.pos_embed.weight       -> sgl-jax visual.pos_embed.embedding
          model.language_model.embed_tokens.weight -> sgl-jax model.embed_tokens.embedding
          model.language_model.norm.weight    -> sgl-jax model.norm.scale
          lm_head.weight                       -> sgl-jax lm_head.embedding
        """
        mappings: dict = {}

        # --- LLM: embed_tokens, final norm, lm_head ---
        mappings["model.language_model.embed_tokens.weight"] = WeightMapping(
            target_path="model.embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        )
        mappings["model.language_model.norm.weight"] = WeightMapping(
            target_path="model.norm.scale",
            sharding=(None,),
            transpose=False,
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # --- LLM: per-layer ---
        num_layers = self.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(self._create_llm_layer_mappings(layer_idx))

        # --- Vision: patch_embed, pos_embed, merger ---
        mappings.update(self._create_vision_non_block_mappings())

        # --- Vision: per-block ---
        num_vision_layers = self.vision_config.depth
        for layer_idx in range(num_vision_layers):
            mappings.update(self._create_vision_block_mappings(layer_idx))

        # --- Vision: deepstack mergers ---
        for ds_idx in range(len(self.vision_config.deepstack_visual_indexes)):
            mappings.update(self._create_vision_merger_mappings(ds_idx, deepstack=True))

        return mappings

    def _create_llm_layer_mappings(self, layer_idx: int) -> dict:
        """Map a single LLM decoder layer's HF keys to sgl-jax targets.

        HF prefix:  model.language_model.layers.{i}
        sgl-jax:    model.layers.{i}
        """
        hf_prefix = f"model.language_model.layers.{layer_idx}"
        tgt_prefix = f"model.layers.{layer_idx}"
        mappings = {
            f"{hf_prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{tgt_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{tgt_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{hf_prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{hf_prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{hf_prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{hf_prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{tgt_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{hf_prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{hf_prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }
        if getattr(self.text_config, "attention_bias", False):
            mappings.update(
                {
                    f"{hf_prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{tgt_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        kv_head_padding=False,
                    ),
                    f"{hf_prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{tgt_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        kv_head_padding=True,
                    ),
                    f"{hf_prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{tgt_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        kv_head_padding=True,
                    ),
                }
            )
        return mappings

    def _create_vision_non_block_mappings(self) -> dict:
        """Map vision-tower non-block params: patch_embed, pos_embed, main merger."""
        mappings = {
            # Conv3D weight: PyTorch [out, in, kt, kh, kw] -> JAX [kt, kh, kw, in, out]
            "model.visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "model.visual.patch_embed.proj.bias": WeightMapping(
                target_path="visual.patch_embed.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # Learned absolute position embedding table (nnx.Embed param)
            "model.visual.pos_embed.weight": WeightMapping(
                target_path="visual.pos_embed.embedding",
                sharding=(None, None),
                transpose=False,
            ),
        }
        # Main merger (use_postshuffle_norm=False)
        mappings.update(self._create_vision_merger_mappings(0, deepstack=False))
        return mappings

    def _create_vision_merger_mappings(self, ds_idx: int, deepstack: bool) -> dict:
        """Map weights for one PatchMerger module."""
        if deepstack:
            hf_prefix = f"model.visual.deepstack_merger_list.{ds_idx}"
            tgt_prefix = f"visual.deepstack_merger_list.{ds_idx}"
        else:
            hf_prefix = "model.visual.merger"
            tgt_prefix = "visual.merger"
        return {
            f"{hf_prefix}.norm.weight": WeightMapping(
                target_path=f"{tgt_prefix}.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.norm.bias": WeightMapping(
                target_path=f"{tgt_prefix}.norm.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.linear_fc1.weight": WeightMapping(
                target_path=f"{tgt_prefix}.linear_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.linear_fc1.bias": WeightMapping(
                target_path=f"{tgt_prefix}.linear_fc1.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.linear_fc2.weight": WeightMapping(
                target_path=f"{tgt_prefix}.linear_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.linear_fc2.bias": WeightMapping(
                target_path=f"{tgt_prefix}.linear_fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

    def _create_vision_block_mappings(self, layer_idx: int) -> dict:
        """Map one ViT block: norm1/norm2, attn.qkv (fused), attn.proj, mlp."""
        hf_prefix = f"model.visual.blocks.{layer_idx}"
        tgt_prefix = f"visual.blocks.{layer_idx}"
        return {
            # LayerNorm has both scale and bias
            f"{hf_prefix}.norm1.weight": WeightMapping(
                target_path=f"{tgt_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.norm1.bias": WeightMapping(
                target_path=f"{tgt_prefix}.norm1.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.norm2.weight": WeightMapping(
                target_path=f"{tgt_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.norm2.bias": WeightMapping(
                target_path=f"{tgt_prefix}.norm2.bias",
                sharding=(None,),
                transpose=False,
            ),
            # Fused QKV - replicated, no TP
            f"{hf_prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{tgt_prefix}.attn.qkv.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{tgt_prefix}.attn.qkv.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{tgt_prefix}.attn.proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{tgt_prefix}.attn.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # MLP fc1/fc2 (no SwiGLU gate in Qwen3-VL ViT)
            f"{hf_prefix}.mlp.linear_fc1.weight": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.linear_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.mlp.linear_fc1.bias": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.linear_fc1.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_prefix}.mlp.linear_fc2.weight": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.linear_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{hf_prefix}.mlp.linear_fc2.bias": WeightMapping(
                target_path=f"{tgt_prefix}.mlp.linear_fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }


EntryClass = Qwen3VLForConditionalGeneration
