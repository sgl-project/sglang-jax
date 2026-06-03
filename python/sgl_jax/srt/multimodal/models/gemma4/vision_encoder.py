"""Gemma4 vision encoder for sglang-jax.

Ported from MaxText ``gemma4_vision.py`` and HF ``Gemma4VisionModel``. Expects
pre-patchified inputs (HF ``Gemma4ImageProcessor`` output) so the encoder itself
is shape-static for a fixed token budget. P0 supports the default 280-token
budget only (2520 patches → 3×3 pool).

No KV cache, no paging — plain bidirectional self-attention. Runs on a single
device by default; sharding annotations are deferred to P2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

PATCH_DIM = 16 * 16 * 3  # 768

# Gemma4 supported token budgets → required input patch count (= budget × pool_k²).
TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def patches_for_budget(budget: int, pool_k: int = 3) -> int:
    return budget * pool_k * pool_k


def select_budget(num_patches: int, pool_k: int = 3) -> int:
    """Smallest budget whose patch capacity ≥ num_patches (pad up)."""
    for b in TOKEN_BUDGETS:
        if patches_for_budget(b, pool_k) >= num_patches:
            return b
    return TOKEN_BUDGETS[-1]


class RMSNorm(nnx.Module):
    """Standard RMSNorm (x * w), matching HF Gemma4RMSNorm. Mesh-free."""

    def __init__(self, dim: int, epsilon: float = 1e-6, use_scale: bool = True, dtype=jnp.bfloat16):
        self.eps = epsilon
        self.dtype = dtype
        self.scale = nnx.Param(jnp.ones((dim,), dtype=dtype)) if use_scale else None

    def __call__(self, x: jax.Array) -> jax.Array:
        x32 = x.astype(jnp.float32)
        normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        if self.scale is not None:
            normed = normed * self.scale.value.astype(jnp.float32)
        return normed.astype(self.dtype)


# --------------------------------------------------------------------------- #
# 2-D multidimensional RoPE
# --------------------------------------------------------------------------- #
def _rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def _apply_rope_1d(x: jax.Array, pos: jax.Array, base_freq: float) -> jax.Array:
    """x: [..., L, H, D_part], pos: [..., L]."""
    dim = x.shape[-1]
    half = dim // 2
    timescale = base_freq ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    sinusoid = pos[..., None, None].astype(jnp.float32) / timescale
    sin = jnp.sin(sinusoid).astype(x.dtype)
    cos = jnp.cos(sinusoid).astype(x.dtype)
    sin = jnp.concatenate([sin, sin], axis=-1)
    cos = jnp.concatenate([cos, cos], axis=-1)
    _ = half  # half computed for clarity; concat above doubles to full dim
    return x * cos + _rotate_half(x) * sin


def apply_2d_rope(x: jax.Array, positions_xy: jax.Array, base_freq: float) -> jax.Array:
    """Split head_dim in two and apply RoPE per spatial axis (x then y).

    x: [..., L, H, D], positions_xy: [..., L, 2].
    """
    d = x.shape[-1]
    x0, x1 = jnp.split(x, 2, axis=-1)
    y0 = _apply_rope_1d(x0, positions_xy[..., 0], base_freq)
    y1 = _apply_rope_1d(x1, positions_xy[..., 1], base_freq)
    out = jnp.concatenate([y0, y1], axis=-1)
    assert out.shape[-1] == d
    return out


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class Gemma4VisionMLP(nnx.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            intermediate_size, hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x))


class Gemma4VisionAttention(nnx.Module):
    def __init__(self, cfg, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_parameters["rope_theta"]
        hidden = cfg.hidden_size
        proj_dim = self.num_heads * self.head_dim

        self.q_proj = nnx.Linear(hidden, proj_dim, use_bias=False, param_dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(hidden, proj_dim, use_bias=False, param_dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(hidden, proj_dim, use_bias=False, param_dtype=dtype, rngs=rngs)
        self.o_proj = nnx.Linear(proj_dim, hidden, use_bias=False, param_dtype=dtype, rngs=rngs)

        self.q_norm = RMSNorm(self.head_dim, epsilon=cfg.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, epsilon=cfg.rms_norm_eps, dtype=dtype)
        self.v_norm = RMSNorm(self.head_dim, epsilon=cfg.rms_norm_eps, use_scale=False, dtype=dtype)

    def __call__(self, x: jax.Array, positions_xy: jax.Array, padding_mask: jax.Array) -> jax.Array:
        b, n, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_norm(self.q_proj(x).reshape(b, n, h, d))
        k = self.k_norm(self.k_proj(x).reshape(b, n, h, d))
        v = self.v_norm(self.v_proj(x).reshape(b, n, h, d))

        q = apply_2d_rope(q, positions_xy, self.rope_theta)
        k = apply_2d_rope(k, positions_xy, self.rope_theta)

        # [B,H,N,D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhnd,bhmd->bhnm", q, k).astype(jnp.float32)
        mask = padding_mask[:, None, None, :]  # [B,1,1,N] mask out padded keys
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1).astype(x.dtype)

        out = jnp.einsum("bhnm,bhmd->bhnd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, h * d)
        return self.o_proj(out)


class Gemma4VisionBlock(nnx.Module):
    def __init__(self, cfg, dtype: jnp.dtype, rngs: nnx.Rngs):
        eps = cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(cfg.hidden_size, epsilon=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, epsilon=eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, epsilon=eps, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, epsilon=eps, dtype=dtype)
        self.self_attn = Gemma4VisionAttention(cfg, dtype, rngs)
        self.mlp = Gemma4VisionMLP(cfg.hidden_size, cfg.intermediate_size, dtype, rngs)

    def __call__(self, x: jax.Array, positions_xy: jax.Array, padding_mask: jax.Array) -> jax.Array:
        h = self.input_layernorm(x)
        h = self.self_attn(h, positions_xy, padding_mask)
        x = x + self.post_attention_layernorm(h)

        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        x = x + self.post_feedforward_layernorm(h)
        return x


class Gemma4VisionPatchEmbedder(nnx.Module):
    """Linear patch projection + learned 2D factorized position embedding."""

    def __init__(self, cfg, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.input_proj = nnx.Linear(
            PATCH_DIM, cfg.hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.position_embedding_table = nnx.Param(
            jnp.zeros((2, cfg.position_embedding_size, cfg.hidden_size), dtype=dtype)
        )

    def __call__(
        self, patches: jax.Array, positions_xy: jax.Array, padding_mask: jax.Array
    ) -> jax.Array:
        x = self.input_proj(patches)
        # Gather per-axis position embeddings then sum. Clamp -1 padding to 0
        # and zero it via padding_mask afterwards.
        clamped = jnp.maximum(positions_xy, 0)
        pe_x = self.position_embedding_table[0][clamped[..., 0]]
        pe_y = self.position_embedding_table[1][clamped[..., 1]]
        pe = (pe_x + pe_y) * padding_mask[..., None].astype(x.dtype)
        return x + pe.astype(x.dtype)


def avg_pool_by_positions(
    x: jax.Array, positions_xy: jax.Array, kernel: int, out_len: int
) -> tuple[jax.Array, jax.Array]:
    """3×3 average pool grouping patches by floor(pos/k). Matches HF Gemma4VisionPooler."""
    max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
    kidx = jnp.floor_divide(positions_xy, kernel)
    flat = kidx[..., 0] + (max_x // kernel) * kidx[..., 1]
    weights = jax.nn.one_hot(flat, out_len, dtype=x.dtype) / (kernel * kernel)
    out = jnp.einsum("bnl,bnd->bld", weights, x)
    mask = jnp.any(weights != 0, axis=1)
    return out, mask


class Gemma4VisionEncoder(nnx.Module):
    """Vision tower: patch embed → N encoder blocks → pool → √d scale → standardize."""

    def __init__(self, cfg, dtype: jnp.dtype = jnp.bfloat16, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.cfg = cfg
        self.dtype = dtype
        self.out_len = cfg.default_output_length
        self.pool_k = cfg.pooling_kernel_size

        self.patch_embedder = Gemma4VisionPatchEmbedder(cfg, dtype, rngs)
        self.layers = nnx.data(
            [Gemma4VisionBlock(cfg, dtype, rngs) for _ in range(cfg.num_hidden_layers)]
        )
        self.std_bias = nnx.Param(jnp.zeros((cfg.hidden_size,), dtype=dtype))
        self.std_scale = nnx.Param(jnp.ones((cfg.hidden_size,), dtype=dtype))

    def __call__(
        self,
        pixel_values: jax.Array,
        pixel_position_ids: jax.Array,
        out_len: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
          pixel_values: [B, num_patches, 768] pre-patchified, range [0, 1].
            num_patches must equal out_len × pool_k² (caller pads with pos=-1).
          pixel_position_ids: [B, num_patches, 2] (x, y), -1 for padding.
          out_len: token budget (70/140/280/560/1120); defaults to config.
        Returns:
          (soft_tokens [B, out_len, hidden], valid_mask [B, out_len]).
        """
        out_len = out_len or self.out_len
        padding_mask = jnp.all(pixel_position_ids != -1, axis=-1)  # [B, N]
        # HF Gemma4 normalizes [0,1] → [-1,1] inside the patch embedder.
        patches = (2.0 * pixel_values - 1.0).astype(self.dtype)

        x = self.patch_embedder(patches, pixel_position_ids, padding_mask)
        for layer in self.layers:
            x = layer(x, pixel_position_ids, padding_mask)

        x, valid_mask = avg_pool_by_positions(x, pixel_position_ids, self.pool_k, out_len)
        x = x * jnp.sqrt(jnp.asarray(self.cfg.hidden_size, x.dtype))
        x = (x - self.std_bias.value.astype(x.dtype)) * self.std_scale.value.astype(x.dtype)
        return x, valid_mask

    def encode_bucketed(
        self, pixel_values: jax.Array, pixel_position_ids: jax.Array
    ) -> tuple[jax.Array, jax.Array, int]:
        """Auto-select token budget and pad patches to the bucket boundary.

        For TPU jit, wrap ``__call__`` with ``jax.jit(static_argnames=['out_len'])``
        so each budget compiles once.
        """
        b, n, _ = pixel_values.shape
        budget = select_budget(n, self.pool_k)
        target_n = patches_for_budget(budget, self.pool_k)
        if n < target_n:
            pad_n = target_n - n
            pixel_values = jnp.pad(pixel_values, ((0, 0), (0, pad_n), (0, 0)))
            pixel_position_ids = jnp.pad(
                pixel_position_ids, ((0, 0), (0, pad_n), (0, 0)), constant_values=-1
            )
        soft, mask = self(pixel_values, pixel_position_ids, out_len=budget)
        return soft, mask, budget


class Gemma4MultimodalEmbedder(nnx.Module):
    """Project pooled vision features into the text embedding space."""

    def __init__(
        self,
        vision_hidden: int,
        text_hidden: int,
        eps: float,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.norm = RMSNorm(vision_hidden, epsilon=eps, use_scale=False, dtype=dtype)
        self.embedding_projection = nnx.Linear(
            vision_hidden, text_hidden, use_bias=False, param_dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.embedding_projection(self.norm(x))


# --------------------------------------------------------------------------- #
# HF safetensors → nnx weight mapping (target_path is the nnx attribute path)
# --------------------------------------------------------------------------- #
def vision_weight_mappings(num_layers: int) -> dict[str, str]:
    """HF key → dotted nnx path. nnx.Linear stores weight as ``kernel`` (in_dim, out_dim)
    so all HF Linear weights need transpose; caller is responsible for that.
    """
    m: dict[str, str] = {
        "model.vision_tower.patch_embedder.input_proj.weight": "patch_embedder.input_proj.kernel",
        "model.vision_tower.patch_embedder.position_embedding_table": "patch_embedder.position_embedding_table",
        "model.vision_tower.std_bias": "std_bias",
        "model.vision_tower.std_scale": "std_scale",
    }
    for i in range(num_layers):
        src = f"model.vision_tower.encoder.layers.{i}"
        dst = f"layers.{i}"
        for ln in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            m[f"{src}.{ln}.weight"] = f"{dst}.{ln}.scale"
        m[f"{src}.self_attn.q_norm.weight"] = f"{dst}.self_attn.q_norm.scale"
        m[f"{src}.self_attn.k_norm.weight"] = f"{dst}.self_attn.k_norm.scale"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m[f"{src}.self_attn.{p}.linear.weight"] = f"{dst}.self_attn.{p}.kernel"
        for p in ("gate_proj", "up_proj", "down_proj"):
            m[f"{src}.mlp.{p}.linear.weight"] = f"{dst}.mlp.{p}.kernel"
    return m
