"""Unfused compressor tail reference for numerical validation."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.attention.dsv4.compressor import (
    interleaved_rope,
    select_window_fields,
)


def _pool_normalize_rope(
    kv_window,
    score_window,
    valid_mask,
    norm_weight,
    cos,
    sin,
    *,
    rope_head_dim: int,
    norm_eps: float,
):
    """Window softmax-pool, RMSNorm, interleaved RoPE.

    Args:
      kv_window: ``[N, W, D]`` content rows of each record's window.
      score_window: ``[N, W, D]`` score rows.
      valid_mask: ``[N, W]`` False where the window runs off the start of the
        sequence (a record near position 0 pools fewer than W rows).
      cos, sin: ``[N, rope_head_dim//2]``.

    The softmax is per feature over the window axis, and masked entries go to
    ``-inf`` so they contribute nothing -- not zero, which would still take a share
    of the normalisation.
    """
    score_window = jnp.where(valid_mask[..., None], score_window, -jnp.inf)
    weights = jax.nn.softmax(score_window, axis=1)
    pooled = jnp.sum(weights * kv_window, axis=1)
    variance = jnp.mean(jnp.square(pooled), axis=-1, keepdims=True)
    normed = pooled * jax.lax.rsqrt(variance + norm_eps) * jnp.asarray(norm_weight, jnp.float32)
    return interleaved_rope(normed, cos, sin, rope_head_dim)


def compressor_tail_ref(
    combined,
    valid_mask,
    norm_weight,
    cos,
    sin,
    *,
    ratio,
    coff,
    head_dim,
    width,
    rope_head_dim,
    norm_eps,
):
    """Select overlap fields, softmax-pool, normalize and apply interleaved RoPE."""
    offsets = jnp.arange(combined.shape[1])
    kv_window, score_window = select_window_fields(
        combined, offsets, ratio=ratio, coff=coff, head_dim=head_dim, width=width
    )
    return _pool_normalize_rope(
        kv_window,
        score_window,
        valid_mask,
        norm_weight,
        cos,
        sin,
        rope_head_dim=rope_head_dim,
        norm_eps=norm_eps,
    )
