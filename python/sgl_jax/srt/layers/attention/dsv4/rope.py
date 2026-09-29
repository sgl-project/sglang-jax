"""A2 -- DeepSeek-V4's two RoPE tables and the per-layer choice between them.

V4 ships two RoPE bases, and the rule for picking one is **per layer**, not per path
inside a layer::

    rope_theta = compress_rope_theta if compress_ratio > 1 else rope_theta
                 # 160000                                       10000

So a layer with compressed history (CSA ratio 4, HCA ratio 128) uses 160000 for
*everything* it does with RoPE -- the forward rotation on q/k and the inverse
rotation in the output projection -- while a SWA-only layer (ratio 0) uses 10000.
Two tables in total, selected by layer type.

Taken from upstream vLLM `vllm/models/deepseek_v4/common/rope.py`, which is the only
place `compress_rope_theta` is consumed. Worth stating because the natural guess --
"the compressed path needs its own cache alongside the normal one" -- is wrong and
would give a layer two tables where it needs one.

Two further details from the same source, both easy to get wrong:

* ``mscale`` and ``mscale_all_dim`` are set to **0**, i.e. YaRN's magnitude scaling
  is disabled. With sgl-jax's `_deepseek_yarn_get_mscale` that lands on a factor of
  exactly 1.0, so passing 0 is not a no-op by accident -- it is the intended value.
* ``is_neox_style=False``: GPT-J interleaved pairs, and only over the trailing
  ``qk_rope_head_dim`` of each head. The rest of the head is left alone.
"""

from __future__ import annotations

import jax.numpy as jnp

from sgl_jax.srt.layers.attention.dsv4.compressor import interleaved_rope
from sgl_jax.srt.layers.embeddings import YarnRotaryEmbedding

__all__ = [
    "apply_dsv4_partial_rope",
    "build_dsv4_rope",
    "dsv4_rope_tables",
    "dsv4_rope_theta",
]


def dsv4_rope_theta(config, compress_ratio: int) -> float:
    """The RoPE base this layer uses.

    ``compress_ratio > 1`` -- every layer that keeps compressed history -- takes
    ``compress_rope_theta``; ratio 0 and 1 take the plain ``rope_theta``.
    """
    if compress_ratio < 0:
        raise ValueError(f"compress_ratio must be non-negative, got {compress_ratio}")
    if compress_ratio > 1:
        return float(config.compress_rope_theta)
    return float(config.rope_theta)


def build_dsv4_rope(config, compress_ratio: int, *, dtype=jnp.bfloat16):
    """A `YarnRotaryEmbedding` for one layer type.

    Reuses sgl-jax's YaRN rather than reimplementing it; the only V4-specific parts
    are the base selection above and the disabled magnitude scaling.
    """
    scaling = config.rope_scaling or {}
    rope_head_dim = int(config.qk_rope_head_dim)
    return YarnRotaryEmbedding(
        head_size=rope_head_dim,
        rotary_dim=rope_head_dim,
        max_position_embeddings=int(config.max_position_embeddings),
        base=dsv4_rope_theta(config, compress_ratio),
        # GPT-J interleaved pairs, per vLLM's `is_neox_style=False`.
        is_neox_style=False,
        dtype=dtype,
        scaling_factor=float(scaling.get("factor", 1.0)),
        original_max_position_embeddings=int(
            scaling.get("original_max_position_embeddings", config.max_position_embeddings)
        ),
        beta_fast=float(scaling.get("beta_fast", 32.0)),
        beta_slow=float(scaling.get("beta_slow", 1.0)),
        # V4 disables YaRN's magnitude scaling.
        mscale=0.0,
        mscale_all_dim=0.0,
    )


def dsv4_rope_tables(config, *, dtype=jnp.bfloat16) -> dict[str, object]:
    """Both tables, keyed by the layer family that uses them.

    ``"plain"`` serves ratio-0 layers and ``"compressed"`` serves ratio 4 and 128.
    Build these once per model, not per layer: there are only ever two.
    """
    return {
        "plain": build_dsv4_rope(config, 0, dtype=dtype),
        "compressed": build_dsv4_rope(config, 128, dtype=dtype),
    }


def apply_dsv4_partial_rope(x, cos, sin, *, rope_head_dim: int, inverse: bool = False):
    """Rotate only the trailing ``rope_head_dim`` features of each head.

    Args:
      x: ``[..., head_dim]``.
      cos, sin: ``[..., rope_head_dim // 2]``.
      inverse: apply the transposed rotation, i.e. negate ``sin``. The output
        projection needs this to un-rotate attention output before ``wo_a``.

    Delegates to the compressor's `interleaved_rope` so there is one implementation
    of the interleaving; `inverse` is exactly a sign flip on sin, and a test pins
    that forward-then-inverse is the identity.
    """
    if inverse:
        sin = -jnp.asarray(sin)
    return interleaved_rope(x, cos, sin, rope_head_dim)
