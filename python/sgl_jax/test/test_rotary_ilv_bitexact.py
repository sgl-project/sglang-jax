"""The interleaved-domain GPT-J rotary (SGLANG_ROTARY_ILV_FIX, default on) must be
bit-identical to the strided-slice formulation, and the neox path must be untouched."""

import importlib
import os

import jax
import jax.numpy as jnp
import pytest

import sgl_jax.srt.layers.embeddings as embeddings


def _apply_with_flag(flag: str, x, cos, sin, is_neox_style: bool):
    old = os.environ.get("SGLANG_ROTARY_ILV_FIX")
    os.environ["SGLANG_ROTARY_ILV_FIX"] = flag
    try:
        mod = importlib.reload(embeddings)
        return mod.apply_rotary_emb(x, cos, sin, is_neox_style)
    finally:
        if old is None:
            os.environ.pop("SGLANG_ROTARY_ILV_FIX", None)
        else:
            os.environ["SGLANG_ROTARY_ILV_FIX"] = old
        importlib.reload(embeddings)


@pytest.mark.parametrize("shape", [(64, 32, 64), (64, 1, 64), (7, 4, 64)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_gptj_interleaved_bitexact(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype=dtype)
    cos = jax.random.normal(key, (shape[0], shape[-1] // 2), dtype=dtype)
    sin = jax.random.normal(key, (shape[0], shape[-1] // 2), dtype=dtype)
    ref = _apply_with_flag("0", x, cos, sin, is_neox_style=False)
    fix = _apply_with_flag("1", x, cos, sin, is_neox_style=False)
    assert ref.shape == fix.shape and ref.dtype == fix.dtype
    assert bool(jnp.all(ref == fix)), "interleaved rotary must be bit-identical"


def test_neox_path_untouched():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (16, 4, 64), dtype=jnp.bfloat16)
    cos = jax.random.normal(key, (16, 32), dtype=jnp.bfloat16)
    sin = jax.random.normal(key, (16, 32), dtype=jnp.bfloat16)
    ref = _apply_with_flag("0", x, cos, sin, is_neox_style=True)
    fix = _apply_with_flag("1", x, cos, sin, is_neox_style=True)
    assert bool(jnp.all(ref == fix))
