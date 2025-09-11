#!/usr/bin/env python3
"""Simple test to verify fused KV syntax is correct"""

import os
import sys

import jax
import jax.numpy as jnp

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def test_import_and_basic_syntax():
    """Test that we can import and basic syntax is correct"""
    print("Testing imports...")

    try:
        from sgl_jax.srt.mem_cache.memory_pool import merge_kv

        print("✅ Memory pool imports successful")
    except Exception as e:
        print(f"❌ Memory pool import failed: {e}")
        return False

    try:
        from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
            ragged_paged_attention,
        )

        print("✅ Flash attention import successful")
    except Exception as e:
        print(f"❌ Flash attention import failed: {e}")
        return False

    print("Testing basic merge/extract functionality...")

    # Test merge_kv and extract functions
    k = jnp.ones((2, 4, 8))  # [tokens, heads, head_dim]
    v = jnp.ones((2, 4, 8)) * 2

    try:
        fused_kv = merge_kv(k, v)
        print(f"Fused KV shape: {fused_kv.shape}")
        assert fused_kv.shape == (
            2,
            8,
            8,
        )  # [tokens, heads * 2, head_dim] head interleaving tpu_commons v3

        # tpu_commons v3 head interleaving: directly test with indexing
        extracted_k = fused_kv[:, ::2, :]  # Even indices: K0, K1, K2...
        extracted_v = fused_kv[:, 1::2, :]  # Odd indices: V0, V1, V2...

        assert jnp.allclose(k, extracted_k)
        assert jnp.allclose(v, extracted_v)

        print("✅ Basic fused KV functionality works")
        return True

    except Exception as e:
        print(f"❌ Fused KV test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_import_and_basic_syntax()
    if success:
        print("\n🎉 All basic tests passed! Syntax is correct.")
        exit(0)
    else:
        print("\n💥 Some tests failed!")
        exit(1)
