"""Tests for AttentionBackendMetadata base type."""

import jax

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackendMetadata


class TestAttentionBackendMetadata:
    def test_default_construction(self):
        meta = AttentionBackendMetadata()
        assert isinstance(meta, AttentionBackendMetadata)

    def test_pytree_roundtrip(self):
        meta = AttentionBackendMetadata()
        leaves, treedef = jax.tree_util.tree_flatten(meta)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, AttentionBackendMetadata)

    def test_pytree_inside_jit(self):
        @jax.jit
        def identity(m):
            return m

        meta = AttentionBackendMetadata()
        out = identity(meta)
        assert isinstance(out, AttentionBackendMetadata)
