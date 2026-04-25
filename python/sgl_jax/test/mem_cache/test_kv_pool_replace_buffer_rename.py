"""Verify KV pool replace_kv_buffer -> replace_buffer rename.

After the rename, MemoryPools.replace_all (which calls pool.replace_buffer(value))
can dispatch to KV pools without AttributeError.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class _MeshFixture:
    """Single-device CPU mesh for unit tests."""

    @staticmethod
    def make():
        from jax.sharding import Mesh

        devices = np.array(jax.devices()[:1]).reshape(1, 1)
        return Mesh(devices, axis_names=("data", "tensor"))


class TestKVCacheAbstractReplaceBuffer(unittest.TestCase):
    """KVCache abstract method must be named replace_buffer (not replace_kv_buffer)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_abstract_method_renamed(self):
        from sgl_jax.srt.mem_cache.memory_pool import KVCache

        # The abstractmethod must be `replace_buffer`, not `replace_kv_buffer`.
        self.assertIn("replace_buffer", KVCache.__abstractmethods__)
        self.assertNotIn("replace_kv_buffer", KVCache.__abstractmethods__)


class TestMHATokenToKVPoolReplaceBuffer(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.mesh = _MeshFixture.make()

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool(
            size=128,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            mesh=self.mesh,
        )

    def test_has_replace_buffer_method(self):
        pool = self._pool()
        self.assertTrue(hasattr(pool, "replace_buffer"))
        self.assertFalse(
            hasattr(pool, "replace_kv_buffer"),
            "Old method name must be removed (not just aliased)",
        )

    def test_replace_buffer_swaps_layers(self):
        pool = self._pool()
        new_buffers = [jnp.ones_like(buf) for buf in pool.kv_buffer]
        pool.replace_buffer(new_buffers)
        for layer in range(pool.layer_num):
            self.assertTrue(bool(jnp.all(pool.kv_buffer[layer] == 1)))


class TestMLATokenToKVPoolReplaceBuffer(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.mesh = _MeshFixture.make()

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool(
            size=128,
            page_size=1,
            dtype=jnp.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            mesh=self.mesh,
        )

    def test_has_replace_buffer_method(self):
        pool = self._pool()
        self.assertTrue(hasattr(pool, "replace_buffer"))
        self.assertFalse(hasattr(pool, "replace_kv_buffer"))

    def test_mla_pytree_roundtrip(self):
        """D2 decision guard: epic c3372a9c added pytree to MLATokenToKVPool;
        this test verifies completeness.

        If roundtrip fails (e.g. start_layer offset lost, sharding fix
        absent, or any aux field missing), STOP and report; user decides
        whether to backfill in Phase 2 (new task) or defer to a later phase.
        Phase 2 must NOT silently rewrite epic's pytree implementation.
        """
        import jax as _jax

        pool = self._pool()
        # Write a non-zero value to verify roundtrip preserves real content.
        pool.kv_buffer[0] = pool.kv_buffer[0].at[0].set(jnp.bfloat16(7.0))

        leaves, treedef = _jax.tree_util.tree_flatten(pool)
        pool2 = _jax.tree_util.tree_unflatten(treedef, leaves)

        # Same logical buffer count
        self.assertEqual(len(pool2.kv_buffer), len(pool.kv_buffer))
        # Buffers must round-trip (content + dtype + shape)
        for layer in range(len(pool.kv_buffer)):
            self.assertTrue(bool(jnp.all(pool.kv_buffer[layer] == pool2.kv_buffer[layer])))
            self.assertEqual(pool2.kv_buffer[layer].shape, pool.kv_buffer[layer].shape)
            self.assertEqual(pool2.kv_buffer[layer].dtype, pool.kv_buffer[layer].dtype)
        # Critical metadata for KV pool slicing must survive (epic adds these in
        # tree_flatten aux; if any are missing the unflatten will raise AttributeError
        # below, which is exactly the guard we want).
        self.assertEqual(pool2.start_layer, pool.start_layer)
        self.assertEqual(pool2.end_layer, pool.end_layer)
        self.assertEqual(pool2.layer_num, pool.layer_num)
        self.assertEqual(pool2.kv_lora_rank, pool.kv_lora_rank)
        self.assertEqual(pool2.qk_rope_head_dim, pool.qk_rope_head_dim)


class TestSWAKVPoolReplaceBuffer(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.mesh = _MeshFixture.make()

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import SWAKVPool

        return SWAKVPool(
            size=128,
            size_swa=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            head_dim=128,
            mesh=self.mesh,
        )

    def test_has_replace_buffer_method(self):
        pool = self._pool()
        self.assertTrue(hasattr(pool, "replace_buffer"))
        self.assertFalse(hasattr(pool, "replace_kv_buffer"))

    def test_replace_buffer_dispatches_to_subpools(self):
        pool = self._pool()
        # Build kv_buffer in the same layer order as layers_mapping expects.
        kv_buffer = [None, None]
        for layer_id, (sub_idx, is_swa) in pool.layers_mapping.items():
            sub = pool.swa_kv_pool if is_swa else pool.full_kv_pool
            kv_buffer[layer_id] = jnp.ones_like(sub.kv_buffer[sub_idx])
        pool.replace_buffer(kv_buffer)
        for sub in (pool.swa_kv_pool, pool.full_kv_pool):
            for buf in sub.kv_buffer:
                self.assertTrue(bool(jnp.all(buf == 1)))


class TestMemoryPoolsCallsReplaceBufferOnKVPool(unittest.TestCase):
    """End-to-end: MemoryPools.replace_all dispatches to MHATokenToKVPool.replace_buffer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.mesh = _MeshFixture.make()

    def test_replace_all_with_real_kv_pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool

        kv = MHATokenToKVPool(
            size=128,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            mesh=self.mesh,
        )
        mp = MemoryPools(token_to_kv_pool=kv)
        new_buffers = [jnp.ones_like(buf) for buf in kv.kv_buffer]
        mp.replace_all({"token_to_kv_pool": new_buffers})
        for buf in kv.kv_buffer:
            self.assertTrue(bool(jnp.all(buf == 1)))


if __name__ == "__main__":
    unittest.main()
