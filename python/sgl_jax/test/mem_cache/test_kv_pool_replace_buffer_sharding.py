"""Phase 5 D2 / issue #233: KV pool replace_buffer must preserve kv_sharding
on tp_size==1 (single-device) so the next JIT trace sees a stable shape.

Phase 2 Task 5 deleted _set_kv_cache_after_forward which had this fix at
the model_runner level; Phase 5 puts it back inside each pool's
replace_buffer (mirroring RecurrentStatePool.replace_buffer Phase 1 pattern).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class _MeshFixture:
    @staticmethod
    def make():
        from jax.sharding import Mesh

        devices = np.array(jax.devices()[:1]).reshape(1, 1)
        return Mesh(devices, axis_names=("data", "tensor"))


class TestMHAReplaceBufferShardingFix(unittest.TestCase):
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

    def test_replace_buffer_preserves_kv_sharding_on_tp1(self):
        """tp_size==1: replace_buffer must device_put incoming arrays into
        kv_sharding before assignment so the next JIT trace's sharding
        constraint is not lost (issue #233)."""
        pool = self._pool()
        original_sharding = pool.kv_buffer[0].sharding

        # Construct new buffers WITHOUT explicit sharding (mimics JIT output
        # whose sharding was not constrained by an out_sharding).
        new_buffers = [
            jax.device_put(jnp.ones_like(buf), device=jax.devices()[0]) for buf in pool.kv_buffer
        ]
        pool.replace_buffer(new_buffers)

        # Each layer's kv_buffer must end up with kv_sharding applied.
        for layer in range(pool.layer_num):
            buf = pool.kv_buffer[layer]
            self.assertEqual(
                buf.sharding,
                original_sharding,
                f"layer {layer}: replace_buffer must preserve kv_sharding "
                f"on tp_size==1; got {buf.sharding} vs expected {original_sharding}",
            )


class TestMLAReplaceBufferShardingFix(unittest.TestCase):
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

    def test_replace_buffer_preserves_kv_sharding_on_tp1(self):
        pool = self._pool()
        original_sharding = pool.kv_buffer[0].sharding

        new_buffers = [
            jax.device_put(jnp.ones_like(buf), device=jax.devices()[0]) for buf in pool.kv_buffer
        ]
        pool.replace_buffer(new_buffers)

        for layer in range(pool.layer_num):
            buf = pool.kv_buffer[layer]
            self.assertEqual(
                buf.sharding,
                original_sharding,
                f"MLA layer {layer}: replace_buffer must preserve kv_sharding " f"on tp_size==1",
            )


if __name__ == "__main__":
    unittest.main()
