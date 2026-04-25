"""TPU platform compatibility tests for RecurrentStatePool.

Constructs pools directly on a single TPU device (no mesh/sharding) and
exercises alloc / replace_buffer (via JIT donate) / clear. CPU runs auto-skip
via skipUnless.

Run on TPU:
    python -m unittest python.sgl_jax.test.mem_cache.test_recurrent_state_pool_tpu -v
"""

import functools
import unittest

import jax
import jax.numpy as jnp


def _has_tpu_devices() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


@unittest.skipUnless(_has_tpu_devices(), "Requires TPU device(s)")
class TestRecurrentStatePoolOnTpu(unittest.TestCase):
    """Single-device TPU platform compatibility tests.

    Full sharding validation (NamedSharding + TP=4 + JIT donate sharding
    preservation) is deferred to a later Phase that wires up mesh integration.
    """

    def _make(self):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        return RecurrentStatePool(
            num_layers=2, max_num_reqs=4, num_heads=2, head_dim=4, conv_kernel_size=4
        )

    def test_buffers_on_tpu_devices(self):
        pool = self._make()
        for layer in range(pool.num_layers):
            for d in pool.recurrent_buffers[layer].devices():
                self.assertEqual(d.platform, "tpu")
            for inner in range(len(pool.conv_buffers[layer])):
                for d in pool.conv_buffers[layer][inner].devices():
                    self.assertEqual(d.platform, "tpu")

    def test_alloc_clears_buffers_on_tpu(self):
        pool = self._make()
        # list element mutation to write non-zero.
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        slots = pool.alloc(2)
        for layer in range(pool.num_layers):
            for idx in slots:
                self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][idx] == 0)))
            for inner in range(len(pool.conv_buffers[layer])):
                for idx in slots:
                    self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner][idx] == 0)))

    def test_replace_buffer_via_jit_donate_on_tpu(self):
        pool = self._make()

        @functools.partial(jax.jit, donate_argnames=["p"])
        def step(p):
            # list element mutation (matches real KDA layer pattern);
            # multi-layer contract guarded in test_memory_pools.py
            # (test_multi_layer_list_element_mutation_propagates).
            p.recurrent_buffers[0] = p.recurrent_buffers[0].at[1].set(7.0)
            p.conv_buffers[0][0] = p.conv_buffers[0][0].at[1].set(jnp.bfloat16(3.0))
            return (p.recurrent_buffers, p.conv_buffers)

        new_buffers = step(pool)
        pool.replace_buffer(new_buffers)
        self.assertEqual(float(pool.recurrent_buffers[0][1, 0, 0, 0]), 7.0)
        self.assertEqual(float(pool.conv_buffers[0][0][1, 0, 0]), 3.0)
        # Buffers stay on TPU (per layer + per inner).
        for layer in range(pool.num_layers):
            for d in pool.recurrent_buffers[layer].devices():
                self.assertEqual(d.platform, "tpu")
            for inner in range(len(pool.conv_buffers[layer])):
                for d in pool.conv_buffers[layer][inner].devices():
                    self.assertEqual(d.platform, "tpu")

    def test_clear_on_tpu(self):
        pool = self._make()
        pool.alloc(2)
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        pool.clear()
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 0)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == 0)))
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
