import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np


def _rsp(max_num_reqs=2):
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

    return RecurrentStatePool(
        linear_recurrent_layer_ids=[0],
        max_num_reqs=max_num_reqs,
        num_heads=1,
        head_dim=2,
        conv_kernel_size=4,
    )


def _copy_conv_buffers(conv_buffers):
    """Force a functional copy of conv_buffers to avoid JIT donate aliasing.

    Donating a buffer that appears unchanged in the output triggers
    "Donated argument is referenced in the output". `+ jnp.bfloat16(0)` forces
    a functional update path so XLA materializes a fresh buffer.
    """
    return [[c + jnp.bfloat16(0) for c in inner] for inner in conv_buffers]


@jax.tree_util.register_pytree_node_class
class _StubKVPool:
    """Pytree-friendly KV pool stub used by multi-pool and replace_all dispatch tests.

    Holds a `last_replace_value` so tests can assert what replace_all forwarded
    to this pool (without unpacking).
    """

    def __init__(self, data):
        self.data = data
        self.last_replace_value = None

    def tree_flatten(self):
        return (self.data,), ()

    @classmethod
    def tree_unflatten(cls, _, children):
        obj = cls.__new__(cls)
        (obj.data,) = children
        obj.last_replace_value = None
        return obj

    def replace_buffer(self, value):
        self.last_replace_value = value
        # Accept either list[Array] or a single array; contract tests only check the
        # forwarded `value` type.
        if isinstance(value, list) and value:
            self.data = value[0]
        else:
            self.data = value


class TestMemoryPoolsContainer(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        return MemoryPools(recurrent_state_pool=rsp), rsp

    def test_attribute_access(self):
        mp, rsp = self._make()
        self.assertIs(mp.recurrent_state_pool, rsp)

    def test_attribute_access_unknown_raises(self):
        mp, _ = self._make()
        with self.assertRaises(AttributeError):
            _ = mp.nonexistent_pool

    def test_dunder_attribute_raises_attribute_error(self):
        mp, _ = self._make()
        with self.assertRaises(AttributeError):
            _ = mp._does_not_exist

    def test_pytree_roundtrip_single_pool(self):
        mp, rsp = self._make()
        # list element mutation to write non-zero (matches implementation style).
        rsp.recurrent_buffers[0] = rsp.recurrent_buffers[0].at[1].set(7.0)
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        mp2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(hasattr(mp2, "recurrent_state_pool"))
        for layer in range(rsp.num_layers):
            self.assertTrue(
                bool(
                    jnp.all(
                        mp2.recurrent_state_pool.recurrent_buffers[layer]
                        == rsp.recurrent_buffers[layer]
                    )
                )
            )

    def test_pytree_roundtrip_two_pools(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        kv = _StubKVPool(jnp.zeros(4))
        mp = MemoryPools(token_to_kv_pool=kv, recurrent_state_pool=rsp)
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        mp2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(hasattr(mp2, "token_to_kv_pool"))
        self.assertTrue(hasattr(mp2, "recurrent_state_pool"))


class TestMemoryPoolsReplaceAll(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _setup(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        return MemoryPools(recurrent_state_pool=rsp), rsp

    def test_replace_all_single_pool_passes_value_unchanged(self):
        mp, rsp = self._setup()
        # value type = (list[Array], list[list[Array]])
        new_recurrent = [jnp.ones_like(b) * 5.0 for b in rsp.recurrent_buffers]
        new_conv = [[jnp.ones_like(c) for c in inner] for inner in rsp.conv_buffers]
        mp.replace_all({"recurrent_state_pool": (new_recurrent, new_conv)})
        for layer in range(rsp.num_layers):
            self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[layer] == 5)))

    def test_replace_all_two_pools_each_gets_its_value_type(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        kv = _StubKVPool(jnp.zeros(2))
        mp = MemoryPools(token_to_kv_pool=kv, recurrent_state_pool=rsp)
        new_recurrent = [jnp.ones_like(b) for b in rsp.recurrent_buffers]
        new_conv = [[jnp.ones_like(c) for c in inner] for inner in rsp.conv_buffers]
        mp.replace_all(
            {
                "token_to_kv_pool": [jnp.zeros(2)],
                "recurrent_state_pool": (new_recurrent, new_conv),
            }
        )
        # KV pool received the original list (contract: no value unpacking).
        self.assertIsInstance(kv.last_replace_value, list)
        for layer in range(rsp.num_layers):
            self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[layer] == 1)))

    def test_replace_all_missing_key_raises(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        mp = MemoryPools(
            token_to_kv_pool=_StubKVPool(jnp.zeros(2)),
            recurrent_state_pool=rsp,
        )
        with self.assertRaises(ValueError):
            mp.replace_all({"token_to_kv_pool": [jnp.zeros(2)]})

    def test_replace_all_extra_key_raises(self):
        mp, rsp = self._setup()
        zero_recurrent = [jnp.zeros_like(b) for b in rsp.recurrent_buffers]
        zero_conv = [[jnp.zeros_like(c) for c in inner] for inner in rsp.conv_buffers]
        with self.assertRaises(ValueError):
            mp.replace_all(
                {
                    "recurrent_state_pool": (zero_recurrent, zero_conv),
                    "phantom_pool": "anything",
                }
            )

    def test_replace_all_empty_dict_raises(self):
        mp, _ = self._setup()
        with self.assertRaises(ValueError):
            mp.replace_all({})


class TestMemoryPoolsJitDonate(unittest.TestCase):
    """End-to-end: JIT donate(memory_pools) -> model returns pool_updates -> replace_all.

    **Write-back style**: every fake_forward in this class follows the real KDA
    layer pattern of list element mutation:

        sp.recurrent_buffers[layer] = sp.recurrent_buffers[layer].at[...].set(...)
        sp.conv_buffers[layer][inner] = sp.conv_buffers[layer][inner].at[...].set(...)

    list is a mutable Python container; multiple layers share the same state_pool
    instance and therefore the same list. Layer N writes list[N], and Layer N+1
    reads list[N] to see the updated value. If a forward function instead writes
    to a local variable (`new = sp.recurrent_buffers[l].at[...].set(...)` without
    storing back to the list), updates from layers 0..N-1 are silently lost in a
    multi-layer setup. `test_multi_layer_list_element_mutation_propagates`
    demonstrates and guards the correct pattern.
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _setup(self):
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

        rsp = _rsp()
        return MemoryPools(recurrent_state_pool=rsp), rsp

    def test_jit_donate_then_replace_updates_state(self):
        mp, rsp = self._setup()

        @functools.partial(jax.jit, donate_argnames=["memory_pools"])
        def fake_forward(memory_pools):
            sp = memory_pools.recurrent_state_pool
            # list element mutation write-back (matches real KDA layer pattern).
            sp.recurrent_buffers[0] = sp.recurrent_buffers[0].at[1].set(42.0)
            sp.conv_buffers[0][0] = sp.conv_buffers[0][0].at[1].set(jnp.bfloat16(7.0))
            return jnp.array([0.0]), {
                "recurrent_state_pool": (sp.recurrent_buffers, sp.conv_buffers),
            }

        _output, pool_updates = fake_forward(mp)
        mp.replace_all(pool_updates)
        self.assertEqual(float(rsp.recurrent_buffers[0][1, 0, 0, 0]), 42.0)
        self.assertEqual(float(rsp.conv_buffers[0][0][1, 0, 0]), 7.0)

    def test_multi_layer_list_element_mutation_propagates(self):
        """Multi-layer list element mutation contract: layer N+1 must see layer N's update.

        Multiple layers share the state_pool instance (same list). Layer N writes
        list[N] in place; layer N+1 reads list[N] back and observes the updated
        value. If KDA layer code accidentally uses local-variable assignment
        (`new = ...at[...].set(...)` without storing back to the list), updates
        from layers 0..N-1 would silently be lost in a multi-layer forward —
        this test catches that regression.
        """
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        # Need linear_recurrent_layer_ids of length >= 2 to exercise the contract.
        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=2,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        mp = MemoryPools(recurrent_state_pool=rsp)

        @functools.partial(jax.jit, donate_argnames=["memory_pools"])
        def two_layer_forward(memory_pools):
            sp = memory_pools.recurrent_state_pool
            # Layer 0: list element mutation write-back.
            sp.recurrent_buffers[0] = sp.recurrent_buffers[0].at[1].set(10.0)
            # Layer 1: must observe layer 0's update via the shared list.
            layer0_value = sp.recurrent_buffers[0][1, 0, 0, 0]
            sp.recurrent_buffers[1] = sp.recurrent_buffers[1].at[1].set(layer0_value + 5.0)
            # conv unchanged -> use copy helper to avoid donate aliasing.
            new_conv = _copy_conv_buffers(sp.conv_buffers)
            return jnp.array([0.0]), {
                "recurrent_state_pool": (sp.recurrent_buffers, new_conv),
            }

        _, upd = two_layer_forward(mp)
        mp.replace_all(upd)
        # Layer 0 / slot 1 = 10
        self.assertEqual(float(rsp.recurrent_buffers[0][1, 0, 0, 0]), 10.0)
        # Layer 1 / slot 1 = 10 + 5 = 15 (proves Layer 1 read Layer 0's update).
        self.assertEqual(float(rsp.recurrent_buffers[1][1, 0, 0, 0]), 15.0)

    def test_zero_invariant_after_alloc_with_dirty_buffer(self):
        """Zero invariant: scatter non-zero -> donate -> replace -> alloc new slot -> slot is zero."""
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool

        mp, rsp = self._setup()
        hybrid = HybridReqToTokenPool(
            size=3, max_context_len=8, dtype=np.int32, recurrent_state_pool=rsp
        )

        @functools.partial(jax.jit, donate_argnames=["memory_pools"])
        def dirty_forward(memory_pools):
            sp = memory_pools.recurrent_state_pool
            # Stamp every slot (including unallocated ones) to 99
            # via per-layer list element mutation.
            for layer in range(len(sp.recurrent_buffers)):
                sp.recurrent_buffers[layer] = jnp.ones_like(sp.recurrent_buffers[layer]) * 99.0
                for inner in range(len(sp.conv_buffers[layer])):
                    sp.conv_buffers[layer][inner] = jnp.ones_like(sp.conv_buffers[layer][inner])
            return jnp.array([0.0]), {
                "recurrent_state_pool": (sp.recurrent_buffers, sp.conv_buffers),
            }

        _, pool_updates = dirty_forward(mp)
        mp.replace_all(pool_updates)

        from types import SimpleNamespace

        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([req])
        # The newly allocated slot is cleared in every layer + every inner.
        for layer in range(rsp.num_layers):
            self.assertTrue(
                bool(jnp.all(rsp.recurrent_buffers[layer][req.recurrent_pool_idx] == 0))
            )
            for inner in range(len(rsp.conv_buffers[layer])):
                self.assertTrue(
                    bool(jnp.all(rsp.conv_buffers[layer][inner][req.recurrent_pool_idx] == 0))
                )

    def test_multi_step_accumulate_then_free_resets(self):
        """Multi-step accumulation, then free + re-alloc starts from zero."""
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool

        mp, rsp = self._setup()
        hybrid = HybridReqToTokenPool(
            size=3, max_context_len=8, dtype=np.int32, recurrent_state_pool=rsp
        )

        from types import SimpleNamespace

        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([req])
        slot = req.recurrent_pool_idx

        @functools.partial(jax.jit, donate_argnames=["memory_pools"])
        def step(memory_pools, slot, delta):
            sp = memory_pools.recurrent_state_pool
            # recurrent update via list element mutation.
            sp.recurrent_buffers[0] = sp.recurrent_buffers[0].at[slot].add(delta)
            # conv unchanged -> use copy helper to avoid donate aliasing.
            new_conv = _copy_conv_buffers(sp.conv_buffers)
            return jnp.array([0.0]), {
                "recurrent_state_pool": (sp.recurrent_buffers, new_conv),
            }

        for _ in range(3):
            _, upd = step(mp, slot, 1.0)
            mp.replace_all(upd)
        self.assertEqual(float(rsp.recurrent_buffers[0][slot, 0, 0, 0]), 3.0)

        hybrid.free_recurrent_cache(req)
        new_req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([new_req])
        self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[0][new_req.recurrent_pool_idx] == 0)))


if __name__ == "__main__":
    unittest.main()
