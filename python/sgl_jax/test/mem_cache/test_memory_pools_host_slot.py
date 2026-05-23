"""Tests for the reserved ``host_pool`` slot on :class:`MemoryPools`.

PR-1-2: the slot must be backward-compatible — existing call sites that
construct ``MemoryPools(**pools)`` keep working — and must survive a
JAX pytree flatten/unflatten round-trip so ``jit(donate_argnames=
["memory_pools"])`` still sees the whole container as donatable.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from sgl_jax.srt.mem_cache.memory_pool import MemoryPools


class _FakePool:
    """Minimal stand-in for ``KVCache`` / ``RecurrentStatePool`` — opaque
    leaf so we don't pull TPU dependencies into this unit test.

    Treated as a JAX leaf by default (no pytree registration), so identity
    is preserved across flatten/unflatten.
    """

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"_FakePool({self.tag!r})"


class TestMemoryPoolsHostPoolSlot(unittest.TestCase):
    def test_default_host_pool_is_none(self):
        mp = MemoryPools(token_to_kv_pool=_FakePool("kv"))
        self.assertIsNone(mp.host_pool)

    def test_existing_construction_still_works(self):
        kv = _FakePool("kv")
        rsp = _FakePool("rsp")
        mp = MemoryPools(token_to_kv_pool=kv, recurrent_state_pool=rsp)
        self.assertIs(mp.token_to_kv_pool, kv)
        self.assertIs(mp.recurrent_state_pool, rsp)
        self.assertIsNone(mp.host_pool)

    def test_pytree_roundtrip_without_host_pool(self):
        kv = _FakePool("kv")
        mp = MemoryPools(token_to_kv_pool=kv)
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(restored.token_to_kv_pool, kv)
        self.assertIsNone(restored.host_pool)

    def test_pytree_roundtrip_with_host_pool(self):
        kv = _FakePool("kv")
        host = jnp.zeros((4, 8), dtype=jnp.bfloat16)
        mp = MemoryPools(token_to_kv_pool=kv, host_pool=host)
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(restored.token_to_kv_pool, kv)
        self.assertTrue(jnp.array_equal(restored.host_pool, host))

    def test_treedef_changes_when_host_pool_present(self):
        kv = _FakePool("kv")
        without = MemoryPools(token_to_kv_pool=kv)
        with_host = MemoryPools(
            token_to_kv_pool=kv,
            host_pool=jnp.zeros((2,), dtype=jnp.float32),
        )
        _, td_without = jax.tree_util.tree_flatten(without)
        _, td_with = jax.tree_util.tree_flatten(with_host)
        # Different PyTreeDef → jit cache will retrace (expected per PR-1-2 ACK).
        self.assertNotEqual(td_without, td_with)

    def test_pool_keys_remain_sorted_in_flatten(self):
        kv = _FakePool("kv")
        rsp = _FakePool("rsp")
        # constructed in non-sorted order but flatten output should be sorted.
        mp = MemoryPools(token_to_kv_pool=kv, recurrent_state_pool=rsp)
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(restored.recurrent_state_pool, rsp)
        self.assertIs(restored.token_to_kv_pool, kv)

    def test_host_pool_not_listed_in_pool_keys(self):
        """``host_pool`` must NOT leak into the inner ``_pools`` dict —
        otherwise ``replace_all`` would demand it in ``updates``."""
        mp = MemoryPools(
            token_to_kv_pool=_FakePool("kv"),
            host_pool=jnp.zeros((1,), dtype=jnp.float32),
        )
        # replace_all only knows about real pools.
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool"})

    def test_replace_all_unaffected_by_host_pool(self):
        kv_orig = _FakePool("kv")
        kv_new = _FakePool("kv_new")
        replaced = []
        kv_orig.replace_buffer = lambda v: replaced.append(v)  # type: ignore[attr-defined]

        mp = MemoryPools(
            token_to_kv_pool=kv_orig,
            host_pool=jnp.zeros((1,), dtype=jnp.float32),
        )
        mp.replace_all({"token_to_kv_pool": kv_new})
        self.assertEqual(replaced, [kv_new])

    def test_unknown_pool_lookup_still_raises(self):
        mp = MemoryPools(token_to_kv_pool=_FakePool("kv"))
        with self.assertRaises(AttributeError) as cm:
            _ = mp.nonexistent_pool
        self.assertIn("MemoryPools has no pool", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
