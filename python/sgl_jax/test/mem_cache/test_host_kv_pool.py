"""Tests for RFC-1.0 HostKVPool (HiCache L2 pinned-host slot pool)."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.host_kv_pool import HostKVPool, make_host_sharding


def _make_mesh() -> Mesh:
    devices = jax.local_devices()
    return Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))


def _make_pool(capacity: int = 8) -> tuple[HostKVPool, NamedSharding]:
    mesh = _make_mesh()
    sharding = make_host_sharding(mesh, P("x"))
    pool = HostKVPool(capacity=capacity, host_sharding=sharding)
    return pool, sharding


class TestAllocFree(unittest.TestCase):
    def test_alloc_returns_unique_ids(self):
        pool, _ = _make_pool(capacity=4)
        ids = pool.alloc(3)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)

    def test_alloc_returns_none_when_full(self):
        pool, _ = _make_pool(capacity=2)
        pool.alloc(2)
        self.assertIsNone(pool.alloc(1))

    def test_free_then_realloc(self):
        pool, _ = _make_pool(capacity=2)
        ids = pool.alloc(2)
        self.assertIsNone(pool.alloc(1))
        pool.free(ids[:1])
        new_ids = pool.alloc(1)
        self.assertIsNotNone(new_ids)
        self.assertEqual(len(new_ids), 1)

    def test_available_size_tracks_alloc_free(self):
        pool, _ = _make_pool(capacity=4)
        self.assertEqual(pool.available_size(), 4)
        ids = pool.alloc(2)
        self.assertEqual(pool.available_size(), 2)
        pool.free(ids)
        self.assertEqual(pool.available_size(), 4)


class TestPutGet(unittest.TestCase):
    def test_put_then_get_returns_same_reference(self):
        pool, sharding = _make_pool(capacity=4)
        ids = pool.alloc(1)
        data = jax.device_put(jnp.ones((2, 3)), sharding)
        pool.put(ids[0], data)
        retrieved = pool.get(ids[0])
        self.assertIs(retrieved, data)

    def test_free_removes_reference(self):
        pool, sharding = _make_pool(capacity=4)
        ids = pool.alloc(1)
        data = jax.device_put(jnp.ones((2, 3)), sharding)
        pool.put(ids[0], data)
        pool.free(ids)
        self.assertNotIn(ids[0], pool.slot_table)

    def test_get_missing_slot_raises(self):
        pool, _ = _make_pool(capacity=4)
        ids = pool.alloc(1)
        with self.assertRaises(KeyError):
            pool.get(ids[0])


class TestD2HRoundTrip(unittest.TestCase):
    def test_device_to_host_to_device_roundtrip(self):
        """D2H -> put -> get -> H2D produces data matching the original."""
        mesh = _make_mesh()
        device_sharding = NamedSharding(mesh, P("x"))
        host_sharding = make_host_sharding(mesh, P("x"))
        pool = HostKVPool(capacity=4, host_sharding=host_sharding)

        original = jax.device_put(
            jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
            device_sharding,
        )

        ids = pool.alloc(1)
        host_array = jax.device_put(original, host_sharding)
        pool.put(ids[0], host_array)

        retrieved = pool.get(ids[0])
        restored = jax.device_put(retrieved, device_sharding)

        np.testing.assert_allclose(np.asarray(restored), np.asarray(original), rtol=0, atol=0)


class TestHostSharding(unittest.TestCase):
    def test_host_sharding_stored_on_pool(self):
        pool, sharding = _make_pool()
        self.assertIs(pool.host_sharding, sharding)


if __name__ == "__main__":
    unittest.main()
