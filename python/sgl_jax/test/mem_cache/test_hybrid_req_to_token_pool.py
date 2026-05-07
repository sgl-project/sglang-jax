# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_hybrid_req_to_token_pool.py -v

import os
import unittest
from types import SimpleNamespace

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.test.test_utils import CustomTestCase


def _mesh_tp_only():
    """DP=1, TP=N mesh. RecurrentStatePool requires both 'data' and 'tensor'
    axes (sharding spec is P('data', 'tensor', None, ...))."""
    devices = np.array(jax.devices()).reshape(1, -1)
    return Mesh(devices, ("data", "tensor"))


def _mesh_dp_tp(dp_size):
    """DP=dp_size, TP=N/dp_size mesh."""
    n = len(jax.devices())
    devices = np.array(jax.devices()).reshape(dp_size, n // dp_size)
    return Mesh(devices, ("data", "tensor"))


def _make_req(req_pool_idx=None, is_chunked=0, kv_committed_len=0):
    """Minimal Req surrogate exposing only the attributes hybrid pool reads."""
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        recurrent_pool_idx=None,
        is_chunked=is_chunked,
        kv_committed_len=kv_committed_len,
    )


def _make_pool(size=4, mesh=None, dp_size=1):
    if mesh is None:
        mesh = _mesh_tp_only()
    rsp = RecurrentStatePool(
        linear_recurrent_layer_ids=[0],
        size=size,
        num_heads=mesh.shape["tensor"],
        head_dim=2,
        conv_kernel_size=4,
        mesh=mesh,
        dp_size=dp_size,
    )
    pool = HybridReqToTokenPool(
        size=size,
        max_context_len=8,
        dtype=np.int32,
        recurrent_state_pool=rsp,
    )
    return pool, rsp


class TestHybridReqToTokenPool(CustomTestCase):
    def test_alloc_fresh(self):
        """Fresh alloc: writes both indices, mapping[req_pool_idx] =
        recurrent_pool_idx, get_linear_recurrent_indices returns the same."""
        pool, _ = _make_pool(size=4)
        reqs = [_make_req(), _make_req(), _make_req(), _make_req()]
        slots = pool.alloc(reqs)

        self.assertEqual(len(slots), 4)
        for req in reqs:
            self.assertIsNotNone(req.req_pool_idx)
            self.assertGreaterEqual(req.recurrent_pool_idx, 1)
            self.assertEqual(
                pool.req_index_to_recurrent_index_mapping[req.req_pool_idx],
                req.recurrent_pool_idx,
            )

        req_pool_indices = np.array([r.req_pool_idx for r in reqs])
        recurrent_indices = pool.get_linear_recurrent_indices(req_pool_indices)
        for req, rec in zip(reqs, recurrent_indices):
            self.assertEqual(rec, req.recurrent_pool_idx)

    def test_alloc_reuse(self):
        """Reuse path: req with recurrent_pool_idx + is_chunked > 0 keeps
        its slot. Buffer content at the held slot must NOT be cleared."""
        for case in ("partial_reuse", "full_reuse"):
            with self.subTest(case=case):
                pool, rsp = _make_pool(size=4)
                req_a = _make_req()
                pool.alloc([req_a])

                # Stamp a sentinel value at the held slot.
                slot = req_a.recurrent_pool_idx
                rsp.recurrent_buffers[0] = rsp.recurrent_buffers[0].at[slot].set(7.5)

                req_a.is_chunked = 1
                rec_before = list(pool.recurrent_free_slots)

                if case == "partial_reuse":
                    req_b = _make_req()
                    pool.alloc([req_a, req_b])
                    self.assertEqual(len(pool.recurrent_free_slots), len(rec_before) - 1)
                    self.assertNotEqual(req_b.recurrent_pool_idx, slot)
                else:
                    pool.alloc([req_a])
                    self.assertEqual(pool.recurrent_free_slots, rec_before)

                # Reused slot retains its sentinel (not cleared on alloc).
                self.assertEqual(req_a.recurrent_pool_idx, slot)
                self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[0][slot] == 7.5)))

    def test_alloc_atomic_on_miss(self):
        """Capacity miss must not partially mutate any state. Covers the
        recurrent-side pre-check, the KV-side parent atomic guarantee, and
        the off-by-one boundary (capacity vs capacity + 1)."""
        for case in ("recurrent_miss", "kv_miss", "off_by_one"):
            with self.subTest(case=case):
                if case == "recurrent_miss":
                    pool, _ = _make_pool(size=2)
                    reqs = [_make_req(), _make_req(), _make_req()]
                elif case == "kv_miss":
                    pool, _ = _make_pool(size=4)
                    pool.free_slots = pool.free_slots[:1]  # leave 1 KV slot
                    reqs = [_make_req(), _make_req()]
                else:  # off_by_one: capacity = N, request N+1
                    pool, _ = _make_pool(size=3)
                    reqs = [_make_req() for _ in range(4)]

                kv_before = list(pool.free_slots)
                rec_before = list(pool.recurrent_free_slots)
                mapping_before = pool.req_index_to_recurrent_index_mapping.copy()

                self.assertIsNone(pool.alloc(reqs))
                for req in reqs:
                    self.assertIsNone(req.req_pool_idx)
                    self.assertIsNone(req.recurrent_pool_idx)
                self.assertEqual(pool.free_slots, kv_before)
                self.assertEqual(pool.recurrent_free_slots, rec_before)
                self.assertTrue(
                    bool((pool.req_index_to_recurrent_index_mapping == mapping_before).all())
                )

    def test_alloc_at_exact_capacity_succeeds(self):
        """Capacity == request: must succeed and drain free pools to empty."""
        pool, _ = _make_pool(size=3)
        reqs = [_make_req() for _ in range(3)]
        self.assertIsNotNone(pool.alloc(reqs))
        self.assertEqual(pool.recurrent_free_slots, [])

    def test_alloc_assert_on_unsanctioned_reuse(self):
        """Inherits parent assert: a req holding req_pool_idx but is_chunked
        == 0 and kv_committed_len == 0 indicates a missed free() — must raise."""
        pool, _ = _make_pool()
        req = _make_req()
        pool.alloc([req])
        # is_chunked still 0, kv_committed_len still 0 -> reuse must raise.
        with self.assertRaises(AssertionError):
            pool.alloc([req])

    def test_free_releases_both_slots(self):
        """free(req) releases KV slot AND recurrent slot; idempotent on the
        recurrent side when req.recurrent_pool_idx is None."""
        for case in ("full", "no_recurrent_slot"):
            with self.subTest(case=case):
                pool, _ = _make_pool()
                req = _make_req()
                pool.alloc([req])
                kv_slot = req.req_pool_idx
                rec_slot = req.recurrent_pool_idx

                if case == "no_recurrent_slot":
                    pool.free_recurrent_cache(req)
                    rec_free_before = list(pool.recurrent_free_slots)

                kv_free_before = len(pool.free_slots)
                pool.free(req)

                self.assertIn(kv_slot, pool.free_slots)
                self.assertEqual(len(pool.free_slots), kv_free_before + 1)
                self.assertIsNone(req.req_pool_idx)
                self.assertIsNone(req.recurrent_pool_idx)
                if case == "full":
                    self.assertIn(rec_slot, pool.recurrent_free_slots)
                else:
                    # no_recurrent_slot: recurrent_free_slots unchanged by free()
                    self.assertEqual(pool.recurrent_free_slots, rec_free_before)

    def test_alloc_after_free_clears_recurrent_buffer(self):
        """clear-on-alloc reuse path: write non-zero, free, re-alloc -> zeroed.
        Critical: protects against stale state leaking into a reused slot."""
        pool, rsp = _make_pool()
        req = _make_req()
        pool.alloc([req])
        rsp.recurrent_buffers[0] = rsp.recurrent_buffers[0].at[req.recurrent_pool_idx].set(99.0)
        pool.free_recurrent_cache(req)

        new_req = _make_req()
        pool.alloc([new_req])
        self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[0][new_req.recurrent_pool_idx] == 0)))

    def test_clear_resets_then_realloc(self):
        """clear() resets KV + recurrent allocators, recurrent + conv buffers,
        and mapping; subsequent alloc must work as on a fresh pool."""
        pool, rsp = _make_pool(size=4)
        reqs = [_make_req(), _make_req()]
        pool.alloc(reqs)
        rsp.recurrent_buffers[0] = jnp.ones_like(rsp.recurrent_buffers[0])

        pool.clear()

        self.assertEqual(pool.free_slots, list(range(pool.size)))
        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3, 4])
        self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[0] == 0)))
        self.assertTrue(bool((pool.req_index_to_recurrent_index_mapping == 0).all()))

        # Realloc after clear must succeed and assign fresh indices.
        new_reqs = [_make_req(), _make_req()]
        self.assertIsNotNone(pool.alloc(new_reqs))
        for req in new_reqs:
            self.assertIsNotNone(req.req_pool_idx)
            self.assertGreaterEqual(req.recurrent_pool_idx, 1)

    def test_allocator_survives_jit_donate_cycle(self):
        """Critical regression: allocator state lives on HybridReqToTokenPool,
        NOT inside the JIT-donated RecurrentStatePool pytree. After a simulated
        JIT-donate cycle on RecurrentStatePool, subsequent allocations must
        observe the persisted free_slots state — otherwise alloc would
        silently re-hand the same slot to two reqs.
        """
        pool, rsp = _make_pool(size=4)
        req_a, req_b = _make_req(), _make_req()
        pool.alloc([req_a, req_b])
        slot_a, slot_b = req_a.recurrent_pool_idx, req_b.recurrent_pool_idx

        # Simulate a JIT-donate cycle on RecurrentStatePool.
        leaves, treedef = jax.tree_util.tree_flatten(rsp)
        pool.recurrent_state_pool = jax.tree_util.tree_unflatten(treedef, leaves)

        # Allocator must NOT have reset to constructor default.
        req_c = _make_req()
        pool.alloc([req_c])
        self.assertNotIn(req_c.recurrent_pool_idx, {slot_a, slot_b})

    def test_size_one_minimal_pool(self):
        """Sanity for the smallest usable pool. Confirms there is no implicit
        assumption that size > 1 in alloc / free / mapping paths."""
        pool, rsp = _make_pool(size=1)
        self.assertEqual(pool.recurrent_free_slots, [1])
        self.assertEqual(pool.req_index_to_recurrent_index_mapping.shape, (1,))

        req = _make_req()
        self.assertIsNotNone(pool.alloc([req]))
        self.assertEqual(req.recurrent_pool_idx, 1)
        self.assertEqual(pool.recurrent_free_slots, [])

        # Second alloc on a drained size=1 pool must fail atomically.
        req2 = _make_req()
        self.assertIsNone(pool.alloc([req2]))
        self.assertIsNone(req2.req_pool_idx)
        self.assertIsNone(req2.recurrent_pool_idx)

        pool.free(req)
        self.assertEqual(pool.recurrent_free_slots, [1])


class TestHybridReqToTokenPoolDP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if len(jax.devices()) < 2:
            raise unittest.SkipTest("DP tests need >= 2 devices")

    def test_dp_capacity_decoupled_from_total_slots(self):
        """Host-side allocator capacity is decoupled from RecurrentStatePool's
        DP-padded total_slots. With dp_size=2 and size=3, total_slots ceils
        to 4 for buffer divisibility; recurrent_free_slots still hands out
        exactly 3 valid slots [1..3]. Full alloc/free roundtrip must work
        identically to the dp_size=1 case.
        """
        mesh = _mesh_dp_tp(dp_size=2)
        pool, rsp = _make_pool(size=3, mesh=mesh, dp_size=2)
        self.assertEqual(rsp.size, 3)
        self.assertEqual(rsp.total_slots, 4)  # ceil(3+1, 2)
        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3])

        reqs = [_make_req(), _make_req(), _make_req()]
        self.assertIsNotNone(pool.alloc(reqs))
        self.assertEqual(pool.recurrent_free_slots, [])

        pool.free(reqs[0])
        req_new = _make_req()
        self.assertIsNotNone(pool.alloc([req_new]))
        self.assertEqual(pool.recurrent_free_slots, [])


if __name__ == "__main__":
    unittest.main()
