""" end-to-end wiring test.

Exercises the  chain:
  scheduler decides free vs keep
    -> ScheduleBatch.alloc_req_slots(reqs)
    -> ReqToTokenPool.alloc(reqs) | HybridReqToTokenPool.alloc(reqs)
    -> reuse vs fresh decision per req

We use SimpleNamespace mock Reqs (matching  test pattern) and the
real pool classes from , so this test covers the integration even
without spinning up a full ScheduleBatch / Scheduler.
"""

import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh


def _mesh():
    """Single-device mesh with the canonical "tensor" axis name; matches the
    sharding axis RecurrentStatePool partitions H / proj_size on."""
    return Mesh(np.array(jax.devices()), ("tensor",))


def _make_req(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        recurrent_pool_idx=recurrent_pool_idx,
        is_chunked=is_chunked,
    )


class TestAllocReqSlotsE2EHybrid(unittest.TestCase):
    """Hybrid model end-to-end: chunked req keeps req_pool_idx +
    recurrent_pool_idx across chunks (no free between chunks)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make_hybrid_pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=4,
            num_heads=2,
            head_dim=4,
            conv_kernel_size=4,
            mesh=_mesh(),
        )
        return HybridReqToTokenPool(
            size=4 + 1,
            max_context_len=16,
            dtype=np.int32,
            recurrent_state_pool=rsp,
        )

    def test_first_chunk_allocates_both_indices(self):
        pool = self._make_hybrid_pool()
        req = _make_req()
        indices = pool.alloc([req])
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 1)
        self.assertEqual(req.req_pool_idx, indices[0])
        self.assertIsNotNone(req.recurrent_pool_idx)

    def test_second_chunk_reuses_both_indices(self):
        """hybrid path: scheduler skipped free, so req keeps both
        indices; second alloc must reuse, not allocate fresh."""
        pool = self._make_hybrid_pool()
        req = _make_req()
        pool.alloc([req])
        first_req_idx = req.req_pool_idx
        first_recurrent_idx = req.recurrent_pool_idx

        # Simulate scheduler hybrid path: skip free; mark is_chunked > 0.
        req.is_chunked = 1
        # Confirm req still holds both indices going into second alloc.
        self.assertEqual(req.req_pool_idx, first_req_idx)
        self.assertEqual(req.recurrent_pool_idx, first_recurrent_idx)

        # Second chunk alloc with the same req.
        pool.alloc([req])
        self.assertEqual(
            req.req_pool_idx,
            first_req_idx,
            "Hybrid path: req_pool_idx must be preserved across chunks",
        )
        self.assertEqual(
            req.recurrent_pool_idx,
            first_recurrent_idx,
            "Hybrid path: recurrent_pool_idx must be preserved across chunks",
        )


class TestAllocReqSlotsE2ENonHybrid(unittest.TestCase):
    """Non-hybrid model end-to-end: scheduler frees + clears req_pool_idx,
    so next alloc treats the req as fresh."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make_pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        return ReqToTokenPool(size=5, max_context_len=16, dtype=np.int32)

    def test_first_chunk_allocates(self):
        pool = self._make_pool()
        req = _make_req()
        indices = pool.alloc([req])
        self.assertIsNotNone(indices)
        self.assertEqual(req.req_pool_idx, indices[0])

    def test_after_free_and_clear_alloc_treats_as_fresh(self):
        """non-hybrid path: scheduler called pool.free + cleared
        req.req_pool_idx; next alloc must allocate a fresh slot."""
        pool = self._make_pool()
        req = _make_req()
        pool.alloc([req])

        # Simulate scheduler non-hybrid path: pool.free + clear ref.
        pool.free(req)
        req.req_pool_idx = None

        # Second alloc: req has no req_pool_idx, treated as fresh.
        second_indices = pool.alloc([req])
        self.assertIsNotNone(second_indices)
        # New slot allocated; may or may not be the same numerical id depending
        # on free_slots LIFO/FIFO; the contract is "fresh allocation happened".
        self.assertIsNotNone(req.req_pool_idx)


class TestSchedulerHelperPlusPoolAllocCombo(unittest.TestCase):
    """Wires pool.free_chunked directly to pool.alloc.
    The smallest reproduction of the actual scheduler chunked-prefill path
    end-state (post-polymorphism refactor)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_hybrid_skip_free_plus_alloc_reuses_same_slot(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=4,
            num_heads=2,
            head_dim=4,
            conv_kernel_size=4,
            mesh=_mesh(),
        )
        pool = HybridReqToTokenPool(
            size=5, max_context_len=16, dtype=np.int32, recurrent_state_pool=rsp
        )
        req = _make_req()
        pool.alloc([req])
        first_req_idx = req.req_pool_idx
        first_recurrent_idx = req.recurrent_pool_idx

        # Scheduler chunked path (hybrid): polymorphic no-op via free_chunked.
        req.is_chunked = 1
        pool.free_chunked(req)

        # Both indices preserved.
        self.assertEqual(req.req_pool_idx, first_req_idx)
        self.assertEqual(req.recurrent_pool_idx, first_recurrent_idx)

        # Next alloc round: same indices reused.
        pool.alloc([req])
        self.assertEqual(req.req_pool_idx, first_req_idx)
        self.assertEqual(req.recurrent_pool_idx, first_recurrent_idx)

    def test_non_hybrid_free_clear_plus_alloc_gives_fresh_slot(self):
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(size=5, max_context_len=16, dtype=np.int32)
        req = _make_req()
        pool.alloc([req])

        # Scheduler chunked path (non-hybrid): free + clear stale ref via free_chunked.
        pool.free_chunked(req)
        self.assertIsNone(req.req_pool_idx, "Non-hybrid path must clear req_pool_idx after free")

        # Next alloc round: fresh allocation succeeds.
        pool.alloc([req])
        self.assertIsNotNone(req.req_pool_idx)


if __name__ == "__main__":
    unittest.main()
