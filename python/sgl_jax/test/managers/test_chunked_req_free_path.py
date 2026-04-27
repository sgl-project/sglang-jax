"""scheduler.get_next_batch_to_run chunked_req release path uses pool.free_chunked.

Replaces the legacy module-level _maybe_free_chunked_req_slot helper with
polymorphic dispatch on the pool subclass:

- ReqToTokenPool.free_chunked(req): free the slot AND clear
  req.req_pool_idx so the new alloc(reqs) list path treats this req as
  fresh (without the clear, alloc would silently return the stale slot
  index that's now back in free_slots).
- HybridReqToTokenPool.free_chunked(req): no-op so both KV slot and
  recurrent_pool_idx survive across chunks; the next alloc(reqs) reuses
  both indices.

Caller (scheduler.get_next_batch_to_run) just invokes
self.req_to_token_pool.free_chunked(self.chunked_req) -- the pool subclass
dispatches.
"""

import inspect
import unittest
from types import SimpleNamespace


class TestSchedulerCallsPoolFreeChunked(unittest.TestCase):
    def test_get_next_batch_uses_pool_free_chunked(self):
        """get_next_batch_to_run must release the chunked slot via the
        pool's polymorphic free_chunked(req)."""
        from sgl_jax.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        self.assertIn(
            "self.req_to_token_pool.free_chunked(self.chunked_req)",
            src,
            "get_next_batch_to_run must release chunked_req via "
            "self.req_to_token_pool.free_chunked(...)",
        )

    def test_legacy_helper_no_longer_referenced(self):
        """The _maybe_free_chunked_req_slot helper has been removed; the
        scheduler must not import or call it."""
        from sgl_jax.srt.managers import scheduler

        src = inspect.getsource(scheduler)
        self.assertNotIn(
            "_maybe_free_chunked_req_slot",
            src,
            "scheduler must not reference the legacy "
            "_maybe_free_chunked_req_slot helper; polymorphism lives in "
            "ReqToTokenPool.free_chunked / HybridReqToTokenPool.free_chunked.",
        )


class TestReqToTokenPoolFreeChunked(unittest.TestCase):
    """Parent ReqToTokenPool.free_chunked: real free + clear req_pool_idx."""

    def test_free_chunked_releases_slot_and_clears_idx(self):
        import numpy as np

        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(size=4, max_context_len=8, dtype=np.int32)
        req = SimpleNamespace(req_pool_idx=None, is_chunked=1)
        pool.alloc([req])
        slot = req.req_pool_idx
        before = list(pool.free_slots)

        pool.free_chunked(req)

        self.assertIn(
            slot,
            pool.free_slots,
            "free_chunked must release the KV slot back to free_slots",
        )
        self.assertEqual(
            len(pool.free_slots),
            len(before) + 1,
            "free_chunked must release exactly one slot (no double-free)",
        )
        self.assertIsNone(
            req.req_pool_idx,
            "free_chunked must clear req.req_pool_idx so alloc(reqs) treats "
            "this req as fresh (otherwise alloc returns the stale slot)",
        )


class TestHybridReqToTokenPoolFreeChunked(unittest.TestCase):
    """HybridReqToTokenPool.free_chunked: no-op so both slots survive."""

    def _make_hybrid_pool(self):
        import jax
        import numpy as np
        from jax.sharding import Mesh

        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        if not jax.devices():
            self.skipTest("JAX not available")
        mesh = Mesh(np.array(jax.devices()), ("tensor",))
        recurrent_state_pool = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=4,
            num_heads=2,
            head_dim=4,
            conv_kernel_size=4,
            mesh=mesh,
        )
        return HybridReqToTokenPool(
            size=4,
            max_context_len=8,
            dtype=np.int32,
            recurrent_state_pool=recurrent_state_pool,
        )

    def _make_req(self):
        return SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=1)

    def test_free_chunked_keeps_both_slots(self):
        pool = self._make_hybrid_pool()
        req = self._make_req()
        pool.alloc([req])
        kv_slot = req.req_pool_idx
        rec_slot = req.recurrent_pool_idx
        kv_free_before = list(pool.free_slots)
        rec_free_before = list(pool.recurrent_free_slots)

        pool.free_chunked(req)

        self.assertEqual(req.req_pool_idx, kv_slot, "Hybrid free_chunked must keep req_pool_idx")
        self.assertEqual(
            req.recurrent_pool_idx,
            rec_slot,
            "Hybrid free_chunked must keep recurrent_pool_idx so recurrent state "
            "survives across chunks",
        )
        self.assertEqual(
            list(pool.free_slots),
            kv_free_before,
            "Hybrid free_chunked must NOT return KV slot to free_slots",
        )
        self.assertEqual(
            list(pool.recurrent_free_slots),
            rec_free_before,
            "Hybrid free_chunked must NOT return recurrent slot to recurrent_free_slots",
        )

    def test_alloc_after_free_chunked_reuses_same_indices(self):
        """End-to-end: free_chunked(req) -> alloc([req]) returns the same
        KV + recurrent indices (no alloc/free roundtrip)."""
        pool = self._make_hybrid_pool()
        req = self._make_req()
        pool.alloc([req])
        kv_slot, rec_slot = req.req_pool_idx, req.recurrent_pool_idx

        pool.free_chunked(req)
        pool.alloc([req])  # next chunk round

        self.assertEqual(
            req.req_pool_idx,
            kv_slot,
            "Next alloc must reuse the same KV slot after free_chunked",
        )
        self.assertEqual(
            req.recurrent_pool_idx,
            rec_slot,
            "Next alloc must reuse the same recurrent slot (recurrent state preserved)",
        )


if __name__ == "__main__":
    unittest.main()
