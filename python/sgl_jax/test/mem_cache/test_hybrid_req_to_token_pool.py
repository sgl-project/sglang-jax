import unittest
from types import SimpleNamespace

import jax
import numpy as np


def _make_req(req_pool_idx=None, is_chunked=0):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        recurrent_pool_idx=None,
        is_chunked=is_chunked,
    )


class TestReqToTokenPoolAllocBackwardCompat(unittest.TestCase):
    """ReqToTokenPool.alloc backwards-compatible signature change
    (RFC §goal 2 / Chunked Prefill Slot Reuse)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        return ReqToTokenPool(size=4, max_context_len=8, dtype=np.int32)

    # --- int path (backwards-compatible) ---
    def test_alloc_int_returns_first_n_slots(self):
        pool = self._pool()
        self.assertEqual(pool.alloc(2), [0, 1])
        self.assertEqual(pool.free_slots, [2, 3])

    def test_alloc_int_returns_none_when_insufficient(self):
        pool = self._pool()
        pool.alloc(3)
        self.assertIsNone(pool.alloc(2))

    def test_alloc_default_int_is_one(self):
        pool = self._pool()
        # Existing implementation is alloc(need_size: int = 1);
        # backwards compatibility requires the no-arg call to still work.
        self.assertEqual(pool.alloc(), [0])

    # --- list[Req] path (new signature) ---
    def test_alloc_reqs_assigns_req_pool_idx(self):
        pool = self._pool()
        reqs = [_make_req(), _make_req()]
        slots = pool.alloc(reqs)
        self.assertEqual(len(slots), 2)
        self.assertEqual(reqs[0].req_pool_idx, slots[0])
        self.assertEqual(reqs[1].req_pool_idx, slots[1])

    def test_alloc_reqs_reuses_existing_idx_when_chunked(self):
        pool = self._pool()
        # Real-world path: first allocate the slot (consuming free_slots),
        # then mark is_chunked > 0.
        existing = _make_req()
        pool.alloc([existing])
        existing.is_chunked = 1
        before_free = list(pool.free_slots)

        new_one = _make_req()
        slots = pool.alloc([existing, new_one])
        self.assertEqual(slots[0], existing.req_pool_idx)
        self.assertEqual(slots[1], new_one.req_pool_idx)
        # Only one slot consumed (for new_one).
        self.assertEqual(len(pool.free_slots), len(before_free) - 1)
        # Critical invariant: an allocated idx must NOT be in free_slots.
        self.assertNotIn(existing.req_pool_idx, pool.free_slots)
        self.assertNotIn(new_one.req_pool_idx, pool.free_slots)

    def test_alloc_reqs_safety_assert_when_reuse_without_chunked(self):
        pool = self._pool()
        # Real-world path: allocate first, leave is_chunked=0 -> reuse must raise.
        existing = _make_req()
        pool.alloc([existing])
        # is_chunked is still 0
        with self.assertRaises(AssertionError):
            pool.alloc([existing])

    def test_alloc_reqs_returns_none_when_insufficient_without_partial_mutation(self):
        pool = self._pool()
        pool.alloc(3)  # consumes [0, 1, 2]; only [3] left
        reqs = [_make_req(), _make_req()]
        result = pool.alloc(reqs)
        self.assertIsNone(result)
        self.assertIsNone(reqs[0].req_pool_idx)
        self.assertIsNone(reqs[1].req_pool_idx)
        self.assertEqual(pool.free_slots, [3])

    def test_alloc_reqs_empty_list_returns_empty(self):
        pool = self._pool()
        self.assertEqual(pool.alloc([]), [])


class TestHybridReqToTokenPoolInit(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self, max_num_reqs=4):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            num_layers=1,
            max_num_reqs=max_num_reqs,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        return (
            HybridReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=8,
                dtype=np.int32,
                recurrent_state_pool=rsp,
            ),
            rsp,
        )

    def test_inherits_req_to_token_pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import (
            HybridReqToTokenPool,
            ReqToTokenPool,
        )

        pool, _ = self._make()
        self.assertIsInstance(pool, ReqToTokenPool)
        self.assertIsInstance(pool, HybridReqToTokenPool)

    def test_holds_recurrent_state_pool_reference(self):
        pool, rsp = self._make()
        self.assertIs(pool.recurrent_state_pool, rsp)

    def test_mapping_initialized_to_zeros(self):
        pool, _ = self._make(max_num_reqs=4)
        self.assertEqual(pool.req_index_to_recurrent_index_mapping.shape, (5,))
        self.assertEqual(pool.req_index_to_recurrent_index_mapping.dtype, np.int32)
        self.assertTrue(bool((pool.req_index_to_recurrent_index_mapping == 0).all()))

    def test_jit_warning_in_docstring(self):
        """Subclass is not pytree-registered; docstring MUST warn against passing it to JIT."""
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool

        doc = HybridReqToTokenPool.__doc__ or ""
        self.assertIn("JIT", doc.upper())
        self.assertTrue(
            any(kw in doc for kw in ("unregistered", "do not", "not register", "must not"))
        )

    def test_inherits_alloc_int_path(self):
        pool, _ = self._make()
        # Hybrid does not override alloc yet (Task 6); int path inherits from parent.
        slots = pool.alloc(2)
        self.assertEqual(slots, [0, 1])
