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
    """Backwards-compatible signature: int (legacy) or list[Req] (new chunked-prefill path)."""

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
            linear_recurrent_layer_ids=[0],
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


class TestHybridReqToTokenPoolAlloc(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self, max_num_reqs=4):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=max_num_reqs,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        pool = HybridReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=8,
            dtype=np.int32,
            recurrent_state_pool=rsp,
        )
        return pool, rsp

    def test_alloc_fresh_reqs_assigns_both_indices(self):
        pool, rsp = self._make()
        reqs = [_make_req(), _make_req()]
        slots = pool.alloc(reqs)
        self.assertEqual(len(slots), 2)
        for req in reqs:
            self.assertIsNotNone(req.req_pool_idx)
            self.assertIsNotNone(req.recurrent_pool_idx)
            self.assertGreaterEqual(req.recurrent_pool_idx, 1)
            self.assertEqual(
                pool.req_index_to_recurrent_index_mapping[req.req_pool_idx],
                req.recurrent_pool_idx,
            )

    def test_alloc_partial_reuse_consumes_only_new_slots(self):
        pool, rsp = self._make()
        req_a = _make_req()
        pool.alloc([req_a])
        old_recurrent = req_a.recurrent_pool_idx

        req_a.is_chunked = 1
        req_b = _make_req()
        before = len(rsp.free_slots)
        pool.alloc([req_a, req_b])
        self.assertEqual(req_a.recurrent_pool_idx, old_recurrent)
        self.assertIsNotNone(req_b.recurrent_pool_idx)
        self.assertNotEqual(req_b.recurrent_pool_idx, old_recurrent)
        self.assertEqual(len(rsp.free_slots), before - 1)

    def test_alloc_full_reuse_does_not_consume_recurrent_slot(self):
        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        before = len(rsp.free_slots)
        req.is_chunked = 1
        pool.alloc([req])
        self.assertEqual(len(rsp.free_slots), before)

    def test_alloc_recurrent_pool_exhausted_returns_none_atomically(self):
        """Critical: on failure neither req fields nor mapping may be partially mutated."""
        pool, rsp = self._make(max_num_reqs=2)
        # rsp capacity is 2; allocating 3 fresh reqs MUST fail.
        reqs = [_make_req(), _make_req(), _make_req()]
        before_free = list(rsp.free_slots)
        before_mapping = pool.req_index_to_recurrent_index_mapping.copy()

        result = pool.alloc(reqs)

        self.assertIsNone(result)
        # No req field touched.
        for req in reqs:
            self.assertIsNone(req.req_pool_idx)
            self.assertIsNone(req.recurrent_pool_idx)
        # rsp.free_slots untouched.
        self.assertEqual(rsp.free_slots, before_free)
        # mapping untouched.
        self.assertTrue(bool((pool.req_index_to_recurrent_index_mapping == before_mapping).all()))

    def test_alloc_safety_assert_when_reuse_without_chunked(self):
        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        # is_chunked still 0 -> reuse must raise (assert from parent alloc).
        with self.assertRaises(AssertionError):
            pool.alloc([req])

    def test_alloc_int_disallowed(self):
        pool, _ = self._make()
        with self.assertRaises(AssertionError):
            pool.alloc(2)

    def test_chunked_prefill_reuse_preserves_buffer_content(self):
        """Chunked prefill reuse must preserve recurrent + conv state across chunks."""
        import jax.numpy as jnp

        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        slot = req.recurrent_pool_idx

        # Simulate first chunk forward: write accumulator values
        # (per-layer + per-inner list element mutation).
        for layer in range(rsp.num_layers):
            rsp.recurrent_buffers[layer] = rsp.recurrent_buffers[layer].at[slot].set(7.5)
            for inner in range(len(rsp.conv_buffers[layer])):
                rsp.conv_buffers[layer][inner] = (
                    rsp.conv_buffers[layer][inner].at[slot].set(jnp.bfloat16(3.25))
                )

        # Enter second chunk: req has both indices + is_chunked > 0.
        req.is_chunked = 1
        pool.alloc([req])

        # Critical assertion: reused-slot content was NOT cleared.
        for layer in range(rsp.num_layers):
            self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[layer][slot] == 7.5)))
            for inner in range(len(rsp.conv_buffers[layer])):
                self.assertTrue(
                    bool(jnp.all(rsp.conv_buffers[layer][inner][slot] == jnp.bfloat16(3.25)))
                )
        self.assertEqual(req.recurrent_pool_idx, slot)


class TestHybridReqToTokenPoolFree(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=4,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        pool = HybridReqToTokenPool(
            size=5, max_context_len=8, dtype=np.int32, recurrent_state_pool=rsp
        )
        return pool, rsp

    def test_free_recurrent_cache_returns_slot_and_resets_field(self):
        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        slot = req.recurrent_pool_idx
        free_before = list(rsp.free_slots)

        pool.free_recurrent_cache(req)

        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIn(slot, rsp.free_slots)
        self.assertEqual(len(rsp.free_slots), len(free_before) + 1)
        self.assertEqual(pool.req_index_to_recurrent_index_mapping[req.req_pool_idx], 0)

    def test_free_recurrent_cache_idempotent_on_no_idx(self):
        pool, rsp = self._make()
        req = _make_req()  # recurrent_pool_idx=None
        before = list(rsp.free_slots)
        pool.free_recurrent_cache(req)
        self.assertEqual(rsp.free_slots, before)
        self.assertIsNone(req.recurrent_pool_idx)

    def test_alloc_recurrent_state_cleared_after_free(self):
        """clear-on-alloc reuse path: write non-zero, free, re-alloc -> zeroed."""
        import jax.numpy as jnp

        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        # Per-layer list element mutation to write non-zero.
        for layer in range(rsp.num_layers):
            rsp.recurrent_buffers[layer] = (
                rsp.recurrent_buffers[layer].at[req.recurrent_pool_idx].set(99.0)
            )
        pool.free_recurrent_cache(req)
        new_req = _make_req()
        pool.alloc([new_req])
        # The reallocated slot is cleared in every layer.
        for layer in range(rsp.num_layers):
            self.assertTrue(
                bool(jnp.all(rsp.recurrent_buffers[layer][new_req.recurrent_pool_idx] == 0))
            )


class TestHybridReqToTokenPoolGetRecurrentIndices(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=4,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        pool = HybridReqToTokenPool(
            size=5, max_context_len=8, dtype=np.int32, recurrent_state_pool=rsp
        )
        return pool, rsp

    def test_get_linear_recurrent_indices_via_mapping(self):
        pool, rsp = self._make()
        reqs = [_make_req(), _make_req()]
        pool.alloc(reqs)
        req_pool_indices = np.array([reqs[0].req_pool_idx, reqs[1].req_pool_idx])
        recurrent_indices = pool.get_linear_recurrent_indices(req_pool_indices)
        self.assertEqual(recurrent_indices[0], reqs[0].recurrent_pool_idx)
        self.assertEqual(recurrent_indices[1], reqs[1].recurrent_pool_idx)


class TestHybridReqToTokenPoolClear(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=4,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
        )
        pool = HybridReqToTokenPool(
            size=5, max_context_len=8, dtype=np.int32, recurrent_state_pool=rsp
        )
        return pool, rsp

    def test_clear_resets_both_pools_and_mapping(self):
        import jax.numpy as jnp

        pool, rsp = self._make()
        reqs = [_make_req(), _make_req()]
        pool.alloc(reqs)
        # Write non-zero (per-layer + per-inner list element mutation).
        for layer in range(rsp.num_layers):
            rsp.recurrent_buffers[layer] = jnp.ones_like(rsp.recurrent_buffers[layer])
            for inner in range(len(rsp.conv_buffers[layer])):
                rsp.conv_buffers[layer][inner] = jnp.ones_like(rsp.conv_buffers[layer][inner])

        pool.clear()

        self.assertEqual(pool.free_slots, list(range(pool.size)))
        self.assertEqual(rsp.free_slots, [1, 2, 3, 4])
        for layer in range(rsp.num_layers):
            self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[layer] == 0)))
            for inner in range(len(rsp.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(rsp.conv_buffers[layer][inner] == 0)))
        self.assertTrue(bool((pool.req_index_to_recurrent_index_mapping == 0).all()))
