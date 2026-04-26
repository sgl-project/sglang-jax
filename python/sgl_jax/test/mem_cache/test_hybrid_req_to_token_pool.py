import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh


def _mesh():
    """Single-device mesh with the canonical "tensor" axis name; matches the
    sharding axis RecurrentStatePool partitions H / proj_size on."""
    return Mesh(np.array(jax.devices()), ("tensor",))


def _make_req(req_pool_idx=None, is_chunked=0):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        recurrent_pool_idx=None,
        is_chunked=is_chunked,
    )


class TestReqToTokenPoolAlloc(unittest.TestCase):
    """Signature: alloc(reqs: list[Req-like]) -> list[int] | None.

    The int path was a transient compat shim and has been removed; callers
    must construct Req-like objects (chunked-prefill semantics depend on
    req_pool_idx / is_chunked, which int can't carry).
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        return ReqToTokenPool(size=4, max_context_len=8, dtype=np.int32)

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
        # Consume 3 slots first so only 1 remains.
        pool.alloc([_make_req(), _make_req(), _make_req()])
        reqs = [_make_req(), _make_req()]
        result = pool.alloc(reqs)
        self.assertIsNone(result)
        self.assertIsNone(reqs[0].req_pool_idx)
        self.assertIsNone(reqs[1].req_pool_idx)
        self.assertEqual(pool.free_slots, [3])

    def test_alloc_reqs_empty_list_returns_empty(self):
        pool = self._pool()
        self.assertEqual(pool.alloc([]), [])

    def test_alloc_int_no_longer_supported(self):
        """The int compat shim is gone; passing an int now raises (TypeError
        from iteration), proving the shim was not silently re-introduced.
        Lock-in for reviewer's request to drop hard-coded compatibility."""
        pool = self._pool()
        with self.assertRaises(TypeError):
            pool.alloc(2)


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
            mesh=_mesh(),
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
            mesh=_mesh(),
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
        before = len(pool.recurrent_free_slots)
        pool.alloc([req_a, req_b])
        self.assertEqual(req_a.recurrent_pool_idx, old_recurrent)
        self.assertIsNotNone(req_b.recurrent_pool_idx)
        self.assertNotEqual(req_b.recurrent_pool_idx, old_recurrent)
        self.assertEqual(len(pool.recurrent_free_slots), before - 1)

    def test_alloc_full_reuse_does_not_consume_recurrent_slot(self):
        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        before = len(pool.recurrent_free_slots)
        req.is_chunked = 1
        pool.alloc([req])
        self.assertEqual(len(pool.recurrent_free_slots), before)

    def test_alloc_recurrent_pool_exhausted_returns_none_atomically(self):
        """Critical: on failure neither req fields nor mapping may be partially mutated."""
        pool, rsp = self._make(max_num_reqs=2)
        # rsp capacity is 2; allocating 3 fresh reqs MUST fail.
        reqs = [_make_req(), _make_req(), _make_req()]
        before_free = list(pool.recurrent_free_slots)
        before_mapping = pool.req_index_to_recurrent_index_mapping.copy()

        result = pool.alloc(reqs)

        self.assertIsNone(result)
        # No req field touched.
        for req in reqs:
            self.assertIsNone(req.req_pool_idx)
            self.assertIsNone(req.recurrent_pool_idx)
        # recurrent_free_slots untouched.
        self.assertEqual(pool.recurrent_free_slots, before_free)
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
        """Hybrid pool inherits the list-only contract from base; int now
        raises TypeError (no defensive isinstance shim)."""
        pool, _ = self._make()
        with self.assertRaises(TypeError):
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
        for layer in range(rsp.num_linear_recurrent_layers):
            rsp.recurrent_buffers[layer] = rsp.recurrent_buffers[layer].at[slot].set(7.5)
            for inner in range(len(rsp.conv_buffers[layer])):
                rsp.conv_buffers[layer][inner] = (
                    rsp.conv_buffers[layer][inner].at[slot].set(jnp.bfloat16(3.25))
                )

        # Enter second chunk: req has both indices + is_chunked > 0.
        req.is_chunked = 1
        pool.alloc([req])

        # Critical assertion: reused-slot content was NOT cleared.
        for layer in range(rsp.num_linear_recurrent_layers):
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
            mesh=_mesh(),
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
        free_before = list(pool.recurrent_free_slots)

        pool.free_recurrent_cache(req)

        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIn(slot, pool.recurrent_free_slots)
        self.assertEqual(len(pool.recurrent_free_slots), len(free_before) + 1)
        self.assertEqual(pool.req_index_to_recurrent_index_mapping[req.req_pool_idx], 0)

    def test_free_recurrent_cache_idempotent_on_no_idx(self):
        pool, rsp = self._make()
        req = _make_req()  # recurrent_pool_idx=None
        before = list(pool.recurrent_free_slots)
        pool.free_recurrent_cache(req)
        self.assertEqual(pool.recurrent_free_slots, before)
        self.assertIsNone(req.recurrent_pool_idx)

    def test_alloc_recurrent_state_cleared_after_free(self):
        """clear-on-alloc reuse path: write non-zero, free, re-alloc -> zeroed."""
        import jax.numpy as jnp

        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        # Per-layer list element mutation to write non-zero.
        for layer in range(rsp.num_linear_recurrent_layers):
            rsp.recurrent_buffers[layer] = (
                rsp.recurrent_buffers[layer].at[req.recurrent_pool_idx].set(99.0)
            )
        pool.free_recurrent_cache(req)
        new_req = _make_req()
        pool.alloc([new_req])
        # The reallocated slot is cleared in every layer.
        for layer in range(rsp.num_linear_recurrent_layers):
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
            mesh=_mesh(),
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
            mesh=_mesh(),
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
        for layer in range(rsp.num_linear_recurrent_layers):
            rsp.recurrent_buffers[layer] = jnp.ones_like(rsp.recurrent_buffers[layer])
            for inner in range(len(rsp.conv_buffers[layer])):
                rsp.conv_buffers[layer][inner] = jnp.ones_like(rsp.conv_buffers[layer][inner])

        pool.clear()

        self.assertEqual(pool.free_slots, list(range(pool.size)))
        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3, 4])
        for layer in range(rsp.num_linear_recurrent_layers):
            self.assertTrue(bool(jnp.all(rsp.recurrent_buffers[layer] == 0)))
            for inner in range(len(rsp.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(rsp.conv_buffers[layer][inner] == 0)))
        self.assertTrue(bool((pool.req_index_to_recurrent_index_mapping == 0).all()))


class TestRecurrentSlotAllocator(unittest.TestCase):
    """Slot allocator state lives on HybridReqToTokenPool (not on
    RecurrentStatePool, which is the JIT-donated pytree leaf). These tests
    cover ownership boundaries, dummy-slot 0 reservation, and persistence
    across calls.
    """

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
            mesh=_mesh(),
        )
        pool = HybridReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=8,
            dtype=np.int32,
            recurrent_state_pool=rsp,
        )
        return pool, rsp

    def test_initial_recurrent_free_slots_starts_from_one(self):
        """Slot 0 is reserved as the dummy slot; free_slots must start from 1
        so mapping default 0 (=> dummy) cannot collide with a real allocation."""
        pool, _ = self._make(max_num_reqs=4)
        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3, 4])

    def test_initial_recurrent_free_slots_size_tracks_recurrent_pool(self):
        """Allocator capacity must match RecurrentStatePool.max_num_reqs, not
        the parent ReqToTokenPool size (which is +1 for the dummy)."""
        pool, rsp = self._make(max_num_reqs=3)
        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3])
        self.assertEqual(len(pool.recurrent_free_slots), rsp.max_num_reqs)

    def test_allocator_state_lives_on_hybrid_pool_not_buffer_pool(self):
        """Buffer pool must not regrow allocator fields after refactor."""
        pool, rsp = self._make()
        self.assertTrue(hasattr(pool, "recurrent_free_slots"))
        self.assertFalse(hasattr(rsp, "free_slots"))
        self.assertFalse(hasattr(rsp, "alloc"))
        self.assertFalse(hasattr(rsp, "free"))

    def test_alloc_pops_first_recurrent_slot_in_order(self):
        pool, _ = self._make()
        req = _make_req()
        pool.alloc([req])
        self.assertEqual(req.recurrent_pool_idx, 1)
        self.assertEqual(pool.recurrent_free_slots, [2, 3, 4])

    def test_alloc_consumes_exactly_one_per_fresh_req(self):
        pool, _ = self._make()
        reqs = [_make_req(), _make_req(), _make_req()]
        pool.alloc(reqs)
        self.assertEqual(pool.recurrent_free_slots, [4])
        self.assertEqual([r.recurrent_pool_idx for r in reqs], [1, 2, 3])

    def test_alloc_capacity_pre_check_does_not_consume(self):
        """Pre-check rejection must NOT pop any slot or write any req field."""
        pool, _ = self._make(max_num_reqs=2)
        before = list(pool.recurrent_free_slots)
        reqs = [_make_req(), _make_req(), _make_req()]
        self.assertIsNone(pool.alloc(reqs))
        self.assertEqual(pool.recurrent_free_slots, before)
        for req in reqs:
            self.assertIsNone(req.recurrent_pool_idx)

    def test_free_then_alloc_reuses_returned_slot(self):
        """A freed slot must come back into rotation."""
        pool, _ = self._make(max_num_reqs=2)
        req_a = _make_req()
        req_b = _make_req()
        pool.alloc([req_a, req_b])
        self.assertEqual(pool.recurrent_free_slots, [])
        slot_a = req_a.recurrent_pool_idx
        pool.free_recurrent_cache(req_a)
        self.assertIn(slot_a, pool.recurrent_free_slots)
        # Now a new req should pick up the freed slot (no available slots otherwise).
        req_c = _make_req()
        result = pool.alloc([req_c])
        self.assertIsNotNone(result)
        self.assertEqual(req_c.recurrent_pool_idx, slot_a)

    def test_alloc_clears_buffer_on_recurrent_slot_reuse(self):
        """Clear-on-alloc: after free, the next allocation must zero the slot."""
        import jax.numpy as jnp

        pool, rsp = self._make()
        req = _make_req()
        pool.alloc([req])
        slot = req.recurrent_pool_idx
        for layer in range(rsp.num_linear_recurrent_layers):
            rsp.recurrent_buffers[layer] = rsp.recurrent_buffers[layer].at[slot].set(99.0)
        pool.free_recurrent_cache(req)
        new_req = _make_req()
        pool.alloc([new_req])
        # Reused slot is zeroed in every layer.
        for layer in range(rsp.num_linear_recurrent_layers):
            self.assertTrue(
                bool(jnp.all(rsp.recurrent_buffers[layer][new_req.recurrent_pool_idx] == 0))
            )

    def test_chunked_prefill_reuse_does_not_pop_recurrent_slot(self):
        """Reuse path (req already has recurrent_pool_idx + is_chunked) must
        leave recurrent_free_slots untouched (the slot is already out)."""
        pool, _ = self._make()
        req = _make_req()
        pool.alloc([req])
        before = list(pool.recurrent_free_slots)
        req.is_chunked = 1
        pool.alloc([req])
        self.assertEqual(pool.recurrent_free_slots, before)

    def test_alloc_then_free_state_persists_across_python_calls(self):
        """Critical regression: allocator state lives on HybridReqToTokenPool,
        NOT on the JIT-donated RecurrentStatePool. After an alloc+free cycle,
        subsequent allocations must observe the persisted free_slots state.

        If allocator state were (incorrectly) carried inside the donated
        pytree, the JIT-cycle simulation below would silently revert
        recurrent_free_slots back to the constructor default and re-hand
        slot 1 a second time — corrupting the recurrent state of two reqs.
        """
        pool, rsp = self._make(max_num_reqs=4)

        # Cycle 1: allocate 2 reqs.
        req_a, req_b = _make_req(), _make_req()
        pool.alloc([req_a, req_b])
        self.assertEqual(pool.recurrent_free_slots, [3, 4])
        slot_a, slot_b = req_a.recurrent_pool_idx, req_b.recurrent_pool_idx
        self.assertEqual({slot_a, slot_b}, {1, 2})

        # Simulate a JIT-donate cycle on RecurrentStatePool (the buffer pool).
        # If the allocator lived inside this pytree it would silently reset to
        # the constructor default; that bug is what this test guards against.
        leaves, treedef = jax.tree_util.tree_flatten(rsp)
        rsp_after_jit = jax.tree_util.tree_unflatten(treedef, leaves)
        # Re-bind rsp inside pool to the post-JIT pool (mirrors what
        # MemoryPools.replace_all leaves callers with after donate).
        pool.recurrent_state_pool = rsp_after_jit

        # Cycle 2: allocate 1 more, free A. The allocator must respect the
        # persisted state from cycle 1: only slots [3, 4] remain free, and the
        # only way slot_a / slot_b can come back is via free_recurrent_cache.
        req_c = _make_req()
        pool.alloc([req_c])
        self.assertNotIn(req_c.recurrent_pool_idx, {slot_a, slot_b})
        self.assertEqual(pool.recurrent_free_slots, [4])
        slot_c = req_c.recurrent_pool_idx

        pool.free_recurrent_cache(req_a)
        self.assertIn(slot_a, pool.recurrent_free_slots)

        # Cycle 3: allocate 2 more — only slots {slot_a (freed), 4 (untouched)}
        # are available; slot_b and slot_c are still held. The post-JIT
        # allocator must NOT silently re-hand slot_b or slot_c.
        req_d, req_e = _make_req(), _make_req()
        result = pool.alloc([req_d, req_e])
        self.assertIsNotNone(result)
        self.assertEqual(
            {req_d.recurrent_pool_idx, req_e.recurrent_pool_idx},
            {slot_a, 4},
            "post-JIT allocator must preserve free_slots state from before the cycle",
        )
        self.assertNotIn(req_d.recurrent_pool_idx, {slot_b, slot_c})
        self.assertNotIn(req_e.recurrent_pool_idx, {slot_b, slot_c})

    def test_clear_resets_recurrent_free_slots_to_full_capacity(self):
        pool, _ = self._make(max_num_reqs=4)
        # Drain the pool.
        reqs = [_make_req() for _ in range(4)]
        pool.alloc(reqs)
        self.assertEqual(pool.recurrent_free_slots, [])

        pool.clear()

        self.assertEqual(pool.recurrent_free_slots, [1, 2, 3, 4])


class TestFreePolymorphism(unittest.TestCase):
    """pool.free(req) polymorphism: parent ReqToTokenPool only releases the KV
    slot; HybridReqToTokenPool override also releases the recurrent slot. The
    parent enforces the Req-object contract via a runtime type check so any
    legacy int / list[int] caller fails loud immediately."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

        return ReqToTokenPool(size=4, max_context_len=8, dtype=np.int32)

    def _make_hybrid(self, max_num_reqs=4):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=max_num_reqs,
            num_heads=1,
            head_dim=2,
            conv_kernel_size=4,
            mesh=_mesh(),
        )
        pool = HybridReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=8,
            dtype=np.int32,
            recurrent_state_pool=rsp,
        )
        return pool, rsp

    def test_free_rejects_legacy_int_caller(self):
        """Defensive type check guards against missed call-site migrations.

        Any caller that still passes a bare int / list[int] (the pre-refactor
        signature) must hit an immediate, clear TypeError so the migration
        gap is visible — never silently AttributeError when free_slots.append
        reaches for the missing req_pool_idx attribute."""
        pool = self._pool()
        # Allocate one req so free_slots is in a non-trivial state; the type
        # check fires before any state mutation regardless.
        pool.alloc([_make_req()])
        with self.assertRaises(TypeError) as cm_int:
            pool.free(0)  # legacy int signature
        self.assertIn("req_pool_idx", str(cm_int.exception))
        with self.assertRaises(TypeError) as cm_list:
            pool.free([0, 1])  # legacy list signature
        self.assertIn("req_pool_idx", str(cm_list.exception))

    def test_free_accepts_req_object(self):
        """Sanity: parent free(req) works on a Req-like object."""
        pool = self._pool()
        req = _make_req()
        pool.alloc([req])
        slot = req.req_pool_idx
        before = list(pool.free_slots)
        pool.free(req)
        self.assertIn(slot, pool.free_slots)
        self.assertEqual(len(pool.free_slots), len(before) + 1)

    def test_hybrid_free_releases_both_slots(self):
        """HybridReqToTokenPool.free(req) releases both the KV slot and the
        recurrent slot in one call — caller never has to remember to also
        drop the recurrent state."""
        pool, _ = self._make_hybrid()
        req = _make_req()
        pool.alloc([req])
        kv_slot = req.req_pool_idx
        rec_slot = req.recurrent_pool_idx
        kv_free_before = len(pool.free_slots)
        rec_free_before = len(pool.recurrent_free_slots)

        pool.free(req)

        # KV slot returned to parent's free_slots.
        self.assertIn(kv_slot, pool.free_slots)
        self.assertEqual(len(pool.free_slots), kv_free_before + 1)
        # Recurrent slot returned to recurrent_free_slots; req field cleared.
        self.assertIn(rec_slot, pool.recurrent_free_slots)
        self.assertEqual(len(pool.recurrent_free_slots), rec_free_before + 1)
        self.assertIsNone(req.recurrent_pool_idx)

    def test_hybrid_free_idempotent_when_no_recurrent_slot(self):
        """When req has no recurrent slot (recurrent_pool_idx is None) the
        hybrid free must still release the KV slot without raising —
        free_recurrent_cache is already idempotent on this case."""
        pool, _ = self._make_hybrid()
        req = _make_req()
        pool.alloc([req])
        # Manually clear the recurrent side so the override hits the
        # idempotent branch in free_recurrent_cache.
        pool.free_recurrent_cache(req)
        self.assertIsNone(req.recurrent_pool_idx)

        kv_slot = req.req_pool_idx
        kv_free_before = len(pool.free_slots)

        # Should not raise and should still release the KV slot.
        pool.free(req)
        self.assertIn(kv_slot, pool.free_slots)
        self.assertEqual(len(pool.free_slots), kv_free_before + 1)

    def test_hybrid_free_rejects_legacy_int_caller(self):
        """Override delegates to super, so the type check is inherited:
        legacy int callers fail loud at the hybrid pool too."""
        pool, _ = self._make_hybrid()
        pool.alloc([_make_req()])
        with self.assertRaises(TypeError):
            pool.free(0)
        with self.assertRaises(TypeError):
            pool.free([0, 1])
