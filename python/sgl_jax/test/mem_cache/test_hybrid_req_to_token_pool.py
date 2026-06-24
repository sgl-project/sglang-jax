# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_hybrid_req_to_token_pool.py -v

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sgl_jax.test.test_utils import CustomTestCase


class FakeRecurrentStatePool:
    """Stub exposing only the interface HybridReqToTokenPool reads.

    `size` is **global** (mirrors MHATokenToKVPool). HybridReqToTokenPool
    internally splits it as size // dp_size per rank.
    """

    def __init__(self, size: int):
        self.size = size

    def clear(self):
        pass


class FakeReq:
    """Minimal Req surrogate for pool tests."""

    def __init__(
        self,
        req_pool_idx=None,
        is_chunked=0,
        kv_committed_len=0,
        dp_rank=None,
        recurrent_pool_idx=None,
    ):
        self.req_pool_idx = req_pool_idx
        self.is_chunked = is_chunked
        self.kv_committed_len = kv_committed_len
        self.dp_rank = dp_rank
        self.recurrent_pool_idx = recurrent_pool_idx
        # Extra-buffer ping-pong track fields.
        self.recurrent_ping_pong_track_buffer = None
        self.recurrent_next_track_idx = None
        self.recurrent_last_track_seqlen = None

    def reset_for_retract(self):
        """Mirror Req.reset_for_retract clears for the ping-pong fields."""
        self.recurrent_pool_idx = None
        self.recurrent_ping_pong_track_buffer = None
        self.recurrent_next_track_idx = None
        self.recurrent_last_track_seqlen = None


# ---------------------------------------------------------------------------
# dp_size=1 tests
# ---------------------------------------------------------------------------


class TestHybridPoolAllocSingleDP(CustomTestCase):
    """Alloc tests with dp_size=1 (default path)."""

    def setUp(self):
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=1,
        )

    def test_alloc_assigns_both_kv_and_recurrent_slots(self):
        reqs = [FakeReq(dp_rank=0), FakeReq(dp_rank=0)]
        indices = self.pool.alloc(reqs)

        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 2)
        for req in reqs:
            self.assertIsNotNone(req.req_pool_idx)
            self.assertIsNotNone(req.recurrent_pool_idx)
        self.assertNotEqual(reqs[0].recurrent_pool_idx, reqs[1].recurrent_pool_idx)

    def test_alloc_updates_mapping(self):
        reqs = [FakeReq(dp_rank=0)]
        self.pool.alloc(reqs)

        req = reqs[0]
        mapped = self.pool.req_index_to_recurrent_index_mapping[req.req_pool_idx]
        self.assertEqual(mapped, req.recurrent_pool_idx)

    def test_alloc_fails_when_recurrent_slots_exhausted(self):
        slots_per_rank = self.state_pool.size // 1  # global // dp_size
        exhaust = [FakeReq(dp_rank=0) for _ in range(slots_per_rank)]
        self.pool.alloc(exhaust)

        overflow = [FakeReq(dp_rank=0)]
        result = self.pool.alloc(overflow)
        self.assertIsNone(result)
        self.assertIsNone(overflow[0].req_pool_idx)
        self.assertIsNone(overflow[0].recurrent_pool_idx)

    def test_alloc_skips_req_with_existing_recurrent_idx(self):
        """A chunked req already has recurrent_pool_idx; alloc must not reassign."""
        chunked = FakeReq(req_pool_idx=0, is_chunked=1, dp_rank=0, recurrent_pool_idx=99)
        fresh = FakeReq(dp_rank=0)

        self.pool.free_slots = [1, 2, 3]
        indices = self.pool.alloc([chunked, fresh])

        self.assertIsNotNone(indices)
        self.assertEqual(chunked.recurrent_pool_idx, 99)
        self.assertIsNotNone(fresh.recurrent_pool_idx)
        self.assertNotEqual(fresh.recurrent_pool_idx, 99)

    def test_alloc_none_dp_rank_defaults_to_zero(self):
        reqs = [FakeReq(dp_rank=None)]
        indices = self.pool.alloc(reqs)

        self.assertIsNotNone(indices)
        self.assertIsNotNone(reqs[0].recurrent_pool_idx)


class TestHybridPoolFreeSingleDP(CustomTestCase):
    """Free tests with dp_size=1."""

    def setUp(self):
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=1,
        )

    def test_free_returns_both_slots(self):
        req = FakeReq(dp_rank=0)
        self.pool.alloc([req])
        kv_idx = req.req_pool_idx
        recurrent_idx = req.recurrent_pool_idx
        recurrent_avail_before = self.pool.recurrent_available_size(0)

        self.pool.free(req)

        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIn(kv_idx, self.pool.free_slots)
        self.assertEqual(self.pool.recurrent_available_size(0), recurrent_avail_before + 1)
        self.assertIn(recurrent_idx, self.pool.recurrent_free_slots[0])

    def test_free_clears_mapping(self):
        req = FakeReq(dp_rank=0)
        self.pool.alloc([req])
        kv_idx = req.req_pool_idx

        self.pool.free(req)

        self.assertEqual(self.pool.req_index_to_recurrent_index_mapping[kv_idx], 0)

    def test_alloc_free_roundtrip(self):
        """Alloc -> free -> alloc must recycle slots correctly."""
        req = FakeReq(dp_rank=0)
        self.pool.alloc([req])
        self.pool.free(req)

        req2 = FakeReq(dp_rank=0)
        indices = self.pool.alloc([req2])
        self.assertIsNotNone(indices)
        self.assertIsNotNone(req2.req_pool_idx)
        self.assertIsNotNone(req2.recurrent_pool_idx)


class TestHybridPoolGetIndicesSingleDP(CustomTestCase):
    """get_linear_recurrent_indices with dp_size=1."""

    def setUp(self):
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=1,
        )

    def test_get_linear_recurrent_indices_returns_correct_mapping(self):
        reqs = [FakeReq(dp_rank=0), FakeReq(dp_rank=0), FakeReq(dp_rank=0)]
        self.pool.alloc(reqs)

        req_pool_indices = np.array([r.req_pool_idx for r in reqs], dtype=np.int32)
        result = self.pool.get_linear_recurrent_indices(req_pool_indices)

        expected = np.array([r.recurrent_pool_idx for r in reqs], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_unmapped_index_returns_zero(self):
        result = self.pool.get_linear_recurrent_indices(np.array([0], dtype=np.int32))
        self.assertEqual(result[0], 0)


class TestHybridPoolClearSingleDP(CustomTestCase):
    """clear() with dp_size=1."""

    def setUp(self):
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=1,
        )

    def test_clear_resets_all_state(self):
        reqs = [FakeReq(dp_rank=0), FakeReq(dp_rank=0)]
        self.pool.alloc(reqs)

        self.pool.clear()

        self.assertEqual(self.pool.available_size(), 8)
        slots_per_rank = self.state_pool.size // 1
        self.assertEqual(self.pool.recurrent_available_size(0), slots_per_rank)
        np.testing.assert_array_equal(
            self.pool.req_index_to_recurrent_index_mapping,
            np.zeros(8, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# dp_size>1 tests
# ---------------------------------------------------------------------------


class TestHybridPoolAllocMultiDP(CustomTestCase):
    """Alloc tests with dp_size=2."""

    def setUp(self):
        # Global size=8, dp_size=2 → 4 slots/rank.
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=2,
        )

    def test_dp_ranks_use_separate_free_lists(self):
        req_dp0 = FakeReq(dp_rank=0)
        req_dp1 = FakeReq(dp_rank=1)

        self.pool.alloc([req_dp0])
        self.pool.alloc([req_dp1])

        self.assertIsNotNone(req_dp0.recurrent_pool_idx)
        self.assertIsNotNone(req_dp1.recurrent_pool_idx)
        # Both start from slot 1 in their respective rank free list
        self.assertEqual(req_dp0.recurrent_pool_idx, 1)
        self.assertEqual(req_dp1.recurrent_pool_idx, 1)

    def test_dp_rank_capacity_independent(self):
        """Exhausting dp_rank=0 must not affect dp_rank=1."""
        slots_per_rank = self.state_pool.size // 2  # 4
        exhaust_dp0 = [FakeReq(dp_rank=0) for _ in range(slots_per_rank)]
        self.pool.alloc(exhaust_dp0)

        overflow_dp0 = [FakeReq(dp_rank=0)]
        self.assertIsNone(self.pool.alloc(overflow_dp0))

        dp1_req = FakeReq(dp_rank=1)
        result = self.pool.alloc([dp1_req])
        self.assertIsNotNone(result)
        self.assertIsNotNone(dp1_req.recurrent_pool_idx)

    def test_free_returns_to_correct_dp_rank(self):
        req_dp0 = FakeReq(dp_rank=0)
        req_dp1 = FakeReq(dp_rank=1)
        self.pool.alloc([req_dp0])
        self.pool.alloc([req_dp1])

        avail_dp0 = self.pool.recurrent_available_size(0)
        avail_dp1 = self.pool.recurrent_available_size(1)

        self.pool.free(req_dp0)

        self.assertEqual(self.pool.recurrent_available_size(0), avail_dp0 + 1)
        self.assertEqual(self.pool.recurrent_available_size(1), avail_dp1)

    def test_cross_dp_alloc_free_roundtrip(self):
        """Alloc on dp0, free, alloc on dp1 — slots are independent."""
        req0 = FakeReq(dp_rank=0)
        self.pool.alloc([req0])
        self.pool.free(req0)

        req1 = FakeReq(dp_rank=1)
        result = self.pool.alloc([req1])
        self.assertIsNotNone(result)

    def test_clear_resets_all_dp_ranks(self):
        req0 = FakeReq(dp_rank=0)
        req1 = FakeReq(dp_rank=1)
        self.pool.alloc([req0])
        self.pool.alloc([req1])

        self.pool.clear()

        slots_per_rank = self.state_pool.size // 2
        for dp_rank in range(2):
            self.assertEqual(self.pool.recurrent_available_size(dp_rank), slots_per_rank)

    def test_mapping_correct_across_dp_ranks(self):
        req0 = FakeReq(dp_rank=0)
        req1 = FakeReq(dp_rank=1)
        self.pool.alloc([req0])
        self.pool.alloc([req1])

        indices = np.array([req0.req_pool_idx, req1.req_pool_idx], dtype=np.int32)
        result = self.pool.get_linear_recurrent_indices(indices)

        self.assertEqual(result[0], req0.recurrent_pool_idx)
        self.assertEqual(result[1], req1.recurrent_pool_idx)

    def test_chunked_reuses_recurrent_slot_across_batches(self):
        chunked = FakeReq(dp_rank=0)
        peer = FakeReq(dp_rank=0)
        self.pool.alloc([chunked, peer])
        chunked_recurrent = chunked.recurrent_pool_idx

        self.pool.free(peer)
        chunked.is_chunked = 1

        new_peer = FakeReq(dp_rank=0)
        indices = self.pool.alloc([chunked, new_peer])

        self.assertIsNotNone(indices)
        self.assertEqual(chunked.recurrent_pool_idx, chunked_recurrent)
        self.assertIsNotNone(new_peer.recurrent_pool_idx)


class TestHybridPoolRecurrentSlotAPI(CustomTestCase):
    """Typed scalar slot API + 3-state (active/tree-owned/free) lifecycle."""

    def setUp(self):
        self.state_pool = FakeRecurrentStatePool(size=8)
        self.pool = HybridReqToTokenPool(
            size=8,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=self.state_pool,
            dp_size=1,
        )

    def test_alloc_free_recurrent_slot_scalars(self):
        slot = self.pool.alloc_recurrent_slot(0)
        self.assertIsInstance(slot, int)
        self.assertNotIn(slot, self.pool.recurrent_free_slots[0])
        self.pool.free_recurrent_slot(slot, 0)
        self.assertIn(slot, self.pool.recurrent_free_slots[0])

    def test_alloc_recurrent_slot_returns_none_when_exhausted(self):
        for _ in range(self.pool.slots_per_rank):
            self.assertIsNotNone(self.pool.alloc_recurrent_slot(0))
        self.assertIsNone(self.pool.alloc_recurrent_slot(0))

    def test_recurrent_value_from_slot_is_length1_array(self):
        v = self.pool.recurrent_value_from_slot(5)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(v.dtype, np.int32)
        self.assertEqual(v.shape, (1,))
        self.assertEqual(int(v[0]), 5)

    def test_commit_to_tree_transfers_ownership_without_freeing(self):
        req = FakeReq(dp_rank=0)
        self.pool.alloc([req])
        slot = req.recurrent_pool_idx
        free_before = self.pool.recurrent_available_size(0)

        self.pool.commit_to_tree(req)

        # Ownership moved to the tree: req handle + mapping cleared, slot NOT
        # returned to the free list.
        self.assertIsNone(req.recurrent_pool_idx)
        self.assertEqual(self.pool.req_index_to_recurrent_index_mapping[req.req_pool_idx], 0)
        self.assertEqual(self.pool.recurrent_available_size(0), free_before)
        self.assertNotIn(slot, self.pool.recurrent_free_slots[0])

    def test_free_skips_donated_slot(self):
        """After commit_to_tree, pool.free must not return the donated slot."""
        req = FakeReq(dp_rank=0)
        self.pool.alloc([req])
        slot = req.recurrent_pool_idx
        self.pool.commit_to_tree(req)
        free_before = self.pool.recurrent_available_size(0)

        self.pool.free(req)  # ownership-based: recurrent_pool_idx is None → no-op

        self.assertEqual(self.pool.recurrent_available_size(0), free_before)
        self.assertNotIn(slot, self.pool.recurrent_free_slots[0])

    def test_ledger_alloc_commit_evict_free(self):
        """Free count tracks the active/tree-owned/free lifecycle."""
        slots = self.pool.slots_per_rank
        reqs = [FakeReq(dp_rank=0) for _ in range(3)]
        self.pool.alloc(reqs)
        self.assertEqual(self.pool.recurrent_available_size(0), slots - 3)

        committed_slot = reqs[0].recurrent_pool_idx
        self.pool.commit_to_tree(reqs[0])
        # commit transfers ownership to the tree: free count unchanged.
        self.assertEqual(self.pool.recurrent_available_size(0), slots - 3)

        # tree eviction returns the committed slot.
        self.pool.free_recurrent_slot(committed_slot, 0)
        self.assertEqual(self.pool.recurrent_available_size(0), slots - 2)

        # request-owned slots freed normally.
        self.pool.free(reqs[1])
        self.pool.free(reqs[2])
        self.assertEqual(self.pool.recurrent_available_size(0), slots)


class TestHybridPoolExtraBufferAPI(CustomTestCase):
    """Ping-pong track-slot API used by the extra-buffer recurrent path.

    Dormant unless ``enable_recurrent_extra_buffer=True``: with it off the pool must
    behave byte-identically to the base page=1 path (running slot only).
    """

    def _make_pool(self, enable_recurrent_extra_buffer, dp_size=1, size=16):
        state_pool = FakeRecurrentStatePool(size=size)
        return HybridReqToTokenPool(
            size=size,
            max_context_len=32,
            dtype=np.int32,
            recurrent_state_pool=state_pool,
            dp_size=dp_size,
            enable_recurrent_extra_buffer=enable_recurrent_extra_buffer,
        )

    # --- Case: extra-buffer OFF is a strict no-op (base page=1 path unchanged) ---

    def test_extra_buffer_off_allocs_only_running_slot(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=False)
        free_before = pool.recurrent_available_size(0)
        req = FakeReq(dp_rank=0)

        pool.alloc([req])

        # Only the running slot is consumed; no track slots, no fields set.
        self.assertIsNotNone(req.recurrent_pool_idx)
        self.assertIsNone(req.recurrent_ping_pong_track_buffer)
        self.assertIsNone(req.recurrent_next_track_idx)
        self.assertIsNone(req.recurrent_last_track_seqlen)
        self.assertEqual(pool.recurrent_available_size(0), free_before - 1)
        # Track mapping row untouched (all zeros).
        np.testing.assert_array_equal(
            pool.req_index_to_recurrent_ping_pong_track_buffer_mapping[req.req_pool_idx],
            np.zeros(2, dtype=np.int32),
        )

    # --- Case 1: alloc gives running + 2 track slots (3 total) ---

    def test_alloc_allocates_running_plus_two_track(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True)
        free_before = pool.recurrent_available_size(0)
        req = FakeReq(dp_rank=0)

        pool.alloc([req])

        self.assertIsNotNone(req.recurrent_pool_idx)
        self.assertIsNotNone(req.recurrent_ping_pong_track_buffer)
        self.assertEqual(len(req.recurrent_ping_pong_track_buffer), 2)
        # Free list dropped by 3 (1 running + 2 track).
        self.assertEqual(pool.recurrent_available_size(0), free_before - 3)
        # next_track_idx starts at 0; last_track_seqlen stays None (no scatter yet).
        self.assertEqual(req.recurrent_next_track_idx, 0)
        self.assertIsNone(req.recurrent_last_track_seqlen)
        # Three distinct non-null slots, none is slot 0.
        used = {req.recurrent_pool_idx, *req.recurrent_ping_pong_track_buffer}
        self.assertEqual(len(used), 3)
        self.assertNotIn(0, used)
        # Mapping row reflects the buffer.
        np.testing.assert_array_equal(
            pool.req_index_to_recurrent_ping_pong_track_buffer_mapping[req.req_pool_idx],
            np.array(req.recurrent_ping_pong_track_buffer, dtype=np.int32),
        )

    def test_alloc_returns_none_when_below_three_free(self):
        # 4 slots/rank; one req takes 3, leaving 1 < 3 for the next.
        pool = self._make_pool(enable_recurrent_extra_buffer=True, size=4)
        first = FakeReq(dp_rank=0)
        self.assertIsNotNone(pool.alloc([first]))
        self.assertEqual(pool.recurrent_available_size(0), 1)

        overflow = FakeReq(dp_rank=0)
        self.assertIsNone(pool.alloc([overflow]))
        self.assertIsNone(overflow.req_pool_idx)
        self.assertIsNone(overflow.recurrent_pool_idx)
        self.assertIsNone(overflow.recurrent_ping_pong_track_buffer)

    # --- Case 2: donate keep slot + keep/other index helpers ---

    def test_donate_keep_slot_replaces_and_returns_value(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True)
        req = FakeReq(dp_rank=0)
        pool.alloc([req])

        # next=0 -> overwrite slot 0; keep slot is the other (index 1).
        self.assertEqual(req.recurrent_next_track_idx, 0)
        keep_idx = pool.get_recurrent_ping_pong_keep_idx(req)
        self.assertEqual(keep_idx, 1)
        self.assertEqual(pool.get_recurrent_ping_pong_other_idx(0), 1)
        self.assertEqual(pool.get_recurrent_ping_pong_other_idx(1), 0)

        keep_slot = req.recurrent_ping_pong_track_buffer[keep_idx]
        new_slot = pool.alloc_recurrent_slot(0)

        value = pool.donate_recurrent_ping_pong_slot(req, new_slot)

        # Returned the keep slot's len-1 int32 value.
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.dtype, np.int32)
        self.assertEqual(value.shape, (1,))
        self.assertEqual(int(value[0]), keep_slot)
        # Keep slot replaced by new slot in buffer + mapping.
        self.assertEqual(req.recurrent_ping_pong_track_buffer[keep_idx], new_slot)
        self.assertEqual(
            pool.req_index_to_recurrent_ping_pong_track_buffer_mapping[req.req_pool_idx, keep_idx],
            new_slot,
        )

    def test_set_recurrent_ping_pong_slot_updates_buffer_and_mapping(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True)
        req = FakeReq(dp_rank=0)
        pool.alloc([req])
        new_slot = pool.alloc_recurrent_slot(0)

        pool.set_recurrent_ping_pong_slot(req, 1, new_slot)

        self.assertEqual(req.recurrent_ping_pong_track_buffer[1], new_slot)
        self.assertEqual(
            pool.req_index_to_recurrent_ping_pong_track_buffer_mapping[req.req_pool_idx, 1],
            new_slot,
        )

    # --- Case 3: free_recurrent_cache frees running + both track slots ---

    def test_free_recurrent_cache_frees_running_and_track(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True)
        free_full = pool.recurrent_available_size(0)
        req = FakeReq(dp_rank=0)
        pool.alloc([req])
        running = req.recurrent_pool_idx
        track = list(req.recurrent_ping_pong_track_buffer)
        req_pool_idx = req.req_pool_idx

        pool.free_recurrent_cache(req)

        # All 3 slots back; mapping rows zeroed; fields cleared.
        self.assertEqual(pool.recurrent_available_size(0), free_full)
        self.assertIn(running, pool.recurrent_free_slots[0])
        for slot in track:
            self.assertIn(slot, pool.recurrent_free_slots[0])
        self.assertEqual(pool.req_index_to_recurrent_index_mapping[req_pool_idx], 0)
        np.testing.assert_array_equal(
            pool.req_index_to_recurrent_ping_pong_track_buffer_mapping[req_pool_idx],
            np.zeros(2, dtype=np.int32),
        )
        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIsNone(req.recurrent_ping_pong_track_buffer)
        self.assertIsNone(req.recurrent_next_track_idx)
        self.assertIsNone(req.recurrent_last_track_seqlen)

    # --- Case 4: retract clears the ping-pong fields after free ---

    def test_retract_clears_ping_pong_fields_for_requeue(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True)
        req = FakeReq(dp_rank=0)
        pool.alloc([req])

        pool.free_recurrent_cache(req)
        req.reset_for_retract()

        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIsNone(req.recurrent_ping_pong_track_buffer)
        self.assertIsNone(req.recurrent_next_track_idx)
        self.assertIsNone(req.recurrent_last_track_seqlen)

        # A requeued req reallocates fresh track slots.
        requeued = FakeReq(dp_rank=0)
        pool.alloc([requeued])
        self.assertIsNotNone(requeued.recurrent_ping_pong_track_buffer)
        self.assertEqual(len(requeued.recurrent_ping_pong_track_buffer), 2)

    # --- Case 5: ledger counts request-owned running + track for dp1 and dp2 ---

    def _assert_ledger(self, pool, live_reqs, dp_rank, tree_owned=0):
        slots = pool.slots_per_rank
        free = pool.recurrent_available_size(dp_rank)
        owned = pool.count_request_owned_recurrent_slots(live_reqs, dp_rank)
        # owned (request) + tree_owned (donated, off-pool) + free == slots.
        self.assertEqual(owned + tree_owned + free, slots)

    def test_ledger_invariant_dp1(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True, dp_size=1, size=16)
        reqs = [FakeReq(dp_rank=0), FakeReq(dp_rank=0)]
        pool.alloc(reqs)
        self._assert_ledger(pool, reqs, 0)

        # donate moves the keep slot to the tree and consumes a fresh free slot
        # for the buffer: request-owned count is unchanged, +1 tree-owned slot.
        new_slot = pool.alloc_recurrent_slot(0)
        donated = pool.donate_recurrent_ping_pong_slot(reqs[0], new_slot)
        self._assert_ledger(pool, reqs, 0, tree_owned=1)

        # Simulate tree eviction returning the donated slot.
        pool.free_recurrent_slot(int(donated[0]), 0)
        self._assert_ledger(pool, reqs, 0)

        pool.free_recurrent_cache(reqs[0])
        live = [reqs[1]]
        self._assert_ledger(pool, live, 0)

    def test_ledger_invariant_dp2(self):
        pool = self._make_pool(enable_recurrent_extra_buffer=True, dp_size=2, size=16)
        r0 = FakeReq(dp_rank=0)
        r1 = FakeReq(dp_rank=1)
        pool.alloc([r0])
        pool.alloc([r1])

        self._assert_ledger(pool, [r0], 0)
        self._assert_ledger(pool, [r1], 1)

        pool.free_recurrent_cache(r0)
        self._assert_ledger(pool, [], 0)
        self._assert_ledger(pool, [r1], 1)

    def test_ledger_detects_leaked_track_slot(self):
        """A track slot left allocated but not returned to the free list must make
        owned + free != slots_per_rank (the leak is caught)."""
        pool = self._make_pool(enable_recurrent_extra_buffer=True, dp_size=1, size=16)
        req = FakeReq(dp_rank=0)
        pool.alloc([req])

        # Simulate a leak: free only the running slot, drop the req handle without
        # returning its track slots.
        pool.free_recurrent_slot(req.recurrent_pool_idx, 0)
        leaked = list(req.recurrent_ping_pong_track_buffer)
        live_reqs = []  # request gone, but its track slots never freed

        slots = pool.slots_per_rank
        free = pool.recurrent_available_size(0)
        owned = pool.count_request_owned_recurrent_slots(live_reqs, 0)
        self.assertEqual(len(leaked), 2)
        self.assertNotEqual(owned + free, slots)


if __name__ == "__main__":
    unittest.main()
