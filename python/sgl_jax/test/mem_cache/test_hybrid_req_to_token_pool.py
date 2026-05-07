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
    """Stub exposing only the interface HybridReqToTokenPool reads."""

    def __init__(self, size: int):
        self.size = size
        self.cleared_slots: list[list[int]] = []

    def clear_slot(self, indices: list[int]):
        self.cleared_slots.append(list(indices))

    def clear(self):
        self.cleared_slots.clear()


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

    def test_alloc_clears_newly_assigned_slots(self):
        reqs = [FakeReq(dp_rank=0), FakeReq(dp_rank=0)]
        self.pool.alloc(reqs)

        self.assertEqual(len(self.state_pool.cleared_slots), 1)
        cleared = self.state_pool.cleared_slots[0]
        self.assertEqual(sorted(cleared), sorted([r.recurrent_pool_idx for r in reqs]))

    def test_alloc_updates_mapping(self):
        reqs = [FakeReq(dp_rank=0)]
        self.pool.alloc(reqs)

        req = reqs[0]
        mapped = self.pool.req_index_to_recurrent_index_mapping[req.req_pool_idx]
        self.assertEqual(mapped, req.recurrent_pool_idx)

    def test_alloc_fails_when_recurrent_slots_exhausted(self):
        slots_per_rank = self.state_pool.size // 1
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


if __name__ == "__main__":
    unittest.main()
