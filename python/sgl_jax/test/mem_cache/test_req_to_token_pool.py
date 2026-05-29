# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_req_to_token_pool.py -v

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.test.test_utils import CustomTestCase


class FakeReq:
    """Minimal Req surrogate exposing only the attributes ReqToTokenPool reads."""

    def __init__(self, req_pool_idx=None, is_chunked=0, kv_committed_len=0):
        self.req_pool_idx = req_pool_idx
        self.is_chunked = is_chunked
        self.kv_committed_len = kv_committed_len


class TestReqToTokenPoolAlloc(CustomTestCase):
    """Contract tests for ReqToTokenPool.alloc(reqs)."""

    def setUp(self):
        self.pool = ReqToTokenPool(size=4, max_context_len=16, dtype=np.int32)

    def test_alloc_fresh_assigns_indices(self):
        reqs = [FakeReq(), FakeReq(), FakeReq()]
        indices = self.pool.alloc(reqs)

        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), len(reqs))
        for req, idx in zip(reqs, indices):
            self.assertIsNotNone(req.req_pool_idx)
            self.assertEqual(req.req_pool_idx, idx)
        self.assertEqual(len(set(indices)), len(indices), "indices must be unique")
        self.assertEqual(self.pool.available_size(), 4 - len(reqs))

    def test_alloc_atomic_on_capacity_miss(self):
        # Shrink free_slots to two slots so a 3-req batch overflows.
        self.pool.free_slots = [0, 1]
        reqs = [FakeReq(), FakeReq(), FakeReq()]

        indices = self.pool.alloc(reqs)

        self.assertIsNone(indices)
        for req in reqs:
            self.assertIsNone(req.req_pool_idx, "req fields must stay untouched")
        self.assertEqual(self.pool.free_slots, [0, 1])

    def test_alloc_mixed_reuse_and_fresh(self):
        retained = FakeReq(req_pool_idx=0, is_chunked=1)
        fresh_a = FakeReq()
        fresh_b = FakeReq()
        self.pool.free_slots = [1, 2, 3]

        indices = self.pool.alloc([retained, fresh_a, fresh_b])

        self.assertIsNotNone(indices)
        self.assertEqual(indices[0], 0)
        self.assertEqual(retained.req_pool_idx, 0)
        self.assertEqual(indices[1], 1)
        self.assertEqual(indices[2], 2)
        self.assertEqual(fresh_a.req_pool_idx, 1)
        self.assertEqual(fresh_b.req_pool_idx, 2)
        self.assertEqual(self.pool.free_slots, [3])

    def test_alloc_atomic_with_partial_reuse(self):
        retained = FakeReq(req_pool_idx=5, is_chunked=1)  # outside [0, size); ok for the test
        # Only one fresh slot left, but two fresh reqs requested -> overflow.
        self.pool.free_slots = [0]
        fresh_a = FakeReq()
        fresh_b = FakeReq()

        indices = self.pool.alloc([retained, fresh_a, fresh_b])

        self.assertIsNone(indices)
        self.assertEqual(retained.req_pool_idx, 5, "reuse req must stay set")
        self.assertIsNone(fresh_a.req_pool_idx)
        self.assertIsNone(fresh_b.req_pool_idx)
        self.assertEqual(self.pool.free_slots, [0])


class TestReqToTokenPoolFree(CustomTestCase):
    def setUp(self):
        self.pool = ReqToTokenPool(size=4, max_context_len=16, dtype=np.int32)

    def test_alloc_then_free_then_alloc_roundtrip(self):
        req = FakeReq()
        self.pool.alloc([req])
        idx = req.req_pool_idx
        self.assertIsNotNone(idx)

        self.pool.free(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertIn(idx, self.pool.free_slots, "freed slot must return to the pool")

        # Next alloc must hand out a slot again (any slot - identity not required).
        self.pool.alloc([req])
        self.assertIsNotNone(req.req_pool_idx)

    def test_double_free_rejected(self):
        """Free of a req without a slot must fail loudly, not pollute free_slots."""
        req = FakeReq()
        self.pool.alloc([req])
        self.pool.free(req)

        free_slots_before = list(self.pool.free_slots)
        with self.assertRaises(AssertionError):
            self.pool.free(req)
        self.assertEqual(self.pool.free_slots, free_slots_before)


class TestReqToTokenPoolChunkedReqLifecycle(CustomTestCase):
    """End-to-end lock-in for chunked-req slot ownership across batches."""

    def setUp(self):
        self.pool = ReqToTokenPool(size=4, max_context_len=16, dtype=np.int32)

    def test_chunked_req_survives_across_batches(self):
        chunked = FakeReq()
        peer = FakeReq()
        self.pool.alloc([chunked, peer])
        chunked_idx = chunked.req_pool_idx
        self.assertIsNotNone(chunked_idx)
        self.assertNotEqual(chunked_idx, peer.req_pool_idx)

        # Scheduler caches the unfinished chunked req without freeing
        # its slot; the peer finishes and is freed normally.
        self.pool.free(peer)
        self.assertIsNone(peer.req_pool_idx)
        chunked.is_chunked = 1

        # Round 2: chunked req comes back with req_pool_idx still set;
        # alloc must treat this as in-place reuse, not a fresh alloc.
        new_peer = FakeReq()
        indices = self.pool.alloc([chunked, new_peer])

        self.assertIsNotNone(indices)
        self.assertEqual(
            chunked.req_pool_idx,
            chunked_idx,
            "chunked req must retain its slot across batches",
        )
        self.assertNotEqual(new_peer.req_pool_idx, chunked_idx)


if __name__ == "__main__":
    unittest.main()
