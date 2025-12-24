import unittest
from unittest.mock import Mock

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestPrefillAdderDP(unittest.TestCase):
    def setUp(self):
        self.page_size = 1
        self.dp_size = 2

        # Mock dependencies
        self.tree_cache = Mock()
        self.tree_cache.evictable_size.return_value = 0
        self.tree_cache.full_evictable_size.return_value = 0
        self.tree_cache.swa_evictable_size.return_value = 0
        self.tree_cache.inc_lock_ref = Mock()
        self.tree_cache.dec_lock_ref = Mock()
        self.tree_cache.disable = True  # Disable radix cache complexity for this test

        self.allocator = Mock()
        # Give plenty of memory
        self.allocator.available_size.return_value = 10000
        self.allocator.full_available_size.return_value = 10000
        self.allocator.swa_available_size.return_value = 10000
        self.allocator.dp_size = self.dp_size

        self.running_batch = None

        # Policy params
        self.new_token_ratio = 1.0
        self.rem_input_tokens = 10000  # Large global limit
        self.chunked_prefill_size = 10  # Small chunk size to force chunking

    def create_req(self, rid, dp_rank, length):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[1] * length,
            sampling_params=SamplingParams(max_new_tokens=10),
            dp_rank=dp_rank,
        )
        # Mock internal state needed by PrefillAdder
        req.host_hit_length = 0
        req.prefix_indices = []
        req.last_node = Mock()
        req.extend_input_len = length
        req.fill_ids = req.origin_input_ids
        return req

    def test_concurrent_chunking(self):
        """Test that two requests on different DPs can chunk concurrently."""
        adder = PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.rem_input_tokens,
            rem_chunk_tokens=self.chunked_prefill_size,
            dp_size=self.dp_size,
        )

        # Create two large requests
        # Chunk size is 10.
        # Req 1 (DP0): 20 tokens -> needs 2 chunks
        # Req 2 (DP1): 20 tokens -> needs 2 chunks
        req0 = self.create_req("req0", 0, 20)
        req1 = self.create_req("req1", 1, 20)

        # Add req0
        adder.add_one_req(req0)

        # Verify req0 was added and chunked
        self.assertIn(req0, adder.can_run_list)
        self.assertEqual(req0.extend_input_len, 10)  # Should be truncated to chunk size
        self.assertEqual(adder.new_chunked_reqs[0], req0)
        self.assertIsNone(adder.new_chunked_reqs[1])

        # Add req1
        adder.add_one_req(req1)

        # Verify req1 was added and chunked INDEPENDENTLY
        self.assertIn(req1, adder.can_run_list)
        self.assertEqual(
            req1.extend_input_len, 10
        )  # Should also be 10, not 0 (if budget was shared/exhausted)
        self.assertEqual(adder.new_chunked_reqs[1], req1)

        # Verify both are tracked
        self.assertEqual(len(adder.can_run_list), 2)
        self.assertEqual(adder.new_chunked_reqs[0], req0)
        self.assertEqual(adder.new_chunked_reqs[1], req1)

    def test_budget_exhaustion(self):
        """Test that adding more requests to a full DP is rejected/handled."""
        adder = PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.rem_input_tokens,
            rem_chunk_tokens=self.chunked_prefill_size,  # 10
            dp_size=self.dp_size,
        )

        req0_a = self.create_req("req0_a", 0, 20)

        # Use up budget for DP0
        adder.add_one_req(req0_a)
        self.assertEqual(req0_a.extend_input_len, 10)
        self.assertEqual(adder.rem_chunk_tokens_list[0], 0)

        # Try to add another req for DP0
        req0_b = self.create_req("req0_b", 0, 5)
        res = adder.add_one_req(req0_b)

        # Should be rejected or result in OTHER (no token)
        self.assertNotIn(req0_b, adder.can_run_list)
        self.assertEqual(res, AddReqResult.OTHER)

        # BUT, should still be able to add for DP1
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        req1 = self.create_req("req1", 1, 5)
        res1 = adder.add_one_req(req1)

        self.assertIn(req1, adder.can_run_list)
        self.assertEqual(res1, AddReqResult.CONTINUE)


if __name__ == "__main__":
    unittest.main()
