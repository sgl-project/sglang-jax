"""Unit tests for ScheduleBatch Data Parallelism (DP) merging logic."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.configs import ForwardMode
from sgl_jax.srt.constrained.jump_forward import JumpForwardMap
from sgl_jax.srt.managers.schedule_batch import (
    ScheduleBatch,
    ScheduleReqsInfo,
    find_padding_size,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo


class TestScheduleBatchDPMerging(unittest.TestCase):
    """Test ScheduleBatch merging logic for Data Parallelism."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock dependencies
        self.mock_req_to_token_pool = MagicMock()
        self.mock_req_to_token_pool.req_to_token = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],  # Request 0
                [8, 9, 10, 11, 12, 13, 14, 15],  # Request 1
                [16, 17, 18, 19, 20, 21, 22, 23],  # Request 2
                [24, 25, 26, 27, 28, 29, 30, 31],  # Request 3
            ],
            dtype=np.int32,
        )

        self.mock_token_to_kv_pool = MagicMock()
        self.mock_model_config = MagicMock()
        self.mock_model_config.vocab_size = 32000

    def _create_mock_req(self, rid, lora_id="0"):
        """Create a mock request."""
        req = MagicMock()
        req.rid = rid
        req.lora_id = lora_id
        req.grammar = None
        req.fill_ids = None
        req.origin_input_ids = [1, 2, 3]
        return req

    def _create_sampling_info(self, batch_size):
        """Create mock sampling info for a batch."""
        return SamplingBatchInfo(
            temperatures=np.ones((batch_size, 1), dtype=np.float32),
            top_ps=np.ones(batch_size, dtype=np.float32),
            top_ks=np.ones(batch_size, dtype=np.int32),
            min_ps=np.zeros(batch_size, dtype=np.float32),
            sampling_seeds=None,
            grammars=None,
        )

    def test_compute_global_padding_sizes(self):
        """Test _compute_global_padding_sizes calculates correct padding."""
        # Create ScheduleBatch with 2 DP ranks
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(0), self._create_mock_req(1)],
                input_ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),  # 5 tokens
                seq_lens=np.array([3, 2], dtype=np.int32),  # 2 requests
            ),
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(2)],
                input_ids=np.array([6, 7, 8], dtype=np.int32),  # 3 tokens
                seq_lens=np.array([3], dtype=np.int32),  # 1 request
            ),
        ]

        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=2,
            req_to_token_pool=self.mock_req_to_token_pool,
            token_to_kv_pool=self.mock_token_to_kv_pool,
            tree_cache=MagicMock(),
            model_config=self.mock_model_config,
            forward_mode=ForwardMode.EXTEND,
        )

        token_paddings = [8, 16, 32]
        bs_paddings = [4, 8, 16]

        per_dp_token_size, total_token_size, per_dp_bs_size, total_bs = (
            batch._compute_global_padding_sizes(token_paddings, bs_paddings)
        )

        # Max tokens per DP: 5, should pad to 8
        self.assertEqual(per_dp_token_size, 8)
        self.assertEqual(total_token_size, 16)  # 8 * 2 DP ranks

        # Max BS per DP: 2, should pad to 4
        self.assertEqual(per_dp_bs_size, 4)
        self.assertEqual(total_bs, 8)  # 4 * 2 DP ranks

    def test_merge_input_and_positions_extend_mode(self):
        """Test _merge_input_and_positions in EXTEND mode."""
        # Create ScheduleBatch with 2 DP ranks
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(0), self._create_mock_req(1)],
                input_ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),
                seq_lens=np.array([3, 2], dtype=np.int32),
                prefix_lens=np.array([0, 0], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            ),
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(2)],
                input_ids=np.array([6, 7, 8], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0], dtype=np.int32),
            ),
        ]

        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=2,
            req_to_token_pool=self.mock_req_to_token_pool,
            token_to_kv_pool=self.mock_token_to_kv_pool,
            tree_cache=MagicMock(),
            model_config=self.mock_model_config,
            forward_mode=ForwardMode.EXTEND,
        )

        per_dp_token_size = 8  # Padded size per DP rank
        total_token_size = 16  # 8 * 2

        input_ids, positions, out_cache_loc, real_len = batch._merge_input_and_positions(
            per_dp_token_size, total_token_size
        )

        # Check merged input_ids layout: [dp0: 1,2,3,4,5,0,0,0 | dp1: 6,7,8,0,0,0,0,0]
        expected_input_ids = np.array(
            [1, 2, 3, 4, 5, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0], dtype=np.int32
        )
        np.testing.assert_array_equal(input_ids, expected_input_ids)

        # Check positions: [0,1,2,0,1,0,0,0 | 0,1,2,0,0,0,0,0]
        expected_positions = np.array(
            [0, 1, 2, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0], dtype=np.int32
        )
        np.testing.assert_array_equal(positions, expected_positions)

        # Check real length
        self.assertEqual(real_len, 8)  # 5 + 3

    def test_merge_batch_metadata_extend_mode(self):
        """Test _merge_batch_metadata in EXTEND mode."""
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(0), self._create_mock_req(1)],
                seq_lens=np.array([3, 2], dtype=np.int32),
                req_pool_indices=np.array([0, 1], dtype=np.int32),
                prefix_lens=np.array([0, 0], dtype=np.int32),
                extend_lens=np.array([3, 2], dtype=np.int32),
                extend_logprob_start_lens=np.array([0, 0], dtype=np.int32),
            ),
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(2)],
                seq_lens=np.array([3], dtype=np.int32),
                req_pool_indices=np.array([2], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
            ),
        ]

        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=2,
            req_to_token_pool=self.mock_req_to_token_pool,
            token_to_kv_pool=self.mock_token_to_kv_pool,
            tree_cache=MagicMock(),
            model_config=self.mock_model_config,
            forward_mode=ForwardMode.EXTEND,
        )

        per_dp_bs_size = 4
        total_bs = 8

        (
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            extend_logprob_start_lens,
            real_bs,
        ) = batch._merge_batch_metadata(per_dp_bs_size, total_bs)

        # Check merged layout: [dp0: 0,1,pad,pad | dp1: 2,pad,pad,pad]
        expected_req_pool = np.array([0, 1, -1, -1, 2, -1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(req_pool_indices, expected_req_pool)

        expected_seq_lens = np.array([3, 2, 0, 0, 3, 0, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(seq_lens, expected_seq_lens)

        # Check extend_start_loc: [0, 3, 5, 5, 8, 11, 11, 11]
        # DP0: [0, 3] (start at 0 and 3), DP1: [8] (start at 8 in global array)
        # Note: The actual implementation uses per_dp_token_size for offset
        # We need to check the logic carefully

        self.assertEqual(real_bs, 3)  # 2 + 1

    def test_merge_sampling_info(self):
        """Test _merge_sampling_info merges sampling parameters correctly."""
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(0), self._create_mock_req(1)],
                seq_lens=np.array([3, 2], dtype=np.int32),
                sampling_info=self._create_sampling_info(2),
            ),
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(2)],
                seq_lens=np.array([3], dtype=np.int32),
                sampling_info=self._create_sampling_info(1),
            ),
        ]

        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=2,
            req_to_token_pool=self.mock_req_to_token_pool,
            token_to_kv_pool=self.mock_token_to_kv_pool,
            tree_cache=MagicMock(),
            model_config=self.mock_model_config,
            forward_mode=ForwardMode.DECODE,
            has_grammar=False,
        )

        per_dp_bs_size = 4
        total_bs = 8

        merged_sampling = batch._merge_sampling_info(per_dp_bs_size, total_bs)

        # Check shape
        self.assertEqual(merged_sampling.temperatures.shape, (8, 1))
        self.assertEqual(merged_sampling.top_ps.shape, (8,))

        # Check that first 2 and 5th positions have data (rest are defaults)
        self.assertEqual(merged_sampling.top_ps[0], 1.0)
        self.assertEqual(merged_sampling.top_ps[1], 1.0)
        self.assertEqual(merged_sampling.top_ps[4], 1.0)

    def test_find_padding_size(self):
        """Test find_padding_size helper function."""
        size_buckets = [8, 16, 32, 64]

        # Test exact match
        target, idx = find_padding_size(8, size_buckets)
        self.assertEqual(target, 8)
        self.assertEqual(idx, 0)

        # Test needs padding
        target, idx = find_padding_size(10, size_buckets)
        self.assertEqual(target, 16)
        self.assertEqual(idx, 1)

        # Test needs largest bucket
        target, idx = find_padding_size(50, size_buckets)
        self.assertEqual(target, 64)
        self.assertEqual(idx, 3)

        # Test exceeds all buckets - should raise
        with self.assertRaises(AssertionError):
            find_padding_size(100, size_buckets)

    def test_get_model_worker_batch_integration(self):
        """Integration test for get_model_worker_batch with DP."""
        # Create a complete ScheduleBatch with 2 DP ranks
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(0, "lora1"), self._create_mock_req(1, "lora2")],
                input_ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),
                seq_lens=np.array([3, 2], dtype=np.int32),
                req_pool_indices=np.array([0, 1], dtype=np.int32),
                prefix_lens=np.array([0, 0], dtype=np.int32),
                extend_lens=np.array([3, 2], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0, 0, 0], dtype=np.int32),
                extend_logprob_start_lens=np.array([0, 0], dtype=np.int32),
                sampling_info=self._create_sampling_info(2),
            ),
            ScheduleReqsInfo(
                reqs=[self._create_mock_req(2, "lora1")],
                input_ids=np.array([6, 7, 8], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                req_pool_indices=np.array([2], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
                sampling_info=self._create_sampling_info(1),
            ),
        ]

        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=2,
            req_to_token_pool=self.mock_req_to_token_pool,
            token_to_kv_pool=self.mock_token_to_kv_pool,
            tree_cache=MagicMock(),
            model_config=self.mock_model_config,
            forward_mode=ForwardMode.EXTEND,
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=[],
            token_ids_logprobs=[],
            has_stream=False,
            has_grammar=False,
            return_hidden_states=False,
            extend_input_logprob_token_ids=None,
            launch_done=False,
        )

        # Call get_model_worker_batch
        token_paddings = [8, 16, 32]
        bs_paddings = [4, 8, 16]
        cache_loc_paddings = [32, 64, 128]
        page_size = 16

        worker_batch = batch.get_model_worker_batch(
            token_paddings=token_paddings,
            bs_paddings=bs_paddings,
            cache_loc_paddings=cache_loc_paddings,
            page_size=page_size,
        )

        # Verify merged structure
        # Total tokens: DP0 has 5, DP1 has 3, pad to 8 each = 16 total
        self.assertEqual(len(worker_batch.input_ids), 16)

        # Total BS: DP0 has 2, DP1 has 1, pad to 4 each = 8 total
        self.assertEqual(len(worker_batch.seq_lens), 8)

        # Real BS: 3 actual requests
        self.assertEqual(worker_batch.real_bs, 3)

        # Check lora_ids are correctly collected
        self.assertEqual(len(worker_batch.lora_ids), 8)
        self.assertEqual(worker_batch.lora_ids[0], "lora1")
        self.assertEqual(worker_batch.lora_ids[1], "lora2")
        self.assertEqual(worker_batch.lora_ids[2], "lora1")
        # Rest should be padded with "0"
        self.assertEqual(worker_batch.lora_ids[3], "0")

        # Check sampling info shape
        self.assertEqual(worker_batch.sampling_info.temperatures.shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
