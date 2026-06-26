"""EXTEND bs-bucket selection + selected-shape backstop for the multi-host
recurrent path (per_dp_bs guard). See get_model_worker_batch / _merge_cache_loc
in schedule_batch.py."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo


def _sampling_info(bs):
    return SamplingBatchInfo(
        temperatures=np.ones((bs, 1), dtype=np.float32),
        top_ps=np.ones(bs, dtype=np.float32),
        top_ks=np.ones(bs, dtype=np.int32),
        min_ps=np.zeros(bs, dtype=np.float32),
        vocab_size=32000,
    )


def _mock_req(rid):
    req = MagicMock()
    req.rid = rid
    req.lora_id = "0"
    req.grammar = None
    req.fill_ids = None
    req.origin_input_ids = [1, 2, 3]
    return req


class TestExtendBsGuard(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock()
        self.pool.req_to_token = np.arange(64, dtype=np.int32).reshape(8, 8)
        self.model_config = MagicMock()
        self.model_config.vocab_size = 32000

    def _extend_batch(self, dp_size=2, is_hybrid_recurrent=False):
        # 2 reqs on rank 0, 1 on rank 1 -> max_bs_per_dp = 2.
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[_mock_req(0), _mock_req(1)],
                input_ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),
                seq_lens=np.array([3, 2], dtype=np.int32),
                req_pool_indices=np.array([0, 1], dtype=np.int32),
                prefix_lens=np.array([0, 0], dtype=np.int32),
                extend_lens=np.array([3, 2], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0, 0, 0], dtype=np.int32),
                extend_logprob_start_lens=np.array([0, 0], dtype=np.int32),
                sampling_info=_sampling_info(2),
            ),
            ScheduleReqsInfo(
                reqs=[_mock_req(2)],
                input_ids=np.array([6, 7, 8], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                req_pool_indices=np.array([2], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                out_cache_loc=np.array([0, 0, 0], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
                sampling_info=_sampling_info(1),
            ),
        ]
        return ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=dp_size,
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            model_config=self.model_config,
            forward_mode=ForwardMode.EXTEND,
            is_hybrid_recurrent=is_hybrid_recurrent,
        )

    def test_backstop_raises_when_selected_per_dp_exceeds_safe(self):
        batch = self._extend_batch(dp_size=2)
        # Non-recurrent -> forced largest bucket 64 -> per_dp 32 > 8 -> raise.
        with self.assertRaises(RuntimeError):
            batch.get_model_worker_batch(
                token_paddings=[8, 16, 32],
                bs_paddings=[64],
                cache_loc_paddings=[64],
                page_size=1,
                extend_guard_per_dp_bs=8,
            )

    def _selected_per_dp_bs(self, batch, bs_paddings, extend_guard_per_dp_bs):
        _, resolved_bs, _, active = batch._resolve_extend_paddings(
            [8, 16, 32], list(bs_paddings), [16, 32, 64, 256], 1, extend_guard_per_dp_bs
        )
        _, _, per_dp_bs_padding, _ = batch._compute_global_padding_sizes([8, 16, 32], resolved_bs)
        return per_dp_bs_padding, active

    def test_recurrent_extend_selects_active_bucket(self):
        batch = self._extend_batch(dp_size=2, is_hybrid_recurrent=True)  # affected path
        # max_bs_per_dp=2 -> total 4 -> smallest bucket 4 -> per_dp 2, not 32.
        per_dp_bs, active = self._selected_per_dp_bs(
            batch, [4, 8, 16, 64], extend_guard_per_dp_bs=8
        )
        self.assertTrue(active)
        self.assertEqual(per_dp_bs, 2)

    def test_guard_off_forces_largest_bucket(self):
        batch = self._extend_batch(dp_size=2, is_hybrid_recurrent=True)
        # guard off -> legacy largest bucket 64 -> per_dp 32 even on a recurrent batch.
        per_dp_bs, active = self._selected_per_dp_bs(
            batch, [4, 8, 16, 64], extend_guard_per_dp_bs=0
        )
        self.assertFalse(active)
        self.assertEqual(per_dp_bs, 32)

    def test_non_recurrent_forces_largest_bucket(self):
        batch = self._extend_batch(dp_size=2, is_hybrid_recurrent=False)
        per_dp_bs, active = self._selected_per_dp_bs(
            batch, [4, 8, 16, 64], extend_guard_per_dp_bs=8
        )
        self.assertFalse(active)
        self.assertEqual(per_dp_bs, 32)


if __name__ == "__main__":
    unittest.main()
