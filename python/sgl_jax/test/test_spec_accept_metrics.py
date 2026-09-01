"""Accept-length telemetry for speculative decoding.

``avg_spec_accept_length`` is read by ``bench_one_batch_server.py`` and
``bench_serving.py`` out of ``/get_server_info``; these tests pin the three
things those clients depend on: the None-vs-float contract, the counters
existing under the names the output processor increments, and the cumulative
counters surviving the per-interval reset in ``log_decode_stats``.
"""

import types
import unittest

from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.scheduler_metrics_mixin import (
    SchedulerMetricsMixin,
    compute_avg_spec_accept_length,
)


class _Batch:
    dp_size = 1
    reqs_info = []
    cache_miss_count = 0

    def is_empty(self):
        return True

    def batch_size(self):
        return 2


class _SpecAlgorithm:
    def is_none(self):
        return False


class _Req:
    def __init__(self, retracted=False):
        self.is_retracted = retracted

    def finished(self):
        return False


class _SpecBatch:
    def __init__(self, reqs, per_dp_bs):
        self.per_dp_bs_size = per_dp_bs
        self.reqs_info = [types.SimpleNamespace(reqs=reqs)]


class _KVCache:
    mem_usage = 1.0


class _Allocator:
    def get_kvcache(self):
        return _KVCache()

    def available_size(self, dp_rank=None):
        return 100


class TestComputeAvgSpecAcceptLength(unittest.TestCase):
    def test_returns_none_before_any_verify_step(self):
        # Not 0.0: the bench clients print any float verbatim and only fall
        # back to "n/a" on None.
        self.assertIsNone(compute_avg_spec_accept_length(0, 0))

    def test_ratio(self):
        self.assertAlmostEqual(compute_avg_spec_accept_length(35, 10), 3.5)

    def test_no_speculative_gain_is_one(self):
        # Every verify step commits the bonus token even if all drafts are
        # rejected, so the floor of the metric is 1.0, not 0.0.
        self.assertAlmostEqual(compute_avg_spec_accept_length(10, 10), 1.0)


class TestSpecAcceptCounters(unittest.TestCase):
    def test_init_metrics_defines_the_counters_the_processor_increments(self):
        obj = types.SimpleNamespace()
        SchedulerMetricsMixin.init_metrics(obj)
        self.assertEqual(obj.cum_spec_accept_length, 0)
        self.assertEqual(obj.cum_spec_accept_count, 0)

    def test_accounting_moves_both_counter_pairs(self):
        scheduler = Scheduler.__new__(Scheduler)
        SchedulerMetricsMixin.init_metrics(scheduler)
        scheduler.num_generated_tokens = 0
        scheduler.accept_token = 0
        scheduler.draft_token = 0
        scheduler.spec_num_forward_ct = 0
        scheduler.enable_overlap = False
        scheduler.draft_worker = types.SimpleNamespace(speculative_num_draft_tokens=4)
        # Two live requests committing 3 and 1 tokens, plus one retracted slot
        # that must not be counted at all.
        batch = _SpecBatch(
            [_Req(), _Req(), _Req(retracted=True)],
            per_dp_bs=3,
        )
        next_token_ids = [[11, 12, 13], [21], [31, 32]]

        scheduler.account_spec_decode_tokens(batch, next_token_ids)

        self.assertEqual(scheduler.cum_spec_accept_length, 4)  # 3 + 1
        self.assertEqual(scheduler.cum_spec_accept_count, 2)  # two live reqs
        # The per-interval pair must move by exactly the same amounts, or the
        # logged accept-len and the exported average would disagree.
        self.assertEqual(scheduler.accept_token, 4)
        self.assertEqual(scheduler.spec_num_forward_ct, 2)
        self.assertEqual(scheduler.num_generated_tokens, 4)
        self.assertEqual(scheduler.draft_token, 8)  # 2 reqs * 4 draft tokens
        self.assertAlmostEqual(
            compute_avg_spec_accept_length(
                scheduler.cum_spec_accept_length, scheduler.cum_spec_accept_count
            ),
            2.0,
        )

    def test_log_decode_stats_reset_does_not_touch_cumulative_counters(self):
        scheduler = Scheduler.__new__(Scheduler)
        SchedulerMetricsMixin.init_metrics(scheduler)
        scheduler.running_batch = _Batch()
        scheduler.last_decode_stats_tic = 0.0
        scheduler.num_generated_tokens = 30
        scheduler.is_hybrid = False
        scheduler._get_token_info = lambda: (0, 0.0, 0, 0)
        scheduler.spec_algorithm = _SpecAlgorithm()
        scheduler.waiting_queue = []
        scheduler.server_args = types.SimpleNamespace(decode_log_interval=1)
        # One interval's worth of the counters log_decode_stats consumes...
        scheduler.accept_token = 30
        scheduler.draft_token = 40
        scheduler.spec_num_forward_ct = 10
        # ...and the cumulative twins covering two intervals.
        scheduler.cum_spec_accept_length = 60
        scheduler.cum_spec_accept_count = 20

        scheduler.log_decode_stats()

        self.assertEqual(scheduler.accept_token, 0)
        self.assertEqual(scheduler.draft_token, 0)
        self.assertEqual(scheduler.spec_num_forward_ct, 0)
        self.assertEqual(scheduler.cum_spec_accept_length, 60)
        self.assertEqual(scheduler.cum_spec_accept_count, 20)
        self.assertAlmostEqual(
            compute_avg_spec_accept_length(
                scheduler.cum_spec_accept_length, scheduler.cum_spec_accept_count
            ),
            3.0,
        )


class TestGetInternalStateExposesAcceptLength(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        SchedulerMetricsMixin.init_metrics(scheduler)
        scheduler.token_to_kv_pool_allocator = _Allocator()
        scheduler.max_total_num_tokens = 1024
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.pending_dp_reqs = []
        scheduler.running_batch = _Batch()
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.chunked_reqs = [None]
        scheduler.tree_cache = None
        scheduler.req_to_token_pool = None
        scheduler.dp_size = 1
        scheduler.num_generated_tokens = 0
        scheduler.forward_ct_decode = 0
        scheduler.new_token_ratio = 0.25
        scheduler.init_new_token_ratio = 0.75
        scheduler.disagg_prefill_queue = None
        scheduler.disagg_prealloc_queue = None
        scheduler.disagg_transfer_queue = None
        return scheduler

    def test_key_is_none_without_spec_decode(self):
        scheduler = self._make_scheduler()
        state = scheduler.get_internal_state(None).internal_state
        self.assertIn("avg_spec_accept_length", state)
        self.assertIsNone(state["avg_spec_accept_length"])

    def test_key_reports_the_running_average(self):
        scheduler = self._make_scheduler()
        scheduler.cum_spec_accept_length = 27
        scheduler.cum_spec_accept_count = 9
        state = scheduler.get_internal_state(None).internal_state
        self.assertAlmostEqual(state["avg_spec_accept_length"], 3.0)


if __name__ == "__main__":
    unittest.main()
