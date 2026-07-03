import unittest
from types import SimpleNamespace

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class _EmptyBatch:
    def is_empty(self):
        return True

    def batch_size(self):
        return 0


class _NonEmptyBatch:
    def is_empty(self):
        return False

    def batch_size(self):
        return 1


class TestSchedulerIdleCheck(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = []
        scheduler.grammar_queue = []
        scheduler.pending_dp_reqs = []
        scheduler.running_batch = _EmptyBatch()
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.chunked_reqs = [None, None]
        scheduler.enable_overlap = False
        scheduler.disagg_prefill_queue = None
        scheduler.disagg_prealloc_queue = None
        scheduler.disagg_transfer_queue = None
        scheduler.init_new_token_ratio = 0.75
        scheduler.new_token_ratio = 0.25
        scheduler.calls = []
        scheduler.check_memory = lambda: scheduler.calls.append("memory")
        scheduler.check_tree_cache = lambda: scheduler.calls.append("tree")
        return scheduler

    def test_idle_check_skips_pending_work(self):
        cases = [
            ("waiting_queue", lambda s: s.waiting_queue.append(object())),
            ("grammar_queue", lambda s: s.grammar_queue.append(object())),
            ("pending_dp_reqs", lambda s: s.pending_dp_reqs.append(object())),
            ("running_batch", lambda s: setattr(s, "running_batch", _NonEmptyBatch())),
            ("cur_batch", lambda s: setattr(s, "cur_batch", _NonEmptyBatch())),
            ("last_batch", lambda s: setattr(s, "last_batch", _NonEmptyBatch())),
            ("chunked_reqs", lambda s: s.chunked_reqs.__setitem__(1, object())),
            (
                "result_queue",
                lambda s: (
                    setattr(s, "enable_overlap", True),
                    setattr(s, "result_queue", [object()]),
                ),
            ),
            ("disagg_prefill_queue", lambda s: setattr(s, "disagg_prefill_queue", [object()])),
            ("disagg_prealloc_queue", lambda s: setattr(s, "disagg_prealloc_queue", [object()])),
            ("disagg_transfer_queue", lambda s: setattr(s, "disagg_transfer_queue", [object()])),
        ]

        for name, setup in cases:
            with self.subTest(name=name):
                scheduler = self._make_scheduler()
                setup(scheduler)

                scheduler.on_idle()

                self.assertEqual(scheduler.calls, [])
                self.assertEqual(scheduler.new_token_ratio, 0.25)

    def test_idle_check_runs_when_fully_idle(self):
        scheduler = self._make_scheduler()

        scheduler.on_idle()

        self.assertEqual(scheduler.calls, ["memory", "tree"])
        self.assertEqual(scheduler.new_token_ratio, scheduler.init_new_token_ratio)

    def test_chunked_cache_skips_parked_chunk_without_new_kv(self):
        scheduler = self._make_scheduler()
        scheduler.cached = []
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda req: scheduler.cached.append(req.rid)
        )
        req = SimpleNamespace(rid="parked", fill_ids=[1, 2], prefix_indices=[10, 11])

        scheduler._cache_chunked_req_if_needed(req)

        self.assertEqual(scheduler.cached, [])

    def test_chunked_cache_stashes_chunk_with_new_kv(self):
        scheduler = self._make_scheduler()
        scheduler.cached = []
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda req: scheduler.cached.append(req.rid)
        )
        req = SimpleNamespace(rid="chunk", fill_ids=[1, 2, 3], prefix_indices=[10, 11])

        scheduler._cache_chunked_req_if_needed(req)

        self.assertEqual(scheduler.cached, ["chunk"])

    def test_prefill_logprob_allows_missing_next_token_logprobs(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model_config = SimpleNamespace(vocab_size=100)
        req = Req(
            rid="rid",
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=True,
            top_logprobs_num=2,
            token_ids_logprob=[5, 6],
        )
        output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=[-0.1, -0.2, -0.3],
            input_top_logprobs_val=[[[-0.4, -0.5], [-0.6, -0.7], [-0.8, -0.9]]],
            input_top_logprobs_idx=[[[11, 12], [13, 14], [15, 16]]],
            input_token_ids_logprobs_val=[[[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]],
            input_token_ids_logprobs_idx=[[[5, 6], [5, 6], [5, 6]]],
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            next_token_token_ids_logprobs_val=None,
            next_token_token_ids_logprobs_idx=None,
        )

        scheduler.add_logprob_return_values(
            0,
            req,
            0,
            [4],
            3,
            output,
        )

        self.assertEqual(req.output_token_logprobs_val, [])
        self.assertEqual(req.output_token_logprobs_idx, [])
        self.assertEqual(req.output_top_logprobs_val, [])
        self.assertEqual(req.output_top_logprobs_idx, [])
        self.assertEqual(req.output_token_ids_logprobs_val, [])
        self.assertEqual(req.output_token_ids_logprobs_idx, [])
        self.assertEqual(req.input_token_logprobs_val, [None, -0.1, -0.2])
        self.assertEqual(req.input_token_logprobs_idx, [1, 2, 3])
        self.assertEqual(
            req.input_top_logprobs_val,
            [None, [-0.4, -0.5], [-0.6, -0.7]],
        )
        self.assertEqual(req.input_top_logprobs_idx, [None, [11, 12], [13, 14]])
        self.assertEqual(
            req.input_token_ids_logprobs_val,
            [None, [-1.1, -1.2], [-1.3, -1.4]],
        )
        self.assertEqual(req.input_token_ids_logprobs_idx, [None, [5, 6], [5, 6]])


if __name__ == "__main__":
    unittest.main()
