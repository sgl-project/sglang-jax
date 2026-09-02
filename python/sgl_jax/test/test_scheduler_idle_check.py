import unittest

from sgl_jax.srt.managers.scheduler import Scheduler


class _Batch:
    reqs_info = []

    def __init__(self, empty):
        self.empty = empty

    def is_empty(self):
        return self.empty

    def batch_size(self):
        return 0 if self.empty else 1


class TestSchedulerIdleCheck(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = []
        scheduler.grammar_queue = []
        scheduler.pending_dp_reqs = []
        scheduler.running_batch = _Batch(empty=True)
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.chunked_reqs = [None, None]
        scheduler.enable_overlap = False
        scheduler.disagg_prefill_queue = None
        scheduler.disagg_prealloc_queue = None
        scheduler.disagg_transfer_queue = None
        scheduler._pd_pending_bootstrap = []
        scheduler.pd = ""
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
            ("running_batch", lambda s: setattr(s, "running_batch", _Batch(empty=False))),
            ("cur_batch", lambda s: setattr(s, "cur_batch", _Batch(empty=False))),
            ("last_batch", lambda s: setattr(s, "last_batch", _Batch(empty=False))),
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
            ("pd_pending_bootstrap", lambda s: s._pd_pending_bootstrap.append(object())),
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


if __name__ == "__main__":
    unittest.main()
