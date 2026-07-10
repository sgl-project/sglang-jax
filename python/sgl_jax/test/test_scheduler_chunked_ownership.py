import unittest
from types import SimpleNamespace

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestSchedulerChunkedOwnership(unittest.TestCase):
    def _make_req(self, rid, fill_ids, prefix_indices, dp_rank=0):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=fill_ids,
            sampling_params=SamplingParams(max_new_tokens=8),
            dp_rank=dp_rank,
            eos_token_ids={2},
            vocab_size=100,
        )
        req.fill_ids = fill_ids
        req.prefix_indices = prefix_indices
        req.req_pool_idx = 6
        return req

    def _make_batch(self, reqs, chunked_reqs):
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=100),
            enable_overlap=False,
            dp_size=len(reqs),
            chunked_reqs=chunked_reqs,
            mesh=None,
            spec_algorithm=None,
        )
        batch.forward_mode = SimpleNamespace(is_extend=lambda: True)
        return batch

    def _make_scheduler(self, req, active_req=None):
        cached = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_size = 1
        scheduler.pd = None
        scheduler.chunked_reqs = [active_req]
        scheduler.last_batch = self._make_batch([[req]], [req])
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda cached_req: cached.append(cached_req)
        )
        return scheduler, cached

    def test_last_batch_chunked_req_restores_scheduler_ownership(self):
        req = self._make_req("chunked", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req)

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertIs(scheduler.chunked_reqs[0], req)
        self.assertIs(excluded[0], req)
        self.assertTrue(scheduler.last_batch.is_empty())
        self.assertEqual(req.req_pool_idx, 6)
        self.assertEqual(cached, [req])

    def test_parked_chunk_without_new_kv_is_not_cached_again(self):
        req = self._make_req("parked", [1, 2], [10, 11])
        scheduler, cached = self._make_scheduler(req, active_req=req)

        excluded = scheduler._prepare_chunked_reqs_to_exclude()

        self.assertIs(excluded[0], req)
        self.assertEqual(cached, [])

    def test_conflicting_chunked_owner_is_rejected(self):
        batch_req = self._make_req("batch", [1, 2, 3], [10, 11])
        active_req = self._make_req("active", [4, 5, 6], [12, 13])
        scheduler, _ = self._make_scheduler(batch_req, active_req=active_req)

        with self.assertRaisesRegex(AssertionError, "Chunked request mismatch"):
            scheduler._prepare_chunked_reqs_to_exclude()

    def test_restoring_one_dp_rank_preserves_the_other_rank_owner(self):
        req0 = self._make_req("rank-0", [1, 2, 3], [10, 11], dp_rank=0)
        req1 = self._make_req("rank-1", [4, 5, 6], [12, 13], dp_rank=1)
        cached = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_size = 2
        scheduler.pd = None
        scheduler.chunked_reqs = [None, req1]
        scheduler.last_batch = self._make_batch([[req0], [req1]], [req0, req1])
        scheduler.tree_cache = SimpleNamespace(cache_unfinished_req=lambda req: cached.append(req))

        excluded = scheduler._prepare_chunked_reqs_to_exclude()

        self.assertEqual(scheduler.chunked_reqs, [req0, req1])
        self.assertIs(excluded[0], req0)
        self.assertIs(excluded[1], req1)
        self.assertEqual(cached, [req0, req1])

    def test_final_chunk_stays_in_batch_without_creating_an_owner(self):
        req = self._make_req("final", [1, 2, 3], [10, 11])
        cached = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_size = 1
        scheduler.pd = None
        scheduler.chunked_reqs = [None]
        scheduler.last_batch = self._make_batch([[req]], [None])
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda cached_req: cached.append(cached_req)
        )

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertEqual(excluded, {})
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [req])
        self.assertEqual(cached, [])

    def test_pd_prefill_caches_chunked_req_in_prefill_tree(self):
        req = self._make_req("pd-prefill", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req, active_req=req)
        pd_cached = []
        scheduler.pd = "prefill"
        scheduler.p_tree = SimpleNamespace(
            cache_unfinished_req=lambda cached_req: pd_cached.append(cached_req)
        )

        excluded = scheduler._prepare_chunked_reqs_to_exclude()

        self.assertIs(excluded[0], req)
        self.assertEqual(pd_cached, [req])
        self.assertEqual(cached, [])


if __name__ == "__main__":
    unittest.main()
