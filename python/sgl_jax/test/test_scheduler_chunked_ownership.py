import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sgl_jax.srt.managers.io_struct import AbortReq, PauseGenerationReqInput
from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestSchedulerChunkedOwnership(unittest.TestCase):
    def _make_req(self, rid, fill_ids, prefix_indices, dp_rank=0, return_logprob=False):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=fill_ids,
            sampling_params=SamplingParams(max_new_tokens=8),
            dp_rank=dp_rank,
            eos_token_ids={2},
            vocab_size=100,
            return_logprob=return_logprob,
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
        batch.per_dp_bs_size = max((len(rank_reqs) for rank_reqs in reqs), default=0)
        return batch

    def _make_scheduler(self, reqs, active_reqs=None, batch_owners=None):
        if isinstance(reqs, Req):
            reqs = [[reqs]]
        dp_size = len(reqs)
        if active_reqs is None:
            active_reqs = [None] * dp_size
        elif isinstance(active_reqs, Req):
            active_reqs = [active_reqs]
        if batch_owners is None:
            batch_owners = [rank_reqs[0] if rank_reqs else None for rank_reqs in reqs]

        cached = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_size = dp_size
        scheduler.pd = ""
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.chunked_reqs = active_reqs
        scheduler._pending_chunked_abort_reqs = [None] * dp_size
        scheduler.last_batch = self._make_batch(reqs, batch_owners)
        scheduler.cur_batch = scheduler.last_batch
        scheduler.running_batch = self._make_batch([[] for _ in range(dp_size)], [None] * dp_size)
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda cached_req: cached.append(cached_req)
        )
        scheduler.spec_algorithm = None
        scheduler.is_generation = True
        scheduler.is_hybrid = False
        scheduler.is_mixed_chunk = False
        scheduler.waiting_queue = []
        scheduler.grammar_queue = []
        scheduler.disagg_prefill_queue = None
        scheduler.disagg_prealloc_queue = None
        scheduler.disagg_transfer_queue = None
        scheduler._pd_pending_bootstrap = []
        scheduler._pending_h2d = []
        scheduler._comm_backend = None
        scheduler.policy = SimpleNamespace(calc_priority=Mock())
        scheduler.page_size = 1
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.new_token_ratio = 0.5
        scheduler.max_prefill_tokens = 8
        scheduler.chunked_prefill_size = 2
        scheduler.lora_paths = None
        scheduler.send_to_tokenizer = SimpleNamespace(send_pyobj=Mock())
        scheduler.send_to_detokenizer = SimpleNamespace(send_pyobj=Mock())
        scheduler._release_prefill_host_buffer = Mock()
        scheduler.set_next_batch_sampling_info_done = Mock()
        scheduler.stream_output = Mock()
        scheduler.skip_tokenizer_init = False
        scheduler.stream_interval = 1
        return scheduler, cached

    def _make_prefill_result(self, next_token_ids, input_token_logprobs=None):
        logits_output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=input_token_logprobs,
            hidden_states=None,
        )
        batch_size = len(next_token_ids)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            extend_input_len_per_req=[2] * batch_size,
            extend_logprob_start_len_per_req=[0] * batch_size,
            cache_miss_count=0,
            bid=1,
        )

    def test_parked_chunk_without_new_kv_is_not_cached_again(self):
        req = self._make_req("parked", [1, 2], [10, 11])
        scheduler, cached = self._make_scheduler(req, active_reqs=req)

        excluded = scheduler._prepare_chunked_reqs_to_exclude()

        self.assertIs(excluded[0], req)
        self.assertEqual(cached, [])

    def test_conflicting_chunked_owner_is_rejected(self):
        batch_req = self._make_req("batch", [1, 2, 3], [10, 11])
        active_req = self._make_req("active", [4, 5, 6], [12, 13])
        scheduler, _ = self._make_scheduler(batch_req, active_reqs=active_req)

        with self.assertRaisesRegex(AssertionError, "Chunked request mismatch"):
            scheduler._prepare_chunked_reqs_to_exclude()

    def test_restoring_one_dp_rank_preserves_the_other_rank_owner(self):
        req0 = self._make_req("rank-0", [1, 2, 3], [10, 11], dp_rank=0)
        req1 = self._make_req("rank-1", [4, 5, 6], [12, 13], dp_rank=1)
        scheduler, cached = self._make_scheduler([[req0], [req1]], active_reqs=[None, req1])

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertEqual(scheduler.chunked_reqs, [req0, req1])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None, None])
        self.assertIs(excluded[0], req0)
        self.assertIs(excluded[1], req1)
        self.assertTrue(scheduler.last_batch.is_empty())
        self.assertEqual(req0.req_pool_idx, 6)
        self.assertEqual(req1.req_pool_idx, 6)
        self.assertEqual(cached, [req0, req1])

    def test_final_chunk_stays_in_batch_without_creating_an_owner(self):
        req = self._make_req("final", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req, batch_owners=[None])

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertEqual(excluded, {})
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [req])
        self.assertEqual(cached, [])

    def test_abort_releases_parked_chunk_and_clears_owner(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req)

        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler.abort_request(AbortReq(rid=req.rid))
            self.assertIs(scheduler.chunked_reqs[0], req)
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            release.assert_not_called()

            excluded = scheduler._prepare_chunked_reqs_to_exclude()
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            self.assertEqual(cached, [])
            scheduler._process_pending_chunked_aborts()
            scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        sent = scheduler.send_to_tokenizer.send_pyobj.call_args.args[0]
        self.assertEqual(sent.rid, req.rid)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])

    def test_pause_retires_consumed_chunk_owner_from_batch(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.abort_request(AbortReq(rid=req.rid))

        with patch("sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])
        self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)
        scheduler._sync_chunked_req_owners()
        self.assertEqual(scheduler.chunked_reqs, [None])

    def test_abort_consumes_parked_chunk_while_engine_is_paused(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler.abort_request(AbortReq(rid=req.rid))

        release.assert_called_once()
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertTrue(req.finished())
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])
        self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)
        self.assertEqual(scheduler.send_to_tokenizer.send_pyobj.call_count, 1)

    def test_pending_abort_chunk_is_not_rescheduled(self):
        req = self._make_req("inflight", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        req.to_finish = FINISH_ABORT()
        scheduler._pending_chunked_abort_reqs[0] = req
        req.init_next_round_input = Mock()
        adder = SimpleNamespace(
            can_run_list={0: []},
            pending_h2d=[],
            new_chunked_reqs=[None],
            add_chunked_req=Mock(),
        )

        with patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder):
            self.assertIsNone(scheduler.get_new_batch_prefill())

        req.init_next_round_input.assert_not_called()
        adder.add_chunked_req.assert_not_called()
        self.assertIs(scheduler.chunked_reqs[0], req)

    def test_abort_finalizes_when_inflight_overlap_chunk_returns(self):
        req = self._make_req("inflight", [1, 2], [10, 11], return_logprob=True)
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.last_batch.return_logprob = True
        scheduler.enable_overlap = True
        req.is_chunked = 1
        result = self._make_prefill_result([3])
        scheduler.tp_worker = SimpleNamespace(
            resolve_last_batch_result=Mock(return_value=(result.logits_output, [3], 0))
        )

        with (
            patch(
                "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
            ) as release,
            patch(
                "sgl_jax.srt.managers.scheduler_output_processor_mixin._complete_precision_trace"
            ) as complete_trace,
        ):
            scheduler.abort_request(AbortReq(rid=req.rid))
            self.assertIsNotNone(req.to_finish)
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            release.assert_not_called()
            complete_trace.assert_not_called()

            scheduler._process_pending_chunked_aborts()
            release.assert_not_called()

            SchedulerOutputProcessorMixin.process_batch_result_prefill(
                scheduler, scheduler.last_batch, result
            )

        self.assertTrue(req.finished())
        self.assertIsNone(req.to_finish)
        self.assertEqual(req.is_chunked, 0)
        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        skip_reqs = scheduler.stream_output.call_args.args[3]
        self.assertNotIn(id(req), skip_reqs)
        self.assertTrue(req.input_logprob_sent)

        SchedulerOutputProcessorMixin.stream_output_generation(
            scheduler,
            [req],
            return_logprob=True,
            return_output_logprob_only=False,
        )

        sent = scheduler.send_to_detokenizer.send_pyobj.call_args.args[0]
        self.assertEqual(sent.rids, [req.rid])
        self.assertEqual(sent.input_token_logprobs_val, [[]])
        self.assertEqual(sent.input_token_logprobs_idx, [[]])
        complete_trace.assert_called_once_with(req)

    def test_aborted_dp_chunk_advances_logprob_cursor_for_later_rank(self):
        aborted_req = self._make_req("aborted", [1, 2], [10], dp_rank=0, return_logprob=True)
        live_req = self._make_req("live", [3, 4], [11], dp_rank=1, return_logprob=True)
        batch = self._make_batch([[aborted_req], [live_req]], [aborted_req, live_req])
        for info in batch.reqs_info:
            info.extend_lens = [2]
            info.extend_logprob_start_lens = [0]

        scheduler, _ = self._make_scheduler(
            [[aborted_req], [live_req]], active_reqs=[aborted_req, live_req]
        )
        aborted_req.is_chunked = 1
        aborted_req.to_finish = FINISH_ABORT()
        scheduler._pending_chunked_abort_reqs[0] = aborted_req
        live_req.is_chunked = 1
        result = self._make_prefill_result([5, 6], input_token_logprobs=(0.1, 0.2, 0.3, 0.4))

        with patch("sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"):
            SchedulerOutputProcessorMixin.process_batch_result_prefill(scheduler, batch, result)

        self.assertEqual(live_req.input_token_logprobs, [0.3, 0.4])

    def test_pd_prefill_result_makes_middle_chunk_abort_releasable(self):
        req = self._make_req("pd-inflight", [1, 2], [10], return_logprob=False)
        req.bootstrap_room = 0
        req.is_chunked = 1
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)

        scheduler.process_prefill_chunk(scheduler.last_batch, SimpleNamespace())

        self.assertEqual(req.is_chunked, 0)
        scheduler.abort_request(AbortReq(rid=req.rid))
        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler._process_pending_chunked_aborts()

        release.assert_called_once()
        self.assertTrue(req.finished())
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])

    def test_retract_requeues_parked_chunk_after_releasing_allocation(self):
        req = self._make_req("parked", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache") as release:
            scheduler._retract_parked_chunked_reqs([])

        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertTrue(req.is_retracted)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])

    def test_retract_does_not_release_batch_owned_chunk_twice(self):
        req = self._make_req("active", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache") as release:
            scheduler._retract_parked_chunked_reqs([req])

        release.assert_not_called()
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])


if __name__ == "__main__":
    unittest.main()
