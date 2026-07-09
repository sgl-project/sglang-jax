"""Scheduler must split prefill at recurrent track boundaries: a scheduled EXTEND
ends on a boundary or is a non-final chunk. The cap composes with chunk-budget
truncation (admitted length = min of the caps; chunked if either truncated);
extra-buffer OFF is byte-identical to today."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class _DummyAllocator:
    def available_size(self, dp_rank: int = 0):
        return 1_000_000


class _RecurrentRadixCache:
    """Fake recurrent tree cache exposing the predicate PrefillAdder reads."""

    disable = False

    def __init__(self, extra_buffer: bool, interval: int | None):
        self._extra_buffer = extra_buffer
        self.recurrent_track_interval = interval
        self.enable_recurrent_extra_buffer = extra_buffer

    def supports_recurrent(self) -> bool:
        return True

    def recurrent_extra_buffer_active(self) -> bool:
        return self._extra_buffer and self.recurrent_track_interval is not None

    def evictable_size(self, dp_rank: int = 0):
        return 0

    def inc_lock_ref(self, node):
        return IncLockRefResult(delta=0)

    def dec_lock_ref(self, node, params=None):
        pass


def _make_req(rid, *, input_len, prefix_len=0, dp_rank=0):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(range(input_len)),
        sampling_params=SamplingParams(max_new_tokens=8),
        dp_rank=dp_rank,
        eos_token_ids={input_len + 1},
        vocab_size=32000,
    )
    req.fill_ids = list(range(input_len))
    req.prefix_indices = list(range(prefix_len))
    req.extend_input_len = input_len - prefix_len
    req.host_hit_length = 0
    req.last_node = object()
    return req


def _adder(tree_cache, *, page_size=128, rem_chunk_tokens=100_000):
    return PrefillAdder(
        page_size=page_size,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=_DummyAllocator(),
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=rem_chunk_tokens,
        dp_size=1,
    )


class TestBoundarySplitAddOneReq(unittest.TestCase):
    def test_split_at_first_boundary_marks_chunked(self):
        # prefix=0, interval=128, input=300 -> cap to 128, land on boundary.
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=300, prefix_len=0)
        res = adder.add_one_req(req)
        self.assertEqual(res, AddReqResult.CONTINUE)
        self.assertEqual(req.extend_input_len, 128)
        self.assertEqual(len(req.fill_ids), 128)
        # Lands exactly on a boundary.
        self.assertEqual((len(req.prefix_indices) + req.extend_input_len) % 128, 0)
        # Marked chunked: more tokens next round from the protected prefix.
        self.assertIs(adder.new_chunked_reqs[0], req)

    def test_split_with_nonzero_prefix(self):
        # prefix=200, interval=128 -> first boundary after 200 is 256.
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=400, prefix_len=200)
        adder.add_one_req(req)
        self.assertEqual(req.extend_input_len, 56)  # 256 - 200
        self.assertEqual((200 + req.extend_input_len) % 128, 0)
        self.assertIs(adder.new_chunked_reqs[0], req)

    def test_no_split_when_already_within_first_interval(self):
        # input lands before the first boundary -> final chunk, NOT marked chunked.
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=100, prefix_len=0)
        adder.add_one_req(req)
        self.assertEqual(req.extend_input_len, 100)
        self.assertIsNone(adder.new_chunked_reqs[0])

    def test_exact_boundary_input_is_final_chunk(self):
        # input exactly = interval -> ends on boundary, final chunk (no more tokens).
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=128, prefix_len=0)
        adder.add_one_req(req)
        self.assertEqual(req.extend_input_len, 128)
        self.assertEqual((0 + req.extend_input_len) % 128, 0)
        self.assertIsNone(adder.new_chunked_reqs[0])

    def test_chunk_budget_tighter_than_boundary(self):
        # boundary cap = 128 but chunk budget = 64 -> min = 64, still chunked,
        # mid-interval is allowed because it is not the final chunk.
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache, page_size=64, rem_chunk_tokens=64)
        req = _make_req("r", input_len=300, prefix_len=0)
        adder.add_one_req(req)
        self.assertEqual(req.extend_input_len, 64)
        self.assertIs(adder.new_chunked_reqs[0], req)

    def test_off_no_boundary_cap(self):
        # extra-buffer OFF -> no cap, no forced chunked marking (regression).
        cache = _RecurrentRadixCache(extra_buffer=False, interval=None)
        adder = _adder(cache)
        req = _make_req("r", input_len=300, prefix_len=0)
        adder.add_one_req(req)
        self.assertEqual(req.extend_input_len, 300)
        self.assertIsNone(adder.new_chunked_reqs[0])


class TestBoundarySplitAddChunkedReq(unittest.TestCase):
    def test_chunked_req_capped_to_boundary(self):
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=300, prefix_len=0)
        out = adder.add_chunked_req(req)
        self.assertEqual(req.extend_input_len, 128)
        self.assertEqual((len(req.prefix_indices) + req.extend_input_len) % 128, 0)
        # boundary truncation -> chunked req returned (more tokens next round).
        self.assertIs(out, req)

    def test_chunked_req_final_chunk_within_interval(self):
        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        adder = _adder(cache)
        req = _make_req("r", input_len=100, prefix_len=0)
        out = adder.add_chunked_req(req)
        self.assertEqual(req.extend_input_len, 100)
        self.assertIsNone(out)

    def test_chunked_req_off_no_cap(self):
        cache = _RecurrentRadixCache(extra_buffer=False, interval=None)
        adder = _adder(cache)
        req = _make_req("r", input_len=300, prefix_len=0)
        out = adder.add_chunked_req(req)
        self.assertEqual(req.extend_input_len, 300)
        self.assertIsNone(out)


class TestSchedulerSingleChunkedPerRankPerRound(unittest.TestCase):
    """One chunked req per rank per round: a boundary-only split leaves chunk
    budget, so without the rank-skip on new_chunked_reqs a second same-rank req
    would truncate and silently overwrite the first."""

    def test_second_boundary_split_req_stays_in_waiting_queue(self):
        from sgl_jax.srt.managers import scheduler as scheduler_mod

        cache = _RecurrentRadixCache(extra_buffer=True, interval=128)
        cache.root_node = object()
        cache.recurrent_evictable_size = lambda dp_rank=0: 0
        cache.match_prefix = lambda params: SimpleNamespace(
            device_indices=[],
            last_device_node=cache.root_node,
            last_host_node=cache.root_node,
            host_hit_length=0,
        )

        req1 = _make_req("r1", input_len=300, prefix_len=0)
        req2 = _make_req("r2", input_len=300, prefix_len=0)

        rank_info = SimpleNamespace(reqs=[], batch_is_full=False)
        fake = SimpleNamespace(
            grammar_queue=[],
            chunked_reqs=[None],
            is_hybrid=False,
            _is_spec_decode_enabled=lambda: False,
            enable_overlap=False,
            spec_algorithm=None,
            running_batch=SimpleNamespace(
                batch_is_full=False,
                is_empty=lambda: True,
                batch_size=lambda: 0,
                reqs_info=[rank_info],
            ),
            waiting_queue=[req1, req2],
            policy=SimpleNamespace(calc_priority=lambda q: None),
            page_size=128,
            tree_cache=cache,
            token_to_kv_pool_allocator=_DummyAllocator(),
            new_token_ratio=1.0,
            max_prefill_tokens=1_000_000,
            chunked_prefill_size=100_000,
            is_mixed_chunk=False,
            dp_size=1,
            lora_paths=None,
            per_dp_max_running_requests=1000,
            req_to_token_pool=SimpleNamespace(
                request_owned_slots=1,
                recurrent_available_size=lambda dp_rank=0: 1_000_000,
            ),
            log_prefill_stats=lambda *a, **k: None,
            model_config=None,
            mesh=None,
        )

        with patch.object(scheduler_mod.ScheduleBatch, "init_new", return_value=MagicMock()):
            batch = scheduler_mod.Scheduler.get_new_batch_prefill(fake)

        self.assertIsNotNone(batch)
        # req1 admitted as the rank's single chunked req; req2 deferred intact.
        self.assertIs(fake.chunked_reqs[0], req1)
        self.assertEqual(req1.is_chunked, 1)
        self.assertEqual(req1.extend_input_len, 128)
        self.assertIn(req2, fake.waiting_queue)
        self.assertNotIn(req1, fake.waiting_queue)
        self.assertEqual(req2.extend_input_len, 300)
        self.assertEqual(req2.is_chunked, 0)


if __name__ == "__main__":
    unittest.main()
