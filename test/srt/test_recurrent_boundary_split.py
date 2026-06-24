"""Scheduler must split prefill chunks at recurrent track boundaries.

When extra-buffer recurrent caching is active, a scheduled EXTEND must never
cross a track boundary: ``(len(prefix_indices) + extend_input_len) % interval``
is 0 (lands on a boundary) OR the req is the final chunk. PrefillAdder caps
``extend_input_len`` to the first boundary after the prefix and marks the req
chunked so the next round continues from the protected FULL prefix.

The boundary cap composes with the existing chunk-budget truncation: the
admitted length is the MINIMUM of the two caps, and the req is chunked if
EITHER cap truncated it. With extra-buffer OFF the path is byte-identical to
today (no cap, no forced chunked marking).
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
