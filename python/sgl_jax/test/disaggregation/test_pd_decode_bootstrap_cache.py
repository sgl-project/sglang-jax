"""Decode intake uses the local prefill cache and defers (never aborts) when
no prefill is registered yet.

Covers ``process_input_requests_disagg_decode``: a cache hit routes the req to
the prealloc queue; a cache miss (``pick_for_room`` returns ``None``) parks the
req in ``_pd_pending_bootstrap`` and retries it on a later tick — it is never
aborted and never stranded in the waiting queue.
"""
from __future__ import annotations

from sgl_jax.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)


class _Req:
    def __init__(self, rid, room=0):
        self.rid = rid
        self.bootstrap_room = room
        self.disagg_transfer_id = None
        self.origin_input_ids = [1, 2, 3, 4]


class _ServerArgs:
    # page_size/kv_dtype left falsy so check_prefill_compat is a no-op.
    page_size = 0
    kv_cache_dtype = ""
    enable_request_time_stats_logging = False


class _FakeCache:
    """Returns queued results per ``pick_for_room`` call (dict or None)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def pick_for_room(self, room):
        self.calls += 1
        return self._results.pop(0) if self._results else None


class _FakeScheduler:
    process_input_requests_disagg_decode = (
        SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode
    )
    _extract_pd_reqs_from_waiting_queue = (
        SchedulerDisaggregationDecodeMixin._extract_pd_reqs_from_waiting_queue
    )

    def __init__(self, cache):
        self.server_args = _ServerArgs()
        self.waiting_queue = []
        self.disagg_prefill_info_cache = cache
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self._pd_pending_bootstrap = []
        self.aborted = []
        self.failures = []

    def process_input_requests(self, recv_reqs):
        self.waiting_queue.extend(recv_reqs)

    def _pd_mark_time(self, req, name):
        pass

    def _record_decode_transfer_failure(self, reason):
        self.failures.append(reason)

    def _abort_decode_request(self, req, reason):
        self.aborted.append((req.rid, reason))


def _p_info():
    return {"host": "1.2.3.4", "transfer_port": 5000, "side_channel_port": 5001}


def test_cache_hit_routes_to_prealloc():
    sched = _FakeScheduler(_FakeCache([_p_info()]))
    sched.process_input_requests_disagg_decode([_Req("a")])

    assert len(sched.disagg_prealloc_queue) == 1
    assert sched._pd_pending_bootstrap == []
    assert sched.aborted == []


def test_cache_miss_defers_without_abort_then_retries():
    # First call: cache miss -> deferred, not aborted, not enqueued.
    cache = _FakeCache([None])
    sched = _FakeScheduler(cache)
    sched.process_input_requests_disagg_decode([_Req("a")])

    assert len(sched.disagg_prealloc_queue) == 0
    assert [r.rid for r in sched._pd_pending_bootstrap] == ["a"]
    assert sched.aborted == []
    # The deferred req must not be left stranded in the waiting queue.
    assert sched.waiting_queue == []

    # Next tick: prefill now registered, no new recv reqs -> pending retried.
    cache._results = [_p_info()]
    sched.process_input_requests_disagg_decode([])

    assert len(sched.disagg_prealloc_queue) == 1
    assert sched._pd_pending_bootstrap == []
    assert sched.aborted == []


def test_pending_retried_ahead_of_new_reqs():
    cache = _FakeCache([None])
    sched = _FakeScheduler(cache)
    sched.process_input_requests_disagg_decode([_Req("old")])
    assert [r.rid for r in sched._pd_pending_bootstrap] == ["old"]

    # Both resolve next tick; FIFO means the deferred "old" is admitted first.
    cache._results = [_p_info(), _p_info()]
    sched.process_input_requests_disagg_decode([_Req("new")])

    admitted = list(sched.disagg_prealloc_queue.items_fifo())
    assert [e.req_id for e in admitted] == ["old", "new"]
    assert sched._pd_pending_bootstrap == []
