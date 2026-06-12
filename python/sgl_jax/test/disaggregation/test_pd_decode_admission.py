"""Decode-side capacity-gated admission: queue, never crash.

Covers ``_admit_decode_prealloc``: KV alloc is deferred to a budget gate
that reserves headroom for in-flight/running reqs (transfer-queue reqs
cannot be retracted), so an over-subscribed decode defers admission (FIFO
requeue) instead of OOM-crashing.
"""
from __future__ import annotations

from sgl_jax.srt.disaggregation.decode import (
    DecodeBookkeeping,
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)


class _Allocator:
    def __init__(self, capacity, page_size=1):
        self.page_size = page_size
        self._capacity = capacity
        self._used = 0
        self.freed = []

    def available_size(self):
        return self._capacity - self._used

    def alloc(self, n):
        if self._used + n > self._capacity:
            return None
        self._used += n
        return object()

    def free(self, idx):
        self.freed.append(idx)


class _Receiver:
    def __init__(self, raise_on_init=False):
        self._raise = raise_on_init
        self.inited = None

    def init(self, pmeta):
        if self._raise:
            raise RuntimeError("boom")
        self.inited = pmeta


class _KVManager:
    def __init__(self, raise_on_init=False):
        self._raise = raise_on_init
        self.created = []

    def create_receiver(self, rid):
        r = _Receiver(raise_on_init=self._raise)
        self.created.append((rid, r))
        return r


class _ServerArgs:
    def __init__(self, reserved, max_inflight=0):
        self.disaggregation_num_reserved_decode_tokens = reserved
        self.disaggregation_max_inflight_transfers = max_inflight
        self.enable_request_time_stats_logging = False


class _Req:
    def __init__(self, rid, seqlen):
        self.rid = rid
        self.disagg_transfer_id = None
        self.origin_input_ids = list(range(seqlen))


class _Batch:
    def __init__(self, n):
        self.reqs = list(range(n))


class _FakeScheduler:
    _admit_decode_prealloc = SchedulerDisaggregationDecodeMixin._admit_decode_prealloc

    def __init__(self, capacity, reserved, page_size=1, n_running=0, raise_on_init=False, max_inflight=0):
        self.token_to_kv_pool_allocator = _Allocator(capacity, page_size)
        self.server_args = _ServerArgs(reserved, max_inflight=max_inflight)
        self.running_batch = _Batch(n_running)
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self.disagg_transfer_queue = DecodeTransferQueue()
        self.disagg_kv_manager = _KVManager(raise_on_init=raise_on_init)
        self.aborted = []
        self.failures = []

    def _build_kv_spec_for_req(self, req):
        return []

    def _pd_mark_time(self, req, name):
        pass

    def _record_decode_transfer_failure(self, reason):
        self.failures.append(reason)

    def _release_decode_kv_indices(self, kv_indices):
        self.token_to_kv_pool_allocator.free(kv_indices)

    def _abort_decode_request(self, req, reason):
        self.aborted.append((req.rid, reason))


def _p_info():
    return {"host": "1.2.3.4", "transfer_port": 5000, "side_channel_port": 5001}


def _enqueue(sched, rid, seqlen):
    entry = DecodeBookkeeping(req_id=rid, req=_Req(rid, seqlen), p_info=_p_info())
    sched.disagg_prealloc_queue.add(entry)
    return entry


def test_admit_when_capacity_sufficient():
    sched = _FakeScheduler(capacity=100, reserved=0)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_prealloc_queue) == 0
    assert len(sched.disagg_transfer_queue) == 2
    assert sched.aborted == []


def test_defer_when_capacity_insufficient_does_not_abort():
    sched = _FakeScheduler(capacity=4, reserved=0)
    _enqueue(sched, "a", seqlen=8)  # needs 8 > 4

    sched._admit_decode_prealloc()

    # Stays queued for a later tick; not admitted, not aborted, no crash.
    assert len(sched.disagg_prealloc_queue) == 1
    assert len(sched.disagg_transfer_queue) == 0
    assert sched.aborted == []
    assert sched.token_to_kv_pool_allocator.available_size() == 4


def test_reserved_headroom_blocks_over_admission():
    # capacity 10, but 2 running reqs each reserve 4 tokens of headroom, so
    # only the first queued req fits even though raw free space (8) could
    # hold both 2-token reqs. Guards against retract-impossible deadlock.
    sched = _FakeScheduler(capacity=10, reserved=4, n_running=2)
    _enqueue(sched, "a", seqlen=2)
    _enqueue(sched, "b", seqlen=2)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 1
    assert len(sched.disagg_prealloc_queue) == 1
    assert sched.aborted == []


def test_recovery_admits_after_capacity_frees():
    sched = _FakeScheduler(capacity=4, reserved=0)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)

    sched._admit_decode_prealloc()
    assert len(sched.disagg_transfer_queue) == 1  # only "a" fits
    assert len(sched.disagg_prealloc_queue) == 1  # "b" deferred

    # A transfer completes and frees its KV slots.
    sched.token_to_kv_pool_allocator._used = 0

    sched._admit_decode_prealloc()
    assert len(sched.disagg_transfer_queue) == 2  # "b" now admitted
    assert len(sched.disagg_prealloc_queue) == 0


def test_receiver_init_failure_aborts_and_frees():
    sched = _FakeScheduler(capacity=100, reserved=0, raise_on_init=True)
    _enqueue(sched, "a", seqlen=4)

    sched._admit_decode_prealloc()

    # Failed setup: dropped from queue, KV freed, aborted, loop did not crash.
    assert len(sched.disagg_prealloc_queue) == 0
    assert len(sched.disagg_transfer_queue) == 0
    assert sched.aborted == [("a", "receiver_init")]
    assert sched.failures == ["receiver_init"]
    assert len(sched.token_to_kv_pool_allocator.freed) == 1


def test_inflight_cap_defers_excess_without_abort():
    # Capacity is ample (each req needs 4, capacity 100), so only the in-flight
    # cap can stop admission. cap=2 means at most 2 transfers admitted per tick;
    # the 3rd+ stay queued (deferral), are not aborted, and KV is not allocated
    # for them (no transient pull buffer reserved beyond the cap).
    sched = _FakeScheduler(capacity=100, reserved=0, max_inflight=2)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)
    _enqueue(sched, "c", seqlen=4)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 2  # a, b admitted up to cap
    assert len(sched.disagg_prealloc_queue) == 1  # c deferred, still queued
    assert sched.aborted == []
    # Only 2 reqs' KV allocated (8 tokens); c's was not reserved.
    assert sched.token_to_kv_pool_allocator.available_size() == 92


def test_inflight_cap_counts_existing_transfers():
    # An already-occupied transfer queue counts against the cap. With one entry
    # in flight and cap=2, only one more may be admitted this tick.
    sched = _FakeScheduler(capacity=100, reserved=0, max_inflight=2)
    # Pre-load the transfer queue with a placeholder to simulate one in-flight.
    existing = DecodeBookkeeping(req_id="x", req=_Req("x", 4), p_info=_p_info())
    sched.disagg_transfer_queue.add(existing)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 2  # placeholder + a
    assert len(sched.disagg_prealloc_queue) == 1  # b deferred
    assert sched.aborted == []


def test_inflight_cap_recovers_after_transfer_drains():
    sched = _FakeScheduler(capacity=100, reserved=0, max_inflight=2)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)
    _enqueue(sched, "c", seqlen=4)

    sched._admit_decode_prealloc()
    assert len(sched.disagg_transfer_queue) == 2
    assert len(sched.disagg_prealloc_queue) == 1

    # A transfer drains (removed from the transfer queue), freeing a slot.
    sched.disagg_transfer_queue.abort_matching("a", abort_all=False)

    sched._admit_decode_prealloc()
    assert len(sched.disagg_transfer_queue) == 2  # "c" now admitted
    assert len(sched.disagg_prealloc_queue) == 0
