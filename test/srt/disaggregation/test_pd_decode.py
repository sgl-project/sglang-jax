"""Tests for PD decode side: watchdog, non-blocking pull, capacity-gated admission, bootstrap cache, orphan reaper."""

from __future__ import annotations

import threading
import time

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.common.core import CommonKVManager
from sgl_jax.srt.disaggregation.decode import (
    DecodeBookkeeping,
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sgl_jax.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sgl_jax.srt.disaggregation.decode_watchdog import EventLoopWatchdog
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVReceiver,
    PMetadata,
)


# ---- from test_decode_watchdog.py ----

class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

def _make(threshold=5.0, clock=None, snapshot=None):
    dumped = {"n": 0}

    def _dumper():
        dumped["n"] += 1

    wd = EventLoopWatchdog(
        stall_threshold_s=threshold,
        clock=clock or _Clock(),
        snapshot_provider=snapshot,
        traceback_dumper=_dumper,
    )
    return wd, dumped

class TestEnabled:
    def test_positive_threshold_enabled(self):
        wd, _ = _make(threshold=5.0)
        assert wd.enabled is True

    def test_zero_threshold_disabled(self):
        wd, _ = _make(threshold=0.0)
        assert wd.enabled is False

    def test_start_is_noop_when_disabled(self):
        clock = _Clock()
        wd, _ = _make(threshold=0.0, clock=clock)
        wd.start()
        assert wd._thread is None

class TestStallDetection:
    def test_no_report_within_threshold(self):
        clock = _Clock()
        wd, dumped = _make(threshold=5.0, clock=clock)
        wd.beat("recv_requests")
        clock.t = 4.9
        assert wd.check_once() is False
        assert dumped["n"] == 0

    def test_report_once_when_stalled(self):
        clock = _Clock()
        wd, dumped = _make(threshold=5.0, clock=clock)
        wd.beat("run_batch")
        clock.t = 10.0
        assert wd.check_once() is True
        assert dumped["n"] == 1

    def test_does_not_report_twice_for_same_stall(self):
        clock = _Clock()
        wd, dumped = _make(threshold=5.0, clock=clock)
        wd.beat("run_batch")
        clock.t = 10.0
        assert wd.check_once() is True
        clock.t = 20.0
        # Same tick -> still stuck on the same phase -> no second report.
        assert wd.check_once() is False
        assert dumped["n"] == 1

    def test_rearms_after_loop_advances(self):
        clock = _Clock()
        wd, dumped = _make(threshold=5.0, clock=clock)
        wd.beat("run_batch")
        clock.t = 10.0
        assert wd.check_once() is True
        # Loop advances (new beat -> new tick), then stalls again.
        wd.beat("process_decode_queue")
        clock.t = 20.0
        assert wd.check_once() is True
        assert dumped["n"] == 2

class TestSnapshot:
    def test_snapshot_provider_included_and_dumped(self):
        clock = _Clock()
        calls = {"n": 0}

        def _snap():
            calls["n"] += 1
            return "prealloc_q=3 transfer_q=2"

        wd, dumped = _make(threshold=5.0, clock=clock, snapshot=_snap)
        wd.beat("get_next_batch")
        clock.t = 10.0
        wd.check_once()
        assert calls["n"] == 1
        assert dumped["n"] == 1

    def test_snapshot_failure_is_suppressed(self):
        clock = _Clock()

        def _snap():
            raise RuntimeError("boom")

        wd, dumped = _make(threshold=5.0, clock=clock, snapshot=_snap)
        wd.beat("get_next_batch")
        clock.t = 10.0
        # Snapshot raises, but report + traceback dump still happen.
        assert wd.check_once() is True
        assert dumped["n"] == 1

# ---- from test_pd_decode_pull_nonblocking.py ----

class _Leaf:
    """Stand-in for a pulled jax.Array leaf; a custom object is a pytree leaf.

    Readiness is read live from the owning wrapper so a test can flip it
    after the pull has been dispatched.
    """

    def __init__(self, wrapper: "_Wrapper"):
        self._wrapper = wrapper

    def is_ready(self) -> bool:
        return self._wrapper.ready

class _Wrapper:
    def __init__(self, *, raise_exc: bool = False, ready: bool = True):
        self._raise = raise_exc
        self.ready = ready
        self.calls = 0
        self.connected: list[str] = []

    def connect(self, remote_addr):
        self.connected.append(remote_addr)

    def pull(self, uuid, spec, remote_addr=None):
        self.calls += 1
        # Mirror the real wrapper: ``pull`` lazily connects (and caches) the
        # link on the calling thread before fetching.
        if remote_addr is not None and remote_addr not in self.connected:
            self.connected.append(remote_addr)
        if self._raise:
            raise RuntimeError("pull boom")
        return _Leaf(self)

class _Notifier:
    def __init__(self):
        self.sent = []

    def send_done(self, uuid_bytes, host, port):
        self.sent.append((uuid_bytes, host, port))

class _FakeMgr:
    """Stands in for the manager. ``enqueue_pull`` records the receiver but
    does NOT run it — tests drive the worker explicitly via ``_run_pull`` so
    the ordering between worker completion and reaper ``fail()`` is fully
    controllable.
    """

    def __init__(self, *, raise_exc=False, ready=True):
        self._wrapper = _Wrapper(raise_exc=raise_exc, ready=ready)
        self._notifier = _Notifier()
        self.terminal = []
        self.pruned = []
        self.enqueued: list[JaxTransferKVReceiver] = []

    @property
    def wrapper(self):
        return self._wrapper

    @property
    def zmq_notifier(self):
        return self._notifier

    def enqueue_pull(self, receiver):
        self.enqueued.append(receiver)

    def record_terminal(self, req_id, *, role, transfer_id, state, reason):
        self.terminal.append((req_id, role, state, reason))

    def _prune_receiver(self, req_id):
        self.pruned.append(req_id)

def _pmeta():
    return PMetadata(
        remote_addr="1.2.3.4:5000",
        uuid="uuid-1",
        specs={"k": object()},
        p_side_channel_host="1.2.3.4",
        p_side_channel_port=5001,
    )

def _make_receiver(**kw):
    mgr = _FakeMgr(**kw)
    recv = JaxTransferKVReceiver(mgr, "req-a")
    recv.init(_pmeta())
    return mgr, recv

def test_init_does_not_connect():
    mgr, recv = _make_receiver()
    # init must NOT touch the link: connect+pull stay on the worker thread.
    assert mgr.wrapper.connected == []
    assert recv.state == KVPoll.WAITING_FOR_INPUT

def test_poll_enqueues_without_pulling():
    mgr, recv = _make_receiver()

    # First poll transitions to TRANSFERRING and enqueues — it must NOT pull.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.enqueued == [recv]
    assert mgr.wrapper.calls == 0
    assert recv.result is None

    # Polling again before the worker runs stays TRANSFERRING, no ack.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr._notifier.sent == []

def test_worker_pull_then_success():
    mgr, recv = _make_receiver()
    assert recv.poll() == KVPoll.TRANSFERRING
    # Enqueue does not connect; the link is created on the worker thread.
    assert mgr.wrapper.connected == []

    # The background worker performs the blocking pull and stores results.
    recv._run_pull()
    assert mgr.wrapper.calls == 1
    assert mgr.wrapper.connected == ["1.2.3.4:5000"]
    assert recv.result is not None

    # Next poll drives ack -> SUCCESS once every leaf is ready.
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr.wrapper.calls == 1  # not re-pulled
    assert mgr._notifier.sent
    assert ("req-a", "decode", KVPoll.SUCCESS, "ack_send") in mgr.terminal

def test_poll_waits_for_ready():
    mgr, recv = _make_receiver(ready=False)

    assert recv.poll() == KVPoll.TRANSFERRING
    recv._run_pull()
    # Results stored but leaves not ready yet -> stays TRANSFERRING, no ack.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr._notifier.sent == []

    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr._notifier.sent

def test_pull_exception_transitions_failed():
    mgr, recv = _make_receiver(raise_exc=True)

    assert recv.poll() == KVPoll.TRANSFERRING
    # The worker hits the exception and drives the terminal transition.
    recv._run_pull()
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "pull_init") in mgr.terminal
    assert "req-a" in mgr.pruned
    assert recv.result is None

    # A subsequent poll stays FAILED.
    assert recv.poll() == KVPoll.FAILED

def test_reaper_fail_then_worker_drops_results():
    mgr, recv = _make_receiver()

    assert recv.poll() == KVPoll.TRANSFERRING

    # Reaper times out the in-flight transfer before the worker finishes.
    recv.fail(reason="timeout")
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "timeout") in mgr.terminal

    # The worker completes late: results must be dropped, not stored, and the
    # terminal state must not be resurrected.
    recv._run_pull()
    assert recv.result is None
    assert recv.state == KVPoll.FAILED

    # Even if leaves are ready, poll stays FAILED and no ack is ever sent.
    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.FAILED
    assert mgr._notifier.sent == []

def test_poll_never_spawns_thread():
    mgr, recv = _make_receiver()

    before = threading.active_count()
    recv.poll()
    recv.poll()
    assert threading.active_count() == before

def test_manager_owns_persistent_pull_worker_pool():
    """The manager starts a pool of long-lived workers that drain the queue
    and run each receiver's blocking pull off the event loop."""

    mgr = JaxTransferKVManager(
        wrapper=object(), zmq_notifier=object(), pull_worker_count=4
    )

    worker_threads = [
        t for t in threading.enumerate() if t.name.startswith("jax-kv-pull-worker")
    ]
    assert len(worker_threads) == 4
    assert all(t.daemon for t in worker_threads)

    ran = threading.Event()

    class _Stub:
        def _run_pull(self):
            ran.set()

    mgr.enqueue_pull(_Stub())
    assert ran.wait(timeout=5.0)

# ---- from test_pd_decode_admission.py ----

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

class _AdmServerArgs:
    def __init__(self, reserved, max_inflight=0):
        self.disaggregation_num_reserved_decode_tokens = reserved
        self.disaggregation_max_inflight_transfers = max_inflight
        self.enable_request_time_stats_logging = False

class _AdmReq:
    def __init__(self, rid, seqlen):
        self.rid = rid
        self.disagg_transfer_id = None
        self.origin_input_ids = list(range(seqlen))

class _Batch:
    def __init__(self, n):
        self.reqs = list(range(n))

class _AdmScheduler:
    _admit_decode_prealloc = SchedulerDisaggregationDecodeMixin._admit_decode_prealloc

    def __init__(self, capacity, reserved, page_size=1, n_running=0, raise_on_init=False, max_inflight=0):
        self.token_to_kv_pool_allocator = _Allocator(capacity, page_size)
        self.server_args = _AdmServerArgs(reserved, max_inflight=max_inflight)
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

def _adm_p_info():
    return {"host": "1.2.3.4", "transfer_port": 5000, "side_channel_port": 5001}

def _enqueue(sched, rid, seqlen):
    entry = DecodeBookkeeping(req_id=rid, req=_AdmReq(rid, seqlen), p_info=_adm_p_info())
    sched.disagg_prealloc_queue.add(entry)
    return entry

def test_admit_when_capacity_sufficient():
    sched = _AdmScheduler(capacity=100, reserved=0)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_prealloc_queue) == 0
    assert len(sched.disagg_transfer_queue) == 2
    assert sched.aborted == []

def test_defer_when_capacity_insufficient_does_not_abort():
    sched = _AdmScheduler(capacity=4, reserved=0)
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
    sched = _AdmScheduler(capacity=10, reserved=4, n_running=2)
    _enqueue(sched, "a", seqlen=2)
    _enqueue(sched, "b", seqlen=2)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 1
    assert len(sched.disagg_prealloc_queue) == 1
    assert sched.aborted == []

def test_recovery_admits_after_capacity_frees():
    sched = _AdmScheduler(capacity=4, reserved=0)
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
    sched = _AdmScheduler(capacity=100, reserved=0, raise_on_init=True)
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
    sched = _AdmScheduler(capacity=100, reserved=0, max_inflight=2)
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
    sched = _AdmScheduler(capacity=100, reserved=0, max_inflight=2)
    # Pre-load the transfer queue with a placeholder to simulate one in-flight.
    existing = DecodeBookkeeping(req_id="x", req=_AdmReq("x", 4), p_info=_adm_p_info())
    sched.disagg_transfer_queue.add(existing)
    _enqueue(sched, "a", seqlen=4)
    _enqueue(sched, "b", seqlen=4)

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 2  # placeholder + a
    assert len(sched.disagg_prealloc_queue) == 1  # b deferred
    assert sched.aborted == []

def test_inflight_cap_recovers_after_transfer_drains():
    sched = _AdmScheduler(capacity=100, reserved=0, max_inflight=2)
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

# ---- from test_pd_decode_bootstrap_cache.py ----

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

# ---- from test_orphan_reaper.py ----

class _FakeParticipant:
    """Duck-types the (transfer_started_at, fail) contract the reaper needs."""

    def __init__(self, started_at):
        self.transfer_started_at = started_at
        self.failed_reason = None

    def fail(self, *, reason):
        self.failed_reason = reason

class _Mgr(CommonKVManager):
    def create_sender(self, req_id):  # pragma: no cover - unused
        raise NotImplementedError

    def create_receiver(self, req_id):  # pragma: no cover - unused
        raise NotImplementedError

def _mgr(**kw):
    return _Mgr(
        ack_timeout_seconds=kw.get("ack", 10.0),
        pull_timeout_seconds=kw.get("pull", 5.0),
        reaper_interval_seconds=kw.get("interval", 0.01),
    )

class TestReapOnce:
    def test_sender_past_ack_timeout_is_failed(self):
        m = _mgr(ack=10.0)
        s = _FakeParticipant(started_at=100.0)
        m.register_sender("r1", s)
        timed_out_s, timed_out_r = m.reap_once(now=111.0)
        assert timed_out_s == ["r1"]
        assert timed_out_r == []
        assert s.failed_reason == "timeout"

    def test_fresh_sender_is_kept(self):
        m = _mgr(ack=10.0)
        s = _FakeParticipant(started_at=100.0)
        m.register_sender("r1", s)
        timed_out_s, _ = m.reap_once(now=105.0)
        assert timed_out_s == []
        assert s.failed_reason is None

    def test_receiver_past_pull_timeout_is_failed(self):
        m = _mgr(pull=5.0)
        r = _FakeParticipant(started_at=100.0)
        m.register_receiver("r1", r)
        _, timed_out_r = m.reap_once(now=106.0)
        assert timed_out_r == ["r1"]
        assert r.failed_reason == "timeout"

    def test_unstarted_participant_is_skipped(self):
        m = _mgr(ack=1.0)
        s = _FakeParticipant(started_at=None)
        m.register_sender("r1", s)
        timed_out_s, _ = m.reap_once(now=1e9)
        assert timed_out_s == []
        assert s.failed_reason is None

    def test_zero_ack_timeout_disables_sender_reaping(self):
        m = _mgr(ack=0.0, pull=5.0)
        s = _FakeParticipant(started_at=0.0)
        m.register_sender("r1", s)
        timed_out_s, _ = m.reap_once(now=1e9)
        assert timed_out_s == []
        assert s.failed_reason is None

class TestRegistry:
    def test_duplicate_sender_raises(self):
        m = _mgr()
        m.register_sender("r1", _FakeParticipant(1.0))
        try:
            m.register_sender("r1", _FakeParticipant(1.0))
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError on duplicate sender")

    def test_inflight_count(self):
        m = _mgr()
        m.register_sender("s1", _FakeParticipant(1.0))
        m.register_receiver("d1", _FakeParticipant(1.0))
        m.register_receiver("d2", _FakeParticipant(1.0))
        assert m.inflight_count() == (1, 2)

class TestReaperLifecycle:
    def test_start_then_stop(self):
        m = _mgr(interval=0.01)
        m.start_reaper()
        assert m._reaper_thread is not None
        assert m._reaper_thread.is_alive()
        m.stop_reaper()
        assert m._reaper_thread is None

    def test_start_noop_when_no_timeouts(self):
        m = _mgr(ack=0.0, pull=0.0)
        m.start_reaper()
        assert m._reaper_thread is None

    def test_reaper_thread_fails_stale_participant(self):
        m = _Mgr(ack_timeout_seconds=0.05, pull_timeout_seconds=0.0, reaper_interval_seconds=0.01)
        s = _FakeParticipant(started_at=time.monotonic() - 1.0)
        m.register_sender("r1", s)
        m.start_reaper()
        try:
            deadline = time.monotonic() + 2.0
            while s.failed_reason is None and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            m.stop_reaper()
        assert s.failed_reason == "timeout"

class TestTerminalRecords:
    def test_record_and_get(self):
        m = _mgr()
        m.record_terminal(
            "r1", role="prefill", transfer_id="t1", state=KVPoll.FAILED, reason="timeout"
        )
        rec = m.get_terminal_record("r1", role="prefill")
        assert rec is not None
        assert rec.state == KVPoll.FAILED
        assert rec.reason == "timeout"

    def test_get_missing_returns_none(self):
        m = _mgr()
        assert m.get_terminal_record("nope", role="prefill") is None

    def test_register_clears_prior_terminal_record(self):
        m = _mgr()
        m.record_terminal(
            "r1", role="prefill", transfer_id="t1", state=KVPoll.FAILED, reason="x"
        )
        m.register_sender("r1", _FakeParticipant(1.0))
        assert m.get_terminal_record("r1", role="prefill") is None
