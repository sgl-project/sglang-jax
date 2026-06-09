"""Tests for graceful shutdown: CommonKVManager.graceful_shutdown + runtime closure."""

from __future__ import annotations

import time

from sgl_jax.srt.disaggregation.common.core import CommonKVManager
from sgl_jax.srt.disaggregation.runtime import _make_disagg_shutdown


class _Participant:
    def __init__(self, started_at=None):
        self.transfer_started_at = started_at
        self.failed_reason = None

    def fail(self, *, reason):
        self.failed_reason = reason


class _Mgr(CommonKVManager):
    def create_sender(self, req_id):  # pragma: no cover - unused
        raise NotImplementedError

    def create_receiver(self, req_id):  # pragma: no cover - unused
        raise NotImplementedError


class TestGracefulShutdown:
    def test_drains_when_no_inflight(self):
        m = _Mgr(ack_timeout_seconds=10.0, pull_timeout_seconds=10.0)
        aborted_s, aborted_r = m.graceful_shutdown(drain_timeout_seconds=0.0)
        assert (aborted_s, aborted_r) == (0, 0)

    def test_aborts_stragglers_after_drain_timeout(self):
        m = _Mgr(ack_timeout_seconds=10.0, pull_timeout_seconds=10.0)
        s = _Participant(started_at=time.monotonic())
        r = _Participant(started_at=time.monotonic())
        m.register_sender("s1", s)
        m.register_receiver("d1", r)
        aborted_s, aborted_r = m.graceful_shutdown(drain_timeout_seconds=0.0)
        assert aborted_s == 1
        assert aborted_r == 1
        assert s.failed_reason == "shutdown"
        assert r.failed_reason == "shutdown"

    def test_stops_reaper(self):
        m = _Mgr(ack_timeout_seconds=10.0, pull_timeout_seconds=10.0, reaper_interval_seconds=0.01)
        m.start_reaper()
        assert m._reaper_thread is not None
        m.graceful_shutdown(drain_timeout_seconds=0.0)
        assert m._reaper_thread is None

    def test_drain_returns_early_when_inflight_clears(self):
        m = _Mgr(ack_timeout_seconds=10.0, pull_timeout_seconds=10.0)
        m.register_sender("s1", _Participant(started_at=time.monotonic()))
        # Drain loop polls inflight_count; prune before deadline -> 0 aborts.
        m._prune_sender("s1")
        aborted_s, aborted_r = m.graceful_shutdown(drain_timeout_seconds=1.0)
        assert (aborted_s, aborted_r) == (0, 0)


class _FakeBootstrapClient:
    def __init__(self):
        self.unregistered = []

    def unregister_prefill(self, key):
        self.unregistered.append(key)


class _FakeHeartbeat:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeZmqNotifier:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeManager:
    def __init__(self):
        self.shutdown_calls = 0
        self.zmq_notifier = _FakeZmqNotifier()

    def graceful_shutdown(self, drain_timeout_seconds=30.0):
        self.shutdown_calls += 1
        return 0, 0


class _FakeScheduler:
    def __init__(self):
        self.disagg_bootstrap_key = "h:1"
        self.disagg_bootstrap_client = _FakeBootstrapClient()
        self.disagg_heartbeat = _FakeHeartbeat()
        self.disagg_kv_manager = _FakeManager()
        self.disagg_decode_watchdog = None


class TestRuntimeShutdownClosure:
    def test_prefill_unregisters_and_stops_heartbeat(self):
        sched = _FakeScheduler()
        shutdown = _make_disagg_shutdown(sched, "prefill")
        shutdown()
        assert sched.disagg_bootstrap_client.unregistered == ["h:1"]
        assert sched.disagg_heartbeat.stopped is True
        assert sched.disagg_kv_manager.shutdown_calls == 1
        assert sched.disagg_kv_manager.zmq_notifier.stopped is True

    def test_idempotent(self):
        sched = _FakeScheduler()
        shutdown = _make_disagg_shutdown(sched, "prefill")
        shutdown()
        shutdown()
        shutdown()
        assert sched.disagg_kv_manager.shutdown_calls == 1
        assert sched.disagg_bootstrap_client.unregistered == ["h:1"]

    def test_decode_skips_unregister(self):
        sched = _FakeScheduler()
        shutdown = _make_disagg_shutdown(sched, "decode")
        shutdown()
        assert sched.disagg_bootstrap_client.unregistered == []
        assert sched.disagg_heartbeat.stopped is False
        assert sched.disagg_kv_manager.shutdown_calls == 1
        assert sched.disagg_kv_manager.zmq_notifier.stopped is True

    def test_manager_shutdown_runs_even_if_unregister_raises(self):
        sched = _FakeScheduler()

        def _boom(key):
            raise RuntimeError("unregister failed")

        sched.disagg_bootstrap_client.unregister_prefill = _boom
        shutdown = _make_disagg_shutdown(sched, "prefill")
        shutdown()
        assert sched.disagg_kv_manager.shutdown_calls == 1
