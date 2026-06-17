"""Tests for PD lifecycle: protocol version, shutdown handler, time stats, KV cache pop idempotency, KV debug snapshot, admission."""

from __future__ import annotations

import socket
import time
import unittest

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation.bootstrap import (
    MIN_COMPATIBLE_VERSION,
    PROTOCOL_VERSION,
    BootstrapClient,
    BootstrapServer,
    PrefillInfo,
    build_app,
)
from sgl_jax.srt.disaggregation.common.core import CommonKVManager
from sgl_jax.srt.disaggregation.debug_utils import build_kv_debug_snapshot
from sgl_jax.srt.disaggregation.req_time_stats import (
    TimeStats,
    format_time_stats,
    maybe_log_time_stats,
)
from sgl_jax.srt.disaggregation.runtime import _make_disagg_shutdown
from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler import _reserve_host_slot_for_pd
from sgl_jax.srt.sampling.sampling_params import SamplingParams

# ---- from test_protocol_version.py ----


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestConstants:
    def test_min_compatible_not_above_current(self):
        assert MIN_COMPATIBLE_VERSION <= PROTOCOL_VERSION

    def test_prefill_info_defaults_to_current_version(self):
        info = PrefillInfo(bootstrap_key="h:1", host="h", transfer_port=1, side_channel_port=2)
        assert info.protocol_version == PROTOCOL_VERSION


class TestServerRoundTrip:
    def test_version_preserved_through_registry(self):
        app, _ = build_app()
        client = TestClient(app)
        client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        body = client.get("/get_prefill_info", params={"bootstrap_room": 0}).json()
        assert body["protocol_version"] == PROTOCOL_VERSION


class TestClientVersionGate:
    @pytest.fixture
    def server(self):
        srv = BootstrapServer(host="127.0.0.1", port=_free_port())
        srv.start()
        yield srv
        srv.stop()

    def test_compatible_peer_accepted(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        client.register_prefill(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            protocol_version=PROTOCOL_VERSION,
        )
        info = client.get_prefill_info(0)
        assert info["protocol_version"] == PROTOCOL_VERSION

    def test_peer_below_floor_rejected(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        client.register_prefill(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            protocol_version=MIN_COMPATIBLE_VERSION - 1,
        )
        with pytest.raises(RuntimeError, match="protocol_version"):
            client.get_prefill_info(0)

    def test_no_prefill_registered_raises(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.get_prefill_info(0)


# ---- from test_disagg_shutdown_handler.py ----


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


# ---- from test_pd_time_stats.py ----


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestMarks:
    def test_mark_records_clock_value(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        clk.advance(1.5)
        ts.mark("queue_entry")
        assert ts.marks["queue_entry"] == 1.5

    def test_mark_is_idempotent_keeps_first(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("queue_entry")
        clk.advance(2.0)
        ts.mark("queue_entry")
        assert ts.marks["queue_entry"] == 0.0

    def test_duration_between_marks(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("forward_start")
        clk.advance(0.25)
        ts.mark("forward_done")
        assert ts.duration("forward_start", "forward_done") == pytest.approx(0.25)

    def test_duration_missing_mark_returns_none(self):
        ts = TimeStats("prefill", clock=_FakeClock())
        ts.mark("forward_start")
        assert ts.duration("forward_start", "forward_done") is None


class TestPrefillPhases:
    def test_prefill_phase_breakdown(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("queue_entry")
        clk.advance(0.1)
        ts.mark("forward_start")
        clk.advance(0.2)
        ts.mark("forward_done")
        clk.advance(0.05)
        ts.mark("transfer_start")
        clk.advance(0.3)
        ts.mark("transfer_done")
        phases = ts.phases()
        assert phases["queue"] == pytest.approx(0.1)
        assert phases["forward"] == pytest.approx(0.2)
        assert phases["stage"] == pytest.approx(0.05)
        assert phases["transfer"] == pytest.approx(0.3)
        assert phases["total"] == pytest.approx(0.65)

    def test_partial_prefill_phases_skip_unset(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("queue_entry")
        clk.advance(0.1)
        ts.mark("forward_start")
        # never marked forward_done / transfer_*
        phases = ts.phases()
        assert phases["queue"] == pytest.approx(0.1)
        assert "forward" not in phases
        assert "total" not in phases


class TestDecodePhases:
    def test_decode_phase_breakdown(self):
        clk = _FakeClock()
        ts = TimeStats("decode", clock=clk)
        ts.mark("bootstrap_start")
        clk.advance(0.02)
        ts.mark("bootstrap_done")
        clk.advance(0.01)
        ts.mark("prealloc_entry")
        clk.advance(0.04)
        ts.mark("transfer_entry")
        clk.advance(0.5)
        ts.mark("first_token")
        clk.advance(1.0)
        ts.mark("completion")
        phases = ts.phases()
        assert phases["bootstrap"] == pytest.approx(0.02)
        assert phases["prealloc_wait"] == pytest.approx(0.04)
        assert phases["kv_wait"] == pytest.approx(0.5)
        assert phases["decode"] == pytest.approx(1.0)
        assert phases["total"] == pytest.approx(1.57)


class TestFormatting:
    def test_format_contains_role_and_phases(self):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("queue_entry")
        clk.advance(0.123)
        ts.mark("forward_start")
        clk.advance(0.2)
        ts.mark("forward_done")
        clk.advance(0.0)
        ts.mark("transfer_start")
        clk.advance(0.3)
        ts.mark("transfer_done")
        s = format_time_stats(ts, req_id="abc")
        assert "prefill" in s
        assert "abc" in s
        assert "queue" in s
        assert "ms" in s

    def test_format_unknown_role_lists_raw_marks(self):
        ts = TimeStats("weird", clock=_FakeClock())
        ts.mark("a")
        s = format_time_stats(ts, req_id="r")
        assert "weird" in s


class TestMaybeLog:
    def test_disabled_does_not_log(self, caplog):
        ts = TimeStats("prefill", clock=_FakeClock())
        ts.mark("queue_entry")
        with caplog.at_level("INFO"):
            maybe_log_time_stats(ts, req_id="r", enabled=False)
        assert "queue" not in caplog.text

    def test_enabled_logs(self, caplog):
        clk = _FakeClock()
        ts = TimeStats("prefill", clock=clk)
        ts.mark("queue_entry")
        clk.advance(0.1)
        ts.mark("forward_start")
        with caplog.at_level("INFO"):
            maybe_log_time_stats(ts, req_id="r", enabled=True)
        assert "prefill" in caplog.text

    def test_none_time_stats_is_noop(self, caplog):
        with caplog.at_level("INFO"):
            maybe_log_time_stats(None, req_id="r", enabled=True)
        assert caplog.text == ""


# ---- from test_kv_cache_pop_idempotent.py ----


def _make_req() -> Req:
    req = Req("rid", "text", [1, 2, 3], SamplingParams(max_new_tokens=1))
    req.kv_committed_len = 10
    req.kv_allocated_len = 10
    return req


class TestKVCachePopIdempotent(unittest.TestCase):
    def test_pop_committed_idempotent(self):
        req = _make_req()
        self.assertEqual(req.pop_committed_kv_cache(), 10)
        # Second call must not raise; returns 0 so the caller frees nothing.
        self.assertEqual(req.pop_committed_kv_cache(), 0)
        self.assertEqual(req.pop_committed_kv_cache(), 0)

    def test_pop_overallocated_idempotent(self):
        req = _make_req()
        self.assertEqual(req.pop_overallocated_kv_cache(), (10, 10))
        self.assertEqual(req.pop_overallocated_kv_cache(), (0, 0))
        self.assertEqual(req.pop_overallocated_kv_cache(), (0, 0))


if __name__ == "__main__":
    unittest.main()

# ---- from test_kv_debug_snapshot_list.py ----


def test_list_matches_stacked():
    rng = np.random.default_rng(0)
    layers = [rng.standard_normal((3, 2, 4)).astype(np.float32) for _ in range(5)]
    stacked = np.stack(layers, axis=0)
    snap_list = build_kv_debug_snapshot(layers)
    snap_stack = build_kv_debug_snapshot(stacked)
    assert snap_list.shape == snap_stack.shape == (5, 3, 2, 4)
    assert snap_list.global_digest == snap_stack.global_digest
    assert snap_list.page_digests == snap_stack.page_digests


# ---- from test_pd_admission.py ----


class _Pool:
    def __init__(self, n):
        self._free = list(range(n))

    def reserve(self):
        return self._free.pop(0) if self._free else None


class _Req:
    def __init__(self, room):
        self.bootstrap_room = room
        self.disagg_host_buffer_id = None


def test_non_pd_req_not_gated():
    pool = _Pool(1)
    req = _Req(room=None)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is True and bid is None  # admitted, no reservation


def test_pd_req_reserves_slot():
    pool = _Pool(1)
    req = _Req(room=7)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is True and bid == 0


def test_pd_req_blocked_when_pool_full():
    pool = _Pool(0)
    req = _Req(room=7)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is False and bid is None  # caller must `continue` (stay in queue)


def test_disabled_or_no_pool_not_gated():
    req = _Req(room=7)
    assert _reserve_host_slot_for_pd(None, True, req) == (True, None)
    assert _reserve_host_slot_for_pd(_Pool(0), False, req) == (True, None)
