"""Tests for the decode event-loop stall watchdog (pure observability)."""

from __future__ import annotations

from sgl_jax.srt.disaggregation.decode_watchdog import EventLoopWatchdog


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
