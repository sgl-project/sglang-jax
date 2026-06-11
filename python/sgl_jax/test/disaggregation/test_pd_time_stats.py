"""Tests for per-request PD time_stats (phase latency breakdown)."""

from __future__ import annotations

import pytest

from sgl_jax.srt.disaggregation.req_time_stats import (
    TimeStats,
    format_time_stats,
    maybe_log_time_stats,
)


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
