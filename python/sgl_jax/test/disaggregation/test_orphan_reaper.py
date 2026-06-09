"""Tests for the orphan/timeout reaper in CommonKVManager."""

from __future__ import annotations

import time

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.common.core import CommonKVManager


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
