"""Stage 4 H-B: orphan reaper + per-phase timeout.

Drives ``JaxTransferKVManager.reap_once(now)`` with a hand-stepped
clock so the tests stay deterministic. No real ZMQ / transfer
sockets are bound — the manager's sender/receiver objects expose
``transfer_started_at`` and ``fail(reason=...)`` that the reaper
calls directly, and we stub the dependencies they touch.
"""

from __future__ import annotations

from typing import Optional
from unittest import mock

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
)


def _make_mgr(ack_timeout=10.0, pull_timeout=5.0):
    wrapper = mock.MagicMock()
    notifier = mock.MagicMock()
    notifier.unregister_callback.return_value = None
    return JaxTransferKVManager(
        wrapper, notifier, host_pool=None,
        ack_timeout_seconds=ack_timeout,
        pull_timeout_seconds=pull_timeout,
        reaper_interval_seconds=60.0,  # avoid background thread
    )


def test_reap_once_does_nothing_when_no_inflight():
    mgr = _make_mgr()
    s, r = mgr.reap_once(now=1000.0)
    assert s == [] and r == []


def test_reap_once_skips_senders_in_bootstrapping():
    mgr = _make_mgr()
    sender = mgr.create_sender("req-1")
    # No call to send() → transfer_started_at is None → not reapable.
    s, r = mgr.reap_once(now=10000.0)
    assert s == [] and r == []
    assert sender.state == KVPoll.BOOTSTRAPPING


def test_reap_once_force_fails_orphan_sender(monkeypatch):
    mgr = _make_mgr(ack_timeout=10.0)
    sender = mgr.create_sender("req-orphan")
    # Simulate the sender being TRANSFERRING since t=0.
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    s, r = mgr.reap_once(now=10.0 + 0.001)
    assert s == ["req-orphan"]
    assert r == []
    assert sender.state == KVPoll.FAILED
    record = mgr.get_terminal_record("req-orphan", role="prefill")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "timeout"


def test_reap_once_force_fails_orphan_receiver():
    mgr = _make_mgr(pull_timeout=5.0)
    receiver = mgr.create_receiver("req-stuck")
    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 100.0

    s, r = mgr.reap_once(now=105.001)
    assert s == []
    assert r == ["req-stuck"]
    assert receiver.state == KVPoll.FAILED
    record = mgr.get_terminal_record("req-stuck", role="decode")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "timeout"


def test_reap_once_below_threshold_is_no_op():
    mgr = _make_mgr(ack_timeout=10.0, pull_timeout=5.0)
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    receiver = mgr.create_receiver("r1")
    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 0.0

    s, r = mgr.reap_once(now=4.0)  # both below threshold
    assert s == []
    assert r == []


def test_disabled_timeouts():
    mgr = _make_mgr(ack_timeout=0.0, pull_timeout=0.0)
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0

    s, r = mgr.reap_once(now=1_000_000.0)
    assert s == [] and r == []


def test_inflight_count_reflects_active():
    mgr = _make_mgr()
    assert mgr.inflight_count() == (0, 0)
    mgr.create_sender("a")
    mgr.create_receiver("b")
    mgr.create_receiver("c")
    assert mgr.inflight_count() == (1, 2)


def test_graceful_shutdown_drains_when_clean():
    mgr = _make_mgr()
    # No inflight → returns 0,0 immediately.
    aborted_s, aborted_r = mgr.graceful_shutdown(drain_timeout_seconds=0.1)
    assert (aborted_s, aborted_r) == (0, 0)


def test_graceful_shutdown_force_fails_after_timeout():
    mgr = _make_mgr()
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    aborted_s, aborted_r = mgr.graceful_shutdown(
        drain_timeout_seconds=0.05
    )
    assert aborted_s == 1
    assert sender.state == KVPoll.FAILED
    record = mgr.get_terminal_record("s1", role="prefill")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "shutdown"


def test_receiver_fail_is_lock_protected_against_concurrent_poll():
    """Stage 4 review I1: receiver.fail() called from the reaper
    while poll() is mid-flight must not produce illegal
    transitions. We simulate the race by force-failing the receiver
    BEFORE the second-leg poll runs; the poll must observe the
    state flip and bail out without re-transitioning to SUCCESS.
    """

    mgr = _make_mgr()
    receiver = mgr.create_receiver("r-race")
    receiver._metadata = mock.MagicMock()
    receiver._metadata.uuid = "r-race"
    receiver._metadata.p_side_channel_host = "127.0.0.1"
    receiver._metadata.p_side_channel_port = 9601
    receiver._metadata.remote_addr = "127.0.0.1:30001"
    receiver._metadata.spec = mock.MagicMock()

    # Drive the first leg manually so we control the wrapper.pull
    # behavior.
    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 0.0
    fake_arr = mock.MagicMock()
    fake_arr.is_ready.return_value = True
    receiver._result = fake_arr

    # Reaper steps in *before* the next poll's success transition.
    receiver.fail(reason="timeout")
    assert receiver.state == KVPoll.FAILED

    # Second poll must not blow up trying to transition FAILED -> SUCCESS.
    final = receiver.poll()
    assert final == KVPoll.FAILED


def test_receiver_abort_failure_exception_and_clear():
    mgr = _make_mgr()
    receiver = mgr.create_receiver("r-abort")

    receiver.abort()
    assert receiver.state == KVPoll.FAILED
    record = mgr.get_terminal_record("r-abort", role="decode")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "abort"

    with pytest.raises(RuntimeError, match="abort"):
        receiver.failure_exception()

    receiver.clear()
    assert mgr.get_terminal_record("r-abort", role="decode") is None
    receiver.clear()


