"""Tests for the transport layer: KVPoll state machine and CommonKVManager."""

from __future__ import annotations

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    KVPoll,
    StateHolder,
    is_legal_transition,
)
from sgl_jax.srt.disaggregation.common.core import (
    CommonKVManager,
    TerminalTransferRecord,
)

# ------------------------------------------------------------------
# KVPoll state machine (from test_kv_manager_state)
# ------------------------------------------------------------------


def test_legal_transitions_set_matches_rfc():
    assert (
        frozenset(
            {
                (KVPoll.BOOTSTRAPPING, KVPoll.WAITING_FOR_INPUT),
                (KVPoll.WAITING_FOR_INPUT, KVPoll.TRANSFERRING),
                (KVPoll.TRANSFERRING, KVPoll.SUCCESS),
                (KVPoll.BOOTSTRAPPING, KVPoll.FAILED),
                (KVPoll.WAITING_FOR_INPUT, KVPoll.FAILED),
                (KVPoll.TRANSFERRING, KVPoll.FAILED),
            }
        )
        == LEGAL_TRANSITIONS
    )


def test_terminal_states():
    assert frozenset({KVPoll.SUCCESS, KVPoll.FAILED}) == TERMINAL_STATES


@pytest.mark.parametrize(
    ("current", "next_state"),
    sorted(LEGAL_TRANSITIONS, key=lambda pair: (pair[0].value, pair[1].value)),
)
def test_legal_transitions_succeed(current: KVPoll, next_state: KVPoll):
    holder = StateHolder(initial=current)
    assert is_legal_transition(current, next_state) is True
    holder._transition_to(next_state)
    assert holder.state == next_state


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (KVPoll.BOOTSTRAPPING, KVPoll.TRANSFERRING),
        (KVPoll.BOOTSTRAPPING, KVPoll.SUCCESS),
        (KVPoll.WAITING_FOR_INPUT, KVPoll.SUCCESS),
        (KVPoll.TRANSFERRING, KVPoll.WAITING_FOR_INPUT),
    ],
)
def test_representative_illegal_transitions_raise(current: KVPoll, next_state: KVPoll):
    holder = StateHolder(initial=current)
    assert is_legal_transition(current, next_state) is False
    with pytest.raises(ValueError, match="illegal KVPoll transition"):
        holder._transition_to(next_state)
    assert holder.state == current


@pytest.mark.parametrize("terminal", [KVPoll.SUCCESS, KVPoll.FAILED])
@pytest.mark.parametrize("target", [KVPoll.BOOTSTRAPPING, KVPoll.WAITING_FOR_INPUT])
def test_terminal_states_reject_outgoing_transitions(terminal: KVPoll, target: KVPoll):
    holder = StateHolder(initial=terminal)
    with pytest.raises(ValueError, match="illegal KVPoll transition"):
        holder._transition_to(target)
    assert holder.state == terminal


def test_self_loops_are_illegal():
    for state in KVPoll:
        holder = StateHolder(initial=state)
        with pytest.raises(ValueError):
            holder._transition_to(state)


# ------------------------------------------------------------------
# CommonKVManager (from test_request_transport_core)
# ------------------------------------------------------------------


class _MockParticipant:
    """Minimal duck-type stand-in for a sender or receiver."""

    def __init__(self, *, transfer_started_at: float | None = None):
        self.transfer_started_at = transfer_started_at
        self.failed = False
        self.fail_reason: str | None = None

    def fail(self, *, reason: str = "test") -> None:
        self.failed = True
        self.fail_reason = reason


class _TestKVManager(CommonKVManager):
    """Concrete subclass for testing (CommonKVManager is abstract)."""

    def create_sender(self, req_id):
        raise NotImplementedError

    def create_receiver(self, req_id):
        raise NotImplementedError


def _make_core(ack_timeout=10.0, pull_timeout=5.0):
    return _TestKVManager(
        ack_timeout_seconds=ack_timeout,
        pull_timeout_seconds=pull_timeout,
        reaper_interval_seconds=60.0,
    )


# Registry


def test_register_and_prune_sender():
    core = _make_core()
    p = _MockParticipant()
    core.register_sender("req-1", p)
    assert core.inflight_count() == (1, 0)
    core._prune_sender("req-1")
    assert core.inflight_count() == (0, 0)


def test_register_and_prune_receiver():
    core = _make_core()
    p = _MockParticipant()
    core.register_receiver("req-1", p)
    assert core.inflight_count() == (0, 1)
    core._prune_receiver("req-1")
    assert core.inflight_count() == (0, 0)


def test_duplicate_sender_raises():
    core = _make_core()
    core.register_sender("dup", _MockParticipant())
    with pytest.raises(ValueError, match="already exists"):
        core.register_sender("dup", _MockParticipant())


def test_duplicate_receiver_raises():
    core = _make_core()
    core.register_receiver("dup", _MockParticipant())
    with pytest.raises(ValueError, match="already exists"):
        core.register_receiver("dup", _MockParticipant())


def test_prune_nonexistent_is_no_op():
    core = _make_core()
    core._prune_sender("ghost")
    core._prune_receiver("ghost")


# Terminal records


def test_record_and_get_terminal():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="prefill",
        transfer_id="tid-1",
        state=KVPoll.SUCCESS,
        reason="ack",
    )
    rec = core.get_terminal_record("req-1", role="prefill")
    assert rec is not None
    assert isinstance(rec, TerminalTransferRecord)
    assert rec.req_id == "req-1"
    assert rec.state == KVPoll.SUCCESS
    assert rec.reason == "ack"


def test_clear_terminal_record():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="decode",
        transfer_id="tid-1",
        state=KVPoll.FAILED,
        reason="timeout",
    )
    core._clear_terminal_record("req-1", role="decode")
    assert core.get_terminal_record("req-1", role="decode") is None


def test_terminal_records_bounded_eviction():
    core = _make_core()
    core._max_terminal_records = 3
    for i in range(5):
        core.record_terminal(
            f"req-{i}",
            role="prefill",
            transfer_id=f"tid-{i}",
            state=KVPoll.SUCCESS,
            reason="ack",
        )
    assert core.get_terminal_record("req-0", role="prefill") is None
    assert core.get_terminal_record("req-1", role="prefill") is None
    assert core.get_terminal_record("req-2", role="prefill") is not None
    assert core.get_terminal_record("req-4", role="prefill") is not None


def test_register_sender_clears_old_terminal():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="prefill",
        transfer_id="tid-old",
        state=KVPoll.FAILED,
        reason="old",
    )
    assert core.get_terminal_record("req-1", role="prefill") is not None
    core.register_sender("req-1", _MockParticipant())
    assert core.get_terminal_record("req-1", role="prefill") is None


# Reaper


def test_reap_once_no_inflight():
    core = _make_core()
    s, r = core.reap_once(now=1000.0)
    assert s == [] and r == []


def test_reap_once_skips_not_started():
    core = _make_core(ack_timeout=10.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=None))
    s, r = core.reap_once(now=10000.0)
    assert s == []


def test_reap_once_force_fails_sender():
    core = _make_core(ack_timeout=10.0)
    p = _MockParticipant(transfer_started_at=0.0)
    core.register_sender("s1", p)
    s, r = core.reap_once(now=10.001)
    assert s == ["s1"]
    assert p.failed
    assert p.fail_reason == "timeout"


def test_reap_once_force_fails_receiver():
    core = _make_core(pull_timeout=5.0)
    p = _MockParticipant(transfer_started_at=100.0)
    core.register_receiver("r1", p)
    s, r = core.reap_once(now=105.001)
    assert r == ["r1"]
    assert p.failed


def test_reap_once_below_threshold():
    core = _make_core(ack_timeout=10.0, pull_timeout=5.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=0.0))
    core.register_receiver("r1", _MockParticipant(transfer_started_at=0.0))
    s, r = core.reap_once(now=4.0)
    assert s == [] and r == []


def test_disabled_timeouts():
    core = _make_core(ack_timeout=0.0, pull_timeout=0.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=0.0))
    s, r = core.reap_once(now=1_000_000.0)
    assert s == [] and r == []


# Inflight count


def test_inflight_count():
    core = _make_core()
    assert core.inflight_count() == (0, 0)
    core.register_sender("a", _MockParticipant())
    core.register_receiver("b", _MockParticipant())
    core.register_receiver("c", _MockParticipant())
    assert core.inflight_count() == (1, 2)


# Graceful shutdown


def test_graceful_shutdown_no_inflight():
    core = _make_core()
    aborted_s, aborted_r = core.graceful_shutdown(drain_timeout_seconds=0.05)
    assert (aborted_s, aborted_r) == (0, 0)


def test_graceful_shutdown_force_fails_stragglers():
    core = _make_core()
    p1 = _MockParticipant(transfer_started_at=0.0)
    p2 = _MockParticipant(transfer_started_at=0.0)
    core.register_sender("s1", p1)
    core.register_receiver("r1", p2)
    aborted_s, aborted_r = core.graceful_shutdown(drain_timeout_seconds=0.05)
    assert aborted_s == 1
    assert aborted_r == 1
    assert p1.failed and p1.fail_reason == "shutdown"
    assert p2.failed and p2.fail_reason == "shutdown"
