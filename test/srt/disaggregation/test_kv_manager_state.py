"""Contract tests for the ``KVPoll`` state machine."""

from __future__ import annotations

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    KVPoll,
    StateHolder,
    is_legal_transition,
)


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
