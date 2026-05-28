"""Exhaustive ``KVPoll`` state-machine tests.

Covers every (current, next) pair across all five states. Legal pairs
must succeed; everything else must raise ``ValueError``. ``SUCCESS`` and
``FAILED`` are terminal — no transition out of them is allowed.
"""

from __future__ import annotations

import itertools

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    KVPoll,
    StateHolder,
    is_legal_transition,
)

ALL_STATES = list(KVPoll)
ALL_PAIRS = list(itertools.product(ALL_STATES, ALL_STATES))


def test_state_space_size():
    assert len(ALL_STATES) == 5
    assert len(ALL_PAIRS) == 25


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


@pytest.mark.parametrize(("current", "next_state"), ALL_PAIRS)
def test_exhaustive_transitions(current: KVPoll, next_state: KVPoll):
    holder = StateHolder(initial=current)
    expected_legal = (current, next_state) in LEGAL_TRANSITIONS
    assert is_legal_transition(current, next_state) == expected_legal

    if expected_legal:
        holder._transition_to(next_state)
        assert holder.state == next_state
    else:
        with pytest.raises(ValueError, match="illegal KVPoll transition"):
            holder._transition_to(next_state)
        assert holder.state == current


@pytest.mark.parametrize("terminal", list(TERMINAL_STATES))
@pytest.mark.parametrize("target", ALL_STATES)
def test_terminal_states_reject_all_outgoing(terminal: KVPoll, target: KVPoll):
    holder = StateHolder(initial=terminal)
    with pytest.raises(ValueError, match="illegal KVPoll transition"):
        holder._transition_to(target)
    assert holder.state == terminal


def test_self_loops_are_illegal():
    for state in ALL_STATES:
        holder = StateHolder(initial=state)
        with pytest.raises(ValueError):
            holder._transition_to(state)
