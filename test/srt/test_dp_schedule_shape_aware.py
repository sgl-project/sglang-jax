"""Unit tests for the ``shape_aware`` DP scheduling decision logic.

Covers the pure ``pick_shape_aware_dp`` policy, which balances prefill
(input) and decode (output) token load jointly:

    score(rank) = max(input_counts[rank] + item_input,
                      output_counts[rank] + item_output)

Route to the eligible rank minimizing that bottleneck dimension. Like the
other policies this lives in the dependency-free helper module, so the test
imports neither Scheduler nor JAX. The end-to-end throughput/latency
improvement is validated separately by a bimodal A/B (see the PR description).
"""

from __future__ import annotations

from sgl_jax.srt.managers.dp_schedule_policy import pick_shape_aware_dp as pick


def test_routes_to_min_bottleneck_when_item_is_small():
    # rank0 output-heavy, rank1 input-heavy, rank2 balanced-low.
    inp = [100, 900, 50]
    out = [900, 100, 50]
    # A tiny request lands on rank2 (lowest max dimension = 50).
    assert pick([0, 1, 2], inp, out, item_input=10, item_output=10) == 2


def test_input_heavy_request_drawn_to_output_heavy_rank():
    # rank0: heavy output, light input. rank1: heavy input, light output.
    inp = [50, 800]
    out = [800, 50]
    # A prefill-heavy request (big input) goes to rank0 (light input), co-locating
    # with its heavy output: rank0 max(50+700,800)=800 < rank1 max(800+700,50)=1500.
    assert pick([0, 1], inp, out, item_input=700, item_output=0) == 0


def test_output_heavy_request_drawn_to_input_heavy_rank():
    inp = [50, 800]
    out = [800, 50]
    # A decode-heavy request (big output) goes to rank1 (light output):
    # rank0 max(50,800+700)=1500 > rank1 max(800,50+700)=800.
    assert pick([0, 1], inp, out, item_input=0, item_output=700) == 1


def test_tiebreak_prefers_lower_total_then_rank():
    # Equal bottleneck (max=100) on both; rank1 has the lower total sum (110<200).
    inp = [100, 100]
    out = [100, 10]
    assert pick([0, 1], inp, out) == 1


def test_full_ranks_excluded_by_caller_eligibility():
    # rank0 would be ideal (empty) but the caller left it out of `eligible`.
    inp = [0, 500]
    out = [0, 500]
    assert pick([1], inp, out, item_input=10, item_output=10) == 1


def test_no_eligible_ranks_returns_none():
    assert pick([], [0, 0], [0, 0], item_input=10, item_output=10) is None


def test_item_shape_changes_the_choice():
    # Same rank state, opposite request shapes route to opposite ranks.
    inp = [0, 600]
    out = [600, 0]
    assert pick([0, 1], inp, out, item_input=500, item_output=0) == 0
    assert pick([0, 1], inp, out, item_input=0, item_output=500) == 1


def test_zero_item_defaults_balance_existing_load():
    # With no item load, pick the rank with the smallest current bottleneck.
    inp = [300, 100]
    out = [100, 250]
    # rank0 max=300, rank1 max=250 -> rank1.
    assert pick([0, 1], inp, out) == 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
