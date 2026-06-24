"""Unit tests for the cache-aware DP scheduling policy decision logic.

Covers the pure ``pick_cache_aware_dp`` policy and ``req_prefix_match_key``
probe-key extraction. These live in a dependency-free helper module, so the test
imports neither Scheduler nor JAX. The end-to-end reuse improvement is validated
separately by the recurrent reuse K-sweep against a multi-host server.
"""

from __future__ import annotations

from types import SimpleNamespace

from sgl_jax.srt.managers.dp_schedule_policy import pick_cache_aware_dp as pick
from sgl_jax.srt.managers.dp_schedule_policy import req_prefix_match_key as match_key


def test_affinity_rank_wins_when_loads_equal():
    counts = [0, 0, 0, 0]
    tokens = [0, 0, 0, 0]
    matches = {0: 0, 1: 512, 2: 0, 3: 0}
    assert pick([0, 1, 2, 3], counts, tokens, matches) == 1


def test_tie_on_match_breaks_by_least_load():
    counts = [5, 2, 7, 9]
    tokens = [50, 20, 70, 90]
    matches = {0: 512, 1: 512, 2: 0, 3: 0}
    assert pick([0, 1, 2, 3], counts, tokens, matches) == 1


def test_no_cached_match_falls_back_to_least_load():
    counts = [3, 1, 2, 4]
    tokens = [30, 10, 20, 40]
    matches = {0: 0, 1: 0, 2: 0, 3: 0}
    assert pick([0, 1, 2, 3], counts, tokens, matches) == 1


def test_hot_prefix_spills_when_owning_rank_is_full():
    # Rank 0 holds the prefix but is full (not eligible): the request spills to
    # the least-loaded eligible rank rather than overloading rank 0.
    counts = [0, 5, 1, 9]
    tokens = [0, 50, 10, 90]
    matches = {0: 512}  # only the full rank had a cached match
    assert pick([1, 2, 3], counts, tokens, matches) == 2


def test_match_and_load_tie_breaks_by_lowest_rank():
    counts = [2, 2, 2, 2]
    tokens = [20, 20, 20, 20]
    matches = {0: 256, 1: 256, 2: 0, 3: 0}
    assert pick([0, 1, 2, 3], counts, tokens, matches) == 0


def test_all_full_returns_none():
    assert pick([], [0, 0], [0, 0], {}) is None


def test_match_key_single_sequence():
    req = SimpleNamespace(input_ids=[1, 2, 3, 4], extra_key="lora-a")
    assert match_key(req) == ([1, 2, 3, 4], "lora-a")


def test_match_key_none_and_empty_are_skipped():
    assert match_key(SimpleNamespace(input_ids=None, extra_key=None)) == (None, None)
    assert match_key(SimpleNamespace(input_ids=[], extra_key=None)) == (None, None)


def test_match_key_batched_input_is_skipped():
    req = SimpleNamespace(input_ids=[[1, 2], [3, 4]], extra_key=None)
    assert match_key(req) == (None, None)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
