"""Unit tests for DP scheduling policy selection.

Covers the pure ``pick_cache_aware_dp`` decision logic,
``req_prefix_match_key`` probe-key extraction, and ServerArgs default policy
normalization.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from sgl_jax.srt.managers.dp_schedule_policy import pick_cache_aware_dp as pick
from sgl_jax.srt.managers.dp_schedule_policy import req_prefix_match_key as match_key
from sgl_jax.srt.server_args import ServerArgs


def _pick(
    eligible,
    counts,
    tokens,
    matches,
    prompt_len,
    *,
    input_counts=None,
    output_counts=None,
    item_input=0,
    item_output=0,
):
    return pick(
        eligible,
        counts,
        tokens,
        matches,
        prompt_len,
        input_counts if input_counts is not None else counts,
        output_counts if output_counts is not None else tokens,
        item_input,
        item_output,
    )


def test_holder_wins_when_loads_equal():
    counts = [0, 0, 0, 0]
    tokens = [0, 0, 0, 0]
    matches = {0: 0, 1: 512, 2: 0, 3: 0}
    assert _pick([0, 1, 2, 3], counts, tokens, matches, prompt_len=512) == 1


def test_least_loaded_among_holders():
    # Both 0 and 1 hold a substantial prefix; rank 1 is less loaded. The longer
    # match (rank 0, 600) does NOT win -- the hot prefix spreads to the lighter holder.
    counts = [3, 1, 0, 0]
    tokens = [30, 10, 0, 0]
    matches = {0: 600, 1: 400}
    assert _pick([0, 1, 2, 3], counts, tokens, matches, prompt_len=512) == 1


def test_no_cached_match_falls_back_to_shape_aware():
    # Rank 0 has fewer requests, so the old least-loaded fallback would pick it.
    # The output-heavy item should instead go to rank 1, whose output side is light.
    counts = [0, 1]
    tokens = [0, 1]
    input_counts = [0, 900]
    output_counts = [900, 0]
    matches = {0: 0, 1: 0}
    assert (
        _pick(
            [0, 1],
            counts,
            tokens,
            matches,
            prompt_len=512,
            input_counts=input_counts,
            output_counts=output_counts,
            item_input=0,
            item_output=700,
        )
        == 1
    )


def test_below_threshold_match_is_not_a_holder():
    # 100/512 < 0.5 -> rank 0 is not a holder; route by load to the idle rank 1.
    counts = [5, 0]
    tokens = [50, 0]
    matches = {0: 100}
    assert _pick([0, 1], counts, tokens, matches, prompt_len=512) == 1


def test_large_load_skew_overrides_cache_affinity():
    # Rank 0 holds the full prefix but is overloaded; the skew (40-0 > 32) forces
    # load balancing to the idle rank 1.
    counts = [40, 0]
    tokens = [400, 0]
    matches = {0: 512}
    assert _pick([0, 1], counts, tokens, matches, prompt_len=512) == 1


def test_small_load_skew_keeps_affinity():
    # Skew 5-0 <= 32: affinity holds, route to the holder rank 0.
    counts = [5, 0]
    tokens = [50, 0]
    matches = {0: 512}
    assert _pick([0, 1], counts, tokens, matches, prompt_len=512) == 0


def test_hot_prefix_spills_when_owning_rank_is_full():
    # Rank 0 holds the prefix but is full (not eligible): route by load among the
    # eligible ranks (none of which hold it).
    counts = [0, 5, 1, 9]
    tokens = [0, 50, 10, 90]
    matches = {0: 512}
    assert _pick([1, 2, 3], counts, tokens, matches, prompt_len=512) == 2


def test_holder_and_load_tie_breaks_by_lowest_rank():
    counts = [2, 2, 2, 2]
    tokens = [20, 20, 20, 20]
    matches = {0: 256, 1: 256, 2: 0, 3: 0}
    assert _pick([0, 1, 2, 3], counts, tokens, matches, prompt_len=256) == 0


def test_prompt_len_zero_falls_back_to_shape_aware():
    # No usable prompt_len -> skip holder math (no zero-division) and use shape-aware fallback.
    counts = [0, 1]
    tokens = [0, 1]
    input_counts = [0, 900]
    output_counts = [900, 0]
    matches = {0: 512}
    assert (
        _pick(
            [0, 1],
            counts,
            tokens,
            matches,
            prompt_len=0,
            input_counts=input_counts,
            output_counts=output_counts,
            item_input=0,
            item_output=700,
        )
        == 1
    )


def test_all_full_returns_none():
    assert _pick([], [0, 0], [0, 0], {}, prompt_len=4) is None


def _req(input_ids, extra_key=None, lora_id=None, return_logprob=False, logprob_start_len=-1):
    return SimpleNamespace(
        input_ids=input_ids,
        extra_key=extra_key,
        lora_id=lora_id,
        return_logprob=return_logprob,
        logprob_start_len=logprob_start_len,
    )


def test_match_key_uses_reusable_prefix_not_full():
    # The real radix lookup keeps >=1 token to generate, so the probe uses the
    # reusable prefix (input_len - 1), not the full prompt.
    assert match_key(_req([1, 2, 3, 4], extra_key="lora-a")) == ([1, 2, 3], "lora-a")


def test_match_key_appends_lora_id():
    # Matches Req.__init__: lora_id is concatenated into extra_key.
    assert match_key(_req([1, 2, 3, 4], extra_key="salt:", lora_id="adapter")) == (
        [1, 2, 3],
        "salt:adapter",
    )


def test_match_key_lora_id_without_extra_key():
    assert match_key(_req([1, 2, 3, 4], lora_id="adapter")) == ([1, 2, 3], "adapter")


def test_match_key_return_logprob_clamps_to_start_len():
    # return_logprob clamps the reusable prefix to logprob_start_len.
    assert match_key(_req([1, 2, 3, 4, 5], return_logprob=True, logprob_start_len=2)) == (
        [1, 2],
        None,
    )


def test_match_key_return_logprob_default_start_len_uses_reusable_prefix():
    # logprob_start_len=-1 is normalized to input_len-1 by the scheduler, so a
    # default logprob request still probes the reusable prefix (not None).
    assert match_key(_req([1, 2, 3, 4], return_logprob=True, logprob_start_len=-1)) == (
        [1, 2, 3],
        None,
    )
    # None on the wire behaves like the -1 sentinel.
    assert match_key(_req([1, 2, 3, 4], return_logprob=True, logprob_start_len=None)) == (
        [1, 2, 3],
        None,
    )


def test_match_key_single_token_has_no_reusable_prefix():
    assert match_key(_req([7])) == (None, None)


def test_match_key_none_and_empty_are_skipped():
    assert match_key(_req(None)) == (None, None)
    assert match_key(_req([])) == (None, None)


def test_match_key_batched_input_is_skipped():
    assert match_key(_req([[1, 2], [3, 4]])) == (None, None)


def _server_args(**kwargs) -> ServerArgs:
    return ServerArgs(model_path="dummy", device="cpu", **kwargs)


def test_unset_dp_schedule_policy_follows_normalized_radix_state():
    assert _server_args().dp_schedule_policy == "cache_aware"
    assert _server_args(disable_radix_cache=True).dp_schedule_policy == "min_running_queue"
    assert _server_args(pd_disaggregation="pathways").dp_schedule_policy == "min_running_queue"

    hicache_args = _server_args(disable_radix_cache=True, hicache_storage="none")
    assert hicache_args.disable_radix_cache is False
    assert hicache_args.dp_schedule_policy == "cache_aware"


def test_explicit_dp_schedule_policy_is_preserved():
    assert (
        _server_args(dp_schedule_policy="min_running_queue").dp_schedule_policy
        == "min_running_queue"
    )
    assert (
        _server_args(
            disable_radix_cache=True,
            dp_schedule_policy="shape_aware",
        ).dp_schedule_policy
        == "shape_aware"
    )


def test_cli_default_keeps_unset_policy_distinct_from_explicit_min_running_queue():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    unset = parser.parse_args(["--model-path", "dummy"])
    explicit = parser.parse_args(
        ["--model-path", "dummy", "--dp-schedule-policy", "min_running_queue"]
    )

    assert unset.dp_schedule_policy is None
    assert explicit.dp_schedule_policy == "min_running_queue"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
