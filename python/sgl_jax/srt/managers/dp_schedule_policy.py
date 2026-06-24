"""Pure DP scheduling-policy decisions.

Kept dependency-free (no JAX / model imports) so the policy logic can be unit
tested without constructing a Scheduler. The Scheduler imports these and supplies
the live tree / load state; these functions hold only the decision math.
"""

from __future__ import annotations


def req_prefix_match_key(req) -> tuple[list[int] | None, str | None]:
    """Token ids + extra_key for a cache-affinity prefix probe.

    Returns ``(None, None)`` when the request carries no usable single-sequence
    token list (e.g. an unexpanded batch), so the caller falls back to load
    balancing.
    """
    input_ids = req.input_ids
    if not isinstance(input_ids, list) or not input_ids:
        return None, None
    if isinstance(input_ids[0], int):
        return input_ids, req.extra_key
    return None, None


def pick_cache_aware_dp(
    eligible: list[int],
    counts: list[int],
    token_counts: list[int],
    matches: dict[int, int],
) -> int | None:
    """Cache-affinity DP policy.

    Among eligible ranks, the one holding the longest cached prefix wins; ties
    (and the no-cached-prefix case) break by least load ``(running, tokens,
    rank)``. Eligibility is decided by the caller, so a hot prefix whose rank is
    full spills to the next-best rank instead of overloading one rank.
    """
    if not eligible:
        return None
    best = max((matches.get(dp_rank, 0) for dp_rank in eligible), default=0)
    pool = (
        [dp_rank for dp_rank in eligible if matches.get(dp_rank, 0) == best]
        if best > 0
        else list(eligible)
    )
    return min(pool, key=lambda dp_rank: (counts[dp_rank], token_counts[dp_rank], dp_rank))
