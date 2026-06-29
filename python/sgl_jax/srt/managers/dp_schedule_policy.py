"""Pure DP scheduling-policy decisions.

Kept dependency-free (no JAX / model imports) so the policy logic can be unit
tested without constructing a Scheduler. The Scheduler imports these and supplies
the live tree / load state; these functions hold only the decision math.
"""

from __future__ import annotations

# Soft affinity-vs-load thresholds (upstream sgl-router defaults). When the load
# skew across eligible ranks is large, ignore cache affinity and balance; else
# route a request to the least-loaded rank that holds a substantial cached prefix
# (match_rate > CACHE_THRESHOLD), so a hot prefix spreads across its holders
# instead of concentrating on the single longest-match rank.
BALANCE_ABS = 32
BALANCE_REL = 1.1
CACHE_THRESHOLD = 0.5


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
    prompt_len: int,
) -> int | None:
    """Cache-affinity DP policy with soft load balancing.

    1. If the load skew across eligible ranks is large
       (``max-min > BALANCE_ABS`` and ``max > min*BALANCE_REL``), ignore cache
       affinity and pick the least-loaded rank.
    2. Otherwise route to the least-loaded rank among the *holders* -- ranks whose
       cached prefix covers more than ``CACHE_THRESHOLD`` of the prompt -- so a hot
       prefix spreads across all its holders by load.
    3. No holders (or no usable ``prompt_len``) -> least-loaded rank.

    Load order is ``(running, tokens, rank)``. Eligibility (the per-rank admission
    cap) is decided by the caller; this only chooses among eligible ranks.
    """
    if not eligible:
        return None

    def least_loaded(ranks: list[int]) -> int:
        return min(ranks, key=lambda r: (counts[r], token_counts[r], r))

    loads = [counts[r] for r in eligible]
    max_load, min_load = max(loads), min(loads)
    if max_load - min_load > BALANCE_ABS and max_load > min_load * BALANCE_REL:
        return least_loaded(eligible)

    if prompt_len > 0:
        holders = [r for r in eligible if matches.get(r, 0) / prompt_len > CACHE_THRESHOLD]
        if holders:
            return least_loaded(holders)

    return least_loaded(eligible)
