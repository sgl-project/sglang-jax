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
    """Effective ``(token_ids, extra_key)`` for a cache-affinity prefix probe.

    Mirrors the key the request will use for its *real* radix lookup, so the
    probe routes on a match the request can actually reuse (and the holder
    denominator is the real probe length):

    - ``lora_id`` is concatenated into ``extra_key`` for cache namespace
      isolation, exactly as ``Req.__init__`` does.
    - The probe length is the *reusable* prefix, not the full prompt: a request
      must keep at least one token to generate, so the lookup uses ``len - 1``
      (``Req.adjust_max_prefix_ids``), further clamped by ``logprob_start_len``
      when ``return_logprob`` is set. Using ``len - 1`` also stops page-aligned
      ``match_prefix`` from reporting a phantom final-page hit the request drops.

    Returns ``(None, None)`` when there is no usable single-sequence reusable
    prefix (an unexpanded batch, an empty/one-token prompt, or a request whose
    reusable prefix clamps to zero), so the caller falls back to load balancing.
    """
    input_ids = req.input_ids
    if not isinstance(input_ids, list) or not input_ids or not isinstance(input_ids[0], int):
        return None, None

    extra_key = req.extra_key
    lora_id = getattr(req, "lora_id", None)
    if lora_id is not None:
        extra_key = (extra_key or "") + lora_id

    max_prefix_len = len(input_ids) - 1
    if getattr(req, "return_logprob", False):
        logprob_start_len = getattr(req, "logprob_start_len", -1)
        if logprob_start_len is None or logprob_start_len == -1:
            # Scheduler.handle_generate_request normalizes the -1 sentinel to
            # input_len - 1, so a default logprob request still reuses the prefix;
            # only an explicit start length clamps it shorter.
            logprob_start_len = len(input_ids) - 1
        max_prefix_len = min(max_prefix_len, logprob_start_len)
    max_prefix_len = max(max_prefix_len, 0)
    if max_prefix_len == 0:
        return None, None

    return input_ids[:max_prefix_len], extra_key


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


def pick_shape_aware_dp(
    eligible: list[int],
    input_counts: list[int],
    output_counts: list[int],
    item_input: int = 0,
    item_output: int = 0,
) -> int | None:
    """Shape-aware DP policy: joint prefill/decode load balancing.

    Balance prefill (input) and decode (output) token load jointly: route to the
    eligible rank whose *bottleneck* dimension stays smallest after admitting the
    request::

        score(r) = max(input_counts[r] + item_input, output_counts[r] + item_output)

    The ``max`` keeps whichever resource is the bottleneck on each rank (prefill
    FLOPs vs. decode/KV HBM) lowest, so a prefill-heavy request is drawn toward a
    decode-heavy rank and vice versa, co-locating complementary shapes rather than
    piling like-shaped work onto one rank. Ties are broken by total load then rank
    index. ``input_counts`` / ``output_counts`` are the per-rank running+pending
    token sums; eligibility (the admission cap) is decided by the caller.
    """
    if not eligible:
        return None

    def score(r: int) -> tuple[int, int, int]:
        after_in = input_counts[r] + item_input
        after_out = output_counts[r] + item_output
        return (max(after_in, after_out), after_in + after_out, r)

    return min(eligible, key=score)
