"""Scheduler-level DP rank assignment for incoming requests."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput

logger = logging.getLogger(__name__)


@dataclass
class DpRankAssignmentResult:
    ready_reqs: list[Any]
    pending_reqs: list[TokenizedGenerateReqInput]


def _valid_dp_rank(dp_rank: int | None, dp_size: int) -> bool:
    return dp_rank is not None and 0 <= dp_rank < dp_size


def assign_dp_ranks(
    *,
    recv_reqs: list[Any] | None,
    pending_dp_reqs: list[TokenizedGenerateReqInput],
    dp_size: int,
    dp_schedule_policy: str,
    select_round_robin_dp: Callable[[], int],
    select_cache_aware_dp: Callable[
        [TokenizedGenerateReqInput, list[int], list[int], list[int], list[int]],
        int | None,
    ],
    select_min_running_dp: Callable[[list[int], list[int]], int | None],
    select_shape_aware_dp: Callable[
        [int, int, list[int], list[int], list[int], list[int]], int | None
    ],
    estimate_req_tokens: Callable[[TokenizedGenerateReqInput], int],
    estimate_req_io_tokens: Callable[[TokenizedGenerateReqInput], tuple[int, int]],
) -> DpRankAssignmentResult:
    """Assign or defer DP ranks for incoming generate requests.

    This keeps request-intake ordering and mutation rules together while the
    Scheduler remains the owner of live load snapshots and cache probing.
    """
    if recv_reqs is None:
        recv_reqs = []

    combined_reqs: list[Any] = []
    if pending_dp_reqs:
        combined_reqs.extend(pending_dp_reqs)
    if recv_reqs:
        combined_reqs.extend(recv_reqs)

    if dp_size == 1:
        for req in combined_reqs:
            if isinstance(req, TokenizedGenerateReqInput):
                req.dp_rank = 0
        return DpRankAssignmentResult(ready_reqs=combined_reqs, pending_reqs=[])

    pending_counts = [0] * dp_size
    pending_token_counts = [0] * dp_size
    # shape_aware balances prefill (input) and decode (output) separately. cache_aware
    # needs the same split for its no-holder/cache-miss shape-aware fallback. Include
    # requests assigned earlier in this intake tick, otherwise bursts route against a
    # stale snapshot and mis-balance.
    tracks_pending_io = dp_schedule_policy in ("shape_aware", "cache_aware")
    pending_input_counts = [0] * dp_size
    pending_output_counts = [0] * dp_size
    ready_reqs: list[Any] = []
    deferred_reqs: list[TokenizedGenerateReqInput] = []

    def track(req, dp_rank):
        pending_counts[dp_rank] += 1
        pending_token_counts[dp_rank] += estimate_req_tokens(req)
        if tracks_pending_io:
            in_tok, out_tok = estimate_req_io_tokens(req)
            pending_input_counts[dp_rank] += in_tok
            pending_output_counts[dp_rank] += out_tok

    for req in combined_reqs:
        if not isinstance(req, TokenizedGenerateReqInput):
            ready_reqs.append(req)
            continue

        if req.dp_rank is not None:
            if _valid_dp_rank(req.dp_rank, dp_size):
                track(req, req.dp_rank)
                ready_reqs.append(req)
                continue

            logger.warning(
                "Ignoring invalid dp_rank=%s for request %s; reassigning with %s policy",
                req.dp_rank,
                getattr(req, "rid", None),
                dp_schedule_policy,
            )
            req.dp_rank = None

        if dp_schedule_policy == "round_robin":
            req.dp_rank = select_round_robin_dp()
            ready_reqs.append(req)
            continue

        if dp_schedule_policy == "cache_aware":
            dp_rank = select_cache_aware_dp(
                req,
                pending_counts,
                pending_token_counts,
                pending_input_counts,
                pending_output_counts,
            )
        elif dp_schedule_policy == "shape_aware":
            item_in, item_out = estimate_req_io_tokens(req)
            dp_rank = select_shape_aware_dp(
                item_in,
                item_out,
                pending_counts,
                pending_token_counts,
                pending_input_counts,
                pending_output_counts,
            )
        else:
            dp_rank = select_min_running_dp(pending_counts, pending_token_counts)

        if dp_rank is None:
            deferred_reqs.append(req)
            continue

        req.dp_rank = dp_rank
        track(req, dp_rank)
        ready_reqs.append(req)

    return DpRankAssignmentResult(ready_reqs=ready_reqs, pending_reqs=deferred_reqs)
