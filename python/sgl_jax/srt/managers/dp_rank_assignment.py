"""Preserve intake order and dispatch generate requests to the per-round router."""

from dataclasses import dataclass

from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput


@dataclass
class DpRankAssignmentResult:
    ready_reqs: list
    pending_reqs: list[TokenizedGenerateReqInput]


def assign_dp_ranks(
    *, recv_reqs: list | None, pending_dp_reqs: list, assign
) -> DpRankAssignmentResult:
    """Retry pending requests first; pass control messages through unchanged."""
    ready_reqs = []
    deferred_reqs = []
    for req in [*pending_dp_reqs, *(recv_reqs or ())]:
        if isinstance(req, TokenizedGenerateReqInput) and assign(req) is None:
            deferred_reqs.append(req)
        else:
            ready_reqs.append(req)
    return DpRankAssignmentResult(ready_reqs, deferred_reqs)
