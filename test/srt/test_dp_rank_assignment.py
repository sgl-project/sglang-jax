"""Characterization tests for scheduler-level DP rank assignment."""

from __future__ import annotations

from sgl_jax.srt.managers.dp_rank_assignment import assign_dp_ranks
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput


def _req(
    rid: str,
    *,
    dp_rank: int | None = None,
    input_len: int = 4,
    max_new_tokens: int = 4,
) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=rid,
        input_ids=list(range(input_len)),
        sampling_params={"max_new_tokens": max_new_tokens},
        dp_rank=dp_rank,
    )


class _DpAssignmentHarness:
    def __init__(
        self,
        *,
        dp_size: int = 2,
        running_counts: list[int] | None = None,
        running_token_counts: list[int] | None = None,
        full_ranks: set[int] | None = None,
        per_dp_max_running_requests: int = 8,
    ):
        self.dp_size = dp_size
        self.running_counts = running_counts or [0] * dp_size
        self.running_token_counts = running_token_counts or [0] * dp_size
        self.full_ranks = full_ranks or set()
        self.per_dp_max_running_requests = per_dp_max_running_requests
        self.round_robin_counter = 0
        # Records (rid, extra_counts, extra_token_counts, item_input, item_output,
        # extra_input_counts, extra_output_counts) per cache_aware dispatch.
        self.cache_aware_calls: list[
            tuple[
                str | list[str] | None,
                list[int],
                list[int],
                int,
                int,
                list[int],
                list[int],
            ]
        ] = []
        # Records (item_input, item_output, extra_input_counts, extra_output_counts)
        # per shape_aware dispatch, so tests can assert the same-tick pending-IO split.
        self.shape_aware_calls: list[tuple[int, int, list[int], list[int]]] = []

    def select_round_robin_dp(self) -> int:
        dp_rank = self.round_robin_counter % self.dp_size
        self.round_robin_counter += 1
        return dp_rank

    def select_min_running_dp(
        self,
        extra_counts: list[int],
        extra_token_counts: list[int],
    ) -> int | None:
        eligible = self._eligible(extra_counts)
        if not eligible:
            return None

        return min(
            eligible,
            key=lambda dp_rank: (
                self.running_counts[dp_rank] + extra_counts[dp_rank],
                self.running_token_counts[dp_rank] + extra_token_counts[dp_rank],
                dp_rank,
            ),
        )

    def select_cache_aware_dp(
        self,
        req: TokenizedGenerateReqInput,
        extra_counts: list[int],
        extra_token_counts: list[int],
        extra_input_counts: list[int],
        extra_output_counts: list[int],
    ) -> int | None:
        item_input, item_output = self.estimate_req_io_tokens(req)
        self.cache_aware_calls.append(
            (
                req.rid,
                list(extra_counts),
                list(extra_token_counts),
                item_input,
                item_output,
                list(extra_input_counts),
                list(extra_output_counts),
            )
        )
        return self.select_min_running_dp(extra_counts, extra_token_counts)

    def select_shape_aware_dp(
        self,
        item_input: int,
        item_output: int,
        extra_counts: list[int],
        extra_token_counts: list[int],
        extra_input_counts: list[int],
        extra_output_counts: list[int],
    ) -> int | None:
        # Record the IO-specific args so the dispatcher's same-tick pending-IO
        # tracking is under test; the routing decision itself (shape-aware math)
        # is covered by test_dp_schedule_shape_aware.py, so reuse min-running here.
        self.shape_aware_calls.append(
            (
                item_input,
                item_output,
                list(extra_input_counts),
                list(extra_output_counts),
            )
        )
        return self.select_min_running_dp(extra_counts, extra_token_counts)

    @staticmethod
    def estimate_req_tokens(req: TokenizedGenerateReqInput) -> int:
        max_new_tokens = req.sampling_params.get("max_new_tokens", 0)
        return len(req.input_ids or []) + max_new_tokens

    @staticmethod
    def estimate_req_io_tokens(req: TokenizedGenerateReqInput) -> tuple[int, int]:
        return len(req.input_ids or []), req.sampling_params.get("max_new_tokens", 0)

    def _eligible(self, extra_counts: list[int]) -> list[int]:
        return [
            dp_rank
            for dp_rank in range(self.dp_size)
            if dp_rank not in self.full_ranks
            and self.running_counts[dp_rank] + extra_counts[dp_rank]
            < self.per_dp_max_running_requests
        ]


def _assign(
    *,
    recv_reqs: list[object] | None,
    pending_dp_reqs: list[TokenizedGenerateReqInput] | None = None,
    dp_size: int = 2,
    policy: str = "min_running_queue",
    harness: _DpAssignmentHarness | None = None,
):
    harness = harness or _DpAssignmentHarness(dp_size=dp_size)
    return assign_dp_ranks(
        recv_reqs=recv_reqs,
        pending_dp_reqs=pending_dp_reqs or [],
        dp_size=dp_size,
        dp_schedule_policy=policy,
        select_round_robin_dp=harness.select_round_robin_dp,
        select_cache_aware_dp=harness.select_cache_aware_dp,
        select_min_running_dp=harness.select_min_running_dp,
        select_shape_aware_dp=harness.select_shape_aware_dp,
        estimate_req_tokens=harness.estimate_req_tokens,
        estimate_req_io_tokens=harness.estimate_req_io_tokens,
    )


def test_min_running_defers_when_all_dp_ranks_are_full():
    harness = _DpAssignmentHarness(
        running_counts=[1, 1],
        full_ranks={0, 1},
        per_dp_max_running_requests=1,
    )
    req = _req("new")

    result = _assign(recv_reqs=[req], harness=harness)

    assert result.ready_reqs == []
    assert result.pending_reqs == [req]
    assert req.dp_rank is None


def test_pending_requests_are_assigned_before_new_arrivals():
    old = _req("old")
    new = _req("new")

    result = _assign(recv_reqs=[new], pending_dp_reqs=[old])

    assert result.ready_reqs == [old, new]
    assert result.pending_reqs == []
    assert [old.dp_rank, new.dp_rank] == [0, 1]


def test_existing_valid_dp_rank_counts_toward_following_assignment():
    sticky = _req("sticky", dp_rank=0)
    new = _req("new")

    result = _assign(recv_reqs=[sticky, new])

    assert result.ready_reqs == [sticky, new]
    assert result.pending_reqs == []
    assert sticky.dp_rank == 0
    assert new.dp_rank == 1


def test_round_robin_preserves_current_no_defer_behavior_for_full_ranks():
    harness = _DpAssignmentHarness(
        running_counts=[1, 1],
        full_ranks={0, 1},
        per_dp_max_running_requests=1,
    )
    req = _req("new")

    result = _assign(recv_reqs=[req], policy="round_robin", harness=harness)

    assert result.ready_reqs == [req]
    assert result.pending_reqs == []
    assert req.dp_rank == 0


def test_invalid_existing_dp_rank_is_reassigned_by_policy(caplog):
    req = _req("bad-rank", dp_rank=-1)

    result = _assign(recv_reqs=[req])

    assert result.ready_reqs == [req]
    assert result.pending_reqs == []
    assert req.dp_rank == 0
    assert "invalid dp_rank" in caplog.text


def test_cache_aware_receives_pending_load_from_prior_requests():
    harness = _DpAssignmentHarness()
    sticky = _req("sticky", dp_rank=0)
    new = _req("new")

    result = _assign(recv_reqs=[sticky, new], policy="cache_aware", harness=harness)

    assert result.ready_reqs == [sticky, new]
    assert result.pending_reqs == []
    assert new.dp_rank == 1
    assert harness.cache_aware_calls == [("new", [1, 0], [8, 0], 4, 4, [4, 0], [4, 0])]


def test_cache_aware_receives_pending_io_split_from_prior_same_tick_assignment():
    # Guards cache_aware's shape-aware miss fallback: a request assigned earlier in
    # the same intake tick must affect the input/output split seen by the next
    # request, not only the aggregate pending token count.
    harness = _DpAssignmentHarness()
    first = _req("first", input_len=10, max_new_tokens=3)  # assigned to rank 0
    second = _req("second", input_len=5, max_new_tokens=7)

    result = _assign(recv_reqs=[first, second], policy="cache_aware", harness=harness)

    assert result.ready_reqs == [first, second]
    assert result.pending_reqs == []
    assert [first.dp_rank, second.dp_rank] == [0, 1]
    assert harness.cache_aware_calls == [
        ("first", [0, 0], [0, 0], 10, 3, [0, 0], [0, 0]),
        ("second", [1, 0], [13, 0], 5, 7, [10, 0], [3, 0]),
    ]


def test_shape_aware_receives_pending_io_split_from_prior_requests():
    # Guards the load-bearing same-tick pending-IO tracking: a request assigned
    # earlier in this intake tick (the sticky request on rank 0) must be reflected
    # in the input/output split passed to select_shape_aware_dp for the next
    # request -- distinct input/output values here catch a swapped or dropped axis.
    harness = _DpAssignmentHarness()
    sticky = _req("sticky", dp_rank=0, input_len=10, max_new_tokens=3)  # io = (10, 3)
    new = _req("new", input_len=5, max_new_tokens=7)  # item io = (5, 7)

    result = _assign(recv_reqs=[sticky, new], policy="shape_aware", harness=harness)

    assert result.ready_reqs == [sticky, new]
    assert result.pending_reqs == []
    # select_shape_aware_dp was called once (for `new`), seeing the incoming item's
    # (input, output) = (5, 7) and the sticky request's IO accumulated on rank 0:
    # per-rank pending input = [10, 0], pending output = [3, 0].
    assert harness.shape_aware_calls == [(5, 7, [10, 0], [3, 0])]


def test_dp_size_one_assigns_generate_reqs_and_preserves_control_reqs():
    control_req = object()
    req = _req("new", dp_rank=7)

    result = _assign(recv_reqs=[control_req, req], dp_size=1)

    assert result.ready_reqs == [control_req, req]
    assert result.pending_reqs == []
    assert req.dp_rank == 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
