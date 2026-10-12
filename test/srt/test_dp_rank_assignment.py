"""Per-round routing updates load atomically while preserving intake order."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sgl_jax.srt.managers.dp_load import DpLoadSnapshot, DpRouter
from sgl_jax.srt.managers.dp_rank_assignment import assign_dp_ranks
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.managers.scheduler import Scheduler


def _req(rid="req", rank=None):
    return TokenizedGenerateReqInput(
        rid=rid, input_ids=[1, 2, 3, 4], sampling_params={"max_new_tokens": 4}, dp_rank=rank
    )


def _router(policy="min_running_queue", dp_size=2, start=0):
    snapshot = DpLoadSnapshot((0,) * dp_size, (0,) * dp_size, (0,) * dp_size, (256,) * dp_size)
    return DpRouter(snapshot, policy, Mock(return_value=(4, 4)), Mock(return_value=0), start)


@pytest.mark.parametrize(
    "policy", ["min_running_queue", "shape_aware", "cache_aware", "force_cache_aware"]
)
def test_assign_updates_load_once_and_preserves_snapshot(policy):
    snapshot = DpLoadSnapshot((0, 0), (0, 0), (0, 0), (256, 256))
    estimate = Mock(return_value=(4, 4))
    router = DpRouter(snapshot, policy, estimate, Mock(return_value=0))
    requests = [_req(str(i)) for i in range(4)]
    assert [router.assign(req) for req in requests] == [0, 1, 0, 1]
    assert [req.dp_rank for req in requests] == [0, 1, 0, 1]
    assert router.request_counts == [2, 2]
    assert router.input_tokens == router.output_tokens == [8, 8]
    assert estimate.call_count == 4
    assert snapshot == DpLoadSnapshot((0, 0), (0, 0), (0, 0), (256, 256))


@pytest.mark.parametrize(
    "policy", ["min_running_queue", "shape_aware", "cache_aware", "force_cache_aware"]
)
def test_sticky_request_contributes_to_next_choice(policy):
    router = _router(policy)
    assert router.assign(_req(rank=0)) == 0
    assert router.assign(_req("next")) == 1


def test_invalid_rank_is_reassigned(caplog):
    router = _router()
    req = _req(rank=99)
    assert router.assign(req) == 0
    assert "Ignoring invalid dp_rank=99" in caplog.text


def test_round_robin_continues_across_rounds_and_skips_sticky_requests():
    router = _router("round_robin", start=1)
    assert router.assign(_req(rank=0)) == 0
    assert router.assign(_req()) == 1
    next_round = _router("round_robin", start=router.round_robin_counter)
    assert next_round.assign(_req()) == 0
    router._estimate_io.assert_not_called()
    next_round._estimate_io.assert_not_called()


def test_single_dp_overrides_explicit_rank_without_estimation():
    router = _router(dp_size=1)
    assert router.assign(_req(rank=99)) == 0
    router._estimate_io.assert_not_called()


def test_pending_first_and_control_messages_pass_through():
    router = _router()
    pending, new, control = _req("pending"), _req("new"), object()
    result = assign_dp_ranks(
        recv_reqs=[control, new], pending_dp_reqs=[pending], assign=router.assign
    )
    assert result.ready_reqs == [pending, control, new]
    assert result.pending_reqs == []
    assert [pending.dp_rank, new.dp_rank] == [0, 1]


def test_deferred_selection_does_not_change_load():
    router = _router("cache_aware")
    router._pick_cache_aware = Mock(return_value=None)
    req = _req()
    result = assign_dp_ranks(recv_reqs=None, pending_dp_reqs=[req], assign=router.assign)
    assert result.pending_reqs == [req]
    assert result.ready_reqs == []
    assert req.dp_rank is None
    assert router.request_counts == router.input_tokens == router.output_tokens == [0, 0]


def test_scheduler_collects_once_per_round_and_keeps_round_robin_cursor():
    scheduler = object.__new__(Scheduler)
    scheduler.dp_size = 2
    scheduler.per_dp_max_running_requests = 256
    scheduler.dp_schedule_policy = "min_running_queue"
    scheduler.dp_round_robin_counter = 0
    scheduler.pending_dp_reqs = []
    scheduler._collect_dp_load = Mock(
        return_value=DpLoadSnapshot((0, 0), (0, 0), (0, 0), (256, 256))
    )
    scheduler._estimate_req_input_output_tokens = Mock(return_value=(4, 4))
    scheduler._lookup_prefix_length = Mock(return_value=0)
    requests = [_req(str(i)) for i in range(4)]
    scheduler.select_dp_for_request(requests)
    assert scheduler._collect_dp_load.call_count == 1
    assert [req.dp_rank for req in requests] == [0, 1, 0, 1]
    scheduler._collect_dp_load.return_value = DpLoadSnapshot((4, 0), (16, 0), (16, 0), (256, 256))
    req = _req()
    scheduler.select_dp_for_request([req])
    assert req.dp_rank == 1
    assert scheduler._collect_dp_load.call_count == 2
    scheduler.dp_schedule_policy = "round_robin"
    scheduler.select_dp_for_request([_req()])
    assert scheduler.dp_round_robin_counter == 1
    req = _req()
    scheduler.select_dp_for_request([req])
    assert req.dp_rank == 1
    scheduler.select_dp_for_request([])
    assert scheduler._collect_dp_load.call_count == 2


def test_chunk_alias_and_parked_chunk_have_same_snapshot():
    scheduler = object.__new__(Scheduler)
    scheduler.dp_size = 2
    scheduler.per_dp_max_running_requests = 256
    req = SimpleNamespace(dp_rank=0, finished=lambda: False)

    def batch(reqs, chunk=None, extend=False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            reqs_info=[
                SimpleNamespace(reqs=reqs, chunked_req=chunk, batch_is_full=False),
                SimpleNamespace(reqs=[], chunked_req=None, batch_is_full=False),
            ],
        )

    scheduler.running_batch = batch([])
    scheduler.last_batch = batch([req], req, True)
    scheduler.chunked_reqs = [req, None]
    scheduler.waiting_queue = scheduler.grammar_queue = []
    scheduler._estimate_req_input_output_tokens = lambda req: (3, 8)
    expected = DpLoadSnapshot((1, 0), (3, 0), (8, 0), (256, 256))
    assert scheduler._collect_dp_load() == expected
    scheduler.last_batch = batch([])
    assert scheduler._collect_dp_load() == expected
    req.finished = lambda: True
    assert scheduler._collect_dp_load().request_counts == (0, 0)


@pytest.mark.parametrize(
    "policy", ["min_running_queue", "shape_aware", "cache_aware", "force_cache_aware"]
)
def test_full_rank_is_skipped_and_request_cap_updates_within_round(policy):
    estimate = Mock(return_value=(4, 4))
    router = DpRouter(
        DpLoadSnapshot((0, 0), (0, 0), (0, 0), (0, 2)),
        policy,
        estimate,
        Mock(return_value=0),
    )
    requests = [_req(str(i)) for i in range(3)]
    result = assign_dp_ranks(recv_reqs=requests, pending_dp_reqs=[], assign=router.assign)
    assert [r.dp_rank for r in result.ready_reqs] == [1, 1]
    assert result.pending_reqs == [requests[2]]
    assert requests[2].dp_rank is None
    assert router.request_counts == [0, 2]
    assert estimate.call_count == 2


@pytest.mark.parametrize(
    "policy", ["min_running_queue", "shape_aware", "cache_aware", "force_cache_aware"]
)
def test_pending_retries_on_next_round_with_room(policy):
    full = DpRouter(
        DpLoadSnapshot((3, 3), (12, 12), (12, 12), (3, 3)),
        policy,
        Mock(side_effect=AssertionError("full ranks need no estimate")),
        Mock(side_effect=AssertionError("full ranks need no cache probe")),
    )
    req = _req()
    first = assign_dp_ranks(recv_reqs=[req], pending_dp_reqs=[], assign=full.assign)
    assert first.pending_reqs == [req]
    assert full.request_counts == [3, 3]
    available = DpRouter(
        DpLoadSnapshot((3, 2), (12, 8), (12, 8), (3, 3)),
        policy,
        Mock(return_value=(4, 4)),
        Mock(return_value=0),
    )
    second = assign_dp_ranks(
        recv_reqs=None, pending_dp_reqs=first.pending_reqs, assign=available.assign
    )
    assert second.ready_reqs == [req]
    assert second.pending_reqs == []
    assert req.dp_rank == 1


def test_force_cache_aware_defers_for_full_best_holder():
    lookup = Mock(side_effect=lambda tokens, key, rank: 3 if rank == 0 else 1)
    router = DpRouter(
        DpLoadSnapshot((0, 0), (0, 0), (0, 0), (0, 256)),
        "force_cache_aware",
        Mock(return_value=(4, 4)),
        lookup,
    )
    assert router.assign(_req()) is None
    assert [call.args[2] for call in lookup.call_args_list] == [0, 1]
    assert router.request_counts == [0, 0]


def test_cache_aware_only_probes_available_ranks():
    lookup = Mock(return_value=3)
    router = DpRouter(
        DpLoadSnapshot((0, 0), (0, 0), (0, 0), (0, 256)),
        "cache_aware",
        Mock(return_value=(4, 4)),
        lookup,
    )
    assert router.assign(_req()) == 1
    assert [call.args[2] for call in lookup.call_args_list] == [1]


@pytest.mark.parametrize("policy", ["min_running_queue", "round_robin"])
def test_explicit_rank_keeps_existing_full_rank_bypass(policy):
    router = DpRouter(
        DpLoadSnapshot((0, 0), (0, 0), (0, 0), (0, 0)),
        policy,
        Mock(return_value=(4, 4)),
        Mock(return_value=0),
    )
    assert router.assign(_req(rank=1)) == 1


def test_scheduler_snapshots_full_flag_and_request_cap():
    scheduler = object.__new__(Scheduler)
    scheduler.dp_size = 2
    scheduler.per_dp_max_running_requests = 8
    scheduler.running_batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(batch_is_full=True), SimpleNamespace(batch_is_full=False)]
    )
    scheduler._iter_dp_requests = lambda: iter(())
    scheduler._estimate_req_input_output_tokens = Mock()
    assert scheduler._collect_dp_load().request_limits == (0, 8)
    scheduler.running_batch.reqs_info[0].batch_is_full = False
    assert scheduler._collect_dp_load().request_limits == (8, 8)
