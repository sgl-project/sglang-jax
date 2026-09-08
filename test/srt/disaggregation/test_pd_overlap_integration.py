"""Shared scheduler/worker integration contracts for opt-in Raiden overlap."""

from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sgl_jax.srt.managers import scheduler as scheduler_module
from sgl_jax.srt.managers import tp_worker_overlap_thread as worker_module


@pytest.mark.parametrize("mode", ["prefill", "decode", "null"])
@pytest.mark.parametrize("overlap", [False, True])
def test_dispatch_selects_role_specific_loop(mode, overlap):
    scheduler = Mock(enable_overlap=overlap, pd=None)
    scheduler_module.dispatch_scheduler_event_loop(
        scheduler, SimpleNamespace(disaggregation_mode=mode)
    )
    suffix = f"_disagg_{mode}" if mode != "null" else ""
    selected = f"event_loop_{'overlap' if overlap else 'normal'}{suffix}"
    assert scheduler.method_calls == [(selected, (), {})]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pause_drains_role_before_retracting_running_requests(role):
    events = []
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.server_args = SimpleNamespace(disaggregation_mode=role)
    scheduler.enable_overlap = True
    scheduler._sync_chunked_req_owners = lambda: events.append("sync")
    scheduler._drain_disagg_prefill_overlap_results = lambda: events.append("drain_prefill")
    scheduler._drain_disagg_decode_overlap_results = lambda: events.append("drain_decode")
    scheduler._process_pending_chunked_aborts = list
    scheduler._retire_chunked_req_batch_owners = lambda _: None
    scheduler._retract_parked_chunked_reqs = lambda _: None
    scheduler._add_request_to_queue = lambda _: events.append("requeue")
    req = object()

    def retract(_):
        events.append("retract")
        return [req]

    scheduler.running_batch = SimpleNamespace(
        filter_batch=lambda: None,
        reqs_info=[SimpleNamespace(reqs=[req])],
        retract_all=retract,
    )
    scheduler.pause_generation(SimpleNamespace(mode="retract"))
    assert events == ["sync", f"drain_{role}", "retract", "requeue"]
    assert scheduler._engine_paused
    assert scheduler.last_batch is scheduler.cur_batch is None


@pytest.mark.parametrize("finished_on_drain", [False, True])
def test_oom_rechecks_after_draining_before_retraction(monkeypatch, finished_on_drain):
    monkeypatch.setattr(scheduler_module, "TEST_RETRACT", False)
    events = []
    state = {"empty": False, "memory": False}
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.enable_overlap = True
    scheduler.server_args = SimpleNamespace(disaggregation_mode="decode")
    scheduler.tree_cache = None
    scheduler.result_queue = deque([object()])
    scheduler.new_token_ratio = 0.5
    scheduler.new_token_ratio_decay = 0.1
    scheduler.min_new_token_ratio = 0.1
    scheduler.dp_size = scheduler.per_dp_max_running_requests = 1

    def drain():
        events.append("drain")
        scheduler.result_queue.clear()
        state["empty"] = finished_on_drain
        state["memory"] = True

    scheduler._drain_disagg_decode_overlap_results = drain
    batch = SimpleNamespace(
        batch_size=lambda: 0 if state["empty"] else 1,
        filter_batch=lambda: events.append("filter"),
        is_empty=lambda: state["empty"],
        check_decode_mem=lambda: state["memory"],
        prepare_for_decode=lambda: events.append("prepare"),
        retract_decode=Mock(side_effect=AssertionError("retired output already freed enough KV")),
        reqs_info=[SimpleNamespace(reqs=[object()])],
    )
    assert scheduler.update_running_batch(batch) is batch
    assert events.count("drain") == 1
    batch.retract_decode.assert_not_called()
    assert ("prepare" in events) is not finished_on_drain


@pytest.mark.parametrize("remaining", [0, 1])
def test_oom_drain_reopens_rank_admission_after_completed_request(monkeypatch, remaining):
    monkeypatch.setattr(scheduler_module, "TEST_RETRACT", False)
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.enable_overlap = True
    scheduler.server_args = SimpleNamespace(disaggregation_mode="decode")
    scheduler.tree_cache = None
    scheduler.result_queue = deque([object()])
    scheduler.new_token_ratio = 0.5
    scheduler.new_token_ratio_decay = scheduler.min_new_token_ratio = 0.1
    scheduler.dp_size = 1
    scheduler.per_dp_max_running_requests = 2
    info = SimpleNamespace(reqs=[object(), object()], batch_is_full=True)
    state = {"completed": False}

    def drain():
        scheduler.result_queue.clear()
        state["completed"] = True

    def filter_batch():
        if state["completed"]:
            info.reqs = info.reqs[:remaining]

    scheduler._drain_disagg_decode_overlap_results = drain
    batch = SimpleNamespace(
        batch_size=lambda: len(info.reqs),
        filter_batch=filter_batch,
        is_empty=lambda: not info.reqs,
        check_decode_mem=lambda: state["completed"],
        prepare_for_decode=lambda: None,
        reqs_info=[info],
    )
    scheduler.update_running_batch(batch)
    assert info.batch_is_full is False


def test_retraction_still_needed_runs_only_after_pending_result_retired(monkeypatch):
    monkeypatch.setattr(scheduler_module, "TEST_RETRACT", False)
    events = []
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.enable_overlap = True
    scheduler.server_args = SimpleNamespace(disaggregation_mode="decode")
    scheduler.tree_cache = None
    scheduler.result_queue = deque([object()])
    scheduler.new_token_ratio = 0.5
    scheduler.dp_size = scheduler.per_dp_max_running_requests = 1
    scheduler._extend_requests_to_queue = lambda *_, **__: events.append("requeue")

    def drain():
        events.append("drain")
        scheduler.result_queue.clear()

    def retract(_):
        assert not scheduler.result_queue
        events.append("retract")
        return [], 0.5, []

    scheduler._drain_disagg_decode_overlap_results = drain
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        filter_batch=lambda: None,
        is_empty=lambda: False,
        check_decode_mem=lambda: False,
        prepare_for_decode=lambda: events.append("prepare"),
        retract_decode=retract,
        reqs_info=[SimpleNamespace(reqs=[object()])],
    )
    scheduler.update_running_batch(batch)
    assert events == ["drain", "retract", "requeue", "prepare"]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_prefill_worker_fences_before_publishing_or_next_forward(monkeypatch, fused, fence_fails):
    events = []
    client = worker_module.ModelWorkerClient.__new__(worker_module.ModelWorkerClient)
    client.input_queue = Queue()
    client.output_queue = SimpleNamespace(put=lambda _: events.append("publish"))
    client.future_token_ids_map = object()
    client.mesh = None
    client.async_gather_fn = lambda x: x
    buffers = object()

    def forward(batch, *_, **__):
        events.append(f"forward{batch.bid}")
        return (None, [42], 0, object()) if fused else (None, [42], 0)

    client.worker = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode="prefill", disaggregation_enable_overlap_schedule=True
        ),
        model_runner=SimpleNamespace(token_to_kv_pool=SimpleNamespace(kv_buffer=buffers)),
        _pd_fuse_for_batch=lambda _: fused,
        forward_batch_generation=forward,
    )
    monkeypatch.setattr(worker_module, "resolve_future_token_ids", lambda ids, *_: ids)
    monkeypatch.setattr(worker_module, "set_future_token_ids", lambda *args: args[0])

    def fence(actual):
        assert actual is buffers
        events.append("fence")
        if fence_fails:
            raise RuntimeError("device fence failed")

    monkeypatch.setattr(worker_module.jax, "block_until_ready", fence)
    for bid in (1, 2):
        batch = SimpleNamespace(
            bid=bid, launch_done=None, forward_batch=SimpleNamespace(input_ids=[0])
        )
        client.input_queue.put((batch, None, None, None))
    client.input_queue.put((None, None, None, None))
    if fence_fails:
        with pytest.raises(RuntimeError, match="device fence failed"):
            client.forward_thread_func_()
        assert events == ["forward1", "fence"]
    else:
        client.forward_thread_func_()
        assert events == ["forward1", "fence", "publish", "forward2", "fence", "publish"]
