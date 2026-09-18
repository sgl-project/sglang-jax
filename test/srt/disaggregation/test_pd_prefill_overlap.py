"""CPU interleaving contracts for Raiden prefill scheduler overlap."""

from collections import deque
from dataclasses import FrozenInstanceError
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.prefill import (
    PendingPrefillResult,
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sgl_jax.srt.managers import tp_worker_overlap_thread as worker_module


class Harness(SchedulerDisaggregationPrefillMixin):
    def __init__(self, chunk=True):
        self.server_args = NS(disaggregation_enable_chunk_prefill_transfer=chunk)
        self.chunked_reqs = []
        self.disagg_prefill_queue = PrefillBootstrapQueue()
        self.disagg_kv_manager = Mock()
        self.disagg_use_d2h_staging = False
        self.token_to_kv_pool_allocator = NS(
            get_kvcache=lambda: NS(page_size=128, kv_buffer="new-pool")
        )
        self._disagg_prefill_compute = {}
        self._disagg_prefill_deferred_releases = {}
        self._extract_req_block_ids_range = Mock(return_value=[7])
        self._pd_mark_time = Mock()
        self.set_next_batch_sampling_info_done = Mock()
        self._release_prefill_kv_pool = Mock()
        self._release_prefill_host_buffer = Mock()
        self._stream_prefill_req = Mock()
        self._retire_chunk_producer_ownership = Mock()


def request():
    return NS(
        rid="request",
        disagg_transfer_id="attempt-1",
        bootstrap_room=3,
        dp_rank=0,
        fill_ids=list(range(128)),
        extend_input_len=128,
        origin_input_ids=list(range(256)),
        is_chunked=1,
        start_send_idx=0,
        disagg_chunk_index=0,
        disagg_chunk_sender=None,
        to_finish=None,
        finished=lambda: False,
    )


def test_snapshot_survives_next_chunk_mutating_req_and_page_mapping():
    scheduler = Harness()
    req = request()
    scheduler.chunked_reqs = [req]
    batch = NS(reqs=[req])
    snapshots = scheduler._snapshot_prefill_handoffs(batch)
    req.fill_ids.extend(range(128))
    req.extend_input_len = 64
    scheduler.chunked_reqs = [None]
    scheduler._extract_req_block_ids_range.return_value = [99]
    scheduler._raiden_handoff_chunk = Mock()

    scheduler.process_prefill_chunk(batch, NS(), handoffs=snapshots)

    snapshot = snapshots[0]
    assert (snapshot.start, snapshot.end, snapshot.block_ids, snapshot.is_final) == (
        0,
        128,
        (7,),
        False,
    )
    scheduler._raiden_handoff_chunk.assert_called_once_with(req, is_final=False, snapshot=snapshot)
    scheduler.disagg_kv_manager.prepare_prefill_batch.assert_not_called()
    with pytest.raises(FrozenInstanceError):
        snapshot.end = 256


def test_snapshot_chunk_publishes_old_pages_without_waiting_new_pool():
    scheduler = Harness()
    req = request()
    scheduler.chunked_reqs = [req]
    snapshot = scheduler._snapshot_prefill_handoffs(NS(reqs=[req]))[0]
    req.fill_ids.extend(range(128))
    scheduler._extract_req_block_ids_range.reset_mock()
    sender = Mock(has_pending_failure=False, has_started_chunks=False)
    scheduler.disagg_kv_manager.create_sender.return_value = sender

    scheduler._raiden_handoff_chunk(req, is_final=False, snapshot=snapshot)

    assert sender.send_chunk.call_args.args == (0, [7])
    assert sender.send_chunk.call_args.kwargs["on_ready"] is None
    assert req.start_send_idx == 128
    scheduler._extract_req_block_ids_range.assert_not_called()
    scheduler.disagg_kv_manager.prepare_prefill_batch.assert_not_called()


def test_changed_attempt_cannot_publish_old_snapshot():
    scheduler = Harness()
    req = request()
    snapshot = scheduler._snapshot_prefill_handoffs(NS(reqs=[req]))
    req.disagg_transfer_id = "attempt-2"
    scheduler._raiden_handoff_chunk = Mock()
    with pytest.raises(RuntimeError, match="identity changed"):
        scheduler.process_prefill_chunk(NS(reqs=[req]), NS(), handoffs=snapshot)
    scheduler._raiden_handoff_chunk.assert_not_called()


@pytest.mark.parametrize("state", [KVPoll.SUCCESS, KVPoll.FAILED])
def test_transfer_terminal_waits_for_all_compute_owners(state):
    scheduler = Harness()
    req = request()
    sender = Mock()
    sender.poll.return_value = state
    terminal = Mock()
    scheduler.disagg_prefill_queue.add(req.rid, sender, req=req, on_terminal=terminal)
    scheduler._disagg_prefill_compute[id(req)] = 2
    scheduler.send_kv_chunk()
    scheduler._disagg_prefill_compute[id(req)] = 1
    scheduler.send_kv_chunk()
    terminal.assert_not_called()
    assert len(scheduler.disagg_prefill_queue) == 1
    scheduler._disagg_prefill_compute.clear()
    scheduler.send_kv_chunk()
    scheduler.send_kv_chunk()
    terminal.assert_called_once()
    assert len(scheduler.disagg_prefill_queue) == 0


def test_failed_handoff_processing_still_retires_compute_owner():
    scheduler = Harness()
    req = request()
    snapshots = scheduler._snapshot_prefill_handoffs(NS(reqs=[req]))
    scheduler._disagg_prefill_compute[id(req)] = 1
    scheduler._release_prefill_req_resources(req)
    scheduler.tp_worker = NS(resolve_last_batch_result=lambda: (None, [1], 0))
    scheduler.process_prefill_chunk = Mock(side_effect=RuntimeError("handoff failed"))
    with pytest.raises(RuntimeError, match="handoff failed"):
        scheduler._resolve_disagg_prefill_result(PendingPrefillResult(NS(), NS(), snapshots))
    scheduler._release_prefill_kv_pool.assert_called_once_with(req)
    assert not scheduler._disagg_prefill_compute


def test_pause_drain_processes_all_results_before_reaping():
    scheduler = Harness()
    first, second = object(), object()
    scheduler.result_queue = deque([first, second])
    events = []
    scheduler._resolve_disagg_prefill_result = events.append
    scheduler.send_kv_chunk = lambda: events.append("reap")
    scheduler.last_batch = scheduler.cur_batch = object()
    scheduler._drain_disagg_prefill_overlap_results()
    assert events == [first, second, "reap"]
    assert scheduler.last_batch is scheduler.cur_batch is None
    assert not scheduler.result_queue


def test_cancelled_snapshot_aborts_existing_sender_without_republishing():
    scheduler = Harness()
    req = request()
    scheduler.chunked_reqs = [req]
    snapshots = scheduler._snapshot_prefill_handoffs(NS(reqs=[req]))
    req.to_finish = object()
    req.disagg_chunk_sender = Mock()
    scheduler._raiden_handoff_chunk = Mock()
    scheduler._disagg_prefill_compute[id(req)] = 1
    scheduler.process_prefill_chunk(NS(reqs=[req]), NS(), handoffs=snapshots)
    scheduler._raiden_handoff_chunk.assert_not_called()
    req.disagg_chunk_sender.abort.assert_called_once()
    assert req.rid in scheduler.disagg_prefill_queue._entries
    scheduler._release_prefill_kv_pool.assert_not_called()
    scheduler._retire_chunk_producer_ownership.assert_called_once_with(req)


@pytest.mark.parametrize("overlap", [False, True])
def test_whole_request_handoff_uses_snapshot_pages_or_normal_barrier(overlap):
    scheduler = Harness(chunk=False)
    req = request()
    req.fill_ids = list(req.origin_input_ids)
    req.extend_input_len = 256
    batch = NS(reqs=[req])
    snapshots = scheduler._snapshot_prefill_handoffs(batch) if overlap else None
    scheduler._extract_req_block_ids = Mock(return_value=[88])
    scheduler.disagg_kv_manager.start_prefill.return_value = NS(
        sender=Mock(), release_device_kv=False
    )
    scheduler.process_prefill_chunk(batch, NS(), handoffs=snapshots)
    context = scheduler.disagg_kv_manager.start_prefill.call_args.args[0]
    assert context.block_ids_factory() == ([7] if overlap else [88])
    if overlap:
        scheduler.disagg_kv_manager.prepare_prefill_batch.assert_not_called()
    else:
        scheduler.disagg_kv_manager.prepare_prefill_batch.assert_called_once_with("new-pool")


def test_backpressured_loop_keeps_resolving_and_polling_then_resumes():
    scheduler = Harness()
    req = request()
    sender = NS(has_pending_chunks=False)
    req.disagg_chunk_sender = sender
    scheduler.chunked_reqs = [req]
    scheduler._comm_backend = None
    scheduler._engine_paused = False
    scheduler.init_new_token_ratio = 0.5
    scheduler._wait_donation_safe = Mock()
    scheduler.select_dp_for_request = lambda values: values
    scheduler.process_input_requests = Mock()
    scheduler.tp_worker = NS(cur_sampling_info=None)
    events = []
    tick = 0
    polls = 0

    def receive():
        nonlocal tick
        tick += 1
        if tick == 5:
            raise StopIteration
        if tick == 2:
            sender.has_pending_chunks = True
        return []

    def poll():
        nonlocal polls
        polls += 1
        events.append(("poll", tick))
        # Native reads finish while CPU scheduling is stopped, and polling
        # drains pending registrations before the next admission check.
        if tick == 3 and polls == 6:
            sender.has_pending_chunks = False

    batch = NS(reqs=[req], copy=lambda: NS(reqs=[req]))

    def schedule():
        events.append(("schedule", tick))
        return batch if tick == 1 else None

    scheduler.recv_requests = receive
    scheduler.send_kv_chunk = poll
    scheduler.get_next_batch_to_run = schedule
    scheduler.run_batch = lambda batch: NS()
    scheduler._resolve_disagg_prefill_result = lambda result: events.append(("resolve", tick))
    with pytest.raises(StopIteration):
        scheduler.event_loop_overlap_disagg_prefill()
    assert [event for event in events if event[0] == "schedule"] == [
        ("schedule", 1),
        ("schedule", 4),
    ]
    assert [event for event in events if event[0] == "resolve"] == [("resolve", 2)]
    assert [event for event in events if event[0] == "poll"] == [
        ("poll", tick) for tick in range(1, 5) for _ in range(2)
    ]
    assert not scheduler.result_queue


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_prefill_worker_fences_before_publishing_or_next_forward(monkeypatch, fused, fence_fails):
    events = []
    client = worker_module.ModelWorkerClient.__new__(worker_module.ModelWorkerClient)
    client.input_queue = Queue()
    client.output_queue = NS(put=lambda _: events.append("publish"))
    client.future_token_ids_map = object()
    client.mesh = None
    client.async_gather_fn = lambda x: x
    buffers = object()

    def forward(batch, *_, **__):
        client.worker.model_runner.token_to_kv_pool.kv_buffer = (buffers, batch.bid)
        events.append(f"forward{batch.bid}")
        return (None, [42], 0, object()) if fused else (None, [42], 0)

    client.worker = NS(
        server_args=NS(disaggregation_mode="prefill", disaggregation_enable_overlap_schedule=True),
        model_runner=NS(token_to_kv_pool=NS(kv_buffer=buffers)),
        _pd_fuse_for_batch=lambda _: fused,
        forward_batch_generation=forward,
    )
    monkeypatch.setattr(worker_module, "resolve_future_token_ids", lambda ids, *_: ids)
    monkeypatch.setattr(worker_module, "set_future_token_ids", lambda *args: args[0])

    def fence(actual):
        assert actual == (buffers, 1 if "publish" not in events else 2)
        events.append("fence")
        if fence_fails:
            raise RuntimeError("device fence failed")

    monkeypatch.setattr(worker_module.jax, "block_until_ready", fence)
    for bid in (1, 2):
        batch = NS(bid=bid, launch_done=None, forward_batch=NS(input_ids=[0]))
        client.input_queue.put((batch, None, None, None))
    client.input_queue.put((None, None, None, None))
    if fence_fails:
        with pytest.raises(RuntimeError, match="device fence failed"):
            client.forward_thread_func_()
        assert events == ["forward1", "fence"]
    else:
        client.forward_thread_func_()
        assert events == ["forward1", "fence", "publish", "forward2", "fence", "publish"]
