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
        if tick == 3 and polls == 3:
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
        ("schedule", 3),
        ("schedule", 4),
    ]
    assert [event for event in events if event[0] == "resolve"] == [("resolve", 2)]
    assert [event for event in events if event[0] == "poll"] == [
        ("poll", tick) for tick in range(1, 5)
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
        batch = NS(
            bid=bid,
            launch_done=None,
            forward_batch=NS(input_ids=[0], seq_lens=[1], req_pool_indices=[0]),
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


def owner_batch(req):
    return NS(
        reqs_info=[NS(reqs=[req], chunked_req=req)],
        forward_mode=NS(is_extend=lambda: True),
    )


def lifecycle_harness(req, *, chunk=True):
    from sgl_jax.srt.managers.scheduler import Scheduler

    scheduler = Harness(chunk=chunk)
    del scheduler._retire_chunk_producer_ownership
    scheduler.chunked_reqs = [req] if chunk else [None]
    scheduler._pending_chunked_abort_reqs = [None]
    scheduler.last_batch = None
    scheduler.cur_batch = owner_batch(req)
    scheduler.pd = "raiden"
    scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler.grammar_queue = []
    scheduler.running_batch = NS(reqs_info=[])
    scheduler.disagg_prealloc_queue = None
    scheduler.disagg_transfer_queue = None
    scheduler._sync_chunked_req_owners = lambda: Scheduler._sync_chunked_req_owners(scheduler)
    scheduler._mark_pending_chunked_aborts = lambda abort: Scheduler._mark_pending_chunked_aborts(
        scheduler, abort
    )
    scheduler.tp_worker = NS(resolve_last_batch_result=lambda: (None, [1], 0))
    req.pd_time_stats = None
    req.finished_reason = None
    req.finished = lambda: req.finished_reason is not None
    req.check_finished = lambda: setattr(req, "finished_reason", req.to_finish)
    return scheduler


@pytest.mark.parametrize("chunk", [False, True])
def test_abort_in_flight_resolves_before_terminal_and_releases_once(chunk):
    from sgl_jax.srt.managers.io_struct import AbortReq
    from sgl_jax.srt.managers.scheduler import Scheduler

    req = request()
    scheduler = lifecycle_harness(req, chunk=chunk)
    snapshots = scheduler._snapshot_prefill_handoffs(scheduler.cur_batch)
    scheduler._disagg_prefill_compute[id(req)] = 1
    sender = Mock(has_pending_failure=False)
    sender.poll.return_value = KVPoll.FAILED
    if chunk:
        req.disagg_chunk_sender = sender
        scheduler._ensure_chunk_sender_queued(req, sender)
    Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
    assert req.to_finish is not None
    scheduler.send_kv_chunk()
    scheduler._release_prefill_kv_pool.assert_not_called()
    scheduler._resolve_disagg_prefill_result(
        PendingPrefillResult(scheduler.cur_batch, NS(), snapshots)
    )
    if chunk:
        scheduler._release_prefill_kv_pool.assert_not_called()
    scheduler.send_kv_chunk()
    scheduler.send_kv_chunk()
    scheduler._stream_prefill_req.assert_called_once_with(req)
    scheduler._release_prefill_kv_pool.assert_called_once_with(req)
    assert not scheduler._disagg_prefill_compute
    assert not scheduler._disagg_prefill_deferred_releases
    assert scheduler.chunked_reqs == [None]
    # Simulate end-of-tick publication and a subsequent AbortReq owner sync.
    scheduler.last_batch = scheduler.cur_batch
    scheduler._sync_chunked_req_owners()
    assert scheduler.chunked_reqs == [None]


@pytest.mark.parametrize("existing_sender", [False, True])
def test_snapshot_error_is_per_request_and_waits_for_compute_and_native_readers(existing_sender):
    req = request()
    scheduler = lifecycle_harness(req)
    good = request()
    good.rid = "good"
    scheduler._extract_req_block_ids_range.side_effect = [ValueError("unaligned token"), [8]]
    snapshots = scheduler._snapshot_prefill_handoffs(NS(reqs=[req, good]))
    assert snapshots[0].error == "unaligned token"
    assert snapshots[1].block_ids == (8,)
    assert snapshots[1].error is None
    scheduler._release_prefill_kv_pool.assert_not_called()
    scheduler._stream_prefill_req.assert_not_called()
    scheduler._disagg_prefill_compute[id(req)] = 1
    sender = Mock()
    sender.poll.return_value = KVPoll.TRANSFERRING
    if existing_sender:
        req.disagg_chunk_sender = sender
    scheduler._resolve_disagg_prefill_result(
        PendingPrefillResult(scheduler.cur_batch, NS(), snapshots[:1])
    )
    if existing_sender:
        sender.fail.assert_called_once_with(reason="chunk_handoff")
        scheduler.send_kv_chunk()
        scheduler._release_prefill_kv_pool.assert_not_called()
        sender.poll.return_value = KVPoll.FAILED
        scheduler.send_kv_chunk()
    scheduler._stream_prefill_req.assert_called_once_with(req)
    scheduler._release_prefill_kv_pool.assert_called_once_with(req)
    assert scheduler.chunked_reqs == [None]
    assert scheduler.cur_batch.reqs_info[0].chunked_req is None


def test_failed_sender_during_later_chunk_cannot_restore_producer():
    req = request()
    scheduler = lifecycle_harness(req)
    snapshots = scheduler._snapshot_prefill_handoffs(scheduler.cur_batch)
    sender = Mock(has_pending_failure=True)
    sender.poll.return_value = KVPoll.FAILED
    req.disagg_chunk_sender = sender
    scheduler._ensure_chunk_sender_queued(req, sender)
    scheduler._disagg_prefill_compute[id(req)] = 1
    scheduler.send_kv_chunk()
    scheduler._release_prefill_kv_pool.assert_not_called()
    scheduler._resolve_disagg_prefill_result(
        PendingPrefillResult(scheduler.cur_batch, NS(), snapshots)
    )
    sender.send_chunk.assert_not_called()
    scheduler.last_batch = scheduler.cur_batch
    scheduler._sync_chunked_req_owners()
    assert scheduler.chunked_reqs == [None]
    scheduler.send_kv_chunk()
    scheduler.send_kv_chunk()
    scheduler._release_prefill_kv_pool.assert_called_once_with(req)
    scheduler._stream_prefill_req.assert_called_once_with(req)


def test_run_batch_to_resolve_to_native_chunk_handoff():
    import numpy as np

    from sgl_jax.srt.managers.scheduler import Scheduler

    req = request()
    scheduler = lifecycle_harness(req)
    scheduler.pd = None
    scheduler.forward_ct = 0
    scheduler.is_generation = scheduler.enable_overlap = True
    scheduler.spec_algorithm = None
    scheduler.page_size = 128
    scheduler.server_args.enable_static_lora = False
    scheduler._profile_batch_predicate = lambda _: None
    scheduler._extract_dp_output_ids = lambda *_: None
    events = []
    sender = Mock(has_pending_failure=False)
    sender.send_chunk.side_effect = lambda *a, **kw: events.append("send_chunk")
    sender.poll.return_value = KVPoll.TRANSFERRING
    scheduler.disagg_kv_manager.create_sender.return_value = sender

    class Worker:
        def get_precompile_paddings(self):
            return [128], [1], [128]

        def forward_batch_generation(self, batch, **kwargs):
            events.append("launch")
            return None, np.array([-1]), 0

        def resolve_last_batch_result(self):
            events.append("resolve")
            return None, [42], 0

    scheduler.tp_worker = Worker()
    batch = scheduler.cur_batch
    batch.return_logprob = False
    batch.get_model_worker_batch = lambda *_: NS(bid=1)
    snapshots = scheduler._snapshot_prefill_handoffs(batch)
    scheduler._disagg_prefill_compute[id(req)] = 1
    result = Scheduler.run_batch(scheduler, batch)
    assert events == ["launch"]
    scheduler._resolve_disagg_prefill_result(PendingPrefillResult(batch, result, snapshots))
    assert events == ["launch", "resolve", "send_chunk"]
    assert sender.send_chunk.call_args.args == (0, [7])
    assert sender.send_chunk.call_args.kwargs["on_ready"] is None
    assert req.start_send_idx == 128
    scheduler._release_prefill_kv_pool.assert_not_called()
    sender.poll.return_value = KVPoll.FAILED
    scheduler.send_kv_chunk()
    scheduler._release_prefill_kv_pool.assert_called_once_with(req)
