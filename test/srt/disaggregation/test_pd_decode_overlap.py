"""CPU orchestration tests; native Raiden/device ordering needs TPU validation."""

import threading
from collections import deque
from copy import copy
from types import SimpleNamespace

import pytest

from sgl_jax.srt.disaggregation import decode
from sgl_jax.srt.managers import scheduler as scheduler_module


class _StopLoop(Exception):
    pass


class _Batch(SimpleNamespace):
    def copy(self):
        return copy(self)


class _Scheduler(decode.SchedulerDisaggregationDecodeMixin):
    def __init__(self, batches, controls=()):
        self.events = []
        self.batches = iter(batches)
        self.controls = iter(controls)
        self._comm_backend = None
        self._engine_paused = False
        self.last_batch = self.cur_batch = None
        self.result_queue = deque()
        self.disagg_prealloc_queue = []
        self.disagg_decode_watchdog = SimpleNamespace(start=lambda: None, beat=lambda _: None)
        self.dp_size = 1
        self.req_to_token_pool = self.tree_cache = self.model_config = None
        self.spec_algorithm = self.mesh = None
        self.enable_overlap = True
        self.init_new_token_ratio = 0.5
        self.new_token_ratio = 0.1
        self.kv_buffers = object()
        self.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(kv_buffer=self.kv_buffers)
        )
        self.owner = SimpleNamespace(cur_sampling_info=None)

    def recv_requests(self):
        return next(self.controls, None)

    def select_dp_for_request(self, reqs):
        return reqs

    def process_input_requests_disagg_decode(self, control):
        if control:
            control(self)

    def _wait_donation_safe(self):
        self.events.append("donation")

    def process_decode_queue(self):
        self.events.append("transfer")

    def _drain_decode_transfer_terminals(self):
        self.events.append("terminal")

    def get_next_batch_to_run(self):
        try:
            return next(self.batches)
        except StopIteration:
            raise _StopLoop from None

    def run_batch(self, batch):
        self.events.append(("launch", batch.name))
        batch.launch_done.set()
        self.owner.cur_sampling_info = SimpleNamespace(sampling_info_done=threading.Event())
        batch.sampling = self.owner.cur_sampling_info
        return batch.name

    def _current_sampling_info_owner(self):
        return self.owner

    def process_batch_result(self, batch, result, launch_done=None):
        if result is None:
            assert batch.forward_mode is decode.ForwardMode.DUMMY_FIRST
            self.events.append("dummy")
        else:
            assert batch.sampling.sampling_info_done.is_set()
            self.events.append(("result", result))
        if launch_done is not None:
            assert launch_done.is_set()
        if batch.next_batch_sampling_info is not None:
            batch.next_batch_sampling_info.sampling_info_done.set()


@pytest.fixture(autouse=True)
def _cpu_fakes(monkeypatch):
    monkeypatch.setattr(decode.ScheduleBatch, "init_new", lambda **_: _Batch())


def _run(scheduler):
    with pytest.raises(_StopLoop):
        scheduler.event_loop_overlap_disagg_decode()


def test_first_extend_decode_and_idle_drain():
    scheduler = _Scheduler([_Batch(name="extend"), _Batch(name="decode"), None, None])
    _run(scheduler)
    work = [event for event in scheduler.events if event not in ("donation", "transfer")]
    assert work == [
        ("launch", "extend"),
        "dummy",
        ("launch", "decode"),
        ("result", "extend"),
        ("result", "decode"),
    ]
    assert not scheduler.result_queue
    assert scheduler.last_batch is None
    assert scheduler.new_token_ratio == scheduler.init_new_token_ratio


def test_pause_drains_once_and_polls_without_admission(monkeypatch):
    fences = []
    monkeypatch.setattr(decode.jax, "block_until_ready", lambda arrays: fences.append(arrays))

    def pause(scheduler):
        scheduler._engine_paused = True

    def stop(scheduler):
        raise _StopLoop

    scheduler = _Scheduler([_Batch(name="extend")], controls=[None, pause, stop])
    _run(scheduler)
    assert scheduler.events.count(("result", "extend")) == 1
    assert scheduler.events.count("transfer") == 1
    assert scheduler.events.count("terminal") == 1
    assert fences == [scheduler.kv_buffers]
    assert not scheduler.result_queue
    assert scheduler.last_batch is scheduler.cur_batch is None


def test_admission_fence_waits_for_donation_then_device_writes(monkeypatch):
    scheduler = _Scheduler([])
    monkeypatch.setattr(
        decode.jax, "block_until_ready", lambda _: scheduler.events.append("device_ready")
    )
    scheduler._wait_decode_admission_safe()
    assert scheduler.events == ["donation", "device_ready"]


def test_retraction_drain_fences_before_processing_result(monkeypatch):
    scheduler = _Scheduler([])
    monkeypatch.setattr(
        decode.jax, "block_until_ready", lambda _: scheduler.events.append("device_ready")
    )
    batch = _Batch(name="decode", sampling=SimpleNamespace(sampling_info_done=threading.Event()))
    batch.sampling.sampling_info_done.set()
    scheduler.result_queue.append((batch, "decode"))
    scheduler.last_batch = scheduler.cur_batch = batch
    scheduler._drain_disagg_decode_overlap_results()
    assert scheduler.events == ["donation", "device_ready", ("result", "decode")]
    assert not scheduler.result_queue
    assert scheduler.last_batch is scheduler.cur_batch is None


def test_single_process_transfer_drain_does_not_issue_multihost_collectives(monkeypatch):
    from sgl_jax.srt.disaggregation.common import multihost_sync

    monkeypatch.setattr(decode.jax, "process_count", lambda: 1)

    def unexpected_collective(*args, **kwargs):
        pytest.fail("single-process PD polling must not issue multihost collectives")

    monkeypatch.setattr(multihost_sync, "synced_terminal_rooms", unexpected_collective)
    completed = [object()]
    scheduler = SimpleNamespace(
        disagg_transfer_queue=SimpleNamespace(drain_terminal=lambda: completed)
    )
    assert (
        decode.SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced(scheduler)
        is completed
    )


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
