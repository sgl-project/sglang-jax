from types import SimpleNamespace


class FakeQueue:
    def __init__(self, entries):
        self._entries = list(entries)

    def __len__(self):
        return len(self._entries)

    def items_fifo(self):
        return list(self._entries)


class FakeAllocator:
    def available_size(self, dp_rank=0):
        return 1000 - dp_rank


class FakeBatch:
    def __init__(self, reqs_by_dp):
        self.reqs_info = [SimpleNamespace(reqs=reqs) for reqs in reqs_by_dp]

    def is_empty(self):
        return not any(info.reqs for info in self.reqs_info)


def test_pd_decode_admission_state_empty_when_no_queue():
    from sgl_jax.srt.managers.scheduler import Scheduler

    scheduler = SimpleNamespace(disagg_prealloc_queue=None)

    assert Scheduler._get_pd_decode_admission_state(scheduler) == {}


def test_pd_decode_admission_state_counts_queue_tokens_and_dp_capacity():
    from sgl_jax.srt.managers.scheduler import Scheduler

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        pd_time_stats=SimpleNamespace(marks={}),
    )
    scheduler = SimpleNamespace(
        disagg_prealloc_queue=FakeQueue([SimpleNamespace(req=req)]),
        disagg_transfer_queue=FakeQueue([object(), object()]),
        token_to_kv_pool_allocator=FakeAllocator(),
        dp_size=2,
        running_batch=FakeBatch([[object()], [object(), object()]]),
        server_args=SimpleNamespace(
            disaggregation_max_inflight_transfers=8,
            disaggregation_num_reserved_decode_tokens=512,
        ),
    )

    state = Scheduler._get_pd_decode_admission_state(scheduler)

    assert state["prealloc_queue_size"] == 1
    assert state["transfer_queue_size"] == 2
    assert state["running_reqs"] == 3
    assert state["max_inflight_transfers"] == 8
    assert state["reserved_decode_tokens"] == 512
    assert state["kv_available_by_dp"] == [1000, 999]
    assert state["oldest_prealloc_wait_ms"] is None
    assert state["pending_prealloc_prompt_tokens"] == 3


def test_pd_decode_admission_state_reports_oldest_prealloc_wait(monkeypatch):
    from sgl_jax.srt.managers import scheduler as scheduler_module
    from sgl_jax.srt.managers.scheduler import Scheduler

    monkeypatch.setattr(scheduler_module.time, "perf_counter", lambda: 12.5)
    req = SimpleNamespace(
        origin_input_ids=[1],
        pd_time_stats=SimpleNamespace(marks={"prealloc_entry": 10.0}),
    )
    scheduler = SimpleNamespace(
        disagg_prealloc_queue=FakeQueue([SimpleNamespace(req=req)]),
        disagg_transfer_queue=FakeQueue([]),
        token_to_kv_pool_allocator=FakeAllocator(),
        dp_size=1,
        running_batch=None,
        server_args=SimpleNamespace(
            disaggregation_max_inflight_transfers=8,
            disaggregation_num_reserved_decode_tokens=512,
        ),
    )

    state = Scheduler._get_pd_decode_admission_state(scheduler)

    assert state["oldest_prealloc_wait_ms"] == 2500.0
