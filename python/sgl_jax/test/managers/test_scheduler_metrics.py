from types import SimpleNamespace
from unittest.mock import patch

from sgl_jax.srt.managers.scheduler_metrics_mixin import (
    SchedulerMetricsMixin,
    record_queue_wait_times,
)


def _req(dp_rank, start, end=None):
    return SimpleNamespace(
        dp_rank=dp_rank,
        queue_time_start=start,
        queue_time_end=end,
    )


def test_record_queue_wait_times_groups_by_dp_rank():
    reqs = [_req(0, 8.0), _req(1, 9.0), _req(0, 9.5)]

    waits = record_queue_wait_times(reqs, now=10.0)

    assert waits == {0: [2.0, 0.5], 1: [1.0]}
    assert [req.queue_time_end for req in reqs] == [10.0, 10.0, 10.0]


def test_record_queue_wait_times_ignores_retracted_and_unstarted_requests():
    already_recorded = _req(0, 1.0, end=2.0)
    no_start = _req(1, None)

    waits = record_queue_wait_times([already_recorded, no_start], now=10.0)

    assert waits == {}
    assert already_recorded.queue_time_end == 2.0
    assert no_start.queue_time_end is None


def test_log_prefill_stats_reports_per_dp_queue_wait():
    scheduler = SimpleNamespace(
        last_prefill_stats_tic=9.0,
        last_prefill_tokens=0,
        last_input_throughput=0.0,
        is_hybrid=False,
        dp_size=2,
        running_batch=SimpleNamespace(
            reqs_info=[SimpleNamespace(reqs=[]), SimpleNamespace(reqs=[])]
        ),
        waiting_queue=[],
        _get_token_info=lambda: (0, 0.0, None, None),
    )
    adder = SimpleNamespace(
        log_input_tokens=2,
        log_hit_tokens=0,
        can_run_list={0: [_req(0, 8.0)], 1: [_req(1, 9.0)]},
    )
    reqs = adder.can_run_list[0] + adder.can_run_list[1]

    with (
        patch(
            "sgl_jax.srt.managers.scheduler_metrics_mixin.time.perf_counter",
            return_value=10.0,
        ),
        patch("sgl_jax.srt.managers.scheduler_metrics_mixin.logger.info") as log_info,
    ):
        SchedulerMetricsMixin.log_prefill_stats(scheduler, adder, reqs, running_bs=0)

    message = log_info.call_args.args[0]
    assert "queue-wait per DP (ms): " in message
    assert "{'avg': 2000.0, 'max': 2000.0}" in message
    assert "{'avg': 1000.0, 'max': 1000.0}" in message
