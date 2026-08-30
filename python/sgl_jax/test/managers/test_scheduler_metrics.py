from types import SimpleNamespace
from unittest.mock import Mock, call

from sgl_jax.srt.managers.scheduler_metrics_mixin import record_queue_wait_times


def _req(dp_rank, start, end=None):
    return SimpleNamespace(
        dp_rank=dp_rank,
        queue_time_start=start,
        queue_time_end=end,
    )


def test_record_queue_wait_times_groups_by_dp_rank():
    reqs = [_req(0, 8.0), _req(1, 9.0), _req(0, 9.5)]
    metric = Mock()

    record_queue_wait_times(reqs, metric, now=10.0)

    assert metric.labels.call_args_list == [
        call(dp_rank="0"),
        call(dp_rank="1"),
        call(dp_rank="0"),
    ]
    assert metric.labels.return_value.observe.call_args_list == [
        call(2.0),
        call(1.0),
        call(0.5),
    ]
    assert [req.queue_time_end for req in reqs] == [10.0, 10.0, 10.0]


def test_record_queue_wait_times_ignores_retracted_and_unstarted_requests():
    already_recorded = _req(0, 1.0, end=2.0)
    no_start = _req(1, None)
    metric = Mock()

    record_queue_wait_times([already_recorded, no_start], metric, now=10.0)

    metric.labels.assert_not_called()
    assert already_recorded.queue_time_end == 2.0
    assert no_start.queue_time_end is None
