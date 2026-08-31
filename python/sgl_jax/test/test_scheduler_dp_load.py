from types import SimpleNamespace
from unittest.mock import Mock

from sgl_jax.srt.managers.scheduler import Scheduler


def test_waiting_and_encoder_waiting_requests_are_included_in_dp_load_snapshots():
    scheduler = object.__new__(Scheduler)
    scheduler.dp_size = 2
    scheduler.running_batch = SimpleNamespace(
        reqs_info=[
            SimpleNamespace(reqs=[], batch_is_full=False),
            SimpleNamespace(reqs=[], batch_is_full=False),
        ]
    )
    scheduler.last_batch = None
    scheduler.waiting_queue = [SimpleNamespace(dp_rank=1)]
    scheduler.encoder_waiting = {
        "encoder": SimpleNamespace(recv_req=SimpleNamespace(dp_rank=0)),
        "unassigned": SimpleNamespace(recv_req=SimpleNamespace(dp_rank=None)),
    }
    scheduler.per_dp_max_running_requests = 8
    scheduler._estimate_req_tokens = Mock(return_value=11)
    scheduler._estimate_req_input_output_tokens = Mock(return_value=(3, 8))

    assert scheduler._get_dp_load_snapshot() == ([1, 1], [11, 11])
    assert scheduler._get_dp_io_snapshot() == ([3, 3], [8, 8])
    assert scheduler._select_min_running_dp() == 0

    scheduler.waiting_queue.clear()

    assert scheduler._select_min_running_dp() == 1
