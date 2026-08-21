from types import SimpleNamespace
from unittest.mock import Mock

from sgl_jax.srt.managers.scheduler import Scheduler


def test_sticky_waiting_request_is_included_in_dp_load_snapshots():
    scheduler = object.__new__(Scheduler)
    scheduler.dp_size = 2
    scheduler.running_batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[]), SimpleNamespace(reqs=[])]
    )
    scheduler.last_batch = None
    scheduler.waiting_queue = [SimpleNamespace(dp_rank=1)]
    scheduler._estimate_req_tokens = Mock(return_value=11)
    scheduler._estimate_req_input_output_tokens = Mock(return_value=(3, 8))

    assert scheduler._get_dp_load_snapshot() == ([0, 1], [0, 11])
    assert scheduler._get_dp_io_snapshot() == ([0, 3], [0, 8])
