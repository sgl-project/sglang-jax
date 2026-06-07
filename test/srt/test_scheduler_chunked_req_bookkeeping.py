from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler import Scheduler


def test_chunked_req_was_scheduled_uses_identity_within_dp_bucket():
    req = object()
    other_req = object()
    adder = SimpleNamespace(can_run_list={0: [other_req], 1: [req]})

    assert not Scheduler._chunked_req_was_scheduled(adder, 0, req)
    assert Scheduler._chunked_req_was_scheduled(adder, 1, req)
    assert not Scheduler._chunked_req_was_scheduled(adder, 2, req)


def test_scheduled_chunked_reqs_for_batch_masks_unscheduled_pending_chunks():
    scheduled_req = object()
    unscheduled_req = object()

    assert Scheduler._scheduled_chunked_reqs_for_batch(
        [scheduled_req, unscheduled_req, None],
        [True, False, True],
    ) == [scheduled_req, None, None]
