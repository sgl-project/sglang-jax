from unittest.mock import MagicMock

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo


def _req(rid):
    req = MagicMock()
    req.rid = rid
    return req


def test_copy_snapshots_reqs_list_but_keeps_req_objects_shared():
    req1 = _req("r1")
    req2 = _req("r2")
    req3 = _req("r3")
    info = ScheduleReqsInfo(reqs=[req1, req2])
    batch = ScheduleBatch(reqs_info=[info], dp_size=1)

    copied = batch.copy()
    info.reqs.append(req3)

    assert copied.reqs_info[0].reqs == [req1, req2]
    assert copied.reqs_info[0].reqs is not info.reqs
    assert copied.reqs_info[0].reqs[0] is req1
    assert copied.reqs_info[0].reqs[1] is req2
