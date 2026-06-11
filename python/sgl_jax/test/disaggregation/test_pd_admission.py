"""D1 admission: reserve a host slot for PD reqs at batch formation."""
from __future__ import annotations

from sgl_jax.srt.managers.scheduler import _reserve_host_slot_for_pd


class _Pool:
    def __init__(self, n):
        self._free = list(range(n))

    def reserve(self):
        return self._free.pop(0) if self._free else None


class _Req:
    def __init__(self, room):
        self.bootstrap_room = room
        self.disagg_host_buffer_id = None


def test_non_pd_req_not_gated():
    pool = _Pool(1)
    req = _Req(room=None)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is True and bid is None  # admitted, no reservation


def test_pd_req_reserves_slot():
    pool = _Pool(1)
    req = _Req(room=7)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is True and bid == 0


def test_pd_req_blocked_when_pool_full():
    pool = _Pool(0)
    req = _Req(room=7)
    ok, bid = _reserve_host_slot_for_pd(pool, True, req)
    assert ok is False and bid is None  # caller must `continue` (stay in queue)


def test_disabled_or_no_pool_not_gated():
    req = _Req(room=7)
    assert _reserve_host_slot_for_pd(None, True, req) == (True, None)
    assert _reserve_host_slot_for_pd(_Pool(0), False, req) == (True, None)
