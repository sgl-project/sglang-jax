"""Decode-side prefill registry cache (sglang-style local per-room resolution).

Covers ``PrefillInfoCache``: a warm cache serves every request with zero
network round-trips (the fix for the conc128 wedge where a per-request
synchronous ``get_prefill_info`` GET froze the decode event loop); a miss
triggers at most one rate-limited ``list_prefills`` refresh and otherwise
returns ``None`` so the caller defers (never abort); selection mirrors the
server's ``sorted(keys)[room % len]``; stale-protocol peers are rejected.
"""
from __future__ import annotations

import pytest

from sgl_jax.srt.disaggregation.bootstrap import PrefillInfoCache


def _pf(key, **kw):
    d = {
        "bootstrap_key": key,
        "host": "h",
        "transfer_port": 1,
        "side_channel_port": 2,
        "protocol_version": 1,
    }
    d.update(kw)
    return d


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class _FakeClient:
    def __init__(self, prefills) -> None:
        self.prefills = list(prefills)
        self.list_calls = 0

    def list_prefills(self):
        self.list_calls += 1
        return list(self.prefills)


def test_warm_cache_serves_with_zero_get_after_first():
    clock = _Clock()
    client = _FakeClient([_pf("a")])
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    first = cache.pick_for_room(0)
    assert first["bootstrap_key"] == "a"
    assert client.list_calls == 1  # one refresh to warm the cache

    # Many subsequent lookups at the same instant: all cache hits, no refresh.
    for _ in range(50):
        assert cache.pick_for_room(0)["bootstrap_key"] == "a"
    assert client.list_calls == 1


def test_room_modulo_selection_matches_server():
    clock = _Clock()
    client = _FakeClient([_pf("c"), _pf("a"), _pf("b")])  # sorted -> a, b, c
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    assert cache.pick_for_room(0)["bootstrap_key"] == "a"
    assert cache.pick_for_room(1)["bootstrap_key"] == "b"
    assert cache.pick_for_room(2)["bootstrap_key"] == "c"
    assert cache.pick_for_room(4)["bootstrap_key"] == "b"  # 4 % 3 == 1


def test_miss_is_rate_limited_then_resolves():
    clock = _Clock()
    client = _FakeClient([])  # no prefill registered yet
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    # First miss: refreshes once, still empty -> None (caller defers).
    assert cache.pick_for_room(0) is None
    assert client.list_calls == 1

    # Same instant: rate-limited, no second refresh.
    assert cache.pick_for_room(0) is None
    assert client.list_calls == 1

    # Prefill registers; advance past the interval -> one more refresh resolves.
    client.prefills = [_pf("a")]
    clock.t += 1.0
    info = cache.pick_for_room(0)
    assert info["bootstrap_key"] == "a"
    assert client.list_calls == 2


def test_stale_protocol_peer_rejected():
    clock = _Clock()
    client = _FakeClient([_pf("a", protocol_version=0)])
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    with pytest.raises(RuntimeError, match="protocol_version"):
        cache.pick_for_room(0)
