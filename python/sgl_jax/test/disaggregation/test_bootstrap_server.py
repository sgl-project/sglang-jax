"""Unit tests for the PD bootstrap server + client.

Uses a real FastAPI app over an in-process HTTP loopback (TestClient
style via the actual server thread, so we exercise the wire format).
A mock clock controls TTL expiry without sleeping.
"""

from __future__ import annotations

import socket
import threading
import time
from unittest import mock

import pytest

import httpx

from sgl_jax.srt.disaggregation.bootstrap import (
    BootstrapClient,
    BootstrapServer,
    MIN_COMPATIBLE_VERSION,
    PROTOCOL_VERSION,
    PrefillInfo,
    _Registry,
)


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server_and_client():
    server = BootstrapServer("127.0.0.1", _free_port())
    server.start()
    client = BootstrapClient(f"http://127.0.0.1:{server.port}")
    yield server, client
    server.stop()


def test_registry_register_list_get(server_and_client):
    _, client = server_and_client
    assert client.health()
    assert client.list_prefills() == []
    client.register_prefill(
        bootstrap_key="p0", host="10.0.0.1", transfer_port=30001,
        side_channel_port=9600, tp_rank=0, tp_size=1, system_dp_rank=0,
    )
    plist = client.list_prefills()
    assert len(plist) == 1
    assert plist[0]["bootstrap_key"] == "p0"
    assert plist[0]["host"] == "10.0.0.1"

    info = client.get_prefill_info(bootstrap_room=42)
    assert info["bootstrap_key"] == "p0"


def test_register_multiple_room_hashing(server_and_client):
    _, client = server_and_client
    for i in range(3):
        client.register_prefill(
            bootstrap_key=f"p{i}", host="10.0.0.1", transfer_port=30001 + i,
            side_channel_port=9600 + i,
        )
    seen: list[str] = []
    for room in range(30):
        info = client.get_prefill_info(bootstrap_room=room)
        seen.append(info["bootstrap_key"])
    # All three peers must be reached.
    assert set(seen) == {"p0", "p1", "p2"}


def test_re_register_overwrites_and_refreshes(server_and_client):
    _, client = server_and_client
    client.register_prefill(
        bootstrap_key="p0", host="10.0.0.1", transfer_port=30001,
        side_channel_port=9600,
    )
    client.register_prefill(
        bootstrap_key="p0", host="10.0.0.2", transfer_port=30002,
        side_channel_port=9601,
    )
    info = client.get_prefill_info(bootstrap_room=0)
    assert info["host"] == "10.0.0.2"
    assert info["transfer_port"] == 30002


def test_heartbeat_unknown_returns_404(server_and_client):
    _, client = server_and_client
    with pytest.raises(Exception) as excinfo:
        client.heartbeat("never-registered")
    assert "404" in str(excinfo.value)


def test_unregister_removes_entry(server_and_client):
    _, client = server_and_client
    client.register_prefill(
        bootstrap_key="p0", host="10.0.0.1", transfer_port=30001,
        side_channel_port=9600,
    )
    client.unregister_prefill("p0")
    assert client.list_prefills() == []


def test_get_prefill_info_no_workers(server_and_client):
    _, client = server_and_client
    with pytest.raises(Exception) as excinfo:
        client.get_prefill_info(bootstrap_room=0)
    assert "503" in str(excinfo.value)


# --- TTL eviction tests use the in-process registry with a mock clock ---


class _ManualClock:
    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t


def test_ttl_evicts_stale_entries():
    clock = _ManualClock()
    registry = _Registry(ttl_seconds=30.0, clock=clock)
    registry.register(PrefillInfo(
        bootstrap_key="p0", host="10.0.0.1",
        transfer_port=30001, side_channel_port=9600,
    ))
    assert len(registry.list_all()) == 1

    # Advance past TTL.
    clock.t += 31.0
    assert registry.list_all() == []
    assert registry.pick_for_room(0) is None


def test_heartbeat_refreshes_ttl():
    clock = _ManualClock()
    registry = _Registry(ttl_seconds=30.0, clock=clock)
    registry.register(PrefillInfo(
        bootstrap_key="p0", host="10.0.0.1",
        transfer_port=30001, side_channel_port=9600,
    ))
    clock.t += 25.0
    assert registry.heartbeat("p0")
    clock.t += 25.0
    # Without heartbeat we'd be at 50s; with the refresh we're at 25s
    # since last_seen, still under 30s.
    assert len(registry.list_all()) == 1


def test_concurrent_register_list():
    """Smoke: 50 threads register, 50 threads list, no exceptions."""

    registry = _Registry()
    barrier = threading.Barrier(100)
    errors: list[BaseException] = []

    def do_register(i):
        barrier.wait()
        try:
            registry.register(PrefillInfo(
                bootstrap_key=f"p{i}", host="10.0.0.1",
                transfer_port=30001 + i, side_channel_port=9600 + i,
            ))
        except BaseException as e:
            errors.append(e)

    def do_list():
        barrier.wait()
        try:
            registry.list_all()
        except BaseException as e:
            errors.append(e)

    threads = []
    for i in range(50):
        threads.append(threading.Thread(target=do_register, args=(i,)))
        threads.append(threading.Thread(target=do_list))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(registry.list_all()) == 50


def test_heartbeat_daemon_keeps_registration_alive(server_and_client):
    """Stage 2 review C2 regression: without a heartbeat daemon the
    bootstrap registration would fall off the registry after the
    TTL. With the daemon, the entry stays available across multiple
    TTL windows.
    """

    from sgl_jax.srt.disaggregation.bootstrap import HeartbeatDaemon

    _, client = server_and_client
    client.register_prefill(
        bootstrap_key="hb-key",
        host="10.0.0.1",
        transfer_port=30001,
        side_channel_port=9600,
    )
    # 50ms interval — far shorter than the 30s TTL but enough to
    # demonstrate continuous beating without slowing the test.
    daemon = HeartbeatDaemon(client, "hb-key", interval_s=0.05)
    daemon.start()
    try:
        time.sleep(0.25)  # ~5 beats
        plist = client.list_prefills()
        assert any(p["bootstrap_key"] == "hb-key" for p in plist)
    finally:
        daemon.stop()


def test_heartbeat_daemon_survives_transient_server_errors():
    """If a beat raises (e.g. transient network), the daemon logs +
    keeps going, doesn't tear down.
    """

    from sgl_jax.srt.disaggregation.bootstrap import HeartbeatDaemon

    failing_client = mock.MagicMock()
    call_count = {"n": 0}

    def _raise_then_succeed(key):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("transient")

    failing_client.heartbeat.side_effect = _raise_then_succeed
    daemon = HeartbeatDaemon(failing_client, "k", interval_s=0.01)
    daemon.start()
    try:
        time.sleep(0.1)
        assert call_count["n"] >= 3, (
            f"daemon should have kept beating after raises, "
            f"saw n={call_count['n']}"
        )
    finally:
        daemon.stop()


# --- Protocol version skew tests ---


def test_prefill_info_defaults_to_current_version():
    info = PrefillInfo(
        bootstrap_key="k", host="h", transfer_port=1, side_channel_port=2,
    )
    assert info.protocol_version == PROTOCOL_VERSION


def test_min_le_current():
    assert MIN_COMPATIBLE_VERSION <= PROTOCOL_VERSION


def test_client_rejects_below_min_version(monkeypatch):
    client = BootstrapClient("http://nowhere", shared_secret=None)

    fake = mock.MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {
        "bootstrap_key": "k",
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
        "protocol_version": MIN_COMPATIBLE_VERSION - 1,
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake)

    with pytest.raises(RuntimeError, match="protocol_version"):
        client.get_prefill_info(42)


def test_client_accepts_current_version(monkeypatch):
    client = BootstrapClient("http://nowhere", shared_secret=None)
    fake = mock.MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {
        "bootstrap_key": "k",
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
        "protocol_version": PROTOCOL_VERSION,
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake)
    info = client.get_prefill_info(42)
    assert info["host"] == "10.0.0.1"


def test_registry_stores_protocol_version():
    reg = _Registry()
    reg.register(
        PrefillInfo(
            bootstrap_key="k", host="h",
            transfer_port=1, side_channel_port=2,
            protocol_version=PROTOCOL_VERSION,
        )
    )
    rows = reg.list_all()
    assert rows[0].protocol_version == PROTOCOL_VERSION
