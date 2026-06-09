"""Tests for PD wire protocol version negotiation."""

from __future__ import annotations

import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation.bootstrap import (
    MIN_COMPATIBLE_VERSION,
    PROTOCOL_VERSION,
    BootstrapClient,
    BootstrapServer,
    PrefillInfo,
    build_app,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestConstants:
    def test_min_compatible_not_above_current(self):
        assert MIN_COMPATIBLE_VERSION <= PROTOCOL_VERSION

    def test_prefill_info_defaults_to_current_version(self):
        info = PrefillInfo(bootstrap_key="h:1", host="h", transfer_port=1, side_channel_port=2)
        assert info.protocol_version == PROTOCOL_VERSION


class TestServerRoundTrip:
    def test_version_preserved_through_registry(self):
        app, _ = build_app()
        client = TestClient(app)
        client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        body = client.get("/get_prefill_info", params={"bootstrap_room": 0}).json()
        assert body["protocol_version"] == PROTOCOL_VERSION


class TestClientVersionGate:
    @pytest.fixture
    def server(self):
        srv = BootstrapServer(host="127.0.0.1", port=_free_port())
        srv.start()
        yield srv
        srv.stop()

    def test_compatible_peer_accepted(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        client.register_prefill(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            protocol_version=PROTOCOL_VERSION,
        )
        info = client.get_prefill_info(0)
        assert info["protocol_version"] == PROTOCOL_VERSION

    def test_peer_below_floor_rejected(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        client.register_prefill(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            protocol_version=MIN_COMPATIBLE_VERSION - 1,
        )
        with pytest.raises(RuntimeError, match="protocol_version"):
            client.get_prefill_info(0)

    def test_no_prefill_registered_raises(self, server):
        client = BootstrapClient(f"http://127.0.0.1:{server.port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.get_prefill_info(0)
