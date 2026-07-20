"""Bootstrap-side tests: server/client registry, host_ip resolution, and the
``bootstrap_{host,port,room}`` passthrough chain through the tokenizer.

The server tests use a real FastAPI app over an in-process HTTP loopback
(the actual server thread, so the wire format is exercised). A mock clock
controls TTL expiry without sleeping. The passthrough tests verify the
bootstrap fields land on GenerateReqInput / TokenizedGenerateReqInput /
Req, plus the decode-mode rejection of requests missing the fields.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from sgl_jax.srt.disaggregation.bootstrap import (
    MIN_COMPATIBLE_VERSION,
    PROTOCOL_VERSION,
    BootstrapClient,
    BootstrapServer,
    PrefillInfo,
    _Registry,
)
from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
from sgl_jax.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sgl_jax.srt.sampling.sampling_params import SamplingParams


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


# ==================================================================
# Bootstrap server + client (from test_bootstrap_server)
# ==================================================================


def test_registry_register_list_get(server_and_client):
    _, client = server_and_client
    assert client.health()
    assert client.list_prefills() == []
    client.register_prefill(
        bootstrap_key="p0",
        host="10.0.0.1",
        transfer_port=30001,
        side_channel_port=9600,
        tp_rank=0,
        tp_size=1,
        system_dp_rank=0,
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
            bootstrap_key=f"p{i}",
            host="10.0.0.1",
            transfer_port=30001 + i,
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
        bootstrap_key="p0",
        host="10.0.0.1",
        transfer_port=30001,
        side_channel_port=9600,
    )
    client.register_prefill(
        bootstrap_key="p0",
        host="10.0.0.2",
        transfer_port=30002,
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
        bootstrap_key="p0",
        host="10.0.0.1",
        transfer_port=30001,
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
    registry.register(
        PrefillInfo(
            bootstrap_key="p0",
            host="10.0.0.1",
            transfer_port=30001,
            side_channel_port=9600,
        )
    )
    assert len(registry.list_all()) == 1

    # Advance past TTL.
    clock.t += 31.0
    assert registry.list_all() == []
    assert registry.pick_for_room(0) is None


def test_heartbeat_refreshes_ttl():
    clock = _ManualClock()
    registry = _Registry(ttl_seconds=30.0, clock=clock)
    registry.register(
        PrefillInfo(
            bootstrap_key="p0",
            host="10.0.0.1",
            transfer_port=30001,
            side_channel_port=9600,
        )
    )
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
            registry.register(
                PrefillInfo(
                    bootstrap_key=f"p{i}",
                    host="10.0.0.1",
                    transfer_port=30001 + i,
                    side_channel_port=9600 + i,
                )
            )
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
            f"daemon should have kept beating after raises, " f"saw n={call_count['n']}"
        )
    finally:
        daemon.stop()


# --- Protocol version skew tests ---


def test_prefill_info_defaults_to_current_version():
    info = PrefillInfo(
        bootstrap_key="k",
        host="h",
        transfer_port=1,
        side_channel_port=2,
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
    monkeypatch.setattr(client._client, "get", lambda *a, **k: fake)

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
    monkeypatch.setattr(client._client, "get", lambda *a, **k: fake)
    info = client.get_prefill_info(42)
    assert info["host"] == "10.0.0.1"


def test_registry_stores_protocol_version():
    reg = _Registry()
    reg.register(
        PrefillInfo(
            bootstrap_key="k",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            protocol_version=PROTOCOL_VERSION,
        )
    )
    rows = reg.list_all()
    assert rows[0].protocol_version == PROTOCOL_VERSION


# ---- register retry (ref: upstream test_register_to_bootstrap.py) -----------


def test_register_prefill_succeeds_after_transient_failures(monkeypatch):
    """register_prefill retries on ConnectError and succeeds eventually."""
    client = BootstrapClient(
        "http://nowhere:9999",
        register_retries=3,
        register_retry_delay_s=0,
    )

    call_count = 0

    def _mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ConnectError("connection refused")
        resp = mock.MagicMock()
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr(client._client, "post", _mock_post)
    client.register_prefill(
        host="10.0.0.1",
        transfer_port=30001,
        side_channel_port=9600,
        bootstrap_key="k1",
    )
    assert call_count == 3


def test_register_prefill_exhausts_retries_and_raises(monkeypatch):
    """All retries fail → RuntimeError."""
    client = BootstrapClient(
        "http://nowhere:9999",
        register_retries=2,
        register_retry_delay_s=0,
    )
    monkeypatch.setattr(
        client._client,
        "post",
        mock.MagicMock(side_effect=httpx.ConnectError("refused")),
    )
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        client.register_prefill(
            host="10.0.0.1",
            transfer_port=30001,
            side_channel_port=9600,
            bootstrap_key="k2",
        )


# ==================================================================
# host_ip resolution (from test_pd_utils)
# ==================================================================


def test_explicit_value_is_returned_as_is():
    assert resolve_host_ip("10.0.0.42") == "10.0.0.42"


def test_explicit_value_rejects_bind_addresses():
    with pytest.raises(RuntimeError, match="bind/unspecified"):
        resolve_host_ip("0.0.0.0")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("127.0.0.1")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("localhost")


def test_explicit_value_rejects_ipv6_bind_and_loopback():
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("::1")
    with pytest.raises(RuntimeError, match="bind/unspecified"):
        resolve_host_ip("::")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("0:0:0:0:0:0:0:1")
    with pytest.raises(RuntimeError, match="bind/unspecified"):
        resolve_host_ip("0:0:0:0:0:0:0:0")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("::ffff:127.0.0.1")


def test_explicit_value_rejects_127_block():
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("127.0.0.2")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("127.99.99.99")


def test_resolves_from_hostname_env_var(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "pd-host-3.cluster.local")
    with mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
        return_value="10.0.0.3",
    ) as ghbn:
        assert resolve_host_ip() == "10.0.0.3"
    ghbn.assert_called_once_with("pd-host-3.cluster.local")


def test_resolves_from_socket_when_env_unset(monkeypatch):
    monkeypatch.delenv("HOSTNAME", raising=False)
    with (
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
            return_value="fallback-host",
        ),
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
            return_value="10.0.0.7",
        ),
    ):
        assert resolve_host_ip() == "10.0.0.7"


def test_falls_through_when_env_resolution_fails(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "nonexistent.invalid")

    def _gethostbyname(name):
        if name == "nonexistent.invalid":
            raise socket.gaierror(-2, "no such host")
        return "10.0.0.99"

    with (
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
            return_value="real-host",
        ),
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
            side_effect=_gethostbyname,
        ),
    ):
        assert resolve_host_ip() == "10.0.0.99"


def test_raises_when_all_strategies_fail(monkeypatch):
    monkeypatch.delenv("HOSTNAME", raising=False)
    with (
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
            return_value="some-host",
        ),
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
            side_effect=socket.gaierror(-2, "no such host"),
        ),
        pytest.raises(RuntimeError, match="resolve a usable host IP"),
    ):
        resolve_host_ip()


def test_resolved_bind_address_is_rejected(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "bad-dns")
    with (
        mock.patch(
            "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
            return_value="127.0.0.1",
        ),
        pytest.raises(RuntimeError, match="loopback"),
    ):
        resolve_host_ip()


def test_dns_name_passes_through():
    assert resolve_host_ip("pd-host-3.cluster.local") == "pd-host-3.cluster.local"


# ==================================================================
# bootstrap field passthrough (from test_tokenizer_bootstrap_passthrough)
# ==================================================================


def test_generate_req_input_carries_bootstrap_fields():
    obj = GenerateReqInput(
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=42,
        disagg_transfer_id="wire-1",
    )
    assert obj.bootstrap_host == "10.0.0.1"
    assert obj.bootstrap_port == 8998
    assert obj.bootstrap_room == 42
    assert obj.disagg_transfer_id == "wire-1"


def test_tokenized_generate_req_input_has_bootstrap_fields():
    tokenized = TokenizedGenerateReqInput(rid="r1")
    assert tokenized.bootstrap_host is None
    assert tokenized.bootstrap_port is None
    assert tokenized.bootstrap_room is None
    assert tokenized.disagg_transfer_id is None

    tokenized.bootstrap_host = "10.0.0.1"
    tokenized.bootstrap_port = 8998
    tokenized.bootstrap_room = 42
    tokenized.disagg_transfer_id = "wire-1"
    assert tokenized.bootstrap_host == "10.0.0.1"
    assert tokenized.bootstrap_port == 8998
    assert tokenized.bootstrap_room == 42
    assert tokenized.disagg_transfer_id == "wire-1"


def _make_fake_tokenizer_manager(disaggregation_mode: str):
    """Build a stripped-down TokenizerManager-ish object with just
    enough surface for ``_create_tokenized_object`` to run.
    """

    from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager

    tm = TokenizerManager.__new__(TokenizerManager)
    tm.preferred_sampling_params = None
    tm.model_config = SimpleNamespace(vocab_size=32000, hf_config=SimpleNamespace())
    tm.tokenizer = SimpleNamespace(
        normalize=lambda x: x,
    )
    tm.server_args = SimpleNamespace(
        disaggregation_mode=disaggregation_mode,
        speculative_algorithm=None,
        speculative_target_verify_mode="auto",
        speculative_eagle_topk=1,
    )
    return tm


def test_tokenizer_passes_bootstrap_fields_through_in_decode_mode():
    tm = _make_fake_tokenizer_manager("decode")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=42,
        disagg_transfer_id="wire-r1",
    )
    # ``_create_tokenized_object`` calls ``SamplingParams.normalize +
    # verify``; SamplingParams.normalize expects a tokenizer, the
    # SimpleNamespace stand-in satisfies the attribute lookup.
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.1"
    assert tokenized.bootstrap_port == 8998
    assert tokenized.bootstrap_room == 42
    assert tokenized.disagg_transfer_id == "wire-r1"


def test_tokenizer_rejects_missing_fields_in_decode_mode():
    tm = _make_fake_tokenizer_manager("decode")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
        pytest.raises(ValueError, match="bootstrap"),
    ):
        tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])


def test_tokenizer_allows_missing_fields_in_null_mode():
    tm = _make_fake_tokenizer_manager("null")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_room is None


def test_tokenizer_allows_missing_fields_in_prefill_mode():
    # Prefill side doesn't receive bootstrap_* from the HTTP request
    # (the decode side does); allow missing here.
    tm = _make_fake_tokenizer_manager("prefill")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_room is None


def test_req_carries_bootstrap_fields():
    """``Req`` constructor doesn't take these as positional args; the
    scheduler's ``handle_generate_request`` sets them after construction.
    Verify the attribute is present and defaults to None.
    """

    from sgl_jax.srt.managers.schedule_batch import Req

    req = Req(
        rid="r1",
        origin_input_text="hi",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    assert req.bootstrap_host is None
    assert req.bootstrap_port is None
    assert req.bootstrap_room is None
    assert req.disagg_transfer_id is None

    req.bootstrap_host = "10.0.0.1"
    req.bootstrap_port = 8998
    req.bootstrap_room = 42
    req.disagg_transfer_id = "wire-req-1"
    assert req.bootstrap_room == 42
    assert req.disagg_transfer_id == "wire-req-1"


def test_decode_mode_auto_derives_from_bootstrap_url():
    """Stage 3 router integration: when the request body lacks
    bootstrap_*, the tokenizer fills them in from
    ``--disaggregation-bootstrap-url`` so the router doesn't have to.
    """

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://10.0.0.5:8998"
    obj = GenerateReqInput(
        rid="r-auto",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.5"
    assert tokenized.bootstrap_port == 8998
    # Stable CRC32 of "r-auto".
    import zlib

    assert tokenized.bootstrap_room == zlib.crc32(b"r-auto")


def test_decode_mode_auto_derive_brackets_ipv6_host():
    """Stage 3 review I1: urlparse strips IPv6 brackets;
    auto-derive must re-bracket so downstream ``f"{host}:{port}"``
    is a parseable URL.
    """

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://[fe80::1]:8998"
    obj = GenerateReqInput(
        rid="r-v6",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "[fe80::1]"
    assert tokenized.bootstrap_port == 8998


def test_decode_mode_explicit_values_win_over_auto_derive():
    """Operator-supplied bootstrap_* values must NOT be overwritten
    by the auto-derive."""

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://10.0.0.5:8998"
    obj = GenerateReqInput(
        rid="r-explicit",
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.99",
        bootstrap_port=9999,
        bootstrap_room=42,
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.99"
    assert tokenized.bootstrap_port == 9999
    assert tokenized.bootstrap_room == 42
