"""PD utility tests: host_ip resolution + shared-secret auth."""

from __future__ import annotations

import socket
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation import pd_auth
from sgl_jax.srt.disaggregation.bootstrap import build_app
from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

# ---- host_ip resolution -----------------------------------------------------


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


# ---- pd_auth: secret resolution + HMAC tags ---------------------------------


def test_resolve_secret_env_wins(monkeypatch):
    monkeypatch.setenv("SGL_JAX_PD_SHARED_SECRET", "from-env")
    assert pd_auth.resolve_secret("from-args") == "from-env"


def test_resolve_secret_falls_back_to_args(monkeypatch):
    monkeypatch.delenv("SGL_JAX_PD_SHARED_SECRET", raising=False)
    assert pd_auth.resolve_secret("from-args") == "from-args"


def test_resolve_secret_none_when_neither(monkeypatch):
    monkeypatch.delenv("SGL_JAX_PD_SHARED_SECRET", raising=False)
    assert pd_auth.resolve_secret(None) is None


def test_verify_tag_disabled_accepts_anything():
    assert pd_auth.verify_tag(None, b"u", None) is True
    assert pd_auth.verify_tag(None, b"u", b"\x00\x01") is True


def test_verify_tag_rejects_missing():
    assert pd_auth.verify_tag("s", b"u", None) is False


def test_verify_tag_rejects_wrong():
    tag = pd_auth.compute_tag("s", b"u")
    bad = bytes([(b ^ 0x55) for b in tag])
    assert pd_auth.verify_tag("s", b"u", bad) is False


def test_verify_tag_accepts_right():
    tag = pd_auth.compute_tag("s", b"u")
    assert pd_auth.verify_tag("s", b"u", tag) is True


def test_verify_bearer_disabled():
    assert pd_auth.verify_bearer(None, None) is True
    assert pd_auth.verify_bearer(None, "Bearer x") is True


def test_verify_bearer_missing_header():
    assert pd_auth.verify_bearer("s", None) is False


def test_verify_bearer_wrong_scheme():
    assert pd_auth.verify_bearer("s", "Basic abc") is False


def test_verify_bearer_wrong_secret():
    assert pd_auth.verify_bearer("s", "Bearer wrong") is False


def test_verify_bearer_right():
    assert pd_auth.verify_bearer("s", "Bearer s") is True


# ---- pd_auth: Bootstrap Bearer enforcement ----------------------------------


def test_bootstrap_health_open_with_auth():
    app, _ = build_app(shared_secret="shh")
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200


def test_bootstrap_rejects_no_auth():
    app, _ = build_app(shared_secret="shh")
    with TestClient(app) as c:
        r = c.get("/list_prefills")
        assert r.status_code == 401


def test_bootstrap_rejects_wrong_auth():
    app, _ = build_app(shared_secret="shh")
    with TestClient(app) as c:
        r = c.get(
            "/list_prefills",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401


def test_bootstrap_accepts_right_auth():
    app, _ = build_app(shared_secret="shh")
    with TestClient(app) as c:
        r = c.get(
            "/list_prefills",
            headers={"Authorization": "Bearer shh"},
        )
        assert r.status_code == 200


def test_bootstrap_disabled_auth_accepts_anything():
    app, _ = build_app(shared_secret=None)
    with TestClient(app) as c:
        r = c.get("/list_prefills")
        assert r.status_code == 200
        r2 = c.get(
            "/list_prefills",
            headers={"Authorization": "Bearer anything"},
        )
        assert r2.status_code == 200
