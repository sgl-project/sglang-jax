"""Unit tests for :func:`resolve_host_ip` (Stage 3)."""

from __future__ import annotations

import socket
from unittest import mock

import pytest

from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip


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
    """Stage 3 review I2: cover the IPv6 forms that the string-set
    implementation missed. ``ipaddress`` catches both compact and
    long forms uniformly.
    """

    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("::1")
    with pytest.raises(RuntimeError, match="bind/unspecified"):
        resolve_host_ip("::")
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("0:0:0:0:0:0:0:1")  # long-form IPv6 loopback
    with pytest.raises(RuntimeError, match="bind/unspecified"):
        resolve_host_ip("0:0:0:0:0:0:0:0")  # long-form IPv6 unspecified
    with pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip("::ffff:127.0.0.1")  # IPv4-mapped IPv6 loopback


def test_explicit_value_rejects_127_block():
    """Stage 3 review I3: not just 127.0.0.1 — entire 127.0.0.0/8."""

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
    with mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
        return_value="fallback-host",
    ), mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
        return_value="10.0.0.7",
    ):
        assert resolve_host_ip() == "10.0.0.7"


def test_falls_through_when_env_resolution_fails(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "nonexistent.invalid")

    def _gethostbyname(name):
        if name == "nonexistent.invalid":
            raise socket.gaierror(-2, "no such host")
        return "10.0.0.99"

    with mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
        return_value="real-host",
    ), mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
        side_effect=_gethostbyname,
    ):
        assert resolve_host_ip() == "10.0.0.99"


def test_raises_when_all_strategies_fail(monkeypatch):
    monkeypatch.delenv("HOSTNAME", raising=False)
    with mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostname",
        return_value="some-host",
    ), mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
        side_effect=socket.gaierror(-2, "no such host"),
    ), pytest.raises(RuntimeError, match="resolve a usable host IP"):
        resolve_host_ip()


def test_resolved_bind_address_is_rejected(monkeypatch):
    """If a misconfigured DNS round-trips ``hostname`` → ``0.0.0.0``
    or similar, we reject the resolution rather than silently
    publishing a useless address.
    """

    monkeypatch.setenv("HOSTNAME", "bad-dns")
    with mock.patch(
        "sgl_jax.srt.disaggregation.host_ip.socket.gethostbyname",
        return_value="127.0.0.1",
    ), pytest.raises(RuntimeError, match="loopback"):
        resolve_host_ip()


def test_dns_name_passes_through():
    """Non-numeric strings (DNS names) pass through unchanged —
    they round-trip correctly through ``f"{host}:{port}"`` for any
    downstream peer that re-resolves them.
    """

    assert (
        resolve_host_ip("pd-host-3.cluster.local")
        == "pd-host-3.cluster.local"
    )
