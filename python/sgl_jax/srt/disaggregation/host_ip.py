"""Host IP resolution for PD multi-host deployment."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket

logger = logging.getLogger(__name__)


# Hostnames that are aliases for loopback even though
# ``ipaddress.ip_address`` would reject them as non-IP.
_REJECTED_HOSTNAMES = frozenset({"localhost"})


def resolve_host_ip(
    explicit: str | None = None,
    *,
    env_name: str = "HOSTNAME",
) -> str:
    """Return the per-host IP a remote PD peer can dial.

    Raises ``RuntimeError`` if every resolution strategy fails or
    yields an unusable address.
    """

    if explicit:
        return _validate(explicit, source="explicit")

    hostname = os.environ.get(env_name)
    if hostname:
        try:
            resolved = socket.gethostbyname(hostname)
            logger.info(
                "resolved host IP from $%s=%r: %s",
                env_name, hostname, resolved,
            )
            return _validate(resolved, source=f"${env_name}")
        except socket.gaierror as e:
            logger.warning(
                "could not resolve $%s=%r: %s; falling back to "
                "socket.gethostname()",
                env_name, hostname, e,
            )

    try:
        fallback = socket.gethostbyname(socket.gethostname())
        logger.info(
            "resolved host IP from socket.gethostname(): %s",
            fallback,
        )
        return _validate(fallback, source="socket.gethostname()")
    except socket.gaierror as e:
        raise RuntimeError(
            "could not resolve a usable host IP for PD: explicit was "
            f"empty, $HOSTNAME was {os.environ.get(env_name)!r}, "
            f"socket.gethostname() raised {e!r}. Please pass "
            "--disaggregation-host-ip explicitly."
        ) from e


def _validate(ip: str, *, source: str) -> str:
    """Reject bind/loopback addresses regardless of textual form.

    Uses :mod:`ipaddress` so all IPv4/IPv6 representations of
    loopback (``127.0.0.0/8``, ``::1``, ``0:0:0:0:0:0:0:1``,
    ``::ffff:127.0.0.1``) and unspecified (``0.0.0.0``, ``::``,
    long-form IPv6 zero) are caught uniformly. Hostnames that aren't
    valid IPs (e.g. ``localhost``) are matched by an explicit small
    alias set.
    """

    if ip in _REJECTED_HOSTNAMES:
        raise RuntimeError(
            f"host {ip!r} (from {source}) is a loopback alias and "
            "cannot be used as a PD peer address. Pass "
            "--disaggregation-host-ip explicitly or set "
            "$HOSTNAME to a routable name."
        )
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        # Not a numeric IP — assume it's a routable DNS name.
        return ip
    if addr.is_loopback or addr.is_unspecified:
        raise RuntimeError(
            f"host IP {ip!r} (from {source}) is a "
            f"{'loopback' if addr.is_loopback else 'bind/unspecified'} "
            "address and cannot be used as a PD peer address. Pass "
            "--disaggregation-host-ip explicitly or set "
            "$HOSTNAME to a routable name."
        )
    return ip
