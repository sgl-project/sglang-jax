"""Shared-secret helpers for PD channels."""

from __future__ import annotations

import hmac
import os
from hashlib import sha256

_ENV_VAR = "SGL_JAX_PD_SHARED_SECRET"


def resolve_secret(server_args_value: str | None) -> str | None:
    """Return the effective PD shared secret.

    The environment override wins so operators can inject or rotate the
    secret without changing CLI/config wiring.
    """

    env = os.environ.get(_ENV_VAR)
    if env:
        return env
    return server_args_value


def compute_tag(secret: str, payload: bytes) -> bytes:
    """Return the HMAC-SHA256 tag for ``payload``."""

    return hmac.new(secret.encode("utf-8"), payload, sha256).digest()


def verify_tag(secret: str | None, payload: bytes, candidate: bytes | None) -> bool:
    """Return whether ``candidate`` matches the expected HMAC tag."""

    if secret is None:
        return True
    if candidate is None:
        return False
    expected = compute_tag(secret, payload)
    return hmac.compare_digest(expected, candidate)
