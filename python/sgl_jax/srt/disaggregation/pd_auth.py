"""Shared-secret auth helpers for PD channels.

The three PD channels — bootstrap HTTP, transfer pull side channel,
and ZMQ ack channel — all share a single secret. Each channel
applies the secret differently (Bearer header for HTTP, HMAC tag
beside the payload for ZMQ / transfer), but the resolution rules and
constant-time compare live here so every channel agrees on the same
edge cases.

Resolution order:
  1. ``SGL_JAX_PD_SHARED_SECRET`` environment variable.
  2. ``ServerArgs.disaggregation_shared_secret``.
  3. ``None`` → auth is disabled.
"""

from __future__ import annotations

import hmac
import os
from hashlib import sha256

_ENV_VAR = "SGL_JAX_PD_SHARED_SECRET"


def resolve_secret(server_args_value: str | None) -> str | None:
    env = os.environ.get(_ENV_VAR)
    if env:
        return env
    return server_args_value


def compute_tag(secret: str, payload: bytes) -> bytes:
    return hmac.new(secret.encode("utf-8"), payload, sha256).digest()


def verify_tag(
    secret: str | None, payload: bytes, candidate: bytes | None
) -> bool:
    if secret is None:
        return True
    if candidate is None:
        return False
    expected = compute_tag(secret, payload)
    return hmac.compare_digest(expected, candidate)


def bearer_header(secret: str | None) -> dict:
    if secret is None:
        return {}
    return {"Authorization": f"Bearer {secret}"}


def verify_bearer(secret: str | None, header_value: str | None) -> bool:
    if secret is None:
        return True
    if not header_value or not header_value.startswith("Bearer "):
        return False
    candidate = header_value[len("Bearer "):]
    return hmac.compare_digest(secret, candidate)
