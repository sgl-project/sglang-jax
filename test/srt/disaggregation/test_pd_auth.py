"""Tests for PD shared-secret auth helpers (bootstrap Bearer + ZMQ HMAC)."""

from __future__ import annotations

from sgl_jax.srt.disaggregation.pd_auth import (
    _ENV_VAR,
    bearer_header,
    compute_tag,
    resolve_secret,
    verify_bearer,
    verify_tag,
)


class TestResolveSecret:
    def test_env_takes_precedence(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "from-env")
        assert resolve_secret("from-args") == "from-env"

    def test_falls_back_to_args(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert resolve_secret("from-args") == "from-args"

    def test_none_when_both_absent(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert resolve_secret(None) is None

    def test_empty_env_ignored(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "")
        assert resolve_secret("from-args") == "from-args"


class TestHmacTag:
    def test_compute_tag_deterministic(self):
        a = compute_tag("secret", b"payload")
        b = compute_tag("secret", b"payload")
        assert a == b

    def test_compute_tag_varies_with_payload(self):
        assert compute_tag("secret", b"a") != compute_tag("secret", b"b")

    def test_compute_tag_varies_with_secret(self):
        assert compute_tag("s1", b"p") != compute_tag("s2", b"p")

    def test_verify_tag_matches(self):
        tag = compute_tag("secret", b"payload")
        assert verify_tag("secret", b"payload", tag) is True

    def test_verify_tag_rejects_tampered_payload(self):
        tag = compute_tag("secret", b"payload")
        assert verify_tag("secret", b"tampered", tag) is False

    def test_verify_tag_disabled_when_secret_none(self):
        assert verify_tag(None, b"payload", None) is True

    def test_verify_tag_rejects_missing_candidate(self):
        assert verify_tag("secret", b"payload", None) is False


class TestBearer:
    def test_bearer_header_none_is_empty(self):
        assert bearer_header(None) == {}

    def test_bearer_header_sets_authorization(self):
        assert bearer_header("tok") == {"Authorization": "Bearer tok"}

    def test_verify_bearer_disabled_when_none(self):
        assert verify_bearer(None, None) is True

    def test_verify_bearer_accepts_matching(self):
        assert verify_bearer("tok", "Bearer tok") is True

    def test_verify_bearer_rejects_wrong_token(self):
        assert verify_bearer("tok", "Bearer nope") is False

    def test_verify_bearer_rejects_missing_prefix(self):
        assert verify_bearer("tok", "tok") is False

    def test_verify_bearer_rejects_empty(self):
        assert verify_bearer("tok", None) is False
