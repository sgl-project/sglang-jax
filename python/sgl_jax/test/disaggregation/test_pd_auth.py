"""Stage 4 H-C: shared-secret auth helper + Bootstrap Bearer.

Each test isolates one shape of failure:
  * wrong / missing tag → reject
  * disabled (secret=None) → accept everything
  * env override beats config
  * Bootstrap 401 on missing / bad / right Bearer

We use the FastAPI ``TestClient`` so the server bind isn't needed.
"""

from __future__ import annotations


from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation import pd_auth
from sgl_jax.srt.disaggregation.bootstrap import build_app


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


# --- Bootstrap server Bearer enforcement ---


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
