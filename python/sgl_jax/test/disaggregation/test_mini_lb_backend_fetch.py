"""Tests for fetch_backend_json in mini_lb.py."""

import asyncio

import pytest

from sgl_jax.srt.disaggregation.mini_lb import fetch_backend_json


class _FakeResponse:
    def __init__(self, status, json_data=None, text=""):
        self.status = status
        self._json = json_data
        self._text = text

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self._responses = responses

    def get(self, url: str):
        for endpoint, resp in self._responses.items():
            if url.endswith(f"/{endpoint}"):
                return resp
        return _FakeResponse(404, text="not found")


def test_first_candidate_succeeds():
    session = _FakeSession({
        "get_server_info": _FakeResponse(200, {"model": "test"}),
    })

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    result = asyncio.run(_run())
    assert result == {"model": "test"}


def test_fallback_to_second_candidate():
    session = _FakeSession({
        "get_server_info": _FakeResponse(404, text="not found"),
        "server_info": _FakeResponse(200, {"model": "fallback"}),
    })

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    result = asyncio.run(_run())
    assert result == {"model": "fallback"}


def test_all_candidates_fail_raises():
    from fastapi import HTTPException

    session = _FakeSession({
        "get_server_info": _FakeResponse(500, text="error1"),
        "server_info": _FakeResponse(503, text="error2"),
    })

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.status_code == 502


def test_single_candidate():
    session = _FakeSession({
        "server_info": _FakeResponse(200, {"status": "ok"}),
    })

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("server_info",)
        )

    result = asyncio.run(_run())
    assert result == {"status": "ok"}
