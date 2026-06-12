"""Tests for mini_lb pending admission cap (--max-concurrent-requests).

Pending semantics: excess requests await the semaphore (held pending at the
proxy); they are never rejected and never aborted. The permit is held for the
whole request, including the full duration of a streaming response.
"""

import asyncio

import pytest

import sgl_jax.srt.disaggregation.mini_lb as m


@pytest.fixture(autouse=True)
def reset_sem():
    saved = m._admission_sem
    m._admission_sem = None
    yield
    m._admission_sem = saved


def test_no_cap_passthrough(monkeypatch):
    """No cap -> semaphore is None and the request passes straight through."""

    async def fake_forward(req, ep):
        return "OK"

    monkeypatch.setattr(m, "_do_forward", fake_forward)
    assert m._admission_sem is None
    result = asyncio.run(m._forward_to_backend({}, "generate"))
    assert result == "OK"


def test_cap_serializes_excess(monkeypatch):
    """cap=1: the second request stays pending until the first releases, then
    both complete successfully (no error)."""

    m._admission_sem = asyncio.Semaphore(1)
    gate = None  # created inside the running loop
    started = []
    finished = []

    async def fake_forward(req, ep):
        started.append(req["id"])
        await gate.wait()
        finished.append(req["id"])
        return f"done-{req['id']}"

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        nonlocal gate
        gate = asyncio.Event()

        t1 = asyncio.create_task(m._forward_to_backend({"id": 1}, "generate"))
        t2 = asyncio.create_task(m._forward_to_backend({"id": 2}, "generate"))

        # Let the tasks run; only the first should enter _do_forward (cap=1),
        # the second is pending on the semaphore.
        await asyncio.sleep(0.05)
        assert started == [1]
        assert t2.done() is False

        # Release the first; the second then acquires the permit and runs.
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
        return r1, r2

    r1, r2 = asyncio.run(scenario())
    assert {r1, r2} == {"done-1", "done-2"}
    assert sorted(finished) == [1, 2]


def test_stream_holds_until_drained(monkeypatch):
    """Streaming: the permit is released only after body_iterator is fully
    drained, not when the StreamingResponse object is first returned."""
    from fastapi.responses import StreamingResponse

    m._admission_sem = asyncio.Semaphore(1)

    async def chunks():
        yield b"a"
        yield b"b"

    async def fake_forward(req, ep):
        return StreamingResponse(chunks())

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        resp = await m._forward_to_backend({"stream": True}, "generate")
        # Permit still held right after return (stream not drained yet).
        assert m._admission_sem.locked() is True

        collected = [chunk async for chunk in resp.body_iterator]
        # Drained -> permit released.
        assert m._admission_sem.locked() is False
        return collected

    collected = asyncio.run(scenario())
    assert collected == [b"a", b"b"]


def test_no_client_error_on_excess(monkeypatch):
    """Many concurrent requests over a small cap all succeed (never 4xx/5xx)."""

    m._admission_sem = asyncio.Semaphore(2)

    async def fake_forward(req, ep):
        await asyncio.sleep(0.01)
        return f"ok-{req['id']}"

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        tasks = [
            m._forward_to_backend({"id": i}, "generate") for i in range(10)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())
    assert sorted(results) == sorted(f"ok-{i}" for i in range(10))
