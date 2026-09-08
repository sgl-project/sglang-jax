"""Arrival cadence must remain independent of service completion."""

import asyncio
import importlib
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("limit", [1, 8])
def test_fixed_rate_does_not_wait_for_responses_or_hide_client_saturation(monkeypatch, limit):
    from aiohttp import web

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/disaggregation/falcon")
    )
    client = importlib.import_module("steady_rate_client")
    received = []
    completed = []

    async def serve(request):
        received.append(time.perf_counter())
        await asyncio.sleep(0.22)
        completed.append(time.perf_counter())
        return web.json_response({"ok": True})

    async def check():
        app = web.Application()
        app.router.add_get("/", serve)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async def send(session, sequence):
            begin = time.perf_counter()
            async with session.get(f"http://127.0.0.1:{port}/") as response:
                response.raise_for_status()
                assert (await response.json())["ok"]
            elapsed = time.perf_counter() - begin
            return {"ttft_s": elapsed, "e2e_s": elapsed}

        try:
            return await client.run(send, 16, 0.125, 0.125, max_inflight=limit)
        finally:
            await runner.cleanup()

    if limit == 1:
        with pytest.raises(ExceptionGroup) as errors:
            asyncio.run(check())
        assert any("in-flight cap reached" in str(e) for e in errors.value.exceptions)
    else:
        result = asyncio.run(check())
        assert result["completed_total"] == 4
        assert sum(at < completed[0] for at in received) >= 3
        assert result["actual_cohort_rps"] == 16
        assert result["peak_client_active"] >= 3
        assert result["drain_s"] > 0
        assert result["failed"] == 0
