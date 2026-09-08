"""Validate accounting boundaries and replenishment against real HTTP streams."""

import asyncio
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "scripts/disaggregation/falcon")
    )
    return importlib.import_module("steady_client")


def test_window_counts_only_observed_deltas_in_half_open_interval(client):
    window = client.Window(10, 20)
    for at, delta in [(9, 5), (10, 2), (15, 3), (19.99, 4), (20, 8), (21, 1)]:
        window.observe(at, delta)
    assert window.tokens == 9
    assert window.bins == {0: 2, 5: 3, 9: 4}
    with pytest.raises(AssertionError, match="decreased"):
        window.observe(15, -1)


@pytest.mark.parametrize("broken", [False, True])
def test_replenishes_before_slow_peer_finishes_and_drains_cohort(client, broken):
    from aiohttp import web

    starts = []
    slow_done = asyncio.Event()
    replenished_during_slow = []

    async def serve(request):
        body = await request.json()
        slow = body["slow"]
        starts.append(slow)
        if not slow and starts.count(False) > 1:
            replenished_during_slow.append(not slow_done.is_set())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"output_ids":[7],"meta_info":{"completion_tokens":1}}\n\n')
        await asyncio.sleep(0.25 if slow else 0.025)
        await response.write(b'data: {"output_ids":[7,8],"meta_info":{"completion_tokens":2}}\n\n')
        if not broken:
            await response.write(b"data: [DONE]\n\n")
        if slow:
            slow_done.set()
        return response

    async def run():
        app = web.Application()
        app.router.add_post("/generate", serve)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        records = []
        try:
            result = await client.run(
                f"http://127.0.0.1:{port}/generate",
                [
                    {"slow": slow, "sampling_params": {"max_new_tokens": 2}}
                    for slow in [False, True]
                ],
                2,
                0.05,
                0.1,
                stagger_s=0,
                on_complete=records.append,
                expected=[[7, 8]] * 2,
            )
            return result, records
        finally:
            await runner.cleanup()

    if broken:
        with pytest.raises(AssertionError, match="incomplete stream"):
            asyncio.run(run())
    else:
        result, records = asyncio.run(run())
        assert any(replenished_during_slow), starts
        assert result["requests_crossing_start"] >= 1
        assert result["requests_crossing_end"] >= 1
        assert result["drain_s"] > 0
        assert result["cohort_requests"] == sum(0.05 <= r["start_s"] < 0.15 for r in records)
        assert result["exact_token_checks"] == len(records)
        assert 0 < result["window_output_tokens"] < 2 * len(records)
        assert result["output_token_per_s"] == result["window_output_tokens"] / 0.1
