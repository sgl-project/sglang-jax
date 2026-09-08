"""Exercise performance SSE parsing against an actual local HTTP endpoint."""

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("complete", [True, False])
def test_performance_client_omits_logprobs_and_requires_complete_stream(
    monkeypatch, tmp_path, complete
):
    import aiohttp
    from aiohttp import web

    scripts = Path(__file__).resolve().parents[3] / "scripts/disaggregation/falcon"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setenv("PD_OUT", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", "/unused")
    monkeypatch.setenv("PD_HOST_IP", "127.0.0.1")
    module = importlib.import_module("performance_suite")
    monkeypatch.setattr(module.driver, "TOKENIZER", SimpleNamespace(encode=lambda *a, **k: [1, 2]))

    async def serve(request):
        body = await request.json()
        assert "return_logprob" not in body and "logprob_start_len" not in body
        assert len(body["input_ids"]) == 128
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"meta_info":{"completion_tokens":1}}\n\n')
        await response.write(b'data: {"meta_info":{"completion_tokens":4}}\n\n')
        if complete:
            await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def run():
        app = web.Application()
        app.router.add_post("/generate", serve)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                return await module.request(session, port, 128, 4, 0)
        finally:
            await runner.cleanup()

    if complete:
        result = asyncio.run(run())
        assert result["output"] == 4
        assert result["ttft_s"] <= result["e2e_s"]
        assert result["max_event_token_delta"] == 3
        assert len(result["itl_s"]) == 1
    else:
        with pytest.raises(AssertionError):
            asyncio.run(run())
