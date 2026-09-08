"""Regression for PD logprob streaming with NumPy scalars (Falcon r2)."""

import asyncio
from types import SimpleNamespace

import numpy as np
import orjson
from sgl_jax.srt.entrypoints import http_server


def test_generate_stream_serializes_numpy_logprobs_without_truncating(monkeypatch):
    async def generate_request(*_):
        for count in (1, 2):
            yield {
                "text": "ok",
                "meta_info": {
                    "completion_tokens": np.int64(count),
                    "output_token_logprobs": [(np.float64(-0.5), np.int64(42), None)] * count,
                },
            }

    manager = SimpleNamespace(generate_request=generate_request, create_abort_task=lambda _: None)
    monkeypatch.setattr(http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager))

    async def consume():
        response = await http_server.generate_request(SimpleNamespace(stream=True), None)
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(consume())
    assert chunks[-1] == b"data: [DONE]\n\n"
    values = [orjson.loads(chunk[6:]) for chunk in chunks[:-1]]
    assert [x["meta_info"]["completion_tokens"] for x in values] == [1, 2]
    assert values[-1]["meta_info"]["output_token_logprobs"] == [[-0.5, 42, None]] * 2
