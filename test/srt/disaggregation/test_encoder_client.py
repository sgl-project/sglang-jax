from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from sgl_jax.srt.disaggregation.encoder import client as encoder_client
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality


def test_encoder_receiver_reuses_http_client(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, *, timeout) -> None:
            self.timeout = timeout
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    class FakeBackend:
        def close(self) -> None:
            pass

    monkeypatch.setattr(encoder_client.httpx, "Client", FakeClient)
    client = encoder_client.EncoderClient(
        host="127.0.0.1",
        backend=FakeBackend(),
        encoder_urls=["http://encoder"],
        executor=ThreadPoolExecutor(max_workers=1),
        registration_timeout=12.0,
    )

    client.close()

    assert len(clients) == 1
    assert clients[0].timeout == 12.0
    assert clients[0].closed


def test_encoder_receiver_registers_parts_concurrently():
    barrier = threading.Barrier(2)
    posts = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def post(self, url, *, json):
            posts.append((url, json["req_id"]))
            barrier.wait(timeout=1)
            return FakeResponse()

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        future = encoder_client.submit_scheduler_receiver_registrations(
            executor,
            [
                ("http://encoder-0", "request-0", Modality.IMAGE),
                ("http://encoder-1", "request-1", Modality.IMAGE),
            ],
            "127.0.0.1:1234",
            FakeClient(),
        )
        future.result(timeout=2)
    finally:
        executor.shutdown(cancel_futures=True)

    assert sorted(posts) == [
        ("http://encoder-0/scheduler_receive_url", "request-0"),
        ("http://encoder-1/scheduler_receive_url", "request-1"),
    ]


def test_encoder_request_dispatcher_reuses_http_client(monkeypatch):
    clients = []
    posts = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeAsyncClient:
        def __init__(self, *, timeout) -> None:
            self.timeout = timeout
            self.closed = False
            clients.append(self)

        async def post(self, url, *, json):
            posts.append((url, json["req_id"]))
            return FakeResponse()

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(encoder_client.httpx, "AsyncClient", FakeAsyncClient)

    async def run() -> None:
        dispatcher = encoder_client.EncoderRequestDispatcher(timeout=12.0)
        tasks = []
        for rid in ("request-0", "request-1"):
            assignments, task = dispatcher.dispatch(
                GenerateReqInput(rid=rid, image_data="https://example.com/image.png"),
                ["http://encoder"],
            )
            assert sum(assignments.values(), []) == [1]
            tasks.append(task)

        await asyncio.gather(*tasks)
        await dispatcher.close()

    asyncio.run(run())

    assert len(clients) == 1
    assert clients[0].timeout == 12.0
    assert clients[0].closed
    assert posts == [
        ("http://encoder/encode", "request-0_local_part_0"),
        ("http://encoder/encode", "request-1_local_part_0"),
    ]
