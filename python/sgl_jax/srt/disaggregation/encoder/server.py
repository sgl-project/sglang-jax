from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from sgl_jax.srt.disaggregation.encoder.bootstrap import EncoderBootstrapClient
from sgl_jax.srt.disaggregation.encoder.runtime import (
    BatchEncodeFn,
    EncoderRuntime,
    EncoderServerTransfer,
)

logger = logging.getLogger(__name__)


class EncoderServer:
    def __init__(
        self,
        batch_encode_fn: BatchEncodeFn,
        transfer: EncoderServerTransfer,
        receiver_timeout: float | None = 300.0,
        encoder_register_urls: list[str] | None = None,
        advertise_url: str | None = None,
        bootstrap_timeout: float = 5.0,
        max_batch_size: int = 8,
        request_timeout: float | None = 300.0,
    ) -> None:
        encoder_register_urls = list(encoder_register_urls or ())
        if bool(encoder_register_urls) != bool(advertise_url):
            raise ValueError("encoder_register_urls and advertise_url must be configured together")

        self.runtime = EncoderRuntime(
            batch_encode_fn,
            transfer,
            receiver_timeout=receiver_timeout,
            max_batch_size=max_batch_size,
            request_timeout=request_timeout,
        )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            bootstrap_clients = [
                (
                    url,
                    EncoderBootstrapClient(url, timeout=bootstrap_timeout),
                )
                for url in encoder_register_urls
            ]
            registration_task = None
            if advertise_url is not None:
                registration_task = asyncio.create_task(
                    self._register_with_bootstraps(
                        bootstrap_clients,
                        advertise_url.rstrip("/"),
                    )
                )
            self.runtime.start()
            try:
                yield
            finally:
                try:
                    await self.runtime.stop()
                finally:
                    if registration_task is not None:
                        registration_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await registration_task
                    if advertise_url is not None:
                        await self._unregister_from_bootstraps(
                            bootstrap_clients,
                            advertise_url.rstrip("/"),
                        )

        self.app = FastAPI(openapi_url=None, lifespan=lifespan)
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/encode", self.encode, methods=["POST"])
        self.app.add_api_route(
            "/scheduler_receive_url",
            self.register_scheduler_receiver,
            methods=["POST"],
        )

    @staticmethod
    async def _register_with_bootstraps(
        clients: list[tuple[str, EncoderBootstrapClient]],
        encoder_url: str,
    ) -> None:
        pending = list(clients)
        for attempt in range(30):
            results = await asyncio.gather(
                *(client.register(encoder_url) for _, client in pending),
                return_exceptions=True,
            )
            pending = [
                pair for pair, result in zip(pending, results) if isinstance(result, Exception)
            ]
            if not pending:
                return
            if attempt < 29:
                await asyncio.sleep(5)

        logger.error(
            "Encoder registration failed after 30 attempts: %s",
            [url for url, _ in pending],
        )

    @staticmethod
    async def _unregister_from_bootstraps(
        clients: list[tuple[str, EncoderBootstrapClient]],
        encoder_url: str,
    ) -> None:
        results = await asyncio.gather(
            *(client.unregister(encoder_url) for _, client in clients),
            return_exceptions=True,
        )
        for (url, _), result in zip(clients, results):
            if isinstance(result, Exception):
                logger.warning("Encoder unregister from %s failed: %s", url, result)
        await asyncio.gather(*(client.close() for _, client in clients))

    async def health(self) -> Response:
        return Response("OK")

    async def register_scheduler_receiver(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.runtime.register_scheduler_receiver(request)

    async def encode(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.runtime.submit(request)

    def run(self, host: str, port: int) -> None:
        uvicorn.run(self.app, host=host, port=port)
