from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager, suppress

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EncoderRegistration(BaseModel):
    url: str = Field(min_length=1)


class EncoderBootstrapServer:
    """In-process Encoder registry shared with the TokenizerManager."""

    def __init__(
        self,
        host: str,
        port: int,
        urls: list[str] | None = None,
        *,
        health_check_interval: float = 10.0,
        health_check_timeout: float = 2.0,
        evicted_ttl: float = 600.0,
    ) -> None:
        self.host = host
        self.port = port
        self._urls = urls if urls is not None else []
        self._urls[:] = list(dict.fromkeys(url.rstrip("/") for url in self._urls))
        self._lock = threading.Lock()
        self._health_check_interval = health_check_interval
        self._health_check_timeout = health_check_timeout
        self._evicted_ttl = evicted_ttl
        self._health_fail_counts: dict[str, int] = {}
        self._evicted_urls: dict[str, float] = {}
        self._server: uvicorn.Server | None = None

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            task = None
            if self._health_check_interval > 0:
                task = asyncio.create_task(self._health_check_loop())
            try:
                yield
            finally:
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

        self.app = FastAPI(openapi_url=None, lifespan=lifespan)
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/register_encoder_url", self.register, methods=["POST"])
        self.app.add_api_route("/unregister_encoder_url", self.unregister, methods=["DELETE"])
        self.app.add_api_route("/list_encoder_urls", self.list_encoders, methods=["GET"])

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EncoderBootstrap",
        )
        self.thread.start()

    async def health(self) -> Response:
        return Response("OK")

    async def list_encoders(self) -> dict[str, list[str]]:
        return {"encoder_urls": self.list_urls()}

    async def register(self, registration: EncoderRegistration) -> Response:
        url = registration.url.rstrip("/")
        with self._lock:
            self._health_fail_counts.pop(url, None)
            self._evicted_urls.pop(url, None)
            if url not in self._urls:
                self._urls.append(url)
        return Response("OK")

    async def unregister(self, registration: EncoderRegistration) -> Response:
        url = registration.url.rstrip("/")
        with self._lock:
            if url in self._urls:
                self._urls.remove(url)
            self._health_fail_counts.pop(url, None)
            self._evicted_urls.pop(url, None)
        return Response("OK")

    def list_urls(self) -> list[str]:
        with self._lock:
            return list(self._urls)

    async def _health_check_loop(self) -> None:
        timeout = httpx.Timeout(self._health_check_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                await asyncio.sleep(self._health_check_interval)
                now = time.monotonic()
                with self._lock:
                    if self._evicted_ttl > 0:
                        expired = [
                            url
                            for url, evicted_at in self._evicted_urls.items()
                            if now - evicted_at >= self._evicted_ttl
                        ]
                        for url in expired:
                            self._evicted_urls.pop(url, None)
                    candidates = list(dict.fromkeys(self._urls + list(self._evicted_urls)))

                results = await asyncio.gather(
                    *(client.get(f"{url}/health") for url in candidates),
                    return_exceptions=True,
                )
                with self._lock:
                    for url, result in zip(candidates, results):
                        healthy = isinstance(result, httpx.Response) and result.status_code == 200
                        if healthy:
                            self._health_fail_counts.pop(url, None)
                            if self._evicted_urls.pop(url, None) is not None:
                                self._urls.append(url)
                            continue

                        if url in self._evicted_urls:
                            continue
                        failures = self._health_fail_counts.get(url, 0) + 1
                        if failures < 3:
                            self._health_fail_counts[url] = failures
                            continue
                        self._health_fail_counts.pop(url, None)
                        if url in self._urls:
                            self._urls.remove(url)
                        self._evicted_urls[url] = now
                        logger.warning("Evicted unhealthy Encoder: %s", url)

    def _run(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)


class EncoderBootstrapClient:
    def __init__(
        self,
        bootstrap_url: str,
        timeout: float | None = 10.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=bootstrap_url.rstrip("/"),
            timeout=timeout,
        )

    async def list_encoders(self) -> list[str]:
        response = await self._client.get("/list_encoder_urls")
        response.raise_for_status()
        return response.json()["encoder_urls"]

    async def register(self, encoder_url: str) -> None:
        response = await self._client.post(
            "/register_encoder_url",
            json={"url": encoder_url},
        )
        response.raise_for_status()

    async def unregister(self, encoder_url: str) -> None:
        response = await self._client.request(
            "DELETE",
            "/unregister_encoder_url",
            json={"url": encoder_url},
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
