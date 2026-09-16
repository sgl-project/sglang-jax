from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager, suppress

import httpx
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import Response

logger = logging.getLogger(__name__)


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
        evicted_probe_interval: float = 60.0,
    ) -> None:
        self.host = host
        self.port = port
        self._urls = urls if urls is not None else []
        self._urls[:] = list(dict.fromkeys(url.rstrip("/") for url in self._urls))
        self._lock = threading.Lock()
        self._health_check_interval = health_check_interval
        self._health_check_timeout = health_check_timeout
        self._evicted_probe_interval = evicted_probe_interval
        # Registered URL -> (consecutive failures, next probe time).
        self._health: dict[str, tuple[int, float]] = {url: (0, 0) for url in self._urls}
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

        self.app = FastAPI(
            openapi_url=None,
            lifespan=lifespan,
        )
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

    async def register(self, url: str = Body(embed=True, min_length=1)) -> Response:
        url = url.rstrip("/")
        with self._lock:
            self._health[url] = (0, 0)
            if url not in self._urls:
                self._urls.append(url)
        return Response("OK")

    async def unregister(self, url: str = Body(embed=True, min_length=1)) -> Response:
        url = url.rstrip("/")
        with self._lock:
            if url in self._urls:
                self._urls.remove(url)
            self._health.pop(url, None)
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
                    candidates = [
                        url for url, (_, next_probe) in self._health.items() if now >= next_probe
                    ]

                results = await asyncio.gather(
                    *(client.get(f"{url}/health") for url in candidates),
                    return_exceptions=True,
                )
                with self._lock:
                    for url, result in zip(candidates, results):
                        if url not in self._health:
                            continue  # Unregistered while the probe was in flight.
                        healthy = isinstance(result, httpx.Response) and result.status_code == 200
                        failures = 0 if healthy else self._health[url][0] + 1
                        interval = (
                            self._evicted_probe_interval
                            if failures >= 3
                            else self._health_check_interval
                        )
                        self._health[url] = (failures, now + interval)
                        if healthy and url not in self._urls:
                            self._urls.append(url)
                        elif failures >= 3 and url in self._urls:
                            self._urls.remove(url)
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
