from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import jax
import jax.profiler
import numpy as np
import orjson
import uvicorn
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import Response
from zmq.constants import LINGER, PUSH

from sgl_jax.srt.disaggregation.encoder.bootstrap import EncoderBootstrapClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.model import EncoderModelRunner
from sgl_jax.srt.disaggregation.encoder.raiden_pool import create_encoder_pool
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
)
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.disaggregation.encoder.scheduler import DisaggEncoderScheduler
from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import configure_logger, set_uvicorn_logging_configs

logger = logging.getLogger(__name__)


class EncoderServer:
    def __init__(
        self,
        encoder: EncoderModelRunner,
        transfer: Any,
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
            encoder,
            transfer,
            max_batch_size=max_batch_size,
        )
        self.scheduler = DisaggEncoderScheduler(
            self.runtime,
            request_timeout=request_timeout,
        )
        self._zmq = zmq.asyncio.Context.instance()
        self._receiver_timeout = receiver_timeout
        self._receiver_addresses: dict[str, str] = {}
        self._receiver_events: dict[str, asyncio.Event] = {}
        self._receiver_sockets: dict[str, zmq.asyncio.Socket] = {}

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
            self.start()
            try:
                yield
            finally:
                try:
                    await self.stop()
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
        self._trace_active = False
        self._trace_dir: str | None = None
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/encode", self.encode, methods=["POST"])
        self.app.add_api_route(
            "/scheduler_receive_url",
            self.register_scheduler_receiver,
            methods=["POST"],
        )
        self.app.add_api_route("/start_profile", self.start_profile, methods=["POST"])
        self.app.add_api_route("/stop_profile", self.stop_profile, methods=["POST"])
        self.app.add_api_route("/profile_status", self.profile_status, methods=["GET"])

    def start(self) -> None:
        self.runtime.start()
        self.scheduler.start()

    async def stop(self) -> None:
        try:
            await self.scheduler.stop()
        finally:
            try:
                await self.runtime.stop()
            finally:
                for socket in self._receiver_sockets.values():
                    socket.close()
                self._receiver_sockets.clear()
                self._receiver_events.clear()
                self._receiver_addresses.clear()

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
        req_id = request["req_id"]
        self._receiver_addresses[req_id] = request["receive_url"]
        self._receiver_events.setdefault(req_id, asyncio.Event()).set()
        return {"req_id": req_id}

    async def encode(self, request: Request) -> dict[str, Any]:
        if not isinstance(request, dict):
            request = orjson.loads(await request.body())
        try:
            data = await self.scheduler.submit(request)
        except Exception as exc:
            try:
                req_id = request["req_id"]
                await self.send_to_scheduler(
                    req_id,
                    EmbeddingData(
                        req_id=req_id,
                        num_parts=request["num_parts"],
                        part_idx=request["part_idx"],
                        grid_dim=None,
                        modality=Modality.from_str(request["modality"]),
                        error_msg=str(exc),
                    ),
                )
            except Exception:
                logger.exception(
                    "Encoder error delivery failed. req_id=%s",
                    request.get("req_id"),
                )
            raise

        try:
            await self.send_to_scheduler(data.req_id, data)
        except Exception:
            self.runtime.release(data.transfer_id)
            raise
        # The response is only an ACK. Metadata travels over ZMQ and the
        # embedding itself travels over the configured transfer backend.
        return {"req_id": request["req_id"]}

    async def send_to_scheduler(self, req_id: str, data: EmbeddingData) -> None:
        try:
            event = self._receiver_events.setdefault(req_id, asyncio.Event())
            if self._receiver_timeout is None or self._receiver_timeout <= 0:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), self._receiver_timeout)
            address = self._receiver_addresses[req_id]
            # Socket creation stays on this event loop without yielding. PyZMQ
            # queues complete messages, so backpressure stays local to each socket.
            socket = self._receiver_sockets.get(address)
            if socket is None:
                socket = self._zmq.socket(PUSH)
                socket.setsockopt(LINGER, 1000)
                socket.connect(f"tcp://{address}")
                self._receiver_sockets[address] = socket
            await socket.send_pyobj(data)
        finally:
            self._receiver_events.pop(req_id, None)
            self._receiver_addresses.pop(req_id, None)

    async def start_profile(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Arm a jax.profiler trace on the encoder process.

        The encoder batch scheduler has no SchedulerProfilerMixin; this minimal
        endpoint lets the EPD driver capture the encoder tier alongside the
        language server's prefill/decode traces.
        """
        request = request or {}
        if self._trace_active:
            return {"status": "in_progress", "output_dir": self._trace_dir}
        base = request.get("output_dir") or os.path.join(
            os.getenv("SGLANG_JAX_PROFILER_DIR", "/tmp"), "encoder"
        )
        Path(base).mkdir(parents=True, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        host_tracer_level = request.get("host_tracer_level")
        python_tracer_level = request.get("python_tracer_level")
        if host_tracer_level is not None:
            options.host_tracer_level = int(host_tracer_level)
        if python_tracer_level is not None:
            options.python_tracer_level = int(python_tracer_level)
        jax.profiler.start_trace(base, profiler_options=options)
        self._trace_active = True
        self._trace_dir = base
        logger.info("Encoder profiling started -> %s", base)
        return {"status": "in_progress", "output_dir": base}

    async def stop_profile(self) -> dict[str, Any]:
        if not self._trace_active:
            return {"status": "idle"}
        jax.profiler.stop_trace()
        self._trace_active = False
        logger.info("Encoder profiling stopped -> %s", self._trace_dir)
        return {"status": "idle", "output_dir": self._trace_dir}

    async def profile_status(self) -> dict[str, Any]:
        return {"status": "in_progress" if self._trace_active else "idle"}


def launch(server_args: ServerArgs) -> None:
    configure_logger(server_args)
    set_uvicorn_logging_configs()
    encoder = EncoderModelRunner(server_args)
    try:
        host_ip = resolve_host_ip(server_args.disaggregation_host_ip)
        devices = {}
        for device, index in encoder.input_sharding.addressable_devices_indices_map(
            (encoder.num_lanes,)
        ).items():
            devices.setdefault(index[0].start or 0, device)
        pool_mesh = jax.sharding.Mesh(
            np.asarray([devices[i] for i in range(encoder.num_lanes)]), ("lane",)
        )
        pool = create_encoder_pool(server_args, encoder.model_config, pool_mesh)
        if not server_args.disable_precompile:
            for capacity in encoder.packed_capacities:
                pool.warmup(capacity)
        transfer = RaidenEncoderServerTransfer(
            host_ip,
            pool,
            parallelism=server_args.disaggregation_channel_number,
            pool_size=server_args.encoder_transfer_pool_size,
            timeout_s=server_args.encoder_request_timeout_seconds,
        )
        advertise_host = f"[{host_ip}]" if ":" in host_ip else host_ip
        advertise_url = (
            f"http://{advertise_host}:{server_args.port}"
            if server_args.encoder_register_urls
            else None
        )
        control_timeout = server_args.encoder_control_timeout_seconds
        server = EncoderServer(
            encoder,
            transfer,
            receiver_timeout=server_args.encoder_request_timeout_seconds,
            encoder_register_urls=server_args.encoder_register_urls,
            advertise_url=advertise_url,
            bootstrap_timeout=control_timeout if control_timeout > 0 else 5.0,
            max_batch_size=server_args.encoder_max_batch_size,
            request_timeout=server_args.encoder_request_timeout_seconds,
        )
        uvicorn.run(server.app, host=server_args.host, port=server_args.port)
    finally:
        encoder.shutdown()
