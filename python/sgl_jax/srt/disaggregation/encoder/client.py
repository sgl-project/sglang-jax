from __future__ import annotations

import asyncio
import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import jax
import zmq

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ImageData,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, flatten_nested_list

logger = logging.getLogger(__name__)


def create_part_req_id(req_id: str, part_idx: int) -> str:
    return f"{req_id}_local_part_{part_idx}"


def plan_encoder_registrations(
    request: TokenizedGenerateReqInput,
    default_encoder_urls: list[str],
) -> list[tuple[str, str, Modality | None]]:
    """Build the encoder receiver registrations for a tokenized request.

    Args:
        request: The request containing the request ID, optional encoder URLs, and
            the number of multimodal items assigned to each encoder.
        default_encoder_urls: Encoder URLs used when the request does not provide
            its own ``encoder_urls``.

    Returns:
        A list of ``(encoder_url, request_part_id, modality)`` tuples, one for
        each non-empty modality/encoder assignment. For a single encoder without
        explicit assignments, the original request ID and ``None`` modality are
        returned.
    """
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    encoder_urls = request.encoder_urls or default_encoder_urls
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

    if request.num_items_assigned is None:
        if len(encoder_urls) != 1:
            raise ValueError("num_items_assigned is required for multiple encoders")
        return [(encoder_urls[0], request.rid, None)]

    registrations: list[tuple[str, str, Modality | None]] = []
    for modality, assignments in request.num_items_assigned.items():
        if len(assignments) != len(encoder_urls):
            raise ValueError(
                f"{modality.name} has {len(assignments)} assignments for "
                f"{len(encoder_urls)} encoders"
            )
        for encoder_idx, count in enumerate(assignments):
            if count < 0:
                raise ValueError("num_items_assigned cannot contain negative values")
            if count == 0:
                continue
            part_idx = len(registrations)
            registrations.append(
                (
                    encoder_urls[encoder_idx],
                    create_part_req_id(request.rid, part_idx),
                    modality,
                )
            )

    if not registrations:
        raise ValueError("num_items_assigned does not assign any multimodal items")
    return registrations


def register_scheduler_receivers(
    registrations: list[tuple[str, str, Modality | None]],
    receive_url: str,
    timeout: float | None,
) -> None:
    async def register() -> None:
        async with httpx.AsyncClient(timeout=timeout) as client:

            async def register_one(
                encoder_url: str,
                req_id: str,
                modality: Modality | None,
            ) -> None:
                payload = {
                    "req_id": req_id,
                    "receive_count": 1,
                    "receive_url": receive_url,
                }
                if modality is not None:
                    payload["modality"] = modality.name
                response = await client.post(
                    f"{encoder_url.rstrip('/')}/scheduler_receive_url",
                    json=payload,
                )
                response.raise_for_status()

            results = await asyncio.gather(
                *(register_one(*registration) for registration in registrations),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result

    asyncio.run(register())


def validate_encoder_response(
    data: Any,
    expected_num_parts: int,
    active_part_indices: set[int],
    completed_part_indices: set[int],
) -> None:
    if not isinstance(data, EmbeddingData):
        raise TypeError(f"expected EmbeddingData, got {type(data).__name__}")
    if data.num_parts != expected_num_parts:
        raise ValueError("inconsistent encoder part metadata")
    if not 0 <= data.part_idx < expected_num_parts:
        raise ValueError(f"invalid part_idx: {data.part_idx}")
    if data.part_idx in active_part_indices or data.part_idx in completed_part_indices:
        raise ValueError(f"duplicate part_idx: {data.part_idx}")
    if data.error_msg is not None:
        raise RuntimeError(data.error_msg)


def build_encoder_result(accumulator: Any) -> dict[str, Any]:
    return {
        "embeddings": accumulator.get_embedding(is_concat=True),
        **accumulator.get_mm_extra_meta(),
    }


class EncoderReceiveSession(Protocol):
    def poll(self) -> jax.Array | None: ...

    def close(self) -> None: ...


class EncoderReceiverBackend(Protocol):
    def start(self, data: EmbeddingData) -> EncoderReceiveSession: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PendingEncoderRequest:
    recv_req: TokenizedGenerateReqInput
    started_at: float
    receiver: zmq.Socket
    register_future: Future[None]
    accumulator: MultiModalEmbeddingData
    backend: EncoderReceiverBackend
    # Keep each part's metadata alongside its in-flight transfer session so the
    # completed embedding can later be assembled with the correct modality and grid.
    sessions: dict[int, tuple[EmbeddingData, EncoderReceiveSession]] = field(default_factory=dict)

    def poll(self) -> dict[str, Any] | None:
        if self.register_future.done():
            self.register_future.result()  # error re-thrown to the scheduler main thread

        # The ZMQ message contains EmbeddingData metadata (part identity,
        # shape/dtype, and transfer endpoints); the backend pulls the actual
        # embedding separately in _start_receive().
        try:
            data = self.receiver.recv_pyobj(zmq.NOBLOCK)
        except zmq.Again:
            data = None
        if data is not None:
            self._start_receive(data)

        for part_idx, (part_data, session) in list(self.sessions.items()):
            embedding = session.poll()
            if embedding is None:
                continue
            self.accumulator.add(part_data, embedding)
            self.sessions.pop(part_idx)
            session.close()

        if not self.accumulator.ready:
            return None
        return build_encoder_result(self.accumulator)

    def _start_receive(self, data: Any) -> None:
        validate_encoder_response(
            data,
            self.accumulator.num_parts,
            set(self.sessions),
            {
                part_idx
                for part_idx in range(self.accumulator.num_parts)
                if self.accumulator.has_part(part_idx)
            },
        )
        self.sessions[data.part_idx] = (data, self.backend.start(data))

    def close(self) -> None:
        self.register_future.cancel()
        for _, session in self.sessions.values():
            session.close()
        self.sessions.clear()
        self.receiver.close()


class EncoderClient:
    def __init__(
        self,
        host: str,
        backend: EncoderReceiverBackend,
        encoder_urls: list[str],
        executor: ThreadPoolExecutor,
        registration_timeout: float | None,
    ) -> None:
        self._host = host
        self._backend = backend
        self._encoder_urls = list(encoder_urls)
        self._executor = executor
        self._registration_timeout = registration_timeout

    def receive(self, request: TokenizedGenerateReqInput) -> PendingEncoderRequest:
        registrations = plan_encoder_registrations(request, self._encoder_urls)

        receiver = zmq.Context.instance().socket(zmq.PULL)
        receiver.setsockopt(zmq.LINGER, 0)
        port = receiver.bind_to_random_port(f"tcp://{self._host}")
        receive_url = f"{self._host}:{port}"
        try:
            register_future = self._executor.submit(
                register_scheduler_receivers,
                registrations,
                receive_url,
                self._registration_timeout,
            )
        except Exception:
            receiver.close()
            raise
        return PendingEncoderRequest(
            recv_req=request,
            started_at=time.monotonic(),
            receiver=receiver,
            register_future=register_future,
            accumulator=MultiModalEmbeddingData(len(registrations)),
            backend=self._backend,
        )

    def close(self) -> None:
        self._backend.close()
        self._executor.shutdown(wait=False, cancel_futures=True)


def dispatch_encoder_request(
    request: GenerateReqInput,
    encoder_urls: list[str],
    timeout: float | None,
) -> tuple[dict[Modality, list[int]], asyncio.Task[None]]:
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

    items_by_modality = {}
    for name, modality in (
        ("image_data", Modality.IMAGE),
        ("video_data", Modality.VIDEO),
        ("audio_data", Modality.AUDIO),
    ):
        data = getattr(request, name, None)
        if data is None:
            continue
        items = []
        for item in flatten_nested_list(data):
            if item is None:
                continue
            if isinstance(item, ImageData):
                item = item.url
            elif isinstance(item, dict) and "url" in item:
                item = item["url"]
            items.append(item)
        if items:
            items_by_modality[modality] = items

    assignments = {}
    encoder_indices = list(range(len(encoder_urls)))
    random.shuffle(encoder_indices)
    offset = 0
    for modality, items in items_by_modality.items():
        base, remainder = divmod(len(items), len(encoder_urls))
        counts = [base] * len(encoder_urls)
        for index in range(remainder):
            counts[encoder_indices[(offset + index) % len(encoder_urls)]] += 1
        assignments[modality] = counts
        offset = (offset + remainder) % len(encoder_urls)

    num_parts = sum(count > 0 for counts in assignments.values() for count in counts)
    encode_requests = []
    for modality, counts in assignments.items():
        items = items_by_modality[modality]
        item_offset = 0
        for encoder_idx, count in enumerate(counts):
            if count == 0:
                continue
            part_idx = len(encode_requests)
            encode_requests.append(
                (
                    encoder_urls[encoder_idx],
                    {
                        "req_id": create_part_req_id(request.rid, part_idx),
                        "mm_items": items[item_offset : item_offset + count],
                        "num_parts": num_parts,
                        "part_idx": part_idx,
                        "modality": modality.name,
                    },
                )
            )
            item_offset += count

    async def send_encode_requests() -> None:
        async with httpx.AsyncClient(timeout=timeout) as client:

            async def send_one(encoder_url: str, payload: dict[str, Any]) -> None:
                response = await client.post(
                    f"{encoder_url.rstrip('/')}/encode",
                    json=payload,
                )
                response.raise_for_status()

            results = await asyncio.gather(
                *(send_one(*encode_request) for encode_request in encode_requests),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result

    task = asyncio.create_task(
        send_encode_requests(),
        name=f"encoder-dispatch-{request.rid}",
    )

    def finish(completed: asyncio.Task[None]) -> None:
        if completed.cancelled():
            return
        try:
            completed.result()
        except Exception:
            logger.exception("Encoder dispatch failed. rid=%s", request.rid)

    task.add_done_callback(finish)

    return assignments, task


def create_encoder_client(
    server_args,
    mesh: Any,
) -> EncoderClient:
    from sgl_jax.srt.disaggregation.encoder.raiden import create_raiden_client

    return create_raiden_client(server_args, mesh)
