from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol

import jax
import zmq.asyncio
from zmq.constants import LINGER, PUSH

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality

EncodeResult = tuple[jax.Array, dict[str, Any]]
BatchEncodeFn = Callable[
    [list[dict[str, Any]]],
    Awaitable[list[EncodeResult]],
]
logger = logging.getLogger(__name__)


class PendingRequest:
    __slots__ = ("future", "request")

    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.future: asyncio.Future[PublishedEmbedding] = asyncio.get_running_loop().create_future()


DispatchBatchFn = Callable[[list[PendingRequest]], Awaitable[None]]


class PublishedEmbedding:
    __slots__ = ("data", "req_id", "transfer_id")

    def __init__(self, req_id: str, transfer_id: str, data: EmbeddingData) -> None:
        self.req_id = req_id
        self.transfer_id = transfer_id
        self.data = data


class EncoderServerTransfer(Protocol):
    async def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]: ...

    async def release_completed(self) -> None: ...

    def release(self, transfer_id: str) -> None: ...

    def close(self) -> None: ...


class EncoderScheduler:
    """Collect queued requests and dispatch one modality group at a time."""

    def __init__(
        self,
        dispatch_batch: DispatchBatchFn,
        max_batch_size: int = 8,
        request_timeout: float | None = 300.0,
    ) -> None:
        self._dispatch_batch = dispatch_batch
        self._max_batch_size = max(1, int(max_batch_size))
        self._request_timeout = request_timeout
        self._pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        while True:
            try:
                pending = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))
            self._pending_queue.task_done()

    async def submit(self, request: dict[str, Any]) -> PublishedEmbedding:
        if self._worker_task is None:
            raise RuntimeError("EncoderScheduler is not running")
        pending = PendingRequest(request)
        await self._pending_queue.put(pending)
        if self._request_timeout is None or self._request_timeout <= 0:
            return await pending.future
        return await asyncio.wait_for(pending.future, self._request_timeout)

    async def _collect_batch(self) -> list[PendingRequest]:
        batch = [await self._pending_queue.get()]
        while len(batch) < self._max_batch_size:
            try:
                batch.append(self._pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch = await self._collect_batch()
            try:
                groups: dict[Modality, list[PendingRequest]] = defaultdict(list)
                for pending in batch:
                    modality = Modality.from_str(pending.request.get("modality", "image"))
                    groups[modality].append(pending)
                for group in groups.values():
                    await self._dispatch_batch(group)
            except asyncio.CancelledError:
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as exc:
                logger.exception("Encoder batch failed")
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
            finally:
                for _ in batch:
                    self._pending_queue.task_done()


class EncoderRuntime:
    """Owns Encoder execution state independently of the HTTP transport."""

    def __init__(
        self,
        batch_encode_fn: BatchEncodeFn,
        transfer: EncoderServerTransfer,
        *,
        receiver_timeout: float | None = 300.0,
        max_batch_size: int = 8,
        request_timeout: float | None = 300.0,
    ) -> None:
        self._batch_encode_fn = batch_encode_fn
        self._transfer = transfer

        self._zmq = zmq.asyncio.Context.instance()
        self._receiver_timeout = receiver_timeout
        self._receiver_addresses: dict[str, str] = {}
        self._receiver_events: dict[str, asyncio.Event] = {}
        self.scheduler = EncoderScheduler(
            self._dispatch_batch,
            max_batch_size,
            request_timeout,
        )
        self._release_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.scheduler.start()
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._transfer.release_completed())

    async def stop(self) -> None:
        await self.scheduler.stop()

        if self._release_task is not None:
            self._release_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._release_task
            self._release_task = None
        self._transfer.close()

    async def register_scheduler_receiver(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        req_id = request["req_id"]
        self._receiver_addresses[req_id] = request["receive_url"]
        self._receiver_events.setdefault(req_id, asyncio.Event()).set()
        return {"req_id": req_id}

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            published = await self.scheduler.submit(request)
        except Exception as exc:
            try:
                await self._send_error(request, exc)
            except Exception:
                logger.exception(
                    "Encoder error delivery failed. req_id=%s",
                    request.get("req_id"),
                )
            raise

        try:
            await self.send_to_scheduler(published.req_id, published.data)
        except Exception:
            self._transfer.release(published.transfer_id)
            raise
        # The return value itself has no meaning; the client will not read it,
        # but it serves as an ACK, ensuring that the request is fully processed before returning.
        return {"req_id": request["req_id"]}

    async def _dispatch_batch(self, batch: list[PendingRequest]) -> None:
        pending_requests = [pending for pending in batch if not pending.future.done()]
        if not pending_requests:
            return

        try:
            results = await self._encode_batch([pending.request for pending in pending_requests])
        except Exception as exc:
            for pending in pending_requests:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return

        published = await asyncio.gather(
            *(
                self._publish_result(pending.request, *result)
                for pending, result in zip(pending_requests, results)
            ),
            return_exceptions=True,
        )
        for pending, result in zip(pending_requests, published):
            if isinstance(result, Exception):
                if not pending.future.done():
                    pending.future.set_exception(result)
                continue

            if pending.future.done():
                self._transfer.release(result.transfer_id)
                continue
            pending.future.set_result(result)

    async def _encode_batch(self, requests: list[dict[str, Any]]) -> list[EncodeResult]:
        results = await self._batch_encode_fn(requests)
        # Preserve existing direct single-request callers while the public
        # runtime contract remains batch-only.
        if (
            len(requests) == 1
            and isinstance(results, tuple)
            and len(results) == 2
            and isinstance(results[1], dict)
        ):
            results = [results]
        if len(results) != len(requests):
            raise RuntimeError(
                f"batch_encode_fn returned {len(results)} results for {len(requests)} requests"
            )
        return results

    async def _publish_result(
        self,
        request: dict[str, Any],
        embedding: jax.Array,
        metadata: dict[str, Any],
    ) -> PublishedEmbedding:
        req_id = request["req_id"]
        modality = Modality.from_str(request["modality"])

        transfer_id = f"{req_id}:{request.get('part_idx', 0)}:embedding"
        transfer_metadata = await self._transfer.publish(transfer_id, embedding)

        metadata = dict(metadata)
        data = EmbeddingData(
            req_id=req_id,
            num_parts=request.get("num_parts", 1),
            part_idx=request.get("part_idx", 0),
            grid_dim=metadata.pop("grid_dim", None),
            modality=modality,
            embedding_shape=embedding.shape,
            dtype=str(embedding.dtype),
            **transfer_metadata,
            **metadata,
        )
        return PublishedEmbedding(req_id, transfer_id, data)

    async def _send_error(
        self,
        request: dict[str, Any],
        exc: Exception,
    ) -> None:
        req_id = request["req_id"]
        await self.send_to_scheduler(
            req_id,
            EmbeddingData(
                req_id=req_id,
                num_parts=request.get("num_parts", 1),
                part_idx=request.get("part_idx", 0),
                grid_dim=None,
                modality=Modality.from_str(request["modality"]),
                error_msg=str(exc),
            ),
        )

    async def send_to_scheduler(self, req_id: str, data: EmbeddingData) -> None:
        event = self._receiver_events.setdefault(req_id, asyncio.Event())
        try:
            if self._receiver_timeout is None or self._receiver_timeout <= 0:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), self._receiver_timeout)
            await self._notify(self._receiver_addresses[req_id], data)
        finally:
            self._receiver_events.pop(req_id, None)
            self._receiver_addresses.pop(req_id, None)

    async def _notify(self, address: str, data: EmbeddingData) -> None:
        socket = self._zmq.socket(PUSH)
        socket.setsockopt(LINGER, 1000)
        try:
            socket.connect(f"tcp://{address}")
            await socket.send_pyobj(data)
        finally:
            socket.close()
