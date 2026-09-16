from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs

if TYPE_CHECKING:
    import jax
    import numpy as np

_STOP = object()


@dataclass(slots=True, eq=False)
class EncoderRequest:
    """One request shared by admission, preprocessing, encoding, and completion."""

    request: dict[str, Any]
    future: asyncio.Future[EmbeddingData]
    inputs: MultimodalInputs | None = None
    token_count: int = 0

    @property
    def modality(self) -> Modality:
        return Modality.from_str(self.request["modality"])

    @property
    def batch_key(self) -> tuple[Modality, int]:
        return self.modality, self.token_count

    @property
    def transfer_id(self) -> str:
        return f"{self.request['req_id']}:{self.request['part_idx']}:embedding"


@dataclass(slots=True)
class EncoderBatch:
    """One batch handed from the encoding thread to the transfer thread."""

    requests: list[EncoderRequest]
    lanes: list[list[int]]
    output_indices: np.ndarray
    # Filled by the encoding thread before handing off to the transfer queue.
    embeddings: jax.Array | None = None


class EncoderRuntime:
    """Collect completed preprocessing and run ViT/transfer pipeline stages."""

    def __init__(
        self,
        encoder: Any,
        transfer: Any,
        *,
        max_batch_size: int = 8,
    ) -> None:
        self._encoder = encoder
        self._transfer = transfer
        self._max_batch_size = max(1, int(max_batch_size))
        # This is the completed-preprocess reservoir. The ViT thread drains it
        # when submitting the next batch, without waiting for device completion.
        self._encode_queue: queue.Queue[EncoderRequest | object] = queue.Queue(self._max_batch_size)
        # Reserve backend capacity before encoding to bound queued outputs.
        # The backend reclaims it after transfer; no second queue limit is needed.
        self._transfer_queue: queue.SimpleQueue[EncoderBatch | object] = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._started = False
        self._accepting = True
        self._encode_thread: threading.Thread | None = None
        self._transfer_thread: threading.Thread | None = None

    @property
    def preprocess_concurrency(self) -> int:
        # CPU preprocessing and ViT batching have different concurrency needs.
        # Keep all configured processor workers available even for small ViT
        # batches; the bounded ready queue still applies backpressure.
        return max(1, int(self._encoder.preprocess_concurrency))

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            if not self._accepting:
                raise RuntimeError("EncoderRuntime cannot be restarted")
            self._started = True
            self._encode_thread = threading.Thread(
                target=self._encode_worker,
                name="sgl-jax-encoder-vit",
                daemon=True,
            )
            self._transfer_thread = threading.Thread(
                target=self._transfer_worker,
                name="sgl-jax-encoder-transfer",
                daemon=True,
            )
            self._encode_thread.start()
            self._transfer_thread.start()

    async def stop(self) -> None:
        with self._start_lock:
            if not self._started:
                self._accepting = False
                self._transfer.close()
                return
            self._accepting = False
        await asyncio.to_thread(self._stop_workers)

    def _stop_workers(self) -> None:
        # Drain each stage before stopping the next one.
        self._encode_queue.put(_STOP)
        if self._encode_thread is not None:
            self._encode_thread.join()

        self._transfer_queue.put(_STOP)
        if self._transfer_thread is not None:
            self._transfer_thread.join()
        self._transfer.close()

        with self._start_lock:
            self._started = False

    async def preprocess_request(self, request: EncoderRequest) -> None:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        await self._encoder.preprocess_request(request)

    async def enqueue_preprocessed(self, request: EncoderRequest) -> None:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        if not self._started:
            self.start()
        try:
            self._encode_queue.put_nowait(request)
        except queue.Full:
            await asyncio.to_thread(self._encode_queue.put, request)

    def _encode_worker(self) -> None:
        backlog: deque[EncoderRequest] = deque()
        stopping = False
        while True:
            if backlog:
                item = backlog.popleft()
            elif stopping:
                return
            else:
                item = self._encode_queue.get()
                if item is _STOP:
                    return
            assert isinstance(item, EncoderRequest)
            batch, saw_stop = self._collect_batch(item, backlog)
            stopping = stopping or saw_stop
            try:
                job = self._encoder.build_batch(batch)
            except Exception as exc:
                self._deliver(batch, [exc] * len(batch))
            else:
                self._encode_batch(job)

    def _collect_batch(
        self,
        first: EncoderRequest,
        backlog: deque[EncoderRequest],
    ) -> tuple[list[EncoderRequest], bool]:
        batch = [first]
        key = first.batch_key
        limit = min(
            self._max_batch_size,
            self._transfer.batch_capacity(first.token_count),
        )

        retained: deque[EncoderRequest] = deque()
        while backlog:
            item = backlog.popleft()
            if (
                item.future.get_loop() is first.future.get_loop()
                and item.batch_key == key
                and len(batch) < limit
            ):
                batch.append(item)
            else:
                retained.append(item)
        backlog.extend(retained)

        saw_stop = False
        while len(batch) < limit:
            try:
                item = self._encode_queue.get_nowait()
            except queue.Empty:
                break

            if item is _STOP:
                saw_stop = True
                break
            assert isinstance(item, EncoderRequest)
            if item.future.get_loop() is first.future.get_loop() and item.batch_key == key:
                batch.append(item)
            else:
                backlog.append(item)
        return batch, saw_stop

    def _encode_batch(self, batch: EncoderBatch) -> None:
        transfer_ids = [request.transfer_id for request in batch.requests]
        reserved = False
        try:
            self._transfer.reserve_batch_sync(
                transfer_ids,
                [request.token_count for request in batch.requests],
                output_indices=batch.output_indices,
            )
            reserved = True
            batch.embeddings = self._encoder.encode(batch)
            self._transfer.stage_batch_sync(transfer_ids, batch.embeddings)
            # Build host metadata in the transfer thread while the next ViT batch runs.
            self._transfer_queue.put(batch)
        except Exception as exc:
            if reserved:
                for transfer_id in transfer_ids:
                    self._transfer.release(transfer_id)
            self._deliver(batch.requests, [exc] * len(batch.requests))

    def _prepare_transfer_metadata(self, batch: EncoderBatch) -> list[EmbeddingData]:
        metadata = self._encoder.metadata_for_batch(batch)
        if len(metadata) != len(batch.requests):
            raise RuntimeError("encoder returned incomplete batch metadata")
        return [
            EmbeddingData(
                req_id=item.request["req_id"],
                num_parts=item.request["num_parts"],
                part_idx=item.request["part_idx"],
                modality=item.modality,
                shape=(item.token_count, int(batch.embeddings.shape[1])),
                dtype=str(batch.embeddings.dtype),
                **part_metadata,
            )
            for item, part_metadata in zip(batch.requests, metadata, strict=True)
        ]

    def _transfer_worker(self) -> None:
        while True:
            batch = self._transfer_queue.get()
            if batch is _STOP:
                return
            assert isinstance(batch, EncoderBatch)
            self._run_transfer_batch(batch)

    def _run_transfer_batch(self, batch: EncoderBatch) -> None:
        try:
            data_items = self._prepare_transfer_metadata(batch)
            metadata = self._transfer.publish_batch_sync(
                [request.transfer_id for request in batch.requests]
            )
            if len(metadata) != len(data_items):
                raise RuntimeError("transfer returned incomplete batch metadata")
        except Exception as exc:
            for request in batch.requests:
                self._transfer.release(request.transfer_id)
            self._deliver(batch.requests, [exc] * len(batch.requests))
            return
        for data, item_metadata in zip(data_items, metadata):
            data.transfer = item_metadata
        self._deliver(batch.requests, data_items)

    def _deliver(
        self, requests: list[EncoderRequest], results: list[EmbeddingData | Exception]
    ) -> None:
        loop = requests[0].future.get_loop()
        loop.call_soon_threadsafe(self._complete_batch, requests, results)

    def _complete_batch(
        self, requests: list[EncoderRequest], results: list[EmbeddingData | Exception]
    ) -> None:
        for request, result in zip(requests, results, strict=True):
            if isinstance(result, Exception):
                if not request.future.done():
                    request.future.set_exception(result)
            elif request.future.done():
                # Timeout/cancellation can happen while encoding or transferring.
                self._transfer.release(result.transfer_id)
            else:
                request.future.set_result(result)

    def release(self, transfer_id: str) -> None:
        self._transfer.release(transfer_id)
