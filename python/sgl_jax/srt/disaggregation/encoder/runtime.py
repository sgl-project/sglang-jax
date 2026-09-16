from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs

if TYPE_CHECKING:
    import jax

_STOP = object()


@dataclass(slots=True, eq=False)
class EncoderRequest:
    """One request shared by admission, preprocessing, encoding, and completion."""

    payload: dict[str, Any]  # Incoming payload: req_id, modality, mm_items, etc.
    future: asyncio.Future[EmbeddingData]
    inputs: MultimodalInputs | None = None
    token_count: int = 0
    release_requested: threading.Event = field(default_factory=threading.Event)

    @property
    def modality(self) -> Modality:
        return Modality.from_str(self.payload["modality"])

    @property
    def transfer_id(self) -> str:
        return f"{self.payload['req_id']}:{self.payload['part_idx']}:embedding"


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
        self._transfer_queue: queue.SimpleQueue[tuple[list[EncoderRequest], jax.Array] | object] = (
            queue.SimpleQueue()
        )
        # Lifecycle state belongs to the server event loop; lifespan stops us once.
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
        self._accepting = False
        if not self._started:
            self._transfer.close()
            return
        await asyncio.to_thread(self._stop_workers)
        self._started = False

    def _stop_workers(self) -> None:
        self._transfer.close()
        self._encode_queue.put(_STOP)
        if self._encode_thread is not None:
            self._encode_thread.join()

        self._transfer_queue.put(_STOP)
        if self._transfer_thread is not None:
            self._transfer_thread.join()

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
        pending: list[EncoderRequest] = []
        stopping = False
        while pending or not stopping:
            # Keep a bounded snapshot; queue waiting and shutdown stay in the worker.
            while not stopping and len(pending) < self._max_batch_size:
                try:
                    item = self._encode_queue.get_nowait() if pending else self._encode_queue.get()
                except queue.Empty:
                    break
                if item is _STOP:
                    stopping = True
                    break
                assert isinstance(item, EncoderRequest)
                pending.append(item)
            try:
                requests, remaining = self._collect_batch(pending)
            except Exception as exc:
                self._deliver(pending, [exc] * len(pending))
                pending = []
                continue
            pending = remaining
            if requests:
                self._encode_batch(requests)

    def _collect_batch(
        self,
        pending: list[EncoderRequest],
    ) -> tuple[list[EncoderRequest], list[EncoderRequest]]:
        """Return selected requests and the unconsumed suffix.

        Only select a compatible prefix; do not mutate pending or access queues.
        """
        pending = [request for request in pending if not request.future.done()]
        requests: list[EncoderRequest] = []
        for item in pending[: self._max_batch_size]:
            if requests and item.modality != requests[0].modality:
                break
            requests.append(item)
        return requests, pending[len(requests) :]

    def _encode_batch(self, requests: list[EncoderRequest]) -> None:
        requests = [request for request in requests if not request.future.done()]
        if not requests:
            return
        reserved = False
        try:
            transfer_ids = [request.transfer_id for request in requests]
            inputs = [request.inputs for request in requests]
            items_by_lane, output_indices = self._encoder.build_batch(inputs)
            self._transfer.reserve_batch_sync(
                transfer_ids,
                [request.token_count for request in requests],
                output_indices=output_indices,
                cancelled=lambda: any(r.release_requested.is_set() for r in requests),
            )
            reserved = True
            embeddings = self._encoder.encode(items_by_lane, modality=requests[0].modality)
            self._transfer.stage_batch_sync(transfer_ids, embeddings)
            # Build host metadata in the transfer thread while the next ViT batch runs.
            self._transfer_queue.put((requests, embeddings))
        except Exception as exc:
            if reserved:
                for transfer_id in transfer_ids:
                    self._transfer.release(transfer_id)
            if isinstance(exc, CancelledError) and not reserved:
                # Retry live siblings with their original layout after a queued abort.
                self._encode_batch([r for r in requests if not r.release_requested.is_set()])
            else:
                self._deliver(requests, [exc] * len(requests))

    def _transfer_worker(self) -> None:
        while True:
            try:
                item = self._transfer_queue.get(timeout=0.1)
            except queue.Empty:
                self._transfer.progress()
                continue
            if item is _STOP:
                return
            # Each queue item contains a list of requests because encoding and transfer operate in batches.
            requests, embeddings = item
            self._run_transfer_batch(requests, embeddings)

    def _run_transfer_batch(self, requests: list[EncoderRequest], embeddings: jax.Array) -> None:
        try:
            data_items = [
                EmbeddingData(
                    req_id=item.payload["req_id"],
                    num_parts=item.payload["num_parts"],
                    part_idx=item.payload["part_idx"],
                    modality=item.modality,
                    shape=(item.token_count, int(embeddings.shape[1])),
                    dtype=str(embeddings.dtype),
                    **self._encoder.metadata_for_request(item),
                )
                for item in requests
            ]
            metadata = self._transfer.publish_batch_sync(
                [request.transfer_id for request in requests],
                cancelled=[request.release_requested for request in requests],
            )
            if len(metadata) != len(data_items):
                raise RuntimeError("transfer returned incomplete batch metadata")
        except Exception as exc:
            for request in requests:
                self._transfer.release(request.transfer_id)
            self._deliver(requests, [exc] * len(requests))
            return
        for data, item_metadata in zip(data_items, metadata):
            data.transfer = item_metadata
        self._deliver(requests, data_items)

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
                self._transfer.release(request.transfer_id)
            else:
                request.future.set_result(result)

    def release(self, transfer_id: str) -> None:
        self._transfer.release(transfer_id)
