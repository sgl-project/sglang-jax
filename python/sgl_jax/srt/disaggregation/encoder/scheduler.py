from __future__ import annotations

import asyncio
import logging

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRequest, EncoderRuntime

logger = logging.getLogger(__name__)


class DisaggEncoderScheduler:
    """Own request admission, preprocessing workers, futures, and timeouts."""

    def __init__(
        self,
        runtime: EncoderRuntime,
        request_timeout: float | None = 300.0,
    ) -> None:
        self._runtime = runtime
        self._request_timeout = request_timeout
        self._preprocess_queue: asyncio.Queue[EncoderRequest] = asyncio.Queue()
        self._preprocess_workers: set[asyncio.Task[None]] = set()
        self._inflight_requests: set[EncoderRequest] = set()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._runtime.preprocess_concurrency):
            task = asyncio.create_task(
                self._preprocess_worker(),
                name=f"encoder-preprocess-{index}",
            )
            self._preprocess_workers.add(task)
            task.add_done_callback(self._preprocess_workers.discard)

    async def stop(self) -> None:
        self._running = False
        workers = tuple(self._preprocess_workers)
        for task in workers:
            task.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._preprocess_workers.clear()

        error = RuntimeError("DisaggEncoderScheduler stopped")
        for pending in self._inflight_requests:
            if not pending.future.done():
                pending.future.set_exception(error)
        while True:
            try:
                self._preprocess_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._preprocess_queue.task_done()

    async def submit(self, request: dict) -> EmbeddingData:
        if not self._running:
            raise RuntimeError("DisaggEncoderScheduler is not running")
        pending = EncoderRequest(request, asyncio.get_running_loop().create_future())
        self._inflight_requests.add(pending)
        await self._preprocess_queue.put(pending)
        try:
            if self._request_timeout is None or self._request_timeout <= 0:
                return await pending.future
            return await asyncio.wait_for(pending.future, self._request_timeout)
        finally:
            self._inflight_requests.discard(pending)

    async def _preprocess_worker(self) -> None:
        while True:
            pending = await self._preprocess_queue.get()
            try:
                if pending.future.done():
                    continue
                await self._runtime.preprocess_request(pending)
                if pending.future.done():
                    continue

                await self._runtime.enqueue_preprocessed(pending)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Encoder preprocessing failed")
                if not pending.future.done():
                    pending.future.set_exception(exc)
            finally:
                self._preprocess_queue.task_done()
