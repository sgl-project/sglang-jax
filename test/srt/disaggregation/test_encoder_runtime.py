from __future__ import annotations

import asyncio

import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import (
    EncoderRuntime,
    EncoderScheduler,
    PendingRequest,
    PublishedEmbedding,
)
from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimEncoderServerTransfer
from sgl_jax.srt.multimodal.common.modality_enum import Modality


def test_sim_server_transfer_publish_is_awaitable():
    async def run() -> None:
        metadata = await SimEncoderServerTransfer().publish(
            "request-0:embedding", jnp.zeros((1, 2))
        )
        assert metadata == {"transfer_id": "request-0:embedding"}

    asyncio.run(run())


def test_runtime_skips_transfer_for_request_cancelled_during_encode():
    published = []

    class FakeTransfer:
        async def publish(self, transfer_id, embedding):
            published.append(transfer_id)
            return {"transfer_id": transfer_id}

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> None:
        pending = PendingRequest({"req_id": "request-0", "modality": "IMAGE"})

        async def encode(_requests):
            pending.future.cancel()
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(encode, FakeTransfer())
        await runtime._dispatch_batch([pending])

    asyncio.run(run())
    assert published == []


def test_scheduler_records_enqueue_and_dequeue_timestamps(caplog):
    captured = []

    async def dispatch(batch):
        captured.extend(batch)
        for pending in batch:
            data = EmbeddingData(
                req_id=pending.request["req_id"],
                num_parts=1,
                part_idx=0,
                grid_dim=None,
                modality=Modality.IMAGE,
            )
            pending.future.set_result(
                PublishedEmbedding(pending.request["req_id"], "transfer-0", data)
            )

    async def run() -> None:
        scheduler = EncoderScheduler(dispatch, log_queue_timing=True)
        scheduler.start()
        try:
            await scheduler.submit({"req_id": "request-0", "modality": "IMAGE"})
        finally:
            await scheduler.stop()

    caplog.set_level("INFO")
    asyncio.run(run())

    assert len(captured) == 1
    pending = captured[0]
    assert isinstance(pending.enqueue_ns, int)
    assert isinstance(pending.dequeue_ns, int)
    assert pending.queue_duration_ns is not None
    assert pending.queue_duration_ns >= 0
    assert "ENCODER-QUEUE-TIME req_id=request-0" in caplog.text


def test_scheduler_pipelines_bounded_inflight_batches():
    started = []

    async def run() -> None:
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(batch):
            started.append(batch[0].request["req_id"])
            if len(started) == 2:
                both_started.set()
            await release.wait()
            for pending in batch:
                data = EmbeddingData(
                    req_id=pending.request["req_id"],
                    num_parts=1,
                    part_idx=0,
                    grid_dim=None,
                    modality=Modality.IMAGE,
                )
                pending.future.set_result(
                    PublishedEmbedding(pending.request["req_id"], "transfer-0", data)
                )

        scheduler = EncoderScheduler(
            dispatch,
            max_batch_size=1,
            max_inflight_batches=2,
        )
        scheduler.start()
        first = asyncio.create_task(scheduler.submit({"req_id": "request-0", "modality": "IMAGE"}))
        second = asyncio.create_task(scheduler.submit({"req_id": "request-1", "modality": "IMAGE"}))
        try:
            await asyncio.wait_for(both_started.wait(), 1)
            release.set()
            await asyncio.gather(first, second)
        finally:
            release.set()
            await scheduler.stop()

    asyncio.run(run())
    assert started == ["request-0", "request-1"]


def test_runtime_publishes_queue_timing_metadata():
    class FakeTransfer:
        async def publish(self, transfer_id, embedding):
            return {"transfer_id": transfer_id}

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> EmbeddingData:
        pending = PendingRequest({"req_id": "request-0", "modality": "IMAGE"})
        pending.mark_dequeued()

        async def encode(_requests):
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(encode, FakeTransfer())
        await runtime._dispatch_batch([pending])
        return pending.future.result().data

    data = asyncio.run(run())
    assert isinstance(data.enqueue_ns, int)
    assert isinstance(data.dequeue_ns, int)
    assert data.queue_duration_ns >= 0
    assert data.queue_ms == data.queue_duration_ns / 1_000_000
