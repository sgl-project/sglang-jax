from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import ClassVar
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zmq

from sgl_jax.raiden import raiden_requested
from sgl_jax.srt.disaggregation.encoder.client import PendingEncoderRequest
from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.disaggregation.encoder.raiden import (
    DeferredRaidenReceiveSession,
    RaidenEncoderServerTransfer,
    RaidenReceiverBackend,
    RaidenReceiveSession,
)
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality


@pytest.fixture(autouse=True)
def _pretend_raiden_is_preloaded(monkeypatch):
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.require_raiden_preloaded",
        lambda: None,
    )


class _FakeRaidenWrapper:
    instances: ClassVar[list[_FakeRaidenWrapper]] = []
    start_barrier: ClassVar[threading.Barrier | None] = None

    def __init__(self, host: str, port: int, *, parallelism: int) -> None:
        self.host = host
        self.port = port
        self.parallelism = parallelism
        self.endpoints = [{"endpoint": "127.0.0.1:7788", "shards": [0]}]
        self.started = None
        self.registered = None
        self.read = None
        self.stats = ([], [], [])
        self.instances.append(self)

    def start(self, buffers, **kwargs) -> None:
        if self.start_barrier is not None:
            self.start_barrier.wait(timeout=1)
        self.started = (buffers, kwargs)

    def register_read(self, *args) -> bool:
        self.registered = args
        return True

    def start_read(self, *args) -> None:
        self.read = args

    def poll_stats(self):
        return self.stats


class _NoMessageReceiver:
    def __init__(self) -> None:
        self.closed = False

    def recv_pyobj(self, _flags):
        raise zmq.Again()

    def close(self) -> None:
        self.closed = True


def test_raiden_loader_recognizes_encoder_backend():
    assert raiden_requested(["--encoder-transfer-backend", "raiden"])
    assert raiden_requested(["--encoder-transfer-backend=raiden"])
    assert not raiden_requested(["--encoder-transfer-backend", "jax_pull"])


def test_raiden_server_binds_the_produced_embedding(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer(
        "10.0.0.4",
        parallelism=3,
        timeout_s=12.0,
    )
    embedding = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)

    metadata = asyncio.run(transfer.publish("part-0:embedding", embedding))

    session = _FakeRaidenWrapper.instances[0]
    buffers, options = session.started
    assert len(buffers) == 1
    np.testing.assert_array_equal(buffers[0][0], embedding)
    assert buffers[0].shape == (1, 4, 3)
    assert options == {"max_blocks": 1, "num_slots": 1, "timeout_s": 12.0}
    assert session.registered == (
        "part-0:embedding",
        metadata["transfer_uuid"],
        [0],
    )
    assert metadata["transfer_address"] == session.endpoints
    assert metadata["transfer_host"] == "10.0.0.4"
    transfer.close()


def test_raiden_server_prepares_batch_transfers_concurrently(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    _FakeRaidenWrapper.start_barrier = threading.Barrier(2)
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", parallelism=2)

    async def publish_batch() -> None:
        await asyncio.gather(
            transfer.publish("part-0:embedding", jnp.zeros((2, 3))),
            transfer.publish("part-1:embedding", jnp.zeros((2, 3))),
        )

    try:
        asyncio.run(publish_batch())
    finally:
        _FakeRaidenWrapper.start_barrier = None
        transfer.close()

    assert len(_FakeRaidenWrapper.instances) == 2


def test_raiden_server_reaps_completed_sender(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4")
    asyncio.run(transfer.publish("part-0:embedding", jnp.zeros((2, 3))))
    _FakeRaidenWrapper.instances[0].stats = (["part-0:embedding"], [], [])

    async def stop_after_poll(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.asyncio.sleep",
        stop_after_poll,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transfer.release_completed())

    assert not transfer._sessions
    transfer.close()


def test_raiden_request_receives_into_matching_jax_buffer(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    register_future: Future[None] = Future()
    register_future.set_result(None)
    receiver = _NoMessageReceiver()
    backend = RaidenReceiverBackend(
        host="10.0.0.9",
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        parallelism=2,
        transfer_timeout_s=30.0,
    )
    request = PendingEncoderRequest(
        recv_req=TokenizedGenerateReqInput(rid="request-0"),
        started_at=0.0,
        receiver=receiver,
        register_future=register_future,
        accumulator=MultiModalEmbeddingData(1),
        backend=backend,
    )
    data = EmbeddingData(
        req_id="part-0",
        num_parts=1,
        part_idx=0,
        grid_dim=None,
        modality=Modality.IMAGE,
        embedding_shape=(2, 3),
        dtype="float32",
        transfer_id="part-0:embedding",
        transfer_uuid=17,
        transfer_address=[{"endpoint": "127.0.0.1:7788", "shards": [0]}],
        transfer_host="10.0.0.8",
        transfer_block_ids=[0],
    )

    request._start_receive(data)

    session = _FakeRaidenWrapper.instances[0]
    buffers, options = session.started
    assert buffers[0].shape == (1, 2, 3)
    assert buffers[0].dtype == jnp.float32
    receive_session = request.sessions[0][1]
    assert isinstance(receive_session, DeferredRaidenReceiveSession)
    session = receive_session._future.result(timeout=1)
    assert session.buffer.shape == (1, 2, 3)
    assert options == {"max_blocks": 1, "num_slots": 1, "timeout_s": 30.0}
    assert session.read == (
        "part-0:embedding",
        17,
        [{"endpoint": "10.0.0.8:7788", "shards": [0]}],
        [0],
        [0],
    )

    session.stats = ([], ["part-0:embedding"], [])
    result = request.poll()

    np.testing.assert_array_equal(result["embeddings"][Modality.IMAGE], np.zeros((2, 3)))
    request.close()
    backend.close()
    assert receiver.closed


def test_raiden_request_surfaces_receive_failure():
    transfer = mock.Mock()
    transfer.poll_stats.return_value = ([], [], ["part-0:embedding"])

    session = RaidenReceiveSession(
        "part-0:embedding",
        jnp.zeros((1, 1)),
        transfer,
    )

    with pytest.raises(RuntimeError, match="Raiden embedding transfer failed"):
        session.poll()
