from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.client import EncoderClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData

logger = logging.getLogger(__name__)


def _embedding_mib(shape, dtype) -> float:
    """Approximate embedding payload size in MiB from shape + dtype."""
    if shape is None or dtype is None:
        return 0.0
    count = 1
    for dim in shape:
        count *= int(dim)
    itemsize = jnp.dtype(dtype).itemsize
    return count * itemsize / (2**20)


class SimEncoderServerTransfer:
    """Stand-in for ``RaidenEncoderServerTransfer`` under --simulate-compute.

    Publishes nothing over the wire: the receiver rebuilds a zero embedding
    from the shape/dtype carried in ``EmbeddingData``. The modeled transfer
    latency lives on the receiver poll side (see ``SimReceiveSession``), so the
    encoder's batch worker is not blocked here — matching Raiden's async
    register-and-return behavior.
    """

    def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]:
        if embedding.ndim != 2 or embedding.shape[0] <= 0:
            raise ValueError("Sim embedding must be a non-empty matrix")
        # No transfer endpoints: the receiver reconstructs zeros from shape/dtype.
        return {"transfer_id": transfer_id}

    async def release_completed(self) -> None:
        # No real sessions to reap; stay alive for the server lifetime.
        while True:
            await asyncio.sleep(3600)

    def release(self, transfer_id: str) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class SimReceiveSession:
    buffer: jax.Array
    ready_at: float

    def poll(self) -> jax.Array | None:
        if time.monotonic() < self.ready_at:
            return None
        return self.buffer

    def close(self) -> None:
        return None


class SimReceiverBackend:
    """Rebuilds a zero embedding and models the transfer time as a poll delay."""

    def __init__(
        self, sharding: jax.sharding.Sharding, ms_per_mb: float, rtt_ms: float = 0.0
    ) -> None:
        self._sharding = sharding
        self._ms_per_mb = float(ms_per_mb)
        self._rtt_ms = float(rtt_ms)

    def start(self, data: EmbeddingData) -> SimReceiveSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(int(dim) for dim in data.shape)
        if len(shape) != 2 or shape[0] <= 0:
            raise ValueError("Sim embedding must be a non-empty matrix")
        buffer = jax.device_put(jnp.zeros(shape, dtype=jnp.dtype(data.dtype)), self._sharding)
        # One-way network RTT (embedding hop) + size-proportional bandwidth term.
        delay_s = (self._rtt_ms + self._ms_per_mb * _embedding_mib(shape, data.dtype)) / 1000.0
        return SimReceiveSession(buffer=buffer, ready_at=time.monotonic() + delay_s)

    def close(self) -> None:
        return None


def create_sim_client(server_args, mesh: jax.sharding.Mesh) -> EncoderClient:
    """EncoderClient wired with the simulated receiver backend (no Raiden)."""
    # The sim topology is always local, so bind the ZMQ receiver on loopback
    # rather than resolving a routable host IP (which fails on machines whose
    # hostname does not resolve, e.g. many laptops).
    host = server_args.disaggregation_host_ip or "127.0.0.1"
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    control_timeout = server_args.encoder_control_timeout_seconds
    executor = ThreadPoolExecutor(max_workers=max(1, server_args.disaggregation_channel_number))
    backend = SimReceiverBackend(
        sharding,
        server_args.simulate_transfer_ms_per_mb,
        server_args.simulate_network_rtt_ms,
    )
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        executor=executor,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
