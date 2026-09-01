from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.client import EncoderClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


def _uuid_to_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


def _normalize_endpoint(endpoint: object, peer_host: str) -> str:
    value = str(endpoint)
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Raiden endpoint: {value!r}") from exc
    if not 0 < port <= 65535:
        raise ValueError(f"invalid Raiden endpoint port: {port}")
    host = host.strip("[]")
    if host in _LOCAL_ENDPOINT_HOSTS:
        host = peer_host
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}"


def _normalize_endpoints(endpoints: object, peer_host: str) -> list[dict[str, Any]]:
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("Raiden encoder did not publish endpoint descriptors")
    result = []
    for item in endpoints:
        if not isinstance(item, Mapping):
            raise TypeError("Raiden endpoint descriptor must be a mapping")
        shards = item.get("shards", [])
        if not isinstance(shards, list):
            raise TypeError("Raiden endpoint shards must be a list")
        result.append(
            {
                "endpoint": _normalize_endpoint(item.get("endpoint", ""), peer_host),
                "shards": [int(shard) for shard in shards],
            }
        )
    return result


class RaidenEncoderServerTransfer:
    """Binds each produced embedding to its own Raiden transfer session."""

    def __init__(
        self,
        host_ip: str,
        *,
        parallelism: int = 1,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        require_raiden_preloaded()
        self._host_ip = host_ip
        self._parallelism = max(1, int(parallelism))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._sessions: dict[str, RaidenTransferWrapper] = {}
        self._preparing: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=self._parallelism)

    async def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]:
        if transfer_id in self._sessions or transfer_id in self._preparing:
            raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
        if embedding.ndim != 2 or embedding.shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")

        self._preparing.add(transfer_id)
        try:
            session, metadata = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._prepare,
                transfer_id,
                embedding,
            )
        finally:
            self._preparing.discard(transfer_id)
        self._sessions[transfer_id] = session
        return metadata

    def _prepare(
        self,
        transfer_id: str,
        embedding: jax.Array,
    ) -> tuple[RaidenTransferWrapper, dict[str, Any]]:
        # Treat one embedding as one physical major slice. The leading transfer
        # axis makes TPU tile padding part of the slice instead of row stride.
        buffer = embedding[jnp.newaxis, ...]
        block_ids = [0]
        transfer_uuid = _uuid_to_int(transfer_id)
        session = RaidenTransferWrapper(
            self._host_ip,
            0,
            parallelism=self._parallelism,
        )
        session.start(
            [buffer],
            max_blocks=1,
            num_slots=1,
            timeout_s=self._timeout_s,
        )
        if not session.register_read(transfer_id, transfer_uuid, block_ids):
            raise RuntimeError(f"Raiden rejected encoder transfer {transfer_id!r}")
        return (
            session,
            {
                "transfer_id": transfer_id,
                "transfer_uuid": transfer_uuid,
                "transfer_address": session.endpoints,
                "transfer_host": self._host_ip,
                "transfer_block_ids": block_ids,
            },
        )

    async def release_completed(self) -> None:
        while True:
            for transfer_id, session in list(self._sessions.items()):
                try:
                    sent, _, _ = session.poll_stats()
                except Exception:
                    logger.exception("Raiden encoder sender poll failed for %s", transfer_id)
                    self._sessions.pop(transfer_id, None)
                    continue
                if transfer_id in sent:
                    self._sessions.pop(transfer_id, None)
            await asyncio.sleep(self._poll_interval_s)

    def release(self, transfer_id: str) -> None:
        self._sessions.pop(transfer_id, None)

    def close(self) -> None:
        self._sessions.clear()
        self._executor.shutdown(cancel_futures=True)


@dataclass(slots=True)
class RaidenReceiveSession:
    transfer_id: str
    buffer: jax.Array
    transfer: RaidenTransferWrapper

    def poll(self) -> jax.Array | None:
        _, received, failed = self.transfer.poll_stats()
        if self.transfer_id in failed:
            raise RuntimeError(f"Raiden embedding transfer failed: {self.transfer_id}")
        if self.transfer_id in received:
            return self.buffer[0]
        return None

    def close(self) -> None:
        # KVCacheManager has no cancellation API. Dropping the per-part manager
        # releases its listener and buffer holds after the current poll returns.
        return None


class DeferredRaidenReceiveSession:
    """Expose a non-blocking session while Raiden setup runs off-loop."""

    def __init__(self, future: Future[RaidenReceiveSession]) -> None:
        self._future = future
        self._session: RaidenReceiveSession | None = None
        self._closed = False

    def poll(self) -> jax.Array | None:
        if self._closed:
            return None
        if self._session is None:
            if not self._future.done():
                return None
            self._session = self._future.result()
        return self._session.poll()

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
        elif not self._future.cancel():
            self._future.add_done_callback(self._close_session)

    @staticmethod
    def _close_session(future: Future[RaidenReceiveSession]) -> None:
        if future.cancelled():
            return
        try:
            future.result().close()
        except Exception:
            logger.exception("Deferred Raiden receiver setup failed during cleanup")


class RaidenReceiverBackend:
    def __init__(
        self,
        host: str,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        transfer_timeout_s: float,
    ) -> None:
        self._host = host
        self._sharding = sharding
        self._parallelism = max(1, int(parallelism))
        self._transfer_timeout_s = float(transfer_timeout_s)
        # Receiver setup contends on JAX/Raiden initialization. Serialize it
        # off-loop while keeping Raiden's data-plane parallelism unchanged.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, data: EmbeddingData) -> DeferredRaidenReceiveSession:
        return DeferredRaidenReceiveSession(self._executor.submit(self._start, data))

    def _start(self, data: EmbeddingData) -> RaidenReceiveSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(int(dim) for dim in data.shape)
        if len(shape) != 2 or shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")

        transfer_id = getattr(data, "transfer_id", None)
        transfer_uuid = getattr(data, "transfer_uuid", None)
        remote_block_ids = getattr(data, "transfer_block_ids", None)
        endpoints = getattr(data, "transfer_address", None)
        if not transfer_id or not isinstance(transfer_uuid, int):
            raise ValueError("Raiden transfer identity is incomplete")
        if not isinstance(remote_block_ids, list) or len(remote_block_ids) != 1:
            raise ValueError("Raiden block metadata does not match embedding shape")
        remote_block_ids = [int(block_id) for block_id in remote_block_ids]
        if len(set(remote_block_ids)) != len(remote_block_ids) or any(
            block_id < 0 for block_id in remote_block_ids
        ):
            raise ValueError("Raiden remote block IDs must be unique and non-negative")

        transfer_host = getattr(data, "transfer_host", None)
        if str(transfer_host).strip("[]") in _LOCAL_ENDPOINT_HOSTS:
            transfer_host = None
        if not transfer_host:
            raise ValueError("Raiden transfer_host is required")
        remote_endpoints = _normalize_endpoints(endpoints, transfer_host)

        buffer = jax.device_put(jnp.zeros((1, *shape), dtype=jnp.dtype(data.dtype)), self._sharding)
        jax.block_until_ready(buffer)
        local_block_ids = [0]
        transfer = RaidenTransferWrapper(
            self._host,
            0,
            parallelism=self._parallelism,
        )
        transfer.start(
            [buffer],
            max_blocks=1,
            num_slots=1,
            timeout_s=self._transfer_timeout_s,
        )
        transfer.start_read(
            transfer_id,
            transfer_uuid,
            remote_endpoints,
            remote_block_ids,
            local_block_ids,
        )
        return RaidenReceiveSession(transfer_id, buffer, transfer)

    def close(self) -> None:
        self._executor.shutdown(cancel_futures=True)


RaidenEncoderClient = EncoderClient


def create_raiden_client(
    server_args,
    mesh: jax.sharding.Mesh,
) -> EncoderClient:
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

    require_raiden_preloaded()
    host = resolve_host_ip(server_args.disaggregation_host_ip)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    control_timeout = server_args.encoder_control_timeout_seconds
    transfer_timeout = server_args.encoder_request_timeout_seconds
    if transfer_timeout <= 0:
        raise ValueError("Raiden requires a positive encoder request timeout")
    executor = ThreadPoolExecutor(max_workers=server_args.disaggregation_channel_number)
    backend = RaidenReceiverBackend(
        host=host,
        sharding=sharding,
        parallelism=server_args.disaggregation_channel_number,
        transfer_timeout_s=transfer_timeout,
    )
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        executor=executor,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
