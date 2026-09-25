"""Request lifecycle for Raiden sends from one global embedding pool."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

import jax
import numpy as np

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper


@dataclass(slots=True)
class _Part:
    rank: int
    transfer_id: str
    token_indices: np.ndarray
    source_rows: np.ndarray
    pages: tuple[int, ...] = ()
    registered: bool = False
    done: bool = False


@dataclass(slots=True)
class _Reservation:
    transfer_id: str
    parts: list[_Part] = field(default_factory=list)
    write: jax.Array | None = None  # actually a jax.ArrayFuture
    cancelled: bool = False


class RaidenEncoderServerTransfer:
    def __init__(self, host_ip, pool, *, parallelism=1, pool_size=32, timeout_s=300):
        require_raiden_preloaded()
        self._max_inflight = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self.pool = pool
        self.num_lanes = pool.num_shards
        self._pages_per_lane = pool.num_pages // self.num_lanes
        # Like PD ranks, each endpoint registers a view of the same global buffer.
        buffers = (
            [part.data for part in self.pool.buffer.addressable_shards]
            if self.num_lanes > 1
            else [self.pool.buffer]
        )
        self._transfers = []
        for buffer in buffers:
            transfer = RaidenTransferWrapper(host_ip, 0, parallelism=parallelism)
            transfer.start(
                [buffer],
                max_blocks=buffer.shape[0],
                num_slots=self._max_inflight,
                timeout_s=self._timeout_s,
            )
            self._transfers.append(transfer)
        self._reservations: dict[str, _Reservation] = {}
        self._lock = threading.Lock()
        self._closed = False

    def batch_capacity(self, token_count):
        pages = max(1, (token_count + self.pool.page_size - 1) // self.pool.page_size)
        return max(1, min(self._max_inflight, self._pages_per_lane // pages))

    def reserve_batch_sync(self, transfer_ids, token_counts, *, output_indices):
        """Reserve a batch atomically; all reservation objects stay inside this backend."""
        if len(transfer_ids) != len(token_counts) or any(count <= 0 for count in token_counts):
            raise ValueError("Invalid encoder transfer IDs or token counts")
        if not transfer_ids:
            return
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id")
        rows = np.asarray(output_indices)
        if (
            rows.ndim != 1
            or rows.size < sum(token_counts)
            or rows.size % self.num_lanes
            or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows[: sum(token_counts)] < 0)
            or np.any(rows[: sum(token_counts)] >= rows.size)
        ):
            raise ValueError("Invalid encoder output index layout")
        local_capacity = rows.size // self.num_lanes
        reservations = []
        offset = 0
        for transfer_id, count in zip(transfer_ids, token_counts, strict=True):
            request = _Reservation(transfer_id)
            request_rows = rows[offset : offset + count]
            for rank in range(self.num_lanes):
                indices = np.flatnonzero(request_rows // local_capacity == rank)
                if indices.size:
                    request.parts.append(
                        _Part(rank, f"{transfer_id}:lane:{rank}", indices, request_rows[indices])
                    )
            reservations.append(request)
            offset += count
        parts = [part for request in reservations for part in request.parts]
        needed = [
            sum(self.pool.pages_needed(len(p.token_indices)) for p in parts if p.rank == rank)
            for rank in range(self.num_lanes)
        ]
        slots = [sum(p.rank == rank for p in parts) for rank in range(self.num_lanes)]
        if max(needed) > self._pages_per_lane or max(slots) > self._max_inflight:
            raise ValueError("encoder batch exceeds Raiden pool capacity")
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Raiden encoder transfer is closed")
                self._reap_locked()
                if any(key in self._reservations for key in transfer_ids):
                    raise ValueError("duplicate Raiden transfer_id")
                available = (
                    self.pool.available_pages_by_shard()
                    if self.num_lanes > 1
                    else [self.pool.available_pages]
                )
                active = [p for r in self._reservations.values() for p in r.parts if p.pages]
                if all(
                    needed[i] <= available[i]
                    and slots[i] + sum(p.rank == i for p in active) <= self._max_inflight
                    for i in range(self.num_lanes)
                ):
                    for part in parts:
                        part.pages = self.pool.allocate(
                            len(part.token_indices), shard=part.rank if self.num_lanes > 1 else None
                        )
                    self._reservations.update((r.transfer_id, r) for r in reservations)
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Raiden encoder transfer capacity")
            time.sleep(0.001)

    def stage_batch_sync(self, transfer_ids, embeddings):
        if not transfer_ids:
            return
        try:
            with self._lock:
                reservations = [self._reservations[key] for key in transfer_ids]
                for request in reservations:
                    self._check_active(request)
                parts = [part for request in reservations for part in request.parts]
                write = self.pool.write(
                    jax.device_put(embeddings, self.pool.sharding),
                    [part.pages for part in parts],
                    tuple(len(part.token_indices) for part in parts),
                    source_rows=np.concatenate([part.source_rows for part in parts]),
                )
                for request in reservations:
                    request.write = write
        except BaseException:
            for transfer_id in transfer_ids:
                self.release(transfer_id)
            raise

    def publish_batch_sync(self, transfer_ids):
        metadata = []
        try:
            with self._lock:
                reservations = [self._reservations[key] for key in transfer_ids]
            for request in reservations:
                if request.write is None:
                    raise RuntimeError("Raiden reservation has no staged copy")
                request.write.block_until_ready()
                with self._lock:
                    self._check_active(request)
                    parts = []
                    for part in request.parts:
                        transfer = self._transfers[part.rank]
                        digest = hashlib.blake2b(part.transfer_id.encode(), digest_size=8).digest()
                        uuid = int.from_bytes(digest, "big") & ((1 << 50) - 1)
                        blocks = [page - part.rank * self._pages_per_lane for page in part.pages]
                        if not transfer.register_read(part.transfer_id, uuid, blocks):
                            raise RuntimeError("Raiden rejected encoder pages")
                        part.registered = True
                        parts.append(
                            {
                                "transfer_id": part.transfer_id,
                                "transfer_uuid": uuid,
                                "transfer_address": transfer.endpoints,
                                "transfer_host": transfer.host_ip,
                                "transfer_block_ids": blocks,
                                "transfer_page_size": self.pool.page_size,
                                "token_indices": part.token_indices.tolist(),
                            }
                        )
                    metadata.append(
                        {
                            "transfer_id": request.transfer_id,
                            "transfer_page_size": self.pool.page_size,
                            "transfer_parts": parts,
                        }
                    )
            return metadata
        except BaseException:
            for transfer_id in transfer_ids:
                self.release(transfer_id)
            raise

    def release(self, transfer_id):
        with self._lock:
            request = self._reservations.get(transfer_id)
            if request is not None:
                request.cancelled = True
                self._reclaim_locked()

    def _check_active(self, request):
        if (
            self._closed
            or request.cancelled
            or self._reservations.get(request.transfer_id) is not request
            or any(part.registered or not part.pages for part in request.parts)
        ):
            raise RuntimeError(f"Raiden reservation is no longer active: {request.transfer_id}")

    def _reap_locked(self):
        for transfer in self._transfers:
            completed, _, _ = transfer.poll_stats()
            sent = set(completed)
            for request in self._reservations.values():
                for part in request.parts:
                    part.done |= part.transfer_id in sent
        self._reclaim_locked()

    def _reclaim_locked(self):
        for key, request in list(self._reservations.items()):
            if request.write is not None and not request.write.is_ready():
                continue
            for part in request.parts:
                # A cancelled request still owns registered pages until Raiden finishes reading.
                if part.pages and (part.done or (request.cancelled and not part.registered)):
                    self.pool.release(part.pages)
                    part.pages = ()
            if not any(part.pages for part in request.parts):
                del self._reservations[key]

    def close(self):
        with self._lock:
            self._closed = True
