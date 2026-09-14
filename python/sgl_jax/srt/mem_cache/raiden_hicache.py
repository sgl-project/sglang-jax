"""Local L2 adapter for tpu-sync. Physical chunk IDs never enter the tree.

The synchronous tree integration waits at producer/consumer boundaries. Native
submission remains asynchronous for a later overlap implementation. This pool
is dedicated to HiCache; mixing its auto allocator with PD staging is unsupported.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from sgl_jax.srt.mem_cache.cache_transfer import DevicePageSpan, TransferOperation


@dataclass
class _HostPage:
    rank: int
    chunk: int | None = None
    readers: int = 0
    writing: bool = False


class RaidenHostKVPool:
    """Logical reservations backed by the native auto allocator.

    Handles are monotonically increasing, so a stale handle cannot alias a new
    allocation. Capacities are per DP rank; physical blocks remain locked until
    the last logical owner releases them. All calls are on the scheduler thread.
    """

    def __init__(self, engines: dict[int, Any], pages_per_rank: int, device_pages: int):
        if not engines or pages_per_rank <= 0 or device_pages <= 0:
            raise ValueError("Raiden HiCache needs positive host/device capacity and engines")
        self.engines = engines
        self.pages_per_rank = pages_per_rank
        self.device_pages = device_pages
        self._pages: dict[int, _HostPage] = {}
        self._used = Counter()
        self._next_handle = 0
        self._operations: list[TransferOperation] = []
        self._error: Exception | None = None

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "Raiden HiCache failed; memory is quarantined until process exit"
            ) from self._error

    def _poison(self, exc: Exception) -> None:
        self._error = exc

    def available_size(self, dp_rank: int | None = None) -> int:
        self.check_health()
        if dp_rank is None:
            return self.total_size() - len(self._pages)
        if dp_rank not in self.engines:
            raise ValueError(f"Unknown Raiden DP rank {dp_rank}")
        return self.pages_per_rank - self._used[dp_rank]

    def total_size(self) -> int:
        return self.pages_per_rank * len(self.engines)

    def alloc(self, need_pages: int, dp_rank: int = 0) -> np.ndarray | None:
        if need_pages < 0:
            raise ValueError("Page count must be nonnegative")
        if need_pages > self.available_size(dp_rank):
            return None
        ids = list(range(self._next_handle, self._next_handle + need_pages))
        self._next_handle += need_pages
        self._pages.update({i: _HostPage(dp_rank) for i in ids})
        self._used[dp_rank] += need_pages
        return np.asarray(ids, dtype=np.int64)

    def _resolve(self, handles) -> list[_HostPage]:
        self.check_health()
        ids = [int(i) for i in handles]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate host page handles")
        try:
            return [self._pages[i] for i in ids]
        except KeyError as exc:
            raise ValueError(f"Unallocated or stale host handle {exc.args[0]}") from exc

    @property
    def failed(self) -> bool:
        return self._error is not None

    def pin(self, handles) -> None:
        for page in self._resolve(handles):
            page.readers += 1

    def unpin(self, handles) -> None:
        pages = self._resolve(handles)
        if any(page.readers <= 0 for page in pages):
            raise RuntimeError("Unbalanced host page pin")
        for page in pages:
            page.readers -= 1

    def has_inflight(self, handles) -> bool:
        return any(p.readers or p.writing for p in self._resolve(handles))

    def free(self, handles) -> None:
        ids = [int(i) for i in handles]
        pages = self._resolve(ids)
        if any(p.readers or p.writing for p in pages):
            raise RuntimeError("Cannot release host pages with in-flight transfers")
        by_rank = defaultdict(list)
        for p in pages:
            if p.chunk is not None:
                by_rank[p.rank].append(p.chunk)
        try:
            for rank, chunks in by_rank.items():
                self.engines[rank].unlock_blocks(chunks)
        except Exception as exc:
            self._poison(exc)
            raise
        for handle, p in zip(ids, pages):
            del self._pages[handle]
            self._used[p.rank] -= 1

    def _validate(self, device: DevicePageSpan, handles) -> list[_HostPage]:
        self._operations = [op for op in self._operations if not op.done()]
        pages = self._resolve(handles)
        if device.dp_rank not in self.engines:
            raise ValueError(f"Unknown Raiden DP rank {device.dp_rank}")
        if len(device.pages) != len(pages):
            raise ValueError("Device and host page counts differ")
        if any(p.rank != device.dp_rank for p in pages):
            raise ValueError("Host and device DP ranks differ")
        if len(set(device.pages)) != len(device.pages):
            raise ValueError("Duplicate device pages")
        if any(p < 0 or p >= self.device_pages for p in device.pages):
            raise ValueError("Device page out of range")
        return pages

    def _track(self, future, release) -> TransferOperation:
        op = TransferOperation(future, release, self._poison)
        self._operations.append(op)
        return op

    def submit_backup(self, device: DevicePageSpan, handles) -> TransferOperation:
        pages = self._validate(device, handles)
        if any(p.chunk is not None or p.writing or p.readers for p in pages):
            raise ValueError("Backup requires empty reserved host pages")
        if not pages:
            return TransferOperation(None)
        for p in pages:
            p.writing = True
        try:
            chunks, future = self.engines[device.dp_rank].d2h_auto_allocate(list(device.pages))
            if len(chunks) != len(pages) or len(set(chunks)) != len(chunks):
                raise RuntimeError("Raiden returned invalid host allocations")
            for p, chunk in zip(pages, chunks):
                p.chunk = int(chunk)
        except Exception as exc:
            self._poison(exc)
            raise

        def complete():
            for p in pages:
                p.writing = False

        return self._track(future, complete)

    def submit_restore(self, handles, device: DevicePageSpan) -> TransferOperation:
        pages = self._validate(device, handles)
        if any(p.chunk is None or p.writing for p in pages):
            raise ValueError("Restore requires completed host pages")
        if not pages:
            return TransferOperation(None)
        for p in pages:
            p.readers += 1
        try:
            future = self.engines[device.dp_rank].h2d([p.chunk for p in pages], list(device.pages))
        except Exception as exc:
            self._poison(exc)
            raise

        def complete():
            for p in pages:
                p.readers -= 1

        return self._track(future, complete)

    def drain(self) -> None:
        # Await every submitted operation even if one fails. Keep failed
        # resources quarantined: Await errors do not certify native quiescence.
        first_error = None
        for op in self._operations:
            try:
                op.wait()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
        self.check_health()
        self._operations.clear()

    def precompile_transfers(self) -> None:
        pass


class RaidenHiCacheController:
    """Native transfer submission; the tree owns the correctness barriers."""

    direct_transfers = True

    def __init__(self, host_pool: RaidenHostKVPool, device_pool, *, check_registered_buffers=False):
        self.host_pool = host_pool
        self.device_pool = device_pool
        self._registered_buffers = self._buffer_addresses() if check_registered_buffers else None

    def _buffer_addresses(self):
        return tuple(
            (
                tuple(buffer.shape),
                str(buffer.dtype),
                tuple(
                    (shard.device.id, shard.data.unsafe_buffer_pointer())
                    for shard in buffer.addressable_shards
                ),
            )
            for buffer in self.device_pool.kv_buffer
        )

    def prepare_transfer(self) -> None:
        self.host_pool.check_health()
        jax.block_until_ready(self.device_pool.kv_buffer)
        if (
            self._registered_buffers is not None
            and self._buffer_addresses() != self._registered_buffers
        ):
            error = RuntimeError("Registered Raiden KV allocation changed; restart is required")
            self.host_pool._poison(error)
            raise error

    def submit_backup(self, device: DevicePageSpan, handles) -> TransferOperation:
        return self.host_pool.submit_backup(device, handles)

    def submit_restore(self, handles, device: DevicePageSpan) -> TransferOperation:
        return self.host_pool.submit_restore(handles, device)

    def drain_pending(self) -> None:
        self.host_pool.drain()

    def drain_loads(self) -> None:
        self.host_pool.drain()

    def check_write_status(self) -> None:
        self.host_pool.check_health()

    def has_inflight(self, handles) -> bool:
        return self.host_pool.has_inflight(handles)

    def evict_callback(self, handles) -> None:
        self.host_pool.free(handles)

    def shutdown(self) -> None:
        self.host_pool.drain()


def create_raiden_hicache(device_pool, num_pages: int, dp_size: int):
    from sgl_jax.raiden import get_raiden_kv_cache_manager, require_raiden_preloaded
    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import (
        _split_kv_caches_by_dp_rank,
    )

    require_raiden_preloaded()
    if num_pages < dp_size:
        raise ValueError("Raiden HiCache requires at least one host page per DP rank")
    if jax.process_count() != 1 or any(d.platform != "tpu" for d in jax.local_devices()):
        raise ValueError("Raiden HiCache currently requires a single-host TPU runtime")
    manager_cls = get_raiden_kv_cache_manager()
    jax.block_until_ready(device_pool.kv_buffer)
    rank_caches = _split_kv_caches_by_dp_rank(list(device_pool.kv_buffer), dp_size)
    capacity = num_pages // dp_size
    engines = {
        rank: manager_cls(
            kv_caches=caches,
            local_control_port=0,
            host_blocks_to_allocate=capacity,
            unsafe_skip_buffer_lock=True,
            parallelism=1,
        )
        for rank, caches in rank_caches.items()
    }
    host_pool = RaidenHostKVPool(
        engines, capacity, int(device_pool.kv_buffer[0].shape[0]) // dp_size
    )
    return host_pool, RaidenHiCacheController(host_pool, device_pool, check_registered_buffers=True)
