"""HiCache control plane for L1 (HBM) <-> L2 (host pinned) KV transfers.

Thin scheduling layer: async D2H (write) offloaded to a worker thread because
D2H bandwidth is ~16x slower than H2D (~2.8 vs ~45.7 GB/s); sync H2D (load)
stays inline. All data movement goes through HostKVPool page-addressed primitives;
the controller only passes int page ids, keeping the storage backend swappable.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.host_kv_pool import HostKVPool

logger = logging.getLogger(__name__)


class HiCacheController:
    """Schedules L1<->L2 KV transfers over a HostKVPool."""

    def __init__(
        self,
        host_pool: HostKVPool,
        device_pool: Any,
        device_allocator: Any = None,
    ) -> None:
        self._host_pool = host_pool
        self._device_pool = device_pool
        self._device_allocator = device_allocator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hicache-d2h")
        self._pending: list[Future] = []
        # Guards against freeing a host page while the worker is still writing it.
        self._inflight_lock = threading.Lock()
        self._inflight: set[int] = set()
        # Mirrors _inflight for the async H2D load path.
        self._inflight_load_lock = threading.Lock()
        self._inflight_load: set[int] = set()
        self._pending_load: list[Future] = []
        self._shutdown_done = False

    def write(self, device_indices: list[int], host_buffer_ids: list[int]) -> Future:
        """Async D2H: copy device pages into reserved host page slots.

        The device-side gather (stage_backup) runs synchronously on the caller's
        thread because the forward path donates kv_buffer every step — reading it
        from the worker would race that deletion. Only the slow host transfer
        (flush_backup) is offloaded.
        """
        host_buffer_ids = list(host_buffer_ids)
        self._host_pool.stage_backup(list(device_indices), host_buffer_ids)
        future = self._executor.submit(self._do_d2h, host_buffer_ids)
        with self._inflight_lock:
            self._inflight.update(host_buffer_ids)
        self._pending.append(future)
        return future

    def load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """Sync H2D: stage host pages onto device, then scatter into kv_buffer.

        Donation-safe default (stage + flush on the forward thread). The overlap
        scheduler can instead split into async stage_load + flush_load to hide
        the slow device_put behind compute.
        """
        host_buffer_ids = list(host_buffer_ids)
        self._host_pool.stage_load(host_buffer_ids)
        self._host_pool.flush_load(host_buffer_ids, list(device_indices))

    def stage_load(self, host_buffer_ids: list[int]) -> Future:
        """Async H2D stage: offload host->device copy to the worker thread.

        Never touches kv_buffer, so it safely overlaps the main-thread forward.
        The cheap scatter is deferred to flush_load.
        """
        host_buffer_ids = list(host_buffer_ids)
        future = self._executor.submit(self._do_stage_load, host_buffer_ids)
        with self._inflight_load_lock:
            self._inflight_load.update(host_buffer_ids)
        self._pending_load.append(future)
        return future

    def flush_load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """Complete the H2D scatter into kv_buffer. Must run donation-safe."""
        host_buffer_ids = list(host_buffer_ids)
        with self._inflight_load_lock:
            busy = [b for b in host_buffer_ids if b in self._inflight_load]
        if busy:
            raise RuntimeError(
                f"flush_load of page id(s) {busy} with in-flight stage_load; "
                f"call check_load_status()/drain_loads() before flushing"
            )
        self._host_pool.flush_load(host_buffer_ids, list(device_indices))

    def _do_d2h(self, host_buffer_ids: list[int]) -> None:
        try:
            self._host_pool.flush_backup(host_buffer_ids)
        finally:
            with self._inflight_lock:
                self._inflight.difference_update(host_buffer_ids)

    def _do_stage_load(self, host_buffer_ids: list[int]) -> None:
        try:
            self._host_pool.stage_load(host_buffer_ids)
        finally:
            with self._inflight_load_lock:
                self._inflight_load.difference_update(host_buffer_ids)

    def check_write_status(self) -> None:
        """Non-blocking poll: drop completed D2H futures, re-raise any error."""
        done = [f for f in self._pending if f.done()]
        first_exc = None
        for f in done:
            self._pending.remove(f)
            exc = f.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def drain_pending(self) -> None:
        """Block until all pending D2H transfers complete."""
        pending, self._pending = self._pending, []
        first_exc = None
        for f in pending:
            exc = f.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def check_load_status(self) -> None:
        """Non-blocking poll of async H2D stages (mirrors check_write_status)."""
        done = [f for f in self._pending_load if f.done()]
        first_exc = None
        for f in done:
            self._pending_load.remove(f)
            exc = f.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def drain_loads(self) -> None:
        """Block until all pending H2D stages complete (mirrors drain_pending)."""
        pending, self._pending_load = self._pending_load, []
        first_exc = None
        for f in pending:
            exc = f.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def evict_callback(self, host_buffer_ids: list[int]) -> None:
        """Free host page slots. Rejects pages with in-flight D2H writes."""
        with self._inflight_lock:
            busy = [b for b in host_buffer_ids if b in self._inflight]
        if busy:
            raise RuntimeError(
                f"evict of page id(s) {busy} with in-flight D2H write; "
                f"call drain_pending() (or wait for the write) before releasing"
            )
        self._host_pool.free(host_buffer_ids)

    def pending_count(self) -> int:
        return len(self._pending)

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self.drain_pending()
            self.drain_loads()
        except Exception:  # noqa: BLE001 - cleanup path must not mask shutdown
            logger.warning("HiCacheController shutdown: pending transfer raised", exc_info=True)
        finally:
            self._executor.shutdown(wait=True)
