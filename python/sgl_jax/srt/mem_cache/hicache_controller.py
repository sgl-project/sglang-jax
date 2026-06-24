"""HiCache control plane for L1 (HBM) <-> L2 (host pinned) KV transfers.

The controller is deliberately thin: it owns the *scheduling* of transfers
(async D2H write, sync H2D load, completion polling) but not the *mechanics*.
All data movement goes through the :class:`HostKVPool` page-addressed
primitives (``stage_backup`` / ``flush_backup`` / ``copy_to_device``); the
controller never calls ``jax.device_put`` and never touches a host ``jax.Array``.
Only ``int`` page ids (host page slots and device page indices) cross its
boundary, which is what keeps the storage backend raiden-switchable — raiden's
block interface is page-grained too — with zero controller changes (decision D4).

write is async, load is sync: the HiCache+PD spike measured a strong D2H/H2D
bandwidth asymmetry (~2.8 GB/s D2H vs ~45.7 GB/s H2D), so the slow backup path
is offloaded to a worker thread (JAX releases the GIL inside device_put, so it
overlaps the main-thread forward) while the fast read-back path stays inline.
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
    """Schedules L1<->L2 KV transfers over a :class:`HostKVPool`."""

    def __init__(
        self,
        host_pool: HostKVPool,
        device_pool: Any,
        device_allocator: Any = None,
    ) -> None:
        self._host_pool = host_pool
        self._device_pool = device_pool
        self._device_allocator = device_allocator
        # 1 worker: D2H copies are serialized but still overlap the main thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hicache-d2h")
        self._pending: list[Future] = []
        # Host page ids whose async D2H has not finished. Guards against the
        # page being freed/re-alloced while the worker is still writing it (the
        # write is async, so the page is not single-owner until the copy lands).
        self._inflight_lock = threading.Lock()
        self._inflight: set[int] = set()
        # Host page ids whose async H2D stage (slow host->device device_put) has
        # not finished. A page must be staged before flush_load() can scatter it
        # into the KV buffer. Mirrors _inflight (D2H) on the load path.
        self._inflight_load_lock = threading.Lock()
        self._inflight_load: set[int] = set()
        self._pending_load: list[Future] = []
        self._shutdown_done = False

    def write(self, device_indices: list[int], host_buffer_ids: list[int]) -> Future:
        """Async D2H: copy device page(s) into reserved host page slot(s).

        Both ``device_indices`` and ``host_buffer_ids`` are PAGE ids (the radix
        layer folds its token-level ``node.value`` to pages before calling in).
        ``device_indices[i]`` pairs with ``host_buffer_ids[i]``. Returns the
        submitted :class:`Future` so the caller (tree cache) can associate it
        with the node being backed up; completion is also observed via
        :meth:`check_write_status` / :meth:`drain_pending`.

        The device-side gather runs *synchronously here* (on the caller's
        thread, which owns the KV buffer): the forward path rewrites
        ``kv_buffer`` with donation every step, so reading it from the worker
        thread races that deletion ("Array has been deleted"). stage_backup
        materializes the pages before we return; only the slow host transfer
        (flush_backup) is offloaded to the worker.
        """
        host_buffer_ids = list(host_buffer_ids)
        self._host_pool.stage_backup(list(device_indices), host_buffer_ids)
        future = self._executor.submit(self._do_d2h, host_buffer_ids)
        with self._inflight_lock:
            self._inflight.update(host_buffer_ids)
        self._pending.append(future)
        return future

    def load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """Sync H2D: stage host page slot(s) onto device, then scatter into the
        device KV buffer. Both arguments are PAGE ids; ``host_buffer_ids[i]``
        pairs with ``device_indices[i]``.

        This inline form (stage + flush on the caller's thread) is the
        donation-safe default: the scatter touches ``kv_buffer`` and runs on the
        forward thread between steps. The overlap scheduler instead splits this
        into an async :meth:`stage_load` (slow device_put, off-thread, overlaps
        forward) plus a :meth:`flush_load` (cheap kernel scatter, in a
        donation-safe window) to hide the slow transfer behind compute.
        """
        host_buffer_ids = list(host_buffer_ids)
        self._host_pool.stage_load(host_buffer_ids)
        self._host_pool.flush_load(host_buffer_ids, list(device_indices))

    def stage_load(self, host_buffer_ids: list[int]) -> Future:
        """Async H2D stage: device_put host page slot(s) into device staging.

        Submits the slow host->device copy to the worker (it never touches
        ``kv_buffer``, so it overlaps the main-thread forward). The cheap scatter
        into the KV buffer is done later by :meth:`flush_load` in a donation-safe
        window. Returns the submitted Future. ``host_buffer_ids`` are PAGE ids.
        """
        host_buffer_ids = list(host_buffer_ids)
        future = self._executor.submit(self._do_stage_load, host_buffer_ids)
        with self._inflight_load_lock:
            self._inflight_load.update(host_buffer_ids)
        self._pending_load.append(future)
        return future

    def flush_load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """Sync H2D scatter: write staged page(s) into the device KV buffer.

        MUST run on the buffer-owning thread in a donation-safe window. Requires
        the pages to have been staged (:meth:`stage_load`) and that stage to have
        completed; raises if a page is still in-flight.
        """
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
        """Non-blocking poll: drop completed D2H futures, re-raise any error.

        Futures are removed before the error is raised, so a failed transfer is
        surfaced exactly once and never blocks subsequent draining.
        """
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
        """Block until all pending D2H complete (shutdown / consistency points).

        Clears the pending list first so a raised error does not leave a failed
        future to be re-raised on the next drain.
        """
        pending, self._pending = self._pending, []
        first_exc = None
        for f in pending:
            exc = f.exception()  # blocks until the future is done; does not raise
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def check_load_status(self) -> None:
        """Non-blocking poll of async H2D stages: drop completed futures, re-raise
        any error (mirrors :meth:`check_write_status`)."""
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
        """Block until all pending H2D stages complete (mirrors
        :meth:`drain_pending`)."""
        pending, self._pending_load = self._pending_load, []
        first_exc = None
        for f in pending:
            exc = f.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def evict_callback(self, host_buffer_ids: list[int]) -> None:
        """L2 eviction: free page slot(s). This stage discards data (no L3 yet).

        ``host_buffer_ids`` are host PAGE ids. Rejects any page with an in-flight
        D2H write — releasing it would let a later alloc hand the page out while
        the worker is still writing it, silently corrupting the new owner's KV.
        Drain or wait first. Frees the whole batch in one :meth:`HostKVPool.free`
        call (deduped) rather than per-id.
        """
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
