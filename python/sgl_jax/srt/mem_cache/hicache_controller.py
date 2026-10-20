"""HiCache control plane for L1 (HBM) <-> L2 (host pinned) KV transfers.

Thin scheduling layer over the HostKVPool page-addressed primitives; the
controller only passes int page ids, keeping the storage backend swappable.
Every device launch is issued through one injectable ``launcher`` (the forward
thread in overlap mode) so multi-host meshes see a deterministic launch order;
see :class:`HiCacheController` for the rationale.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

import jax

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.host_kv_pool import HostKVPool

logger = logging.getLogger(__name__)


def _inline_launcher(fn: Callable[[], Any]) -> Future:
    """Run ``fn`` on the calling thread and wrap its outcome in a resolved Future."""
    future: Future = Future()
    try:
        future.set_result(fn())
    except BaseException as e:  # noqa: BLE001 - propagate via the future like an executor would
        future.set_exception(e)
    return future


class HiCacheController:
    """Schedules L1<->L2 KV transfers over a HostKVPool.

    Multi-host safety
    -----------------
    On a multi-controller TPU mesh every process must enqueue the same device
    programs in the same order (each launch joins a cross-host "launch group").
    Transfers therefore never launch device work from their own thread: every
    phase that touches the device (gather, D2H put, H2D put, scatter) is handed
    to ``self._launcher`` -- in overlap mode the forward thread's FIFO, i.e. the
    one thread per host that already owns all launches, so HiCache programs are
    ordered identically relative to forwards on every host. The small worker
    thread only *waits* for readiness (``block_until_ready`` is not a launch)
    and settles bookkeeping. Without a launcher (unit tests, non-overlap
    scheduler) jobs run inline on the caller's thread, which is then the sole
    launcher by construction.
    """

    def __init__(
        self,
        host_pool: HostKVPool,
        device_pool: Any,
        device_allocator: Any = None,
        launcher: Callable[[Callable[[], Any]], Future] | None = None,
    ) -> None:
        self._host_pool = host_pool
        self._device_pool = device_pool
        self._device_allocator = device_allocator
        self._launcher = launcher or _inline_launcher
        # Readiness waiter: never issues device programs.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hicache-wait")
        self._pending: list[Future] = []
        # Guards against freeing a host page while a write to it is still in flight.
        self._inflight_lock = threading.Lock()
        self._inflight: set[int] = set()
        # Mirrors _inflight for the async H2D load path.
        self._inflight_load_lock = threading.Lock()
        self._inflight_load: set[int] = set()
        self._pending_load: list[Future] = []
        self._shutdown_done = False

    def set_launcher(self, launcher: Callable[[Callable[[], Any]], Future] | None) -> None:
        """Install the single device-launch thread's submit function (or None for inline)."""
        self._launcher = launcher or _inline_launcher

    def run_on_launcher(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the launcher thread and wait for its result."""
        return self._launcher(fn).result()

    def write(self, device_indices: list[int], host_buffer_ids: list[int]) -> Future:
        """Async D2H: copy device pages into reserved host page slots.

        The gather + host put are *issued* on the launcher thread (ordered with
        forwards, donation-safe); the returned future resolves once the host
        copies are ready, which the waiter thread observes without launching.
        """
        host_buffer_ids = list(host_buffer_ids)
        device_indices = list(device_indices)
        # Register in-flight ids before submit: if the waiter finishes before
        # registration, its cleanup has nothing to remove and the ids leak.
        with self._inflight_lock:
            self._inflight.update(host_buffer_ids)

        def _issue() -> list:
            self._host_pool.stage_backup(device_indices, host_buffer_ids)
            return self._host_pool.flush_backup(host_buffer_ids, block=False)

        launched = self._launcher(_issue)
        future = self._executor.submit(self._await_d2h, launched, host_buffer_ids)
        self._pending.append(future)
        return future

    def load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """Sync H2D: stage host pages onto device, then scatter into kv_buffer.

        Issued as one launcher job (ordered with forwards); blocks until done.
        """
        host_buffer_ids = list(host_buffer_ids)
        device_indices = list(device_indices)

        def _issue():
            self._host_pool.stage_load(host_buffer_ids)
            self._host_pool.flush_load(host_buffer_ids, device_indices)

        self._launcher(_issue).result()

    def stage_load(self, host_buffer_ids: list[int]) -> Future:
        """Async H2D stage: issue the host->device copies on the launcher thread.

        Never touches kv_buffer. The returned future resolves when the staged
        device copies are ready; the cheap scatter is deferred to flush_load.
        """
        host_buffer_ids = list(host_buffer_ids)
        # Register in-flight ids before submit (same race-window fix as write).
        with self._inflight_load_lock:
            self._inflight_load.update(host_buffer_ids)
        launched = self._launcher(
            lambda: self._host_pool.stage_load(host_buffer_ids, block=False)
        )
        future = self._executor.submit(self._await_stage_load, launched, host_buffer_ids)
        self._pending_load.append(future)
        return future

    def flush_load(self, host_buffer_ids: list[int], device_indices: list[int]) -> Future:
        """Issue the H2D scatter into kv_buffer on the launcher thread.

        FIFO order on the launcher guarantees it lands before any forward
        enqueued afterwards, so callers need not block on the returned future
        (the scheduler enqueues it before the next ``run_batch``).
        """
        host_buffer_ids = list(host_buffer_ids)
        device_indices = list(device_indices)
        with self._inflight_load_lock:
            busy = [b for b in host_buffer_ids if b in self._inflight_load]
        if busy:
            raise RuntimeError(
                f"flush_load of page id(s) {busy} with in-flight stage_load; "
                f"call check_load_status()/drain_loads() before flushing"
            )
        return self._launcher(
            lambda: self._host_pool.flush_load(host_buffer_ids, device_indices, block=False)
        )

    # -- waiter-thread bodies (no device launches here) ----------------------

    def _await_d2h(self, launched: Future, host_buffer_ids: list[int]) -> None:
        try:
            staged = launched.result()
            if staged:
                jax.block_until_ready(staged)
        finally:
            with self._inflight_lock:
                self._inflight.difference_update(host_buffer_ids)

    def _await_stage_load(self, launched: Future, host_buffer_ids: list[int]) -> None:
        try:
            staged = launched.result()
            if staged:
                jax.block_until_ready(staged)
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
        """Free host page slots. Rejects pages with in-flight D2H writes or H2D loads."""
        with self._inflight_lock:
            busy_d2h = [b for b in host_buffer_ids if b in self._inflight]
        if busy_d2h:
            raise RuntimeError(
                f"evict of page id(s) {busy_d2h} with in-flight D2H write; "
                f"call drain_pending() (or wait for the write) before releasing"
            )
        with self._inflight_load_lock:
            busy_h2d = [b for b in host_buffer_ids if b in self._inflight_load]
        if busy_h2d:
            raise RuntimeError(
                f"evict of page id(s) {busy_h2d} with in-flight H2D load; "
                f"call drain_loads() before releasing"
            )
        self._host_pool.free(host_buffer_ids)

    def has_inflight(self, host_buffer_ids: list[int]) -> bool:
        """Check whether any of the given host pages have in-flight transfers."""
        with self._inflight_lock:
            if any(b in self._inflight for b in host_buffer_ids):
                return True
        with self._inflight_load_lock:
            if any(b in self._inflight_load for b in host_buffer_ids):
                return True
        return False

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
