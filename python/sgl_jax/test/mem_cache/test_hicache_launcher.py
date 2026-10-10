"""HiCacheController launch-thread contract (multi-host safety).

On a multi-controller mesh all hosts must issue device programs in the same
order, so the controller must never launch from its own worker thread. These
tests install a recording launcher and assert that every pool method that
touches the device is invoked on the launcher, in FIFO order, while the
controller's internal thread only waits for readiness.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import Future
from queue import Queue

from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController


class _RecordingPool:
    """Fake HostKVPool recording which thread each device-touching call ran on."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []  # (method, thread ident)
        self.freed: list[list[int]] = []

    def _rec(self, name):
        self.calls.append((name, threading.get_ident()))

    def stage_backup(self, device_indices, host_buffer_ids):
        self._rec("stage_backup")

    def flush_backup(self, host_buffer_ids, *, block=True):
        self._rec("flush_backup")
        return []

    def stage_load(self, host_buffer_ids, *, block=True):
        self._rec("stage_load")
        return []

    def flush_load(self, host_buffer_ids, device_indices, *, block=True):
        self._rec("flush_load")

    def precompile_transfers(self):
        self._rec("precompile_transfers")

    def free(self, ids):
        self.freed.append(list(ids))


class _SerialLauncher:
    """Single-thread FIFO launcher, like the overlap forward thread."""

    def __init__(self):
        self.q: Queue = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            fn, fut = self.q.get()
            if fn is None:
                return
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)

    def __call__(self, fn) -> Future:
        fut: Future = Future()
        self.q.put((fn, fut))
        return fut

    def stop(self):
        self.q.put((None, None))
        self.thread.join(timeout=5)


class TestLauncherContract(unittest.TestCase):
    def setUp(self):
        self.pool = _RecordingPool()
        self.launcher = _SerialLauncher()
        self.ctrl = HiCacheController(self.pool, device_pool=None, launcher=self.launcher)

    def tearDown(self):
        self.ctrl.shutdown()
        self.launcher.stop()

    def test_all_device_calls_run_on_launcher_thread_in_fifo_order(self):
        self.ctrl.write([1, 2], [10, 11]).result(timeout=5)
        self.ctrl.stage_load([10]).result(timeout=5)
        self.ctrl.drain_loads()
        self.ctrl.flush_load([10], [3]).result(timeout=5)
        self.ctrl.run_on_launcher(self.pool.precompile_transfers)

        names = [n for n, _ in self.pool.calls]
        self.assertEqual(
            names,
            ["stage_backup", "flush_backup", "stage_load", "flush_load", "precompile_transfers"],
        )
        threads = {t for _, t in self.pool.calls}
        self.assertEqual(threads, {self.launcher.thread.ident}, "device work left the launcher")
        self.assertNotIn(threading.get_ident(), threads)

    def test_inflight_cleared_only_after_launched_job_completes(self):
        gate = threading.Event()
        orig = self.pool.flush_backup

        def slow_flush(host_buffer_ids, *, block=True):
            gate.wait(5)
            return orig(host_buffer_ids, block=block)

        self.pool.flush_backup = slow_flush
        fut = self.ctrl.write([1], [10])
        self.assertTrue(self.ctrl.has_inflight([10]))
        with self.assertRaises(RuntimeError):
            self.ctrl.evict_callback([10])
        gate.set()
        fut.result(timeout=5)
        # The waiter clears in-flight after the launched job resolved.
        for _ in range(100):
            if not self.ctrl.has_inflight([10]):
                break
            threading.Event().wait(0.01)
        self.assertFalse(self.ctrl.has_inflight([10]))

    def test_launcher_exception_surfaces_on_write_future(self):
        def boom(*a, **k):
            raise ValueError("gather failed")

        self.pool.stage_backup = boom
        fut = self.ctrl.write([1], [10])
        with self.assertRaises(ValueError):
            fut.result(timeout=5)
        self.assertFalse(self.ctrl.has_inflight([10]))

    def test_inline_launcher_when_none_installed(self):
        ctrl = HiCacheController(self.pool, device_pool=None)
        ctrl.write([1], [10]).result(timeout=5)
        self.assertEqual(
            {t for _, t in self.pool.calls}, {threading.get_ident()}, "inline must run on caller"
        )
        ctrl.shutdown()


if __name__ == "__main__":
    unittest.main()
