"""Unit tests for :class:`HiCacheController` (HiCache Stage 2).

CPU-runnable. Uses a real :class:`LRUHostKVPool` over a small device pool for
round-trip tests, and a tiny fake pool to exercise async scheduling/error
propagation deterministically.
"""

from __future__ import annotations

import threading
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


# head_dim must be 128-aligned: the H2D load path now writes via the in-place
# Pallas kernel (write_kv_layer -> update_fused_kv_cache_vectorized), which
# requires it -- and production KV pools are always 128-aligned.
def _make_device_pool(*, size=16, page_size=1, head_num=4, head_dim=128, layer_num=3, dtype=jnp.float32):
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=_MESH,
        dp_size=1,
    )


def _make_pool(device_pool, *, pool_size=4):
    return LRUHostKVPool(
        device_pool=device_pool,
        pool_size=pool_size,
        page_size=device_pool.page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:]),
        dtype=device_pool.dtype,
        mesh=_MESH,
        partition_spec=device_pool.kv_sharding.spec,
    )


class _FakePool:
    """Records calls; flush_backup can block on an event to make 'pending'
    observable. The controller now splits a backup into a synchronous
    stage_backup (device gather, caller thread) and an async flush_backup
    (host transfer, worker thread), so the blocking/raising behaviour that
    models the slow worker lives in flush_backup."""

    def __init__(self, *, block_event: threading.Event | None = None, raise_on_copy=False):
        self.block_event = block_event
        self.raise_on_copy = raise_on_copy
        self.copy_into_calls = []
        self.staged = []
        self.released = []

    def stage_backup(self, device_indices, host_buffer_ids):
        # Synchronous (caller thread): just record the gather request.
        self.staged.append((list(device_indices), list(host_buffer_ids)))

    def flush_backup(self, host_buffer_ids):
        # Async (worker thread): this is where the slow transfer / errors land.
        if self.block_event is not None:
            self.block_event.wait(timeout=5.0)
        if self.raise_on_copy:
            raise RuntimeError("boom")
        # Pair each flushed buffer_id back with the device indices it staged.
        for dev, host in self.staged:
            if host == list(host_buffer_ids):
                self.copy_into_calls.append((dev, host))
                break

    def copy_into(self, device_indices, host_buffer_ids):
        self.stage_backup(device_indices, host_buffer_ids)
        self.flush_backup(host_buffer_ids)

    def copy_to_device(self, host_buffer_ids, device_indices):
        pass

    def free(self, host_buffer_ids):
        self.released.extend(host_buffer_ids)


class _FakeLoadPool:
    """Records H2D stage/flush calls. ``stage_load`` can block on an event so the
    controller's in-flight guard (flush before the stage finishes) is observable,
    or raise to exercise error propagation through drain_loads/check_load_status."""

    def __init__(self, *, stage_gate: threading.Event | None = None, raise_on_stage=False):
        self.stage_gate = stage_gate
        self.raise_on_stage = raise_on_stage
        self.staged = []
        self.flushed = []

    def stage_load(self, host_buffer_ids):
        if self.stage_gate is not None:
            self.stage_gate.wait(timeout=5.0)
        if self.raise_on_stage:
            raise RuntimeError("stage boom")
        self.staged.append(list(host_buffer_ids))

    def flush_load(self, host_buffer_ids, device_indices):
        self.flushed.append((list(host_buffer_ids), list(device_indices)))

    def free(self, host_buffer_ids):
        pass



class TestHiCacheControllerAsync(unittest.TestCase):
    def test_write_returns_immediately_and_pending_observable(self):
        gate = threading.Event()
        pool = _FakePool(block_event=gate)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.write([1], [0])
            self.assertEqual(ctrl.pending_count(), 1)  # still blocked in worker
            gate.set()
            ctrl.drain_pending()
            self.assertEqual(ctrl.pending_count(), 0)
            self.assertEqual(pool.copy_into_calls, [([1], [0])])
        finally:
            gate.set()
            ctrl.shutdown()

    def test_check_write_status_clears_completed(self):
        pool = _FakePool()
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.write([1], [0])
            # Poll until the worker finishes; check_write_status drops done futures.
            deadline = time.time() + 5.0
            while ctrl.pending_count() > 0 and time.time() < deadline:
                ctrl.check_write_status()
                time.sleep(0.005)
            self.assertEqual(ctrl.pending_count(), 0)
        finally:
            ctrl.shutdown()

    def test_check_write_status_reraises_worker_error(self):
        pool = _FakePool(raise_on_copy=True)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.write([1], [0])
            with self.assertRaises(RuntimeError):
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    ctrl.check_write_status()
                    if ctrl.pending_count() == 0:
                        break
                    time.sleep(0.005)
        finally:
            ctrl.shutdown()

    def test_drain_pending_reraises_worker_error(self):
        pool = _FakePool(raise_on_copy=True)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.write([1], [0])
            with self.assertRaises(RuntimeError):
                ctrl.drain_pending()
        finally:
            ctrl.shutdown()


class TestHiCacheControllerEvict(unittest.TestCase):
    def test_evict_callback_releases_slots(self):
        pool = _FakePool()
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.evict_callback([0, 1, 2])
            self.assertEqual(pool.released, [0, 1, 2])
        finally:
            ctrl.shutdown()

    def test_evict_rejects_inflight_buffer(self):
        gate = threading.Event()
        pool = _FakePool(block_event=gate)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.write([1], [0])  # worker blocks on gate, slot 0 in-flight
            with self.assertRaises(RuntimeError):
                ctrl.evict_callback([0])
            self.assertEqual(pool.released, [])  # nothing released while in-flight
            gate.set()
            ctrl.drain_pending()
            ctrl.evict_callback([0])  # now allowed
            self.assertEqual(pool.released, [0])
        finally:
            gate.set()
            ctrl.shutdown()


class TestHiCacheControllerLoadAsync(unittest.TestCase):
    """The async H2D split used by the overlap scheduler: stage_load (off-thread
    device_put, registers the page in-flight) + flush_load (donation-safe scatter,
    guarded against flushing a page whose stage has not finished)."""

    def test_flush_before_stage_done_raises_then_succeeds_after_drain(self):
        gate = threading.Event()
        pool = _FakeLoadPool(stage_gate=gate)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.stage_load([0])  # worker blocks on gate -> page 0 in-flight
            with self.assertRaises(RuntimeError):
                ctrl.flush_load([0], [3])  # guard trips: stage not done
            self.assertEqual(pool.flushed, [])  # nothing scattered while in-flight
            gate.set()
            ctrl.drain_loads()
            ctrl.flush_load([0], [3])  # now allowed
            self.assertEqual(pool.staged, [[0]])
            self.assertEqual(pool.flushed, [([0], [3])])
        finally:
            gate.set()
            ctrl.shutdown()

    def test_drain_loads_reraises_stage_error(self):
        pool = _FakeLoadPool(raise_on_stage=True)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.stage_load([1])
            with self.assertRaises(RuntimeError):
                ctrl.drain_loads()
        finally:
            ctrl.shutdown()

    def test_check_load_status_reraises_stage_error(self):
        pool = _FakeLoadPool(raise_on_stage=True)
        ctrl = HiCacheController(pool, device_pool=None)
        try:
            ctrl.stage_load([1])
            with self.assertRaises(RuntimeError):
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    ctrl.check_load_status()
                    time.sleep(0.005)
        finally:
            ctrl.shutdown()


class TestHiCacheControllerRoundTrip(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_device_pool(layer_num=3)
        self.host_pool = _make_pool(self.device_pool, pool_size=4)
        self.ctrl = HiCacheController(self.host_pool, self.device_pool)

    def tearDown(self):
        self.ctrl.shutdown()

    def _fill(self, device_idx, seed):
        orig = []
        for layer in range(self.device_pool.layer_num):
            buf = self.device_pool.kv_buffer[layer]
            vals = jax.random.normal(
                jax.random.PRNGKey(seed * 100 + layer), buf[device_idx].shape, buf.dtype
            )
            self.device_pool.kv_buffer[layer] = buf.at[device_idx].set(
                vals, out_sharding=buf.sharding
            )
            orig.append(np.asarray(self.device_pool.kv_buffer[layer][device_idx]))
        return orig

    def test_write_then_load_roundtrip(self):
        src, dst = 5, 9
        orig = self._fill(src, seed=3)
        b = self.host_pool.reserve()
        self.ctrl.write([src], [b])
        self.ctrl.drain_pending()
        self.ctrl.load([b], [dst])
        for layer in range(self.device_pool.layer_num):
            got = np.asarray(self.device_pool.kv_buffer[layer][dst])
            np.testing.assert_allclose(got, orig[layer])

    def test_write_then_stage_drain_flush_roundtrip(self):
        # The split path the overlap scheduler drives: init_load_back issues
        # stage_load (async device_put), finish_load_back does drain_loads then
        # flush_load (donation-safe scatter). Must be bit-exact like inline load.
        src, dst = 5, 9
        orig = self._fill(src, seed=6)
        b = self.host_pool.reserve()
        self.ctrl.write([src], [b])
        self.ctrl.drain_pending()
        self.ctrl.stage_load([b])
        self.ctrl.drain_loads()
        self.ctrl.flush_load([b], [dst])
        for layer in range(self.device_pool.layer_num):
            got = np.asarray(self.device_pool.kv_buffer[layer][dst])
            np.testing.assert_allclose(got, orig[layer])

    def test_evict_callback_frees_real_slot(self):
        b = self.host_pool.reserve()
        self.ctrl.write([5], [b])
        self.ctrl.drain_pending()
        before = self.host_pool.available_size()
        self.ctrl.evict_callback([b])
        self.assertEqual(self.host_pool.available_size(), before + 1)


if __name__ == "__main__":
    unittest.main()
