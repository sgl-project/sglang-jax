"""Unit tests for :class:`LRUHostKVPool` (HiCache Stage 0).

CPU-runnable: ``_make_host_sharding`` falls back to default sharding when
``pinned_host`` is unsupported, so the round-trip semantics are exercised
without a TPU. The pinned-host in-place invariant is covered separately by
``test_single_chip_host_pool`` on a real pod.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.host_kv_pool import (
    LRUHostKVPool,
    QueueHostKVPool,
    make_unit_mesh,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


def _make_device_pool(
    *, size=16, page_size=1, head_num=4, head_dim=128, layer_num=3, dtype=jnp.float32
):
    # head_dim must be 128-aligned: copy_to_device now writes via the in-place
    # Pallas kernel (update_fused_kv_cache_vectorized), which requires it -- and
    # production KV pools are always 128-aligned.
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


def _make_pool(device_pool, *, pool_size=4, page_size=1):
    return LRUHostKVPool(
        device_pool=device_pool,
        pool_size=pool_size,
        page_size=page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:]),
        dtype=device_pool.dtype,
        mesh=_MESH,
        partition_spec=device_pool.kv_sharding.spec,
    )


class TestLRUHostKVPoolFreeList(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_device_pool()
        self.pool = _make_pool(self.device_pool, pool_size=3)

    def test_reserve_returns_distinct_ids(self):
        ids = {self.pool.reserve() for _ in range(3)}
        self.assertEqual(ids, {0, 1, 2})

    def test_exhaustion_returns_none(self):
        for _ in range(3):
            self.pool.reserve()
        self.assertIsNone(self.pool.reserve())

    def test_release_re_enables_reserve(self):
        b = self.pool.reserve()
        for _ in range(2):
            self.pool.reserve()
        self.assertIsNone(self.pool.reserve())
        self.pool.release(b)
        self.assertIsNotNone(self.pool.reserve())

    def test_available_size_tracks_usage(self):
        self.assertEqual(self.pool.available_size(), 3)
        b = self.pool.reserve()
        self.assertEqual(self.pool.available_size(), 2)
        self.pool.release(b)
        self.assertEqual(self.pool.available_size(), 3)

    def test_total_size(self):
        self.assertEqual(self.pool.total_size(), 3)

    def test_double_free_raises(self):
        b = self.pool.reserve()
        self.pool.release(b)
        with self.assertRaises(RuntimeError):
            self.pool.release(b)

    def test_release_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.pool.release(99)


class TestLRUHostKVPoolLockRef(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.pool = _make_pool(_make_device_pool(), pool_size=2)

    def test_locked_slot_cannot_be_released(self):
        b = self.pool.reserve()
        self.pool.inc_lock_ref(b)
        with self.assertRaises(RuntimeError):
            self.pool.release(b)

    def test_dec_to_zero_allows_release(self):
        b = self.pool.reserve()
        self.pool.inc_lock_ref(b)
        self.pool.inc_lock_ref(b)
        self.pool.dec_lock_ref(b)
        with self.assertRaises(RuntimeError):
            self.pool.release(b)
        self.pool.dec_lock_ref(b)
        self.pool.release(b)  # no raise

    def test_dec_underflow_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(RuntimeError):
            self.pool.dec_lock_ref(b)


class TestLRUHostKVPoolBoundaryInvariants(unittest.TestCase):
    """Page-id boundary guards: an unallocated (free-list) host id or an
    out-of-range device page must be rejected at the entry, not silently written
    -- otherwise a transfer/lock corrupts a slot the next alloc() reuses, or DMAs
    to a wrapped/negative device page."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.pool = _make_pool(_make_device_pool(size=16, page_size=1), pool_size=4)

    def test_transfer_rejects_unallocated_host_id(self):
        # id 0 is in the free-list (never alloc'd): staging it would write _slots[0]
        # while alloc() can still hand 0 to another owner.
        with self.assertRaises(RuntimeError):
            self.pool.stage_backup([0], [0])

    def test_copy_from_device_rejects_unallocated_host_id(self):
        # Same invariant on the ABC borrow path: a free-list id must not be
        # written into _slots while alloc() can still hand it out.
        layers = [self.pool._device_pool.kv_buffer[L][0] for L in range(3)]
        with self.assertRaises(RuntimeError):
            self.pool.copy_from_device(layers, 0)

    def test_inc_lock_ref_rejects_unallocated(self):
        with self.assertRaises(RuntimeError):
            self.pool.inc_lock_ref(0)

    def test_lock_ref_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            self.pool.inc_lock_ref(99)

    def test_stage_backup_rejects_out_of_range_device_page(self):
        page = int(self.pool.alloc(1)[0])
        with self.assertRaises(ValueError):
            self.pool.stage_backup([16], [page])  # device has 16 pages: [0, 16)

    def test_flush_load_rejects_negative_device_page(self):
        page = int(self.pool.alloc(1)[0])
        with self.assertRaises(ValueError):
            self.pool.flush_load([page], [-1])


class TestLRUHostKVPoolRoundTrip(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_device_pool(layer_num=3)
        self.pool = _make_pool(self.device_pool, pool_size=4)

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

    def test_copy_into_then_copy_to_device_roundtrip(self):
        src, dst = 5, 9
        orig = self._fill(src, seed=1)
        b = self.pool.reserve()
        self.pool.copy_into([src], [b])
        self.pool.copy_to_device([b], [dst])
        for layer in range(self.device_pool.layer_num):
            got = np.asarray(self.device_pool.kv_buffer[layer][dst])
            np.testing.assert_allclose(got, orig[layer])

    def test_batched_roundtrip(self):
        pairs = [(2, 10), (3, 11), (4, 12)]
        origs = {src: self._fill(src, seed=src) for src, _ in pairs}
        bufs = [self.pool.reserve() for _ in pairs]
        srcs = [src for src, _ in pairs]
        dsts = [dst for _, dst in pairs]
        self.pool.copy_into(srcs, bufs)
        self.pool.copy_to_device(bufs, dsts)
        for (src, dst), b in zip(pairs, bufs):
            for layer in range(self.device_pool.layer_num):
                got = np.asarray(self.device_pool.kv_buffer[layer][dst])
                np.testing.assert_allclose(got, origs[src][layer])

    def test_copy_into_length_mismatch_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(ValueError):
            self.pool.copy_into([1, 2], [b])

    def test_copy_to_device_from_empty_slot_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(RuntimeError):
            self.pool.copy_to_device([b], [0])

    def test_copy_to_device_out_of_range_buffer_raises(self):
        with self.assertRaises(ValueError):
            self.pool.copy_to_device([99], [0])

    def test_release_clears_slot_data(self):
        self._fill(5, seed=7)
        b = self.pool.reserve()
        self.pool.copy_into([5], [b])
        self.pool.release(b)
        # After release the slot reference is dropped: reading it back must fail.
        with self.assertRaises(RuntimeError):
            self.pool.copy_to_device([b], [9])


class TestLRUHostKVPoolAllocFree(unittest.TestCase):
    """The page-addressed control-plane interface used by HiCache: batched
    ``alloc(need_pages)`` / ``free(page_ids)`` (vs the single-slot reserve/
    release kept only to satisfy the ABC)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.pool = _make_pool(_make_device_pool(), pool_size=4)

    def test_alloc_returns_distinct_page_ids(self):
        pages = self.pool.alloc(3)
        self.assertEqual(len(pages), 3)
        self.assertEqual(len(set(int(p) for p in pages)), 3)
        self.assertEqual(self.pool.available_size(), 1)

    def test_alloc_insufficient_returns_none(self):
        self.assertIsNotNone(self.pool.alloc(4))
        self.assertIsNone(self.pool.alloc(1))

    def test_alloc_all_or_nothing(self):
        self.pool.alloc(3)
        # Only 1 free left; asking for 2 must return None and free nothing.
        self.assertIsNone(self.pool.alloc(2))
        self.assertEqual(self.pool.available_size(), 1)

    def test_free_re_enables_alloc(self):
        pages = self.pool.alloc(4)
        self.assertIsNone(self.pool.alloc(1))
        self.pool.free([int(p) for p in pages[:2]])
        self.assertEqual(self.pool.available_size(), 2)
        self.assertIsNotNone(self.pool.alloc(2))

    def test_free_dedupes(self):
        pages = self.pool.alloc(2)
        ids = [int(pages[0]), int(pages[0]), int(pages[1])]
        self.pool.free(ids)
        self.assertEqual(self.pool.available_size(), 4)

    def test_free_locked_page_raises(self):
        pages = self.pool.alloc(1)
        self.pool.inc_lock_ref(int(pages[0]))
        with self.assertRaises(RuntimeError):
            self.pool.free([int(pages[0])])


class TestLRUHostKVPoolPageSize(unittest.TestCase):
    """page_size>1: one host slot holds a whole device page. The control plane
    addresses device + host by PAGE id, and the round-trip must be bit-exact."""

    def _setup(self, page_size):
        self.device_pool = _make_device_pool(size=16, page_size=page_size, layer_num=3)
        self.pool = _make_pool(self.device_pool, pool_size=4, page_size=page_size)

    def _fill_page(self, device_page, seed):
        orig = []
        for layer in range(self.device_pool.layer_num):
            buf = self.device_pool.kv_buffer[layer]
            vals = jax.random.normal(
                jax.random.PRNGKey(seed * 100 + layer), buf[device_page].shape, buf.dtype
            )
            self.device_pool.kv_buffer[layer] = buf.at[device_page].set(
                vals, out_sharding=buf.sharding
            )
            orig.append(np.asarray(self.device_pool.kv_buffer[layer][device_page]))
        return orig

    def test_page4_batched_roundtrip(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self._setup(page_size=4)
        pairs = [(0, 2), (1, 3)]  # 16 tokens / page_size 4 == 4 pages
        origs = {s: self._fill_page(s, seed=s + 1) for s, _ in pairs}
        host_pages = [int(p) for p in self.pool.alloc(len(pairs))]
        srcs = [s for s, _ in pairs]
        dsts = [d for _, d in pairs]
        self.pool.stage_backup(srcs, host_pages)
        self.pool.flush_backup(host_pages)
        self.pool.copy_to_device(host_pages, dsts)
        for s, d in pairs:
            for layer in range(self.device_pool.layer_num):
                got = np.asarray(self.device_pool.kv_buffer[layer][d])
                np.testing.assert_allclose(got, origs[s][layer])


class TestABCExtensionDoesNotBreakPD(unittest.TestCase):
    """Adding copy_into/copy_to_device must leave QueueHostKVPool concrete."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.queue = QueueHostKVPool(
            pool_size=2,
            max_padded_pages=4,
            layer_num=2,
            per_layer_shape=(1, 4, 1, 8),
            dtype=jnp.float32,
            mesh=make_unit_mesh(),
            partition_spec=jax.sharding.PartitionSpec(),
        )

    def test_queue_pool_still_instantiates_and_reserves(self):
        b = self.queue.reserve()
        self.assertIsNotNone(b)
        self.queue.release(b)

    def test_queue_pool_copy_into_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.queue.copy_into([0], [0])

    def test_queue_pool_copy_to_device_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.queue.copy_to_device([0], [0])


class TestLRUHostKVPoolBucketPadding(unittest.TestCase):
    """n between gather buckets (3 -> padded to 4): padded gather rows must be
    discarded and padded scatter slots (loc=-1) skipped, so the round-trip stays
    bit-exact AND untouched device pages (the padding target, page 0) are not
    clobbered. All other tests use exactly 1/2 pages == bucket boundaries, so
    this is the only coverage of the padding path that the batching work added."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_device_pool(size=16, page_size=1, layer_num=3)
        self.pool = _make_pool(self.device_pool, pool_size=4, page_size=1)

    def _fill(self, idx, seed):
        orig = []
        for layer in range(self.device_pool.layer_num):
            buf = self.device_pool.kv_buffer[layer]
            vals = jax.random.normal(
                jax.random.PRNGKey(seed * 100 + layer), buf[idx].shape, buf.dtype
            )
            self.device_pool.kv_buffer[layer] = buf.at[idx].set(vals, out_sharding=buf.sharding)
            orig.append(np.asarray(self.device_pool.kv_buffer[layer][idx]))
        return orig

    def test_three_page_roundtrip_bit_exact(self):
        self.assertEqual(self.pool._pad_to_page_bucket(3), 4)  # padding exercised
        srcs, dsts = [1, 2, 3], [8, 9, 10]
        origs = [self._fill(s, seed=s) for s in srcs]
        host = [int(p) for p in self.pool.alloc(3)]
        self.pool.stage_backup(srcs, host)
        self.pool.flush_backup(host)
        self.pool.copy_to_device(host, dsts)
        for k, d in enumerate(dsts):
            for layer in range(self.device_pool.layer_num):
                got = np.asarray(self.device_pool.kv_buffer[layer][d])
                np.testing.assert_allclose(got, origs[k][layer])

    def test_padding_does_not_clobber_other_pages(self):
        # Device page 0 is the gather/scatter padding target; a 3-page transfer
        # excluding page 0 must leave it untouched (loc=-1 skip works).
        guard = self._fill(0, seed=42)
        srcs, dsts = [1, 2, 3], [8, 9, 10]
        for s in srcs:
            self._fill(s, seed=s)
        host = [int(p) for p in self.pool.alloc(3)]
        self.pool.stage_backup(srcs, host)
        self.pool.flush_backup(host)
        self.pool.copy_to_device(host, dsts)
        for layer in range(self.device_pool.layer_num):
            got = np.asarray(self.device_pool.kv_buffer[layer][0])
            np.testing.assert_allclose(got, guard[layer])


class TestLRUHostKVPoolStageDrainErrors(unittest.TestCase):
    """The async two-phase contract: flush before stage must fail loudly rather
    than silently transfer garbage."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.pool = _make_pool(_make_device_pool(), pool_size=4)

    def test_flush_backup_without_stage_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(RuntimeError):
            self.pool.flush_backup([b])

    def test_flush_load_without_stage_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(RuntimeError):
            self.pool.flush_load([b], [0])

    def test_stage_load_from_empty_slot_raises(self):
        b = self.pool.reserve()
        with self.assertRaises(RuntimeError):
            self.pool.stage_load([b])

    def test_free_drops_pending_gather(self):
        # free() must drop a slot's orphaned D2H gather so a reused id can't have
        # a later flush_backup pop stale data into it.
        pages = [int(p) for p in self.pool.alloc(1)]
        self.pool.stage_backup([0], pages)
        self.pool.free(pages)
        with self.assertRaises(RuntimeError):
            self.pool.flush_backup(pages)

    def test_release_drops_pending_load(self):
        # release() must drop a slot's orphaned H2D staged page for the same reason.
        b = self.pool.reserve()
        self.pool.copy_into([0], [b])
        self.pool.stage_load([b])
        self.pool.release(b)
        with self.assertRaises(RuntimeError):
            self.pool.flush_load([b], [0])


class TestLRUHostKVPoolPrecompile(unittest.TestCase):
    """precompile_transfers warms one shape per page count serving can hit and
    must always restore the pool (free its scratch slots), never raise, and warm
    the top partial bucket even when pool_size is not a bucket boundary."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_runs_and_restores_pool(self):
        pool = _make_pool(_make_device_pool(size=16, layer_num=2), pool_size=8)
        pool.precompile_transfers()
        self.assertEqual(pool.available_size(), 8)

    def test_non_bucket_pool_size_warms_without_alloc_failure(self):
        # pool_size=3 is between buckets (2, 4). The real-count warmup must
        # alloc 3 (which fits) instead of bucket 4 (which would not), so it
        # warms fully and restores the pool with no "pool too small" early stop.
        pool = _make_pool(_make_device_pool(size=16, layer_num=2), pool_size=3)
        pool.precompile_transfers()
        self.assertEqual(pool.available_size(), 3)


if __name__ == "__main__":
    unittest.main()
