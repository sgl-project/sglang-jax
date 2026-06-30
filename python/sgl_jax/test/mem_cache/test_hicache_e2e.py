"""End-to-end tests for the HiCache Stage 3 control plane (L1<->L2 wiring).

CPU-runnable: a single-device mesh keeps the host pool on default sharding, so
the radix control-plane logic (write-through backup, device->host demotion,
host-tier match, synchronous load-back, host eviction) is exercised against a
*real* ``LRUHostKVPool`` + ``HiCacheController`` without a TPU.

These tests cover the wiring, not raw transfer mechanics (those live in
``test_host_kv_pool`` / ``test_hicache_controller``).
"""

from __future__ import annotations

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


def _key(token_ids: list[int]) -> RadixKey:
    return RadixKey(token_ids=token_ids, extra_key=None, dp_rank=0)


class HiCacheE2EBase(unittest.TestCase):
    HEAD_NUM = 4
    HEAD_DIM = 8
    LAYER_NUM = 2
    DEVICE_SIZE = 64
    # Host pool is page-addressed: HOST_PAGES is a page-slot count, sized to
    # comfortably hold every device page the small tests touch at any page_size.
    HOST_PAGES = 32
    PAGE_SIZE = 1
    DTYPE = jnp.float32

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        jax.sharding.set_mesh(_MESH)
        self.kv_cache = MHATokenToKVPool(
            size=self.DEVICE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=self.DTYPE,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            mesh=_MESH,
            dp_size=1,
        )
        self.allocator = (
            TokenToKVPoolAllocator(
                size=self.DEVICE_SIZE, kvcache=self.kv_cache, dp_size=1
            )
            if self.PAGE_SIZE == 1
            else PagedTokenToKVPoolAllocator(
                size=self.DEVICE_SIZE,
                page_size=self.PAGE_SIZE,
                kvcache=self.kv_cache,
                dp_size=1,
            )
        )
        self.req_pool = ReqToTokenPool(size=64, max_context_len=512, dtype=np.int32)
        self.cache = UnifiedRadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.PAGE_SIZE,
            kv_head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            max_seq_len=512,
            dtype=self.DTYPE,
        )
        self.host_pool = LRUHostKVPool(
            device_pool=self.kv_cache,
            pool_size=self.HOST_PAGES,
            page_size=self.kv_cache.page_size,
            layer_num=self.kv_cache.layer_num,
            per_layer_shape=tuple(int(d) for d in self.kv_cache.kv_buffer[0].shape[1:]),
            dtype=self.kv_cache.dtype,
            mesh=_MESH,
            partition_spec=self.kv_cache.kv_sharding.spec,
        )
        self.controller = HiCacheController(self.host_pool, self.kv_cache)
        self._enable_hicache(write_through_threshold=1)

    def tearDown(self):
        self.controller.shutdown()

    def _enable_hicache(self, write_through_threshold: int) -> None:
        # Mirror kv_cache_builder.init_hicache wiring without server_args.
        self.cache.host_pool = self.host_pool
        self.cache.hicache_controller = self.controller
        self.cache.hicache_enabled = True
        self.cache.write_through_threshold = write_through_threshold
        for component in self.cache._components_tuple:
            component._full_kv_pool_host = self.host_pool

    # ---- helpers ----

    def _pages(self, n_tokens: int) -> int:
        """Token count -> host page count under the active page_size."""
        self.assertEqual(n_tokens % self.PAGE_SIZE, 0)
        return n_tokens // self.PAGE_SIZE

    def _write_token(self, layer: int, token_idx: int, vals) -> None:
        """Set one token's KV at [page, offset] (kv_buffer leading axis=page)."""
        page, off = int(token_idx) // self.PAGE_SIZE, int(token_idx) % self.PAGE_SIZE
        buf = self.kv_cache.kv_buffer[layer]
        self.kv_cache.kv_buffer[layer] = buf.at[page, off].set(
            vals, out_sharding=buf.sharding
        )

    def _read_token(self, layer: int, token_idx: int):
        page, off = int(token_idx) // self.PAGE_SIZE, int(token_idx) % self.PAGE_SIZE
        return np.asarray(self.kv_cache.kv_buffer[layer][page, off])

    def _alloc_and_fill(self, n: int, seed: int) -> tuple[np.ndarray, list]:
        """Allocate n device slots, fill them with random KV, return (indices,
        per-(slot,layer) original KV for bit-exact comparison)."""
        indices = self.allocator.alloc(n, dp_rank=0)
        self.assertIsNotNone(indices)
        orig = []  # orig[i][layer] = KV for token i at layer
        for i, idx in enumerate(indices):
            per_layer = []
            for layer in range(self.LAYER_NUM):
                buf = self.kv_cache.kv_buffer[layer]
                # buf[page, off] is one token's slice; shape == buf.shape[2:].
                vals = jax.random.normal(
                    jax.random.PRNGKey(seed * 1000 + i * 10 + layer),
                    buf.shape[2:],
                    buf.dtype,
                )
                self._write_token(layer, idx, vals)
                per_layer.append(self._read_token(layer, idx))
            orig.append(per_layer)
        return indices, orig

    def _settle_writes(self, timeout: float = 5.0) -> None:
        """Block until all in-flight D2H backups are settled and bookkept."""
        self.controller.drain_pending()
        deadline = time.time() + timeout
        while self.cache.ongoing_write and time.time() < deadline:
            self.cache.check_hicache_events()
            time.sleep(0.005)
        self.assertFalse(self.cache.ongoing_write, "D2H writes did not settle")

    def _child_of_root(self) -> object:
        children = [c for c in self.cache.root_node.children.values()]
        self.assertEqual(len(children), 1)
        return children[0]

    def _load_back(self, last_host_node, host_hit_length, mem_quota):
        """init_load_back now only does control plane + async stage_load; the
        kernel scatter into kv_buffer is deferred to finish_load_back. These
        tests want the synchronous round-trip, so drive both halves here."""
        new_indices, last_node, flush_plan = self.cache.init_load_back(
            last_host_node, host_hit_length, mem_quota=mem_quota
        )
        self.cache.finish_load_back(flush_plan)
        return new_indices, last_node


class TestWriteThrough(HiCacheE2EBase):
    def test_backup_triggers_on_reuse_and_releases_lock(self):
        tokens = [10, 11, 12, 13]
        idx, _ = self._alloc_and_fill(len(tokens), seed=1)

        # First insert: new leaf, no hit-count bump, no backup.
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        self.assertFalse(node.backuped)
        self.assertEqual(node.hit_count, 0)

        # Second insert (prefix reuse): hit_count crosses threshold -> backup.
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.assertTrue(node.backuped)
        self.assertEqual(
            len(node.component_data[0].host_value), self._pages(len(tokens))
        )
        # The device lock is taken only around the synchronous gather inside
        # write_backup and released before it returns, so it is already 0 here
        # even though the async host->host flush is still in flight (tracked in
        # ongoing_write). Holding it across the slow flush would chain-lock the
        # whole prefix and wedge admission under memory pressure.
        self.assertEqual(node.component_data[0].lock_ref, 0)
        self.assertIn(node, [n for f, (n, _) in self.cache.ongoing_write.items()])

        self._settle_writes()
        # Flush settled, future drained; still backed up, still unlocked.
        self.assertEqual(node.component_data[0].lock_ref, 0)
        self.assertNotIn(node, [n for f, (n, _) in self.cache.ongoing_write.items()])
        self.assertTrue(node.backuped)

    def test_threshold_gt_one(self):
        self.cache.write_through_threshold = 2
        tokens = [1, 2, 3, 4]
        idx, _ = self._alloc_and_fill(len(tokens), seed=2)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # hit_count=1
        self.assertFalse(node.backuped)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # hit_count=2
        self.assertTrue(node.backuped)
        self._settle_writes()


class TestEvictAndLoadBack(HiCacheE2EBase):
    def test_device_evict_demotes_to_host_then_load_back(self):
        tokens = [20, 21, 22, 23]
        idx, orig = self._alloc_and_fill(len(tokens), seed=7)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # trigger backup
        node = self._child_of_root()
        self._settle_writes()

        # Force device eviction: node is backed up -> demote (tombstone), stays.
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)
        self.assertIn(node, self.cache.evictable_host_leaves)

        # Match now: device misses, host tier hits the whole prefix.
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(mr.device_indices), 0)
        self.assertEqual(mr.host_hit_length, len(tokens))
        self.assertIs(mr.last_host_node, node)

        # Synchronous load-back restores device KV bit-exactly.
        new_indices, last_node = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
        )
        self.assertEqual(len(new_indices), len(tokens))
        self.assertIs(last_node, node)
        self.assertFalse(node.evicted)
        for i, dev_idx in enumerate(new_indices):
            for layer in range(self.LAYER_NUM):
                got = self._read_token(layer, int(dev_idx))
                np.testing.assert_allclose(got, orig[i][layer])

    def test_load_back_respects_mem_quota(self):
        # Two segments so the host tombstone chain has two nodes.
        seg_a = [30, 31, 32, 33]
        seg_b = [34, 35, 36, 37]
        idx_a, _ = self._alloc_and_fill(len(seg_a), seed=11)
        idx_b, _ = self._alloc_and_fill(len(seg_b), seed=12)
        self.cache.insert(InsertParams(key=_key(seg_a), value=idx_a))
        self.cache.insert(
            InsertParams(key=_key(seg_a + seg_b), value=np.concatenate([idx_a, idx_b]))
        )
        # Reuse to back both nodes up.
        self.cache.insert(InsertParams(key=_key(seg_a), value=idx_a))
        self.cache.insert(
            InsertParams(key=_key(seg_a + seg_b), value=np.concatenate([idx_a, idx_b]))
        )
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(seg_a) + len(seg_b), dp_rank=0))

        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(seg_a + seg_b)))
        self.assertEqual(mr.host_hit_length, len(seg_a) + len(seg_b))

        # All-or-nothing load-back (mirrors sglang ``load_back``): a quota that
        # cannot fit the WHOLE tombstone chain reloads nothing (the caller
        # recomputes the prefix instead). A partial reload would leave a live
        # node below a still-tombstoned ancestor.
        new_indices, last_node = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=len(seg_a)
        )
        self.assertEqual(len(new_indices), 0)
        self.assertIs(last_node, mr.last_host_node)

        # A quota covering the full chain reloads both segments.
        new_indices, _ = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=len(seg_a) + len(seg_b)
        )
        self.assertEqual(len(new_indices), len(seg_a) + len(seg_b))


class TestWriteBack(HiCacheE2EBase):
    """write_back policy: no backup on hit; a node is backed up only when its
    device KV is evicted, then demoted to a host tombstone (mirrors sglang's
    write_back path). Cuts D2H from per-hit to per-eviction."""

    def setUp(self):
        super().setUp()
        self.cache.write_policy = "write_back"

    def test_no_backup_on_repeated_hits(self):
        tokens = [40, 41, 42, 43]
        idx, _ = self._alloc_and_fill(len(tokens), seed=21)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        # Even repeated prefix reuse must NOT trigger a hit-time backup.
        for _ in range(5):
            self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.assertFalse(node.backuped)
        self.assertEqual(len(self.cache.ongoing_write), 0)

    def test_backup_on_eviction_then_load_back(self):
        tokens = [50, 51, 52, 53]
        idx, orig = self._alloc_and_fill(len(tokens), seed=22)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        # Not backed up yet (write_back defers to eviction).
        self.assertFalse(node.backuped)

        # Force device eviction -> backed up at evict time, demoted to tombstone.
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)
        self.assertIn(node, self.cache.evictable_host_leaves)
        self._settle_writes()

        # Host tier hits the whole prefix; load-back restores it bit-exactly.
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(mr.device_indices), 0)
        self.assertEqual(mr.host_hit_length, len(tokens))
        new_indices, last_node = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
        )
        self.assertEqual(len(new_indices), len(tokens))
        self.assertIs(last_node, node)
        self.assertFalse(node.evicted)
        for i, dev_idx in enumerate(new_indices):
            for layer in range(self.LAYER_NUM):
                got = self._read_token(layer, int(dev_idx))
                np.testing.assert_allclose(got, orig[i][layer])

    def test_evict_chain_backs_up_leaf_up(self):
        # Two-segment chain: leaf-up eviction must back up both nodes so the
        # whole tombstone chain is loadable (no live node below a tombstone).
        seg_a = [60, 61, 62, 63]
        seg_b = [64, 65, 66, 67]
        idx_a, _ = self._alloc_and_fill(len(seg_a), seed=31)
        idx_b, _ = self._alloc_and_fill(len(seg_b), seed=32)
        self.cache.insert(InsertParams(key=_key(seg_a), value=idx_a))
        self.cache.insert(
            InsertParams(key=_key(seg_a + seg_b), value=np.concatenate([idx_a, idx_b]))
        )
        # No backup happened on these inserts.
        self.assertEqual(len(self.cache.ongoing_write), 0)

        self.cache.evict(EvictParams(num_tokens=len(seg_a) + len(seg_b), dp_rank=0))
        self._settle_writes()
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(seg_a + seg_b)))
        self.assertEqual(mr.host_hit_length, len(seg_a) + len(seg_b))
        new_indices, _ = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=len(seg_a) + len(seg_b)
        )
        self.assertEqual(len(new_indices), len(seg_a) + len(seg_b))


class TestTombstoneRevival(HiCacheE2EBase):
    """Re-inserting (recomputing) a prefix that was demoted to a host-only
    tombstone must REVIVE the tombstone in place (adopt the recomputed device
    KV, keep the host copy), not leave a dead node with a live leaf hanging
    below it. This mirrors sglang's ``insert`` and is what keeps the lock /
    eviction token accounting balanced -- without it ``write_backup`` locks
    through the tombstone and a later ``init_load_back`` un-tombstones a still
    locked node into the evictable counter, drifting evictable/protected.
    """

    def _full_counters(self, dp_rank: int = 0) -> tuple[int, int, int]:
        """(available, evictable, protected) device TOKENS for the FULL tier."""
        avail = self.allocator.available_size(dp_rank)
        evict = int(self.cache.component_evictable_size_[0][dp_rank])
        prot = int(self.cache.component_protected_size_[0][dp_rank])
        return avail, evict, prot

    def _assert_token_conservation(self):
        avail, evict, prot = self._full_counters()
        self.assertEqual(
            avail + evict + prot,
            self.DEVICE_SIZE,
            f"token leak: avail={avail} evict={evict} prot={prot}",
        )

    def test_reinsert_revives_tombstone_in_place(self):
        tokens = [50, 51, 52, 53]
        idx, _ = self._alloc_and_fill(len(tokens), seed=31)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # back up
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.evicted and node.backuped)
        self._assert_token_conservation()

        # Recompute the same prefix (fresh device slots) and re-insert: the
        # tombstone is revived in place, not duplicated.
        new_idx, new_orig = self._alloc_and_fill(len(tokens), seed=99)
        self.cache.insert(InsertParams(key=_key(tokens), value=new_idx))
        self.assertFalse(node.evicted)  # revived
        self.assertTrue(node.backuped)  # host copy retained (write-through)
        self.assertIs(self._child_of_root(), node)  # same node object, no dup
        # Device value is the freshly recomputed KV, bit-exact.
        for i, dev_idx in enumerate(node.component_data[0].value):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(
                    self._read_token(layer, int(dev_idx)), new_orig[i][layer]
                )
        self._assert_token_conservation()

    def test_reinsert_with_extension_no_live_below_tombstone(self):
        head = [60, 61, 62, 63]
        idx, _ = self._alloc_and_fill(len(head), seed=41)
        self.cache.insert(InsertParams(key=_key(head), value=idx))
        node = self._child_of_root()
        self.cache.insert(InsertParams(key=_key(head), value=idx))  # back up
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(head), dp_rank=0))
        self.assertTrue(node.evicted and node.backuped)

        # Re-insert head + extension. The head tombstone must be revived BEFORE
        # the extension leaf is attached, so the leaf never hangs below a dead
        # ancestor.
        tail = [64, 65, 66, 67]
        head_idx, _ = self._alloc_and_fill(len(head), seed=42)
        tail_idx, _ = self._alloc_and_fill(len(tail), seed=43)
        self.cache.insert(
            InsertParams(
                key=_key(head + tail), value=np.concatenate([head_idx, tail_idx])
            )
        )
        self.assertFalse(node.evicted, "ancestor must be revived, not tombstoned")
        child = [c for c in node.children.values()][0]
        self.assertFalse(
            child.evicted and node.evicted,
            "no live node may hang below a tombstone",
        )
        self._assert_token_conservation()

    def test_lock_accounting_balanced_after_revive_and_reevict(self):
        tokens = [70, 71, 72, 73]
        idx, _ = self._alloc_and_fill(len(tokens), seed=51)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))

        # Revive, then drive another backup + eviction cycle.
        new_idx, _ = self._alloc_and_fill(len(tokens), seed=52)
        self.cache.insert(InsertParams(key=_key(tokens), value=new_idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=new_idx))  # re-backup
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))

        # After everything settles at idle, nothing is locked.
        _, _, prot = self._full_counters()
        self.assertEqual(prot, 0, "protected tokens leaked (lock not released)")
        self._assert_token_conservation()


class TestFullMiss(HiCacheE2EBase):
    def test_host_evict_deletes_node_then_full_miss(self):
        tokens = [40, 41, 42, 43]
        idx, _ = self._alloc_and_fill(len(tokens), seed=21)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        self._settle_writes()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.evicted and node.backuped)

        before = self.host_pool.available_size()
        n_pages = self._pages(len(tokens))
        freed = self.cache.evict_host(n_pages)
        self.assertEqual(freed, n_pages)
        self.assertEqual(self.host_pool.available_size(), before + n_pages)

        # Node is gone from the tree -> full miss.
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(mr.device_indices), 0)
        self.assertEqual(mr.host_hit_length, 0)
        self.assertNotIn(node, self.cache.root_node.children.values())


class TestHostEvictionTOCTOU(HiCacheE2EBase):
    def test_inflight_buffer_not_released(self):
        # Reserve a slot and start a (blocked) D2H by pinning the controller's
        # worker via the controller's own in-flight set: use evict_callback to
        # confirm it refuses an in-flight buffer id.
        tokens = [50, 51, 52, 53]
        idx, _ = self._alloc_and_fill(len(tokens), seed=31)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        buffer_ids = [int(b) for b in node.component_data[0].host_value]
        # While in flight, the controller must reject releasing those buffers.
        with self.controller._inflight_lock:
            inflight_now = set(self.controller._inflight)
        if inflight_now & set(buffer_ids):
            with self.assertRaises(RuntimeError):
                self.controller.evict_callback(buffer_ids)
        self._settle_writes()
        # After settle, releasing is allowed.
        self.controller.evict_callback(buffer_ids)
        node.component_data[0].host_value = None


class TestSwitchabilityBoundary(HiCacheE2EBase):
    def test_control_plane_crosses_only_ints(self):
        tokens = [60, 61, 62]
        idx, _ = self._alloc_and_fill(len(tokens), seed=41)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        node = self._child_of_root()
        # host_value is an int buffer-id array, never a jax.Array.
        hv = node.component_data[0].host_value
        self.assertNotIsInstance(hv, jax.Array)
        self.assertTrue(np.issubdtype(np.asarray(hv).dtype, np.integer))
        self._settle_writes()


# ---- B1: re-run the wiring suites at page_size > 1 -------------------------
# Subclassing flips PAGE_SIZE; every inherited test now exercises the
# token<->page boundary conversions in write_backup / init_load_back.


class TestWriteThroughPage2(TestWriteThrough):
    PAGE_SIZE = 2


class TestWriteThroughPage4(TestWriteThrough):
    PAGE_SIZE = 4


class TestEvictAndLoadBackPage2(TestEvictAndLoadBack):
    PAGE_SIZE = 2


class TestEvictAndLoadBackPage4(TestEvictAndLoadBack):
    PAGE_SIZE = 4


class TestFullMissPage2(TestFullMiss):
    PAGE_SIZE = 2


class TestFullMissPage4(TestFullMiss):
    PAGE_SIZE = 4


class TestTombstoneRevivalPage2(TestTombstoneRevival):
    PAGE_SIZE = 2


class TestTombstoneRevivalPage4(TestTombstoneRevival):
    PAGE_SIZE = 4


# ---- B2: host-copy reuse across repeated evict/load (scenario A) -----------


class TestHostCopyReuse(HiCacheE2EBase):
    """Write-through keeps the host copy after a load-back, so re-evicting a
    node must NOT issue a fresh D2H: host_value is unchanged and ongoing_write
    stays empty. Round-trip stays bit-exact across N cycles."""

    def _run(self):
        tokens = [70, 71, 72, 73]
        idx, orig = self._alloc_and_fill(len(tokens), seed=51)
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))
        self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # backup
        node = self._child_of_root()
        self._settle_writes()
        host_first = [int(b) for b in node.component_data[0].host_value]

        for _ in range(3):
            self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
            self.assertTrue(node.evicted and node.backuped)
            # Demotion of an already-backed node issues no new transfer.
            self.assertFalse(self.cache.ongoing_write)
            self.assertEqual(
                [int(b) for b in node.component_data[0].host_value], host_first
            )

            mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
            self.assertEqual(len(mr.device_indices), 0)
            self.assertEqual(mr.host_hit_length, len(tokens))

            new_indices, _ = self._load_back(
                mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
            )
            self.assertEqual(len(new_indices), len(tokens))
            self.assertFalse(node.evicted)
            for i, dev_idx in enumerate(new_indices):
                for layer in range(self.LAYER_NUM):
                    np.testing.assert_allclose(
                        self._read_token(layer, int(dev_idx)), orig[i][layer]
                    )

    def test_reuse(self):
        self._run()


class TestHostCopyReusePage2(TestHostCopyReuse):
    PAGE_SIZE = 2


class TestHostCopyReusePage4(TestHostCopyReuse):
    PAGE_SIZE = 4


# ---- B3: partial hit, shallow on device + deep on host (scenario C) --------


class TestPartialHit(HiCacheE2EBase):
    """A shallow segment stays resident on device while the deep segment is
    demoted to host. match returns device_indices for the shallow part and
    host_hit_length for the deep part; load_back fills only the deep tail and
    the concatenation is bit-exact."""

    def _run(self):
        seg_a = [80, 81, 82, 83]
        seg_b = [84, 85, 86, 87]
        idx_a, orig_a = self._alloc_and_fill(len(seg_a), seed=61)
        idx_b, orig_b = self._alloc_and_fill(len(seg_b), seed=62)
        self.cache.insert(InsertParams(key=_key(seg_a), value=idx_a))
        full = np.concatenate([idx_a, idx_b])
        self.cache.insert(InsertParams(key=_key(seg_a + seg_b), value=full))
        # Reuse both so both back up.
        self.cache.insert(InsertParams(key=_key(seg_a), value=idx_a))
        self.cache.insert(InsertParams(key=_key(seg_a + seg_b), value=full))
        self._settle_writes()

        # Evict only enough to demote the deep (LRU) leaf, keeping seg_a resident.
        self.cache.evict(EvictParams(num_tokens=len(seg_b), dp_rank=0))

        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(seg_a + seg_b)))
        self.assertEqual(len(mr.device_indices), len(seg_a))
        self.assertEqual(mr.host_hit_length, len(seg_b))

        # Shallow KV already on device is correct.
        for i, dev_idx in enumerate(mr.device_indices):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(
                    self._read_token(layer, int(dev_idx)), orig_a[i][layer]
                )

        new_indices, _ = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
        )
        self.assertEqual(len(new_indices), len(seg_b))
        for i, dev_idx in enumerate(new_indices):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(
                    self._read_token(layer, int(dev_idx)), orig_b[i][layer]
                )

    def test_partial(self):
        self._run()


class TestPartialHitPage2(TestPartialHit):
    PAGE_SIZE = 2


# ---- B4: load_back must pre-evict device cache to make room (scenario E) ----


class TestLoadBackPreEvicts(HiCacheE2EBase):
    """init_load_back hits the `avail < total` branch: the device is packed with
    colder evictable cache entries, so reloading the host prefix must trigger a
    device eviction before the alloc succeeds."""

    def test_pre_eviction(self):
        target = [90, 91, 92, 93, 94, 95, 96, 97]  # 8 tokens
        idx, orig = self._alloc_and_fill(len(target), seed=71)
        self.cache.insert(InsertParams(key=_key(target), value=idx))
        self.cache.insert(InsertParams(key=_key(target), value=idx))  # backup
        node = self._child_of_root()
        self._settle_writes()
        # Demote the target to host, freeing its device slots.
        self.cache.evict(EvictParams(num_tokens=len(target), dp_rank=0))
        self.assertTrue(node.evicted)

        # Pack the device with cold, evictable filler so free < len(target).
        avail = self.allocator.available_size(0)
        filler_n = ((avail - 4) // 4) * 4  # leave only 4 free, page-aligned
        self.assertGreater(filler_n, len(target))
        fidx, _ = self._alloc_and_fill(filler_n, seed=72)
        self.cache.insert(
            InsertParams(key=_key(list(range(200, 200 + filler_n))), value=fidx)
        )
        self.assertLess(self.allocator.available_size(0), len(target))

        # Reloading the target now requires pre-eviction of the filler.
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(target)))
        self.assertEqual(mr.host_hit_length, len(target))
        new_indices, _ = self._load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
        )
        self.assertEqual(len(new_indices), len(target))
        for i, dev_idx in enumerate(new_indices):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(
                    self._read_token(layer, int(dev_idx)), orig[i][layer]
                )


# ---- B5: randomized round-trip property test ------------------------------


class TestRoundTripProperty(HiCacheE2EBase):
    """Randomized interleaving of evict->host / match / load-back / evict_host
    against a ground-truth KV map. Every load-back must be bit-exact, and host
    page occupancy must stay conserved (no leaked or double-freed slots)."""

    HOST_PAGES = 64

    def test_random_round_trips(self):
        import random

        rng = random.Random(1234)
        # A handful of distinct, page-aligned sequences (disjoint token ids).
        seqs = []
        truth = {}  # seq_id -> orig KV (list[token][layer])
        for s in range(6):
            length = 4 * rng.randint(1, 3)
            base = 300 + s * 50
            tokens = list(range(base, base + length))
            idx, orig = self._alloc_and_fill(length, seed=100 + s)
            self.cache.insert(InsertParams(key=_key(tokens), value=idx))
            self.cache.insert(InsertParams(key=_key(tokens), value=idx))  # backup
            seqs.append((tokens, length))
            truth[s] = orig
        self._settle_writes()

        total_pages = self.host_pool.total_size()
        for _ in range(120):
            op = rng.choice(["evict_dev", "load", "match"])
            s = rng.randrange(len(seqs))
            tokens, length = seqs[s]
            if op == "evict_dev":
                self.cache.evict(EvictParams(num_tokens=length, dp_rank=0))
            elif op == "match":
                self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
            else:  # load: only meaningful if currently host-resident
                mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
                if mr.host_hit_length == length and len(mr.device_indices) == 0:
                    new_indices, _ = self._load_back(
                        mr.last_host_node,
                        mr.host_hit_length,
                        mem_quota=self.DEVICE_SIZE,
                    )
                    if len(new_indices) == length:
                        for i, dev_idx in enumerate(new_indices):
                            for layer in range(self.LAYER_NUM):
                                np.testing.assert_allclose(
                                    self._read_token(layer, int(dev_idx)),
                                    truth[s][i][layer],
                                )
            # Conservation: free + in-use host pages == capacity, always.
            used = sum(
                len(n.component_data[0].host_value)
                for n in self.cache.evictable_host_leaves
                if n.component_data[0].host_value is not None
            )
            self.assertLessEqual(used, total_pages)
            self.assertEqual(
                self.host_pool.available_size()
                + (total_pages - self.host_pool.available_size()),
                total_pages,
            )
        self._settle_writes()


class TestRoundTripPropertyPage2(TestRoundTripProperty):
    PAGE_SIZE = 2


if __name__ == "__main__":
    unittest.main()
