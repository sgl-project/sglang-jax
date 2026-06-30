"""TPU E2E tests for HiCache L1<->L2 control plane (Stage 3).

Validates the full UnifiedRadixCache + HiCacheController + HostKVPool pipeline
on real TPU with pinned_host memory. Runtime target: <1 minute.

Covers: write-through D2H backup, device eviction + tombstone demotion,
host-tier match, H2D load-back bit-exactness, tombstone revival, host eviction,
and token accounting conservation.
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


def _is_tpu() -> bool:
    try:
        return jax.default_backend() == "tpu"
    except RuntimeError:
        return False


@unittest.skipUnless(_is_tpu(), "requires TPU")
class TestHiCacheE2ETPU(unittest.TestCase):
    """Core HiCache L1<->L2 pipeline on a single TPU chip (tp=1, dp=1)."""

    PAGE_SIZE = 1
    HEAD_NUM = 4
    HEAD_DIM = 128
    LAYER_NUM = 2
    DEVICE_SIZE = 64
    HOST_PAGES = 32

    def setUp(self):
        self.mesh = create_device_mesh(ici_parallelism=[1, 1], dcn_parallelism=[1, 1])
        self.kv_cache = MHATokenToKVPool(
            size=self.DEVICE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=jnp.bfloat16,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            mesh=self.mesh,
            dp_size=1,
        )
        self.allocator = (
            PagedTokenToKVPoolAllocator(
                size=self.DEVICE_SIZE,
                page_size=self.PAGE_SIZE,
                kvcache=self.kv_cache,
                dp_size=1,
            )
            if self.PAGE_SIZE > 1
            else TokenToKVPoolAllocator(size=self.DEVICE_SIZE, kvcache=self.kv_cache, dp_size=1)
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
            dtype=jnp.bfloat16,
        )
        per_layer_shape = tuple(int(d) for d in self.kv_cache.kv_buffer[0].shape[1:])
        self.host_pool = LRUHostKVPool(
            device_pool=self.kv_cache,
            pool_size=self.HOST_PAGES,
            page_size=self.PAGE_SIZE,
            layer_num=self.LAYER_NUM,
            per_layer_shape=per_layer_shape,
            dtype=self.kv_cache.dtype,
            mesh=self.mesh,
            partition_spec=self.kv_cache.kv_sharding.spec,
        )
        self.controller = HiCacheController(self.host_pool, self.kv_cache)
        self.cache.host_pool = self.host_pool
        self.cache.hicache_controller = self.controller
        self.cache.hicache_enabled = True
        self.cache.write_through_threshold = 1
        self.cache.write_policy = "write_through"
        for c in self.cache._components_tuple:
            c._full_kv_pool_host = self.host_pool

    def tearDown(self):
        self.controller.shutdown()

    # ---- helpers ----

    def _settle(self, timeout=5.0):
        self.controller.drain_pending()
        deadline = time.time() + timeout
        while self.cache.ongoing_write and time.time() < deadline:
            self.cache.check_hicache_events()
            time.sleep(0.005)

    def _key(self, token_ids):
        return RadixKey(token_ids=token_ids, extra_key=None, dp_rank=0)

    def _read_token(self, layer, idx):
        page, off = int(idx) // self.PAGE_SIZE, int(idx) % self.PAGE_SIZE
        return np.asarray(jax.device_get(self.kv_cache.kv_buffer[layer]))[page, off]

    def _fill(self, n, seed):
        indices = self.allocator.alloc(n, dp_rank=0)
        self.assertIsNotNone(indices)
        orig = []
        for i, idx in enumerate(indices):
            per_layer = []
            for layer in range(self.LAYER_NUM):
                buf = self.kv_cache.kv_buffer[layer]
                vals = jax.random.normal(
                    jax.random.PRNGKey(seed * 1000 + i * 10 + layer),
                    buf.shape[2:],
                    jnp.float32,
                ).astype(buf.dtype)
                page, off = int(idx) // self.PAGE_SIZE, int(idx) % self.PAGE_SIZE
                self.kv_cache.kv_buffer[layer] = buf.at[page, off].set(
                    vals, out_sharding=buf.sharding
                )
                per_layer.append(self._read_token(layer, idx))
            orig.append(per_layer)
        return indices, orig

    def _token_counters(self):
        avail = self.allocator.available_size(0)
        evict = int(self.cache.component_evictable_size_[0][0])
        prot = int(self.cache.component_protected_size_[0][0])
        return avail, evict, prot

    def _load_back(self, host_node, hit_len, mem_quota):
        new_idx, last, plan = self.cache.init_load_back(host_node, hit_len, mem_quota=mem_quota)
        self.cache.finish_load_back(plan)
        return new_idx, last

    # ---- tests ----

    def test_write_through_backup_and_loadback(self):
        """D2H backup on hit, evict to tombstone, H2D load-back bit-exact."""
        tokens = [10, 11, 12, 13]
        idx, orig = self._fill(len(tokens), seed=1)
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))  # trigger backup
        self._settle()

        # Verify host_value is set and memory_kind is pinned_host.
        node = list(self.cache.root_node.children.values())[0]
        self.assertTrue(node.backuped)
        slot = self.host_pool._slots[int(node.component_data[0].host_value[0])]
        self.assertEqual(slot.sharding.memory_kind, "pinned_host")

        # Device eviction demotes to tombstone (stays in tree).
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.evicted and node.backuped)
        self.assertIn(node, self.cache.evictable_host_leaves)

        # Host match hits the full prefix.
        mr = self.cache.match_prefix(MatchPrefixParams(key=self._key(tokens)))
        self.assertEqual(len(mr.device_indices), 0)
        self.assertEqual(mr.host_hit_length, len(tokens))

        # Load-back restores bit-exact.
        new_idx, _ = self._load_back(mr.last_host_node, mr.host_hit_length, self.DEVICE_SIZE)
        self.assertEqual(len(new_idx), len(tokens))
        self.assertFalse(node.evicted)
        for i, dev_idx in enumerate(new_idx):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(self._read_token(layer, dev_idx), orig[i][layer])

    def test_tombstone_revival_and_token_conservation(self):
        """Re-insert revives tombstone in place; token accounting stays balanced."""
        tokens = [20, 21, 22, 23]
        idx, _ = self._fill(len(tokens), seed=2)
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        self._settle()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))

        # Revive with freshly allocated KV.
        new_idx, _ = self._fill(len(tokens), seed=99)
        self.cache.insert(InsertParams(key=self._key(tokens), value=new_idx))
        node = list(self.cache.root_node.children.values())[0]
        self.assertFalse(node.evicted)
        self.assertTrue(node.backuped)

        # Token conservation: avail + evictable + protected == DEVICE_SIZE.
        avail, evict, prot = self._token_counters()
        self.assertEqual(avail + evict + prot, self.DEVICE_SIZE)

        # Evict again, verify locks are released.
        self.cache.insert(InsertParams(key=self._key(tokens), value=new_idx))
        self._settle()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        _, _, prot = self._token_counters()
        self.assertEqual(prot, 0, "protected tokens leaked")

    def test_host_eviction_frees_and_full_miss(self):
        """Host eviction frees pages; subsequent match is a full miss."""
        tokens = [30, 31, 32, 33]
        idx, _ = self._fill(len(tokens), seed=3)
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        self._settle()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))

        n_pages = len(tokens) // self.PAGE_SIZE
        before = self.host_pool.available_size()
        freed = self.cache.evict_host(n_pages)
        self.assertEqual(freed, n_pages)
        self.assertEqual(self.host_pool.available_size(), before + n_pages)

        mr = self.cache.match_prefix(MatchPrefixParams(key=self._key(tokens)))
        self.assertEqual(len(mr.device_indices), 0)
        self.assertEqual(mr.host_hit_length, 0)

    def test_write_back_on_eviction(self):
        """write_back policy: no backup on hit; backed up only at device eviction."""
        self.cache.write_policy = "write_back"
        tokens = [40, 41, 42, 43]
        idx, orig = self._fill(len(tokens), seed=4)
        self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        for _ in range(3):
            self.cache.insert(InsertParams(key=self._key(tokens), value=idx))
        node = list(self.cache.root_node.children.values())[0]
        self.assertFalse(node.backuped, "write_back must not backup on reuse")
        self._settle()

        # Eviction triggers backup, then demotes to tombstone.
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.assertTrue(node.backuped)
        self.assertTrue(node.evicted)
        self._settle()

        # Load-back restores bit-exact.
        mr = self.cache.match_prefix(MatchPrefixParams(key=self._key(tokens)))
        self.assertEqual(mr.host_hit_length, len(tokens))
        new_idx, _ = self._load_back(mr.last_host_node, mr.host_hit_length, self.DEVICE_SIZE)
        for i, dev_idx in enumerate(new_idx):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_allclose(self._read_token(layer, dev_idx), orig[i][layer])


class TestHiCacheE2ETPUPage4(TestHiCacheE2ETPU):
    PAGE_SIZE = 4


if __name__ == "__main__":
    unittest.main()
