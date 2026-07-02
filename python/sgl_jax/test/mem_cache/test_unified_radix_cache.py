# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_unified_radix_cache.py -v
# specific shard information can be appended -s

import os

# Set up multi-device simulation for tensor parallelism
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    # Set JAX to use CPU for testing with simulated devices
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.common import release_kv_cache
from sgl_jax.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestUnifiedRadixCache(CustomTestCase):
    def setUp(self):
        self.devices = jax.devices()
        # Small KV pool dims: the radix tree only exercises the allocator's
        # index bookkeeping, so the actual KV buffers can stay tiny.
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    def _create_auto_device_setup(self, dp_size: int = 1):
        # create memory pool
        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
        )

        # create KV cache
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            dp_size=dp_size,
        )

        # create allocator
        allocator = TokenToKVPoolAllocator(
            size=self.pool_size,
            kvcache=kv_cache,
            dp_size=dp_size,
        )

        return req_pool, allocator

    def _create_unified_cache(self, req_pool, allocator, **kwargs):
        cache = UnifiedRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=kwargs.get("page_size", 1),
            disable=kwargs.get("disable", False),
            enable_kv_cache_events=kwargs.get("enable_kv_cache_events", False),
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        return cache

    def _create_radix_cache(self, req_pool, allocator, **kwargs):
        cache = RadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=kwargs.get("page_size", 1),
            disable=kwargs.get("disable", False),
            enable_kv_cache_events=kwargs.get("enable_kv_cache_events", False),
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        return cache

    def _assert_sizes_equal(self, radix, unified, dp_rank: int = 0):
        self.assertEqual(radix.evictable_size(dp_rank), unified.evictable_size(dp_rank))
        self.assertEqual(radix.protected_size(dp_rank), unified.protected_size(dp_rank))
        self.assertEqual(radix.total_size(), unified.total_size())

    def test_basic_insert_and_match(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value = np.arange(100, 108, dtype=np.int32)
        prefix_len = cache.insert(InsertParams(key=RadixKey(key), value=value)).prefix_len
        self.assertEqual(prefix_len, 0)  # new inserted, no prefix

        # full hit returns all 8 indices
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key)))
        self.assertTrue(np.array_equal(match_result.device_indices, value))
        # full hit: best_match_node is the last matched device node (HiCache off)
        self.assertIs(match_result.best_match_node, match_result.last_device_node)

        # second insert of the same key matches the full prefix
        prefix_len = cache.insert(InsertParams(key=RadixKey(key), value=value.copy())).prefix_len
        self.assertEqual(prefix_len, 8)
        self.assertEqual(cache.total_size(), 8)

    def test_partial_match_splits_node(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key1 = [1, 2, 3, 4, 5, 6, 7, 8]
        value1 = np.arange(100, 108, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key1), value=value1))

        # shares only the first 4 tokens, then diverges
        key2 = [1, 2, 3, 4, 99, 98, 97, 96]
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key2)))
        self.assertTrue(np.array_equal(match_result.device_indices, value1[:4]))
        self.assertIs(match_result.best_match_node, match_result.last_device_node)
        # the 8-token node was split: the matched node holds exactly 4 tokens
        self.assertEqual(len(match_result.last_device_node.key), 4)

        # inserting the diverging key reuses the 4-token shared prefix
        value2 = np.arange(200, 208, dtype=np.int32)
        prefix_len = cache.insert(InsertParams(key=RadixKey(key2), value=value2)).prefix_len
        self.assertEqual(prefix_len, 4)
        self.assertEqual(cache.total_size(), 12)

    def test_empty_and_no_match(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4])))

        # empty key
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey([])))
        self.assertIsInstance(match_result.device_indices, np.ndarray)
        self.assertEqual(match_result.device_indices.dtype, np.int32)
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertIs(match_result.best_match_node, cache.root_node)

        # completely disjoint key
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey([999, 888, 777])))
        self.assertIsInstance(match_result.device_indices, np.ndarray)
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertIs(match_result.best_match_node, cache.root_node)

    def test_disabled_cache(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator, disable=True)

        key = [1, 2, 3, 4, 5]
        insert_result = cache.insert(InsertParams(key=RadixKey(key))).prefix_len
        self.assertEqual(insert_result, 0)

        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key)))
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertIs(match_result.best_match_node, cache.root_node)

        evict_result = cache.evict(EvictParams(num_tokens=100))
        self.assertEqual(evict_result.num_tokens_evicted, 0)

        lock_result = cache.inc_lock_ref(cache.root_node)
        self.assertEqual(lock_result.delta, 0)

    def test_lock_protects_from_eviction(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value = np.arange(100, 108, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key), value=value))
        self.assertEqual(cache.evictable_size(0), 8)

        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key)))
        last_node = match_result.last_device_node

        # lock: delta is the evictable-size change (negative on acquire)
        lock_result = cache.inc_lock_ref(last_node)
        self.assertEqual(lock_result.delta, -8)
        self.assertEqual(cache.evictable_size(0), 0)
        self.assertEqual(cache.protected_size(0), 8)

        # locked node survives eviction
        evict_result = cache.evict(EvictParams(num_tokens=100, dp_rank=0))
        self.assertEqual(evict_result.num_tokens_evicted, 0)
        self.assertEqual(cache.total_size(), 8)

        # unlock restores the evictable/protected accounting
        cache.dec_lock_ref(last_node, lock_result.to_dec_params())
        self.assertEqual(cache.evictable_size(0), 8)
        self.assertEqual(cache.protected_size(0), 0)

        # now eviction frees the tokens
        evict_result = cache.evict(EvictParams(num_tokens=100, dp_rank=0))
        self.assertEqual(evict_result.num_tokens_evicted, 8)
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(0), 0)

    def test_lru_eviction_order(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key_a = [1, 2, 3, 4, 5, 6, 7, 8]
        value_a = np.arange(100, 108, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key_a), value=value_a))

        key_b = [101, 102, 103, 104, 105, 106, 107, 108]
        value_b = np.arange(200, 208, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key_b), value=value_b))

        # touch A so B becomes the LRU leaf
        cache.match_prefix(MatchPrefixParams(key=RadixKey(key_a)))

        evict_result = cache.evict(EvictParams(num_tokens=len(key_b)))
        self.assertEqual(evict_result.num_tokens_evicted, 8)
        self.assertEqual(cache.total_size(), 8)

        # B is gone, A is still fully matchable
        match_b = cache.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        self.assertEqual(len(match_b.device_indices), 0)
        self.assertIs(match_b.best_match_node, cache.root_node)

        match_a = cache.match_prefix(MatchPrefixParams(key=RadixKey(key_a)))
        self.assertTrue(np.array_equal(match_a.device_indices, value_a))

    def test_equivalence_with_radix_cache_page1(self):
        # fresh pools per cache so allocator state doesn't cross-contaminate
        radix_pool, radix_alloc = self._create_auto_device_setup()
        radix = self._create_radix_cache(radix_pool, radix_alloc, page_size=1)
        unified_pool, unified_alloc = self._create_auto_device_setup()
        unified = self._create_unified_cache(unified_pool, unified_alloc, page_size=1)

        key_a = [1, 2, 3, 4, 5, 6, 7, 8]
        value_a = np.arange(100, 108, dtype=np.int32)
        # shares the first 4 tokens with A
        key_b = [1, 2, 3, 4, 20, 21, 22, 23]
        value_b = np.arange(200, 208, dtype=np.int32)
        # disjoint
        key_c = [50, 51, 52, 53, 54, 55, 56, 57]
        value_c = np.arange(300, 308, dtype=np.int32)

        for key, value, expected_prefix in (
            (key_a, value_a, 0),
            (key_b, value_b, 4),
            (key_c, value_c, 0),
        ):
            radix_prefix = radix.insert(InsertParams(key=RadixKey(key), value=value.copy()))
            unified_prefix = unified.insert(
                InsertParams(key=RadixKey(key), value=value.copy())
            ).prefix_len
            self.assertEqual(radix_prefix, unified_prefix)
            self.assertEqual(radix_prefix, expected_prefix)
            self._assert_sizes_equal(radix, unified)

        # matches return identical indices
        radix_match_a = radix.match_prefix(MatchPrefixParams(key=RadixKey(key_a)))
        unified_match_a = unified.match_prefix(MatchPrefixParams(key=RadixKey(key_a)))
        self.assertTrue(np.array_equal(radix_match_a.device_indices, value_a))
        self.assertTrue(
            np.array_equal(radix_match_a.device_indices, unified_match_a.device_indices)
        )
        self._assert_sizes_equal(radix, unified)

        # lock the matched node in both caches and compare the accounting
        radix_lock = radix.inc_lock_ref(radix_match_a.last_device_node)
        unified_lock = unified.inc_lock_ref(unified_match_a.last_device_node)
        self.assertEqual(radix_lock.delta, unified_lock.delta)
        self.assertEqual(radix_lock.delta, -8)
        self._assert_sizes_equal(radix, unified)
        radix.dec_lock_ref(radix_match_a.last_device_node, radix_lock.to_dec_params())
        unified.dec_lock_ref(unified_match_a.last_device_node, unified_lock.to_dec_params())
        self._assert_sizes_equal(radix, unified)

        for key in ([1, 2, 3, 4], key_c):
            radix_match = radix.match_prefix(MatchPrefixParams(key=RadixKey(key)))
            unified_match = unified.match_prefix(MatchPrefixParams(key=RadixKey(key)))
            self.assertEqual(len(radix_match.device_indices), len(key))
            self.assertTrue(
                np.array_equal(radix_match.device_indices, unified_match.device_indices)
            )
            self._assert_sizes_equal(radix, unified)

        # B's tail is the LRU leaf: both caches evict the same 4 tokens
        radix_evicted = radix.evict(EvictParams(num_tokens=4))
        unified_evicted = unified.evict(EvictParams(num_tokens=4))
        self.assertEqual(radix_evicted.num_tokens_evicted, unified_evicted.num_tokens_evicted)
        self.assertEqual(radix_evicted.num_tokens_evicted, 4)
        self._assert_sizes_equal(radix, unified)

        radix_match_b = radix.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        unified_match_b = unified.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        self.assertEqual(len(radix_match_b.device_indices), 4)
        self.assertTrue(
            np.array_equal(radix_match_b.device_indices, unified_match_b.device_indices)
        )

        # evict everything remaining
        radix_evicted = radix.evict(EvictParams(num_tokens=1000))
        unified_evicted = unified.evict(EvictParams(num_tokens=1000))
        self.assertEqual(radix_evicted.num_tokens_evicted, unified_evicted.num_tokens_evicted)
        self.assertEqual(radix_evicted.num_tokens_evicted, 16)
        self.assertEqual(unified.total_size(), 0)
        self._assert_sizes_equal(radix, unified)

    def test_equivalence_with_radix_cache_page128(self):
        page_size = 128
        radix_pool, radix_alloc = self._create_auto_device_setup()
        radix = self._create_radix_cache(radix_pool, radix_alloc, page_size=page_size)
        unified_pool, unified_alloc = self._create_auto_device_setup()
        unified = self._create_unified_cache(unified_pool, unified_alloc, page_size=page_size)

        # 300 tokens, page-aligned to 256
        key_a = list(range(1000, 1300))
        aligned_len = len(key_a) // page_size * page_size
        self.assertEqual(aligned_len, 256)
        value_a = np.arange(1, 1 + aligned_len, dtype=np.int32)

        # RadixCache.insert does not page-align (its callers pre-align), so the
        # reference gets the pre-aligned key; the unified cache gets the raw
        # un-aligned key to exercise its internal page-align truncation.
        radix_prefix = radix.insert(
            InsertParams(key=RadixKey(key_a[:aligned_len]), value=value_a.copy())
        )
        unified_prefix = unified.insert(
            InsertParams(key=RadixKey(key_a), value=value_a.copy())
        ).prefix_len
        self.assertEqual(radix_prefix, unified_prefix)
        self.assertEqual(radix_prefix, 0)
        self.assertEqual(unified.total_size(), 256)
        self._assert_sizes_equal(radix, unified)

        # match with a 270-token shared-prefix key: aligned down to 256
        key_b = key_a[:270]
        radix_match = radix.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        unified_match = unified.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        self.assertEqual(len(radix_match.device_indices), 256)
        self.assertEqual(len(unified_match.device_indices), 256)
        self.assertTrue(np.array_equal(radix_match.device_indices, value_a))
        self.assertTrue(np.array_equal(radix_match.device_indices, unified_match.device_indices))
        self.assertIs(unified_match.best_match_node, unified_match.last_device_node)
        self._assert_sizes_equal(radix, unified)

        # re-insert: the whole aligned prefix is matched
        radix_prefix = radix.insert(
            InsertParams(key=RadixKey(key_a[:aligned_len]), value=value_a.copy())
        )
        unified_prefix = unified.insert(
            InsertParams(key=RadixKey(key_a), value=value_a.copy())
        ).prefix_len
        self.assertEqual(radix_prefix, unified_prefix)
        self.assertEqual(radix_prefix, 256)
        self._assert_sizes_equal(radix, unified)

        # diverging key sharing exactly one 128-token page: splits the node
        key_c = key_a[:128] + list(range(5000, 5172))
        value_c = np.arange(301, 301 + aligned_len, dtype=np.int32)
        radix_prefix = radix.insert(
            InsertParams(key=RadixKey(key_c[:aligned_len]), value=value_c.copy())
        )
        unified_prefix = unified.insert(
            InsertParams(key=RadixKey(key_c), value=value_c.copy())
        ).prefix_len
        self.assertEqual(radix_prefix, unified_prefix)
        self.assertEqual(radix_prefix, 128)
        self.assertEqual(unified.total_size(), 384)
        self._assert_sizes_equal(radix, unified)

        # touch A so C's tail becomes the LRU leaf, then evict one page
        radix_match = radix.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        unified_match = unified.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        self.assertTrue(np.array_equal(radix_match.device_indices, unified_match.device_indices))

        radix_evicted = radix.evict(EvictParams(num_tokens=128))
        unified_evicted = unified.evict(EvictParams(num_tokens=128))
        self.assertEqual(radix_evicted.num_tokens_evicted, unified_evicted.num_tokens_evicted)
        self.assertEqual(radix_evicted.num_tokens_evicted, 128)
        self._assert_sizes_equal(radix, unified)

        # A is still fully matchable after C's tail eviction
        radix_match = radix.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        unified_match = unified.match_prefix(MatchPrefixParams(key=RadixKey(key_b)))
        self.assertTrue(np.array_equal(radix_match.device_indices, value_a))
        self.assertTrue(np.array_equal(radix_match.device_indices, unified_match.device_indices))

        radix_evicted = radix.evict(EvictParams(num_tokens=1000))
        unified_evicted = unified.evict(EvictParams(num_tokens=1000))
        self.assertEqual(radix_evicted.num_tokens_evicted, unified_evicted.num_tokens_evicted)
        self.assertEqual(radix_evicted.num_tokens_evicted, 256)
        self.assertEqual(unified.total_size(), 0)
        self._assert_sizes_equal(radix, unified)

    def test_dp_rank_isolation(self):
        req_pool, allocator = self._create_auto_device_setup(dp_size=2)
        cache = self._create_unified_cache(req_pool, allocator)

        # same token ids on both ranks: dp_rank namespaces them
        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value_rank0 = np.arange(10, 18, dtype=np.int32)
        value_rank1 = np.arange(20, 28, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key, None, 0), value=value_rank0))
        cache.insert(InsertParams(key=RadixKey(key, None, 1), value=value_rank1))

        self.assertEqual(cache.evictable_size(0), 8)
        self.assertEqual(cache.evictable_size(1), 8)
        self.assertEqual(cache.total_size(), 16)

        match_rank0 = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, None, 0)))
        self.assertTrue(np.array_equal(match_rank0.device_indices, value_rank0))
        match_rank1 = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, None, 1)))
        self.assertTrue(np.array_equal(match_rank1.device_indices, value_rank1))

        # evicting rank 0 leaves rank 1 untouched
        evict_result = cache.evict(EvictParams(num_tokens=100, dp_rank=0))
        self.assertEqual(evict_result.num_tokens_evicted, 8)
        self.assertEqual(cache.evictable_size(0), 0)
        self.assertEqual(cache.evictable_size(1), 8)
        self.assertEqual(cache.total_size(), 8)

        match_rank0 = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, None, 0)))
        self.assertEqual(len(match_rank0.device_indices), 0)
        self.assertIs(match_rank0.best_match_node, cache.root_node)

        match_rank1 = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, None, 1)))
        self.assertTrue(np.array_equal(match_rank1.device_indices, value_rank1))

    def test_extra_key_isolation(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        # same token ids under two extra_keys: extra_key namespaces them
        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value_a = np.arange(10, 18, dtype=np.int32)
        value_b = np.arange(20, 28, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key, "lora-a"), value=value_a))
        cache.insert(InsertParams(key=RadixKey(key, "lora-b"), value=value_b))
        self.assertEqual(cache.total_size(), 16)

        match_a = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, "lora-a")))
        self.assertTrue(np.array_equal(match_a.device_indices, value_a))
        match_b = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, "lora-b")))
        self.assertTrue(np.array_equal(match_b.device_indices, value_b))

        # an unknown extra_key matches nothing
        match_c = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, "lora-c")))
        self.assertEqual(len(match_c.device_indices), 0)
        self.assertIs(match_c.best_match_node, cache.root_node)

    def test_evict_device_leaf_tombstones_and_unlinks(self):
        # Deliberate white-box coverage of the tombstone contract:
        # _cascade_evict sets the FULL component value to None ("tombstone")
        # after the component data is freed. In stage 1 the public API can
        # never observe an evicted-but-alive node (eviction immediately
        # unlinks the leaf), so this drives _evict_device_leaf directly until
        # the host tier (HiCache) lands and makes tombstones reachable.
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value = np.arange(100, 108, dtype=np.int32)
        cache.insert(InsertParams(key=RadixKey(key), value=value))
        leaf = cache.match_prefix(MatchPrefixParams(key=RadixKey(key))).last_device_node

        # Precondition: the leaf is live and evictable.
        self.assertIn(leaf, cache.evictable_device_leaves)
        self.assertIs(leaf.evicted, False)

        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_device_leaf(leaf, tracker)

        self.assertEqual(tracker[ComponentType.FULL], 8)
        # Tombstone: parent link kept, FULL value is None.
        self.assertIs(leaf.evicted, True)
        # The leaf is detached from the tree (key is a single chain off root).
        self.assertNotIn(leaf, cache.root_node.children.values())
        self.assertNotIn(leaf, cache.evictable_device_leaves)
        self.assertEqual(cache.evictable_size(0), 0)
        self.assertEqual(cache.total_size(), 0)

        # A fresh match no longer sees the key.
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key)))
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertIs(match_result.best_match_node, cache.root_node)

    def test_reset(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_unified_cache(req_pool, allocator)

        key1 = [1, 2, 3, 4, 5, 6, 7, 8]
        cache.insert(InsertParams(key=RadixKey(key1), value=np.arange(100, 108, dtype=np.int32)))
        cache.insert(
            InsertParams(key=RadixKey([20, 21, 22]), value=np.arange(200, 203, dtype=np.int32))
        )

        # lock one path so both buckets are non-empty before the reset
        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key1)))
        cache.inc_lock_ref(match_result.last_device_node)
        self.assertEqual(cache.protected_size(0), 8)
        self.assertEqual(cache.evictable_size(0), 3)
        self.assertEqual(cache.total_size(), 11)

        cache.reset()

        self.assertEqual(cache.evictable_size(0), 0)
        self.assertEqual(cache.protected_size(0), 0)
        self.assertEqual(cache.total_size(), 0)

        match_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(key1)))
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertIs(match_result.best_match_node, cache.root_node)


class MockRequest:
    """mock request object for testing cache request functionality"""

    def __init__(
        self,
        req_pool_idx,
        origin_input_ids,
        output_ids,
        fill_ids,
        prefix_indices,
        last_node,
        extra_key=None,
        dp_rank=None,
    ):
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = origin_input_ids
        self.output_ids = output_ids
        self.fill_ids = fill_ids
        self.prefix_indices = prefix_indices
        self.last_node = last_node
        self.extra_key = extra_key
        self.dp_rank = dp_rank
        self.rid = "mock-req"
        # Match legacy length formula so cache_finished_req frees the same range.
        self.kv_committed_len = len(origin_input_ids) + max(len(output_ids) - 1, 0)
        self.kv_allocated_len = self.kv_committed_len
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        # Mirrors prepare_for_extend: page-aligned matched-prefix length at
        # extend time. Tests construct mock reqs without going through extend,
        # so default to len(prefix_indices) (== matched prefix in the simple
        # mock setup; no unaligned tail because tests use page_size=1).
        self.cache_protected_len = len(prefix_indices)

    def pop_committed_kv_cache(self) -> int:
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        assert not self.kv_overallocated_freed
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


class TestUnifiedRadixCacheWithRequests(CustomTestCase):
    """Drive cache_unfinished_req / cache_finished_req through RadixCache and
    UnifiedRadixCache with identical inputs and compare every observable."""

    def setUp(self):
        self.devices = jax.devices()
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    def _create_stack(self, cache_cls, disable: bool = False):
        """One independent req-pool + allocator + cache stack (page_size=1)."""
        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
        )
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
        )
        allocator = TokenToKVPoolAllocator(
            size=self.pool_size,
            kvcache=kv_cache,
        )
        cache = cache_cls(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=disable,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        return req_pool, allocator, cache

    def _assert_sizes_equal(self, radix, unified, dp_rank: int = 0):
        self.assertEqual(radix.evictable_size(dp_rank), unified.evictable_size(dp_rank))
        self.assertEqual(radix.protected_size(dp_rank), unified.protected_size(dp_rank))
        self.assertEqual(radix.total_size(), unified.total_size())

    def test_cache_unfinished_then_finished_equivalence(self):
        radix_pool, radix_alloc, radix = self._create_stack(RadixCache)
        unified_pool, unified_alloc, unified = self._create_stack(UnifiedRadixCache)
        stacks = (
            (radix, radix_pool, radix_alloc),
            (unified, unified_pool, unified_alloc),
        )

        # --- chunked prefill: cache_unfinished_req ---
        prefill_ids = list(range(1, 13))  # 12 tokens
        reqs = []
        for cache, pool, alloc in stacks:
            kv_indices = alloc.alloc(len(prefill_ids), dp_rank=0)
            self.assertIsNotNone(kv_indices)
            pool.write((0, slice(0, len(prefill_ids))), kv_indices)
            req = MockRequest(
                req_pool_idx=0,
                origin_input_ids=list(prefill_ids),
                output_ids=[],
                fill_ids=list(prefill_ids),
                prefix_indices=np.empty((0,), dtype=np.int32),
                last_node=cache.root_node,
            )
            req.last_matched_prefix_len = 0
            cache.cache_unfinished_req(req)
            reqs.append(req)
        radix_req, unified_req = reqs

        # Request observables match across implementations.
        self.assertEqual(radix_req.cache_protected_len, unified_req.cache_protected_len)
        self.assertEqual(radix_req.cache_protected_len, 12)
        self.assertEqual(radix_req.last_matched_prefix_len, unified_req.last_matched_prefix_len)
        self.assertEqual(radix_req.last_matched_prefix_len, 12)
        self.assertEqual(len(radix_req.prefix_indices), len(unified_req.prefix_indices))
        self.assertEqual(len(radix_req.prefix_indices), 12)
        # Cache accounting matches: the cached prefix is locked for the
        # still-running request.
        self._assert_sizes_equal(radix, unified)
        self.assertEqual(unified.evictable_size(0), 0)
        self.assertEqual(unified.protected_size(0), 12)
        self.assertEqual(unified.total_size(), 12)
        # The unified req now points at the locked path, not root.
        self.assertIsNot(unified_req.last_node, unified.root_node)

        # --- completion: cache_finished_req on the same req objects ---
        # Production flow: cache_unfinished during chunked prefill, then
        # cache_finished at completion on the same req; pop_committed_kv_cache
        # is called exactly once, at finish.
        output_ids = [13, 14, 15, 16, 17]
        # kv is committed for all but the last output token.
        num_new_kv = len(output_ids) - 1
        for (cache, pool, alloc), req in zip(stacks, reqs):
            new_kv_indices = alloc.alloc(num_new_kv, dp_rank=0)
            self.assertIsNotNone(new_kv_indices)
            pool.write(
                (0, slice(len(prefill_ids), len(prefill_ids) + num_new_kv)),
                new_kv_indices,
            )
            req.output_ids = list(output_ids)
            req.fill_ids = req.origin_input_ids + req.output_ids
            # Keep the MockRequest committed-kv formula in sync after mutation.
            req.kv_committed_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            req.kv_allocated_len = req.kv_committed_len
            cache.cache_finished_req(req)

        # The lock is released and the full committed range is cached.
        self._assert_sizes_equal(radix, unified)
        self.assertEqual(unified.protected_size(0), 0)
        self.assertEqual(unified.evictable_size(0), 16)
        self.assertEqual(unified.total_size(), 16)

        full_sequence = prefill_ids + output_ids
        radix_match = radix.match_prefix(MatchPrefixParams(key=RadixKey(full_sequence)))
        unified_match = unified.match_prefix(MatchPrefixParams(key=RadixKey(full_sequence)))
        self.assertEqual(len(radix_match.device_indices), len(unified_match.device_indices))
        self.assertEqual(len(unified_match.device_indices), 16)
        self.assertIs(unified_match.best_match_node, unified_match.last_device_node)

    def test_cache_finished_req_disabled_unified(self):
        """test cache finished request disabled"""
        _, _, disabled_cache = self._create_stack(UnifiedRadixCache, disable=True)

        # create mock request
        mock_req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
            fill_ids=[1, 2, 3, 4],
            prefix_indices=jnp.array([1, 2, 3]),
            last_node=disabled_cache.root_node,
        )

        # should execute normally without throwing exception
        try:
            disabled_cache.cache_finished_req(mock_req)
        except Exception as e:
            self.fail(f"cache_finished_req raised an exception: {e}")
        # the committed kv range was popped (and freed) exactly once
        self.assertTrue(mock_req.kv_committed_freed)

    def test_cache_unfinished_req_disabled_unified(self):
        _, _, disabled_cache = self._create_stack(UnifiedRadixCache, disable=True)

        # create mock request
        mock_req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3],
            output_ids=[4],
            fill_ids=[1, 2, 3, 4],
            prefix_indices=jnp.array([1, 2, 3]),
            last_node=disabled_cache.root_node,
        )

        # should execute normally without throwing exception
        try:
            disabled_cache.cache_unfinished_req(mock_req)
        except Exception as e:
            self.fail(f"cache_unfinished_req raised an exception: {e}")


class TestUnifiedRadixCacheEffectiveCacheLen(CustomTestCase):
    """A prepare_for_caching_req cap truncates the inserted key, or skips the insert when 0."""

    def setUp(self):
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    def _create_stack(self):
        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
        )
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
        )
        allocator = TokenToKVPoolAllocator(size=self.pool_size, kvcache=kv_cache)
        cache = UnifiedRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        return req_pool, allocator, cache

    @staticmethod
    def _cap_to(cache, cl):
        """Cap the effective cache length at ``cl``, standing in for a recurrent cap."""
        full = cache.components[ComponentType.FULL]
        full.prepare_for_caching_req = lambda req, insert_params, token_ids_len, is_finished: cl

    def test_no_op_equivalence_unfinished_inserts_full_length(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0

        cache.cache_unfinished_req(req)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(match.device_indices), 12)
        self.assertEqual(cache.total_size(), 12)

    def test_no_op_equivalence_finished_inserts_full_length(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)

        cache.cache_finished_req(req)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(match.device_indices), 12)
        self.assertEqual(cache.total_size(), 12)

    def test_truncation_caps_inserted_key_unfinished_keeps_tail(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 8)
        cache.cache_unfinished_req(req)

        self.assertEqual(cache.total_size(), 8)
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids[:8])))
        self.assertEqual(len(match.device_indices), 8)
        full_match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(full_match.device_indices), 8)
        self.assertEqual(req.cache_protected_len, 8)
        self.assertEqual(allocator.available_size(0), avail_before)
        self.assertEqual(len(req.prefix_indices), 12)

    def test_truncation_caps_inserted_key_and_frees_tail_finished(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 8)
        cache.cache_finished_req(req)

        self.assertEqual(cache.total_size(), 8)
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids[:8])))
        self.assertEqual(len(match.device_indices), 8)
        self.assertEqual(allocator.available_size(0), avail_before + 4)

    def test_skip_insert_on_zero_unfinished(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0
        req.cache_protected_len = 0

        cleanup_calls = []
        full = cache.components[ComponentType.FULL]
        orig_cleanup = full.cleanup_after_caching_req
        full.cleanup_after_caching_req = lambda *a, **k: cleanup_calls.append(True) or orig_cleanup(
            *a, **k
        )

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 0)
        cache.cache_unfinished_req(req)

        self.assertEqual(cache.total_size(), 0)
        self.assertIs(req.last_node, cache.root_node)
        self.assertTrue(cleanup_calls)
        self.assertEqual(allocator.available_size(0), avail_before)
        self.assertEqual(req.cache_protected_len, 0)
        # Leak fix: prefix_indices advanced to the committed KV, not left stale.
        np.testing.assert_array_equal(req.prefix_indices, kv_indices)
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)
        self._cap_to(cache, 0)
        cache.cache_finished_req(req)
        self.assertEqual(allocator.available_size(0), avail_before + len(token_ids))

    def test_skip_insert_on_zero_unfinished_preserves_protected_prefix(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 4
        req.cache_protected_len = 4

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 0)
        cache.cache_unfinished_req(req)

        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(allocator.available_size(0), avail_before)
        self.assertEqual(req.cache_protected_len, 4)
        np.testing.assert_array_equal(req.prefix_indices, kv_indices)
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)
        self._cap_to(cache, 0)
        cache.cache_finished_req(req)
        self.assertEqual(allocator.available_size(0), avail_before + (len(token_ids) - 4))

    def test_skip_insert_on_zero_finished(self):
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)
        req.cache_protected_len = 0

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 0)
        cache.cache_finished_req(req)

        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(allocator.available_size(0), avail_before + 12)

    def test_zero_cache_len_skip_then_continuation_extends_without_overlap(self):
        # A stale prefix after an off-boundary skip would re-extend over round 1's
        # pages and orphan them (round 2 continues without re-matching).
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))  # 12-token prompt, chunked 6 + 6
        avail_start = allocator.available_size(0)

        # --- round 1: off-boundary skip ---
        chunk1 = allocator.alloc(6, dp_rank=0)
        pool.write((0, slice(0, 6)), chunk1)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids[:6]),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
        )
        req.last_matched_prefix_len = 0
        req.cache_protected_len = 0

        self._cap_to(cache, 0)
        cache.cache_unfinished_req(req)

        np.testing.assert_array_equal(req.prefix_indices, chunk1)

        # --- round 2: continuation (mirrors prepare_for_extend) ---
        offset = len(req.prefix_indices)
        extend_len = len(token_ids) - offset
        self.assertEqual(offset, 6)
        self.assertEqual(extend_len, 6)
        chunk2 = allocator.alloc(extend_len, dp_rank=0)
        self.assertEqual(set(chunk1.tolist()) & set(chunk2.tolist()), set())
        pool.write((0, slice(offset, offset + extend_len)), chunk2)
        req.fill_ids = list(token_ids)

        # --- finish ---
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)
        self._cap_to(cache, 0)
        cache.cache_finished_req(req)
        self.assertEqual(allocator.available_size(0), avail_start)


class _CowReq:
    """Minimal Req surrogate for recurrent CoW match recording."""

    def __init__(self, dp_rank=0):
        self.dp_rank = dp_rank
        self.recurrent_cow_src_index = None


class _ReleaseReq:
    """Req surrogate with a request-owned recurrent slot for release_kv_cache."""

    def __init__(self, req_pool_idx, recurrent_pool_idx, origin_input_ids, dp_rank=0):
        self.req_pool_idx = req_pool_idx
        self.recurrent_pool_idx = recurrent_pool_idx
        self.origin_input_ids = list(origin_input_ids)
        self.output_ids = []
        self.fill_ids = list(origin_input_ids)
        self.dp_rank = dp_rank
        self.extra_key = None
        self.last_node = None
        self.rid = "release-req"
        self.kv_committed_len = len(origin_input_ids)
        self.kv_allocated_len = self.kv_committed_len
        self.cache_protected_len = 0
        self.last_matched_prefix_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False

    def pop_committed_kv_cache(self) -> int:
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        assert not self.kv_overallocated_freed
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


class _RunningRecurrentReq:
    """Req surrogate for a running recurrent request mid chunked-prefill."""

    def __init__(self, req_pool_idx, recurrent_pool_idx, fill_ids, dp_rank=0):
        self.req_pool_idx = req_pool_idx
        self.recurrent_pool_idx = recurrent_pool_idx
        self.origin_input_ids = list(fill_ids)
        self.output_ids = []
        self.fill_ids = list(fill_ids)
        self.dp_rank = dp_rank
        self.extra_key = None
        self.last_node = None
        self.rid = "running-rec"
        self.prefix_indices = np.empty((0,), dtype=np.int32)
        self.cache_protected_len = 0
        self.last_matched_prefix_len = 0
        self.recurrent_cow_src_index = None
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False


class TestUnifiedRadixCacheRecurrent(CustomTestCase):
    """Cache-level recurrent component behavior at page_size=1."""

    def setUp(self):
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192
        self.recurrent_size = 8
        self.rec_num_heads = 8
        self.rec_head_dim = 16
        self.conv_kernel_size = 4

    def _create_recurrent_setup(self, dp_size: int = 1):
        state_pool = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            size=self.recurrent_size,
            num_heads=self.rec_num_heads,
            head_dim=self.rec_head_dim,
            conv_kernel_size=self.conv_kernel_size,
            mesh=mesh,
            dp_size=dp_size,
        )
        hybrid_pool = HybridReqToTokenPool(
            size=64,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
            recurrent_state_pool=state_pool,
            dp_size=dp_size,
        )
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            dp_size=dp_size,
        )
        allocator = TokenToKVPoolAllocator(size=self.pool_size, kvcache=kv_cache, dp_size=dp_size)
        cache = UnifiedRadixCache(
            req_to_token_pool=hybrid_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            tree_components=(ComponentType.FULL, ComponentType.RECURRENT),
        )
        return state_pool, hybrid_pool, allocator, cache

    def _commit(self, cache, pool, key, value):
        """Donate a running slot to the tree via insert (mirrors cache_finished_req)."""
        slot = pool.alloc_recurrent_slot(0)
        params = InsertParams(
            key=RadixKey(key, None, 0),
            value=value,
            recurrent_value=pool.recurrent_value_from_slot(slot),
        )
        result = cache.insert(params)
        return slot, result

    def _admit(self, cache, allocator, pool, key, dp_rank=0):
        """Admit a running request the way prefill does, so release can free it."""
        req = _ReleaseReq(
            req_pool_idx=None, recurrent_pool_idx=None, origin_input_ids=key, dp_rank=dp_rank
        )
        pool.alloc([req])  # assigns req_pool_idx + a recurrent running slot
        kv_indices = allocator.alloc(len(key), dp_rank=dp_rank)
        self.assertIsNotNone(kv_indices)
        pool.write((req.req_pool_idx, slice(0, len(key))), kv_indices)
        req.last_node = cache.root_node
        return req

    def test_supports_recurrent(self):
        _, _, _, cache = self._create_recurrent_setup()
        self.assertTrue(cache.supports_recurrent())

    def test_commit_match_cow_evict_ledger(self):
        state_pool, pool, _, cache = self._create_recurrent_setup()
        slots = pool.slots_per_rank

        key = [1, 2, 3, 4]
        value = np.arange(100, 104, dtype=np.int32)
        slot, result = self._commit(cache, pool, key, value)

        self.assertTrue(result.recurrent_committed)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        req = _CowReq(dp_rank=0)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(key, None, 0), cow_recurrent=True, req=req)
        )
        self.assertTrue(np.array_equal(match.device_indices, value))
        self.assertEqual(req.recurrent_cow_src_index, slot)

        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(pool.recurrent_available_size(0), slots)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_abort_and_retract_release_recurrent_slots(self):
        """Abort and retract must return request-owned recurrent slots to the free list."""
        state_pool, pool, allocator, cache = self._create_recurrent_setup()
        slots = pool.slots_per_rank

        # A committed leaf makes the aborted request's sequence a duplicate, so
        # its running slot is freed rather than donated.
        committed_key = [1, 2, 3, 4]
        committed_val = np.arange(100, 104, dtype=np.int32)
        _, commit_res = self._commit(cache, pool, committed_key, committed_val)
        self.assertTrue(commit_res.recurrent_committed)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        abort_req = self._admit(cache, allocator, pool, committed_key)  # duplicate seq
        retract_req = self._admit(cache, allocator, pool, [9, 10, 11])  # distinct seq
        self.assertIsNotNone(abort_req.recurrent_pool_idx)
        self.assertIsNotNone(retract_req.recurrent_pool_idx)
        self.assertEqual(pool.recurrent_available_size(0), slots - 3)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 2)

        release_kv_cache(abort_req, cache, is_insert=True)
        self.assertIsNone(abort_req.recurrent_pool_idx)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 1)

        release_kv_cache(retract_req, cache, is_insert=False)
        self.assertIsNone(retract_req.recurrent_pool_idx)

        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), slots)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_continuation_match_hits_committed_leaf(self):
        """A prompt extending a committed sequence stops on the leaf and records the CoW src."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value = np.arange(200, 208, dtype=np.int32)
        slot, result = self._commit(cache, pool, key, value)
        self.assertTrue(result.recurrent_committed)

        req = _CowReq(dp_rank=0)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(key + [9, 10], None, 0), cow_recurrent=True, req=req)
        )
        self.assertTrue(np.array_equal(match.device_indices, value))
        self.assertEqual(req.recurrent_cow_src_index, slot)

    def test_page1_running_recurrent_chunked_continuation_full_only_preserves_prefix(self):
        """A running request's re-match must use full_only=True, else the leaf-only
        recurrent validator collapses its cached FULL prefix to root."""
        state_pool, pool, allocator, cache = self._create_recurrent_setup()

        chunk1 = list(range(1, 9))
        req = _RunningRecurrentReq(req_pool_idx=None, recurrent_pool_idx=None, fill_ids=chunk1)
        pool.alloc([req])  # assigns req_pool_idx + a running recurrent slot
        self.assertIsNotNone(req.recurrent_pool_idx)
        running_slot = req.recurrent_pool_idx
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 1)

        kv1 = allocator.alloc(len(chunk1), dp_rank=0)
        self.assertIsNotNone(kv1)
        pool.write((req.req_pool_idx, slice(0, len(chunk1))), kv1)
        req.last_node = cache.root_node
        cache.cache_unfinished_req(req)

        # Unfinished caching donates nothing: the cached FULL nodes hold no
        # recurrent value.
        self.assertEqual(req.recurrent_pool_idx, running_slot)
        self.assertEqual(req.cache_protected_len, len(chunk1))
        self.assertEqual(len(req.prefix_indices), len(chunk1))
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 1)

        prefix_key = RadixKey(chunk1, None, 0)

        # (a) fixed params preserve the FULL prefix
        cow_probe = _CowReq(dp_rank=0)
        fixed = cache.match_prefix(
            MatchPrefixParams(key=prefix_key, full_only=True, cow_recurrent=False, req=cow_probe)
        )
        self.assertEqual(len(fixed.device_indices), len(chunk1))
        self.assertTrue(np.array_equal(fixed.device_indices, np.asarray(req.prefix_indices)))
        self.assertIsNone(cow_probe.recurrent_cow_src_index)

        # (b) a fresh prefill's params collapse the match to root
        wrong_probe = _CowReq(dp_rank=0)
        wrong = cache.match_prefix(
            MatchPrefixParams(key=prefix_key, full_only=False, cow_recurrent=True, req=wrong_probe)
        )
        self.assertEqual(len(wrong.device_indices), 0)
        self.assertIsNone(wrong_probe.recurrent_cow_src_index)

        # (c) second continuation
        chunk2 = list(range(9, 13))
        req.fill_ids = chunk1 + chunk2
        kv2 = allocator.alloc(len(chunk2), dp_rank=0)
        self.assertIsNotNone(kv2)
        pool.write((req.req_pool_idx, slice(len(chunk1), len(chunk1) + len(chunk2))), kv2)
        protected_before = req.cache_protected_len
        cache.cache_unfinished_req(req)

        self.assertEqual(req.cache_protected_len, len(chunk1) + len(chunk2))
        self.assertGreater(req.cache_protected_len, protected_before)
        self.assertEqual(len(req.prefix_indices), len(chunk1) + len(chunk2))
        self.assertEqual(req.recurrent_pool_idx, running_slot)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 1)

        full_key = RadixKey(chunk1 + chunk2, None, 0)
        again = cache.match_prefix(MatchPrefixParams(key=full_key, full_only=True))
        self.assertEqual(len(again.device_indices), len(chunk1) + len(chunk2))

    def test_shorter_match_misses_recurrent_leaf(self):
        """A match stopping one token short of the leaf (identical-prompt resubmit)
        lands on a recurrent-less split parent and falls back to root."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        key = [1, 2, 3, 4, 5, 6, 7, 8]
        value = np.arange(200, 208, dtype=np.int32)
        slot, result = self._commit(cache, pool, key, value)
        self.assertTrue(result.recurrent_committed)

        req = _CowReq(dp_rank=0)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(key[:-1], None, 0), cow_recurrent=True, req=req)
        )
        self.assertEqual(len(match.device_indices), 0)
        self.assertIsNone(req.recurrent_cow_src_index)

    def test_shorter_prefix_after_longer_skips_internal(self):
        """Committing on an internal node is skipped (leaf-only); the caller frees the slot."""
        state_pool, pool, _, cache = self._create_recurrent_setup()

        long_key = [1, 2, 3, 4]
        long_val = np.arange(10, 14, dtype=np.int32)
        long_slot, long_res = self._commit(cache, pool, long_key, long_val)
        self.assertTrue(long_res.recurrent_committed)

        short_key = [1, 2]
        short_val = np.arange(20, 22, dtype=np.int32)
        short_slot, short_res = self._commit(cache, pool, short_key, short_val)

        self.assertFalse(short_res.recurrent_committed)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        pool.free_recurrent_slot(short_slot, 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_lock_protects_recurrent_leaf(self):
        state_pool, pool, _, cache = self._create_recurrent_setup()
        key = [7, 8, 9, 10]
        value = np.arange(50, 54, dtype=np.int32)
        slot, _ = self._commit(cache, pool, key, value)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(key, None, 0)))
        leaf = match.last_device_node
        lock_result = cache.inc_lock_ref(leaf)

        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 1)

        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 0)

        cache.dec_lock_ref(leaf, lock_result.to_dec_params())
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 0)
        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), pool.slots_per_rank)

    def test_longer_prefix_after_shorter_frees_parent(self):
        """A leaf turning internal drops its recurrent slot instead of stranding it."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        slots = pool.slots_per_rank

        ab_slot, ab_res = self._commit(cache, pool, [1, 2], np.arange(10, 12, dtype=np.int32))
        self.assertTrue(ab_res.recurrent_committed)
        _, abcd_res = self._commit(cache, pool, [1, 2, 3, 4], np.arange(20, 24, dtype=np.int32))
        self.assertTrue(abcd_res.recurrent_committed)

        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), slots)

    def test_internal_transition_while_locked_frees_on_unlock(self):
        """A locked node turning internal defers the recurrent free to the final unlock."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        self._commit(cache, pool, [1, 2], np.arange(10, 12, dtype=np.int32))
        ab = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2], None, 0))).last_device_node

        lock = cache.inc_lock_ref(ab)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 1)

        self._commit(cache, pool, [1, 2, 3, 4], np.arange(20, 24, dtype=np.int32))
        self.assertIsNotNone(ab.component_data[ComponentType.RECURRENT].value)
        free_before = pool.recurrent_available_size(0)

        cache.dec_lock_ref(ab, lock.to_dec_params())
        self.assertIsNone(ab.component_data[ComponentType.RECURRENT].value)
        self.assertEqual(pool.recurrent_available_size(0), free_before + 1)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_copy_slots_bitwise_clone_all_layers(self):
        """copy_slots clones src->dst bitwise across all layers; src==0 rows are no-ops."""
        state_pool, _, _, _ = self._create_recurrent_setup()
        src, dst = 2, 5

        # out_sharding is required because the test runs under an explicit mesh.
        rec_spec = state_pool.recurrent_sharding.spec
        conv_spec = state_pool.conv_sharding.spec
        for layer in range(state_pool.num_linear_recurrent_layers):
            state_pool.recurrent_buffers[layer] = (
                state_pool.recurrent_buffers[layer]
                .at[src]
                .set(float(layer + 1), out_sharding=rec_spec)
            )
            state_pool.conv_buffers[layer][0] = (
                state_pool.conv_buffers[layer][0]
                .at[src]
                .set(float(layer + 7), out_sharding=conv_spec)
            )

        # Mirror production: P("data")-sharded indices + jit, so shard_map in_specs match.
        data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        src_idx = jax.device_put(np.array([src], dtype=np.int32), data_sh)
        dst_idx = jax.device_put(np.array([dst], dtype=np.int32), data_sh)
        copy = jax.jit(state_pool.copy_slots)
        new_rec, new_conv = copy(src_idx, dst_idx)

        for layer in range(state_pool.num_linear_recurrent_layers):
            rec = np.asarray(new_rec[layer])
            conv = np.asarray(new_conv[layer][0])
            src_rec = np.asarray(state_pool.recurrent_buffers[layer])[src]
            src_conv = np.asarray(state_pool.conv_buffers[layer][0])[src]
            np.testing.assert_array_equal(rec[dst], src_rec)
            np.testing.assert_array_equal(conv[dst], src_conv)
            np.testing.assert_array_equal(rec[src], src_rec)

        zero_idx = jax.device_put(np.array([0], dtype=np.int32), data_sh)
        noop_rec, _ = copy(zero_idx, dst_idx)
        for layer in range(state_pool.num_linear_recurrent_layers):
            dst_before = np.asarray(state_pool.recurrent_buffers[layer])[dst]
            np.testing.assert_array_equal(np.asarray(noop_rec[layer])[dst], dst_before)

    def test_copy_slots_multi_dp_locality(self):
        """src/dst are per-rank-LOCAL indices; clones must not cross DP shards."""
        dp_mesh = create_device_mesh(ici_parallelism=[2, -1], dcn_parallelism=[1, 1])
        with jax.sharding.set_mesh(dp_mesh):
            state_pool = RecurrentStatePool(
                linear_recurrent_layer_ids=[0],
                size=8,  # 4 slots/rank, total_slots=10 (rank0 rows 0-4, rank1 5-9)
                num_heads=8,
                head_dim=16,
                conv_kernel_size=self.conv_kernel_size,
                mesh=dp_mesh,
                dp_size=2,
            )
            buf = state_pool.recurrent_buffers[0]
            # Seed global row i = i so each rank's local slot has distinct data.
            seed = np.broadcast_to(
                np.arange(buf.shape[0], dtype=np.float32).reshape(-1, 1, 1, 1), buf.shape
            )
            state_pool.recurrent_buffers[0] = jax.device_put(
                seed.astype(np.asarray(buf).dtype), state_pool.recurrent_sharding
            )

            data_sh = jax.sharding.NamedSharding(dp_mesh, jax.sharding.PartitionSpec("data"))
            # Both ranks clone local 2 -> local 3 (global 2->3 on rank0, 7->8 on rank1).
            src = jax.device_put(np.array([2, 2], dtype=np.int32), data_sh)
            dst = jax.device_put(np.array([3, 3], dtype=np.int32), data_sh)
            new_rec, _ = jax.jit(state_pool.copy_slots)(src, dst)
            rec = np.asarray(new_rec[0])

        # Cross-rank leakage would make global row 3 read 7.0 (or vice versa).
        self.assertEqual(float(rec[3].reshape(-1)[0]), 2.0)
        self.assertEqual(float(rec[8].reshape(-1)[0]), 7.0)


if __name__ == "__main__":
    unittest.main()
