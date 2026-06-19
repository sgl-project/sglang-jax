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
    """Effective-cache-length consumption in cache_finished_req /
    cache_unfinished_req. A component returns an int from
    prepare_for_caching_req to cap the inserted key; the cache must truncate
    (and free the dropped KV tail) or skip the insert entirely when 0. All
    components return None today, so capping is a no-op unless a stub injects
    a value -- the truncation/skip tests drive that path with a monkeypatch."""

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
        """Monkeypatch the FULL component's prepare_for_caching_req to return a
        fixed effective length, standing in for a future recurrent cap."""
        full = cache.components[ComponentType.FULL]
        full.prepare_for_caching_req = lambda req, insert_params, token_ids_len, is_finished: cl

    def test_no_op_equivalence_unfinished_inserts_full_length(self):
        # No component caps (all return None): the inserted key is the full
        # page-aligned length, identical to today's behavior.
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))  # 12 tokens
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
        # committed_kv_len == len(token_ids): finished caches the full sequence.
        req.kv_committed_len = len(token_ids)
        req.kv_allocated_len = len(token_ids)

        cache.cache_finished_req(req)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(match.device_indices), 12)
        self.assertEqual(cache.total_size(), 12)

    def test_truncation_caps_inserted_key_unfinished_keeps_tail(self):
        # A running request whose tree key is capped keeps its full KV (it is
        # still generating); only the tree key is shortened, the dropped tail is
        # retained in prefix_indices, not freed.
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))  # 12 tokens
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
        self._cap_to(cache, 8)  # cap 12 -> 8
        cache.cache_unfinished_req(req)

        # Only the first 8 tokens were materialized into the tree.
        self.assertEqual(cache.total_size(), 8)
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids[:8])))
        self.assertEqual(len(match.device_indices), 8)
        # The full key no longer fully matches; only the capped prefix does.
        full_match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(full_match.device_indices), 8)
        # The protected prefix tracks the capped length.
        self.assertEqual(req.cache_protected_len, 8)
        # The running request keeps its full KV (tail retained in prefix_indices,
        # nothing freed beyond the duplicate overlap, which is 0 here).
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
        # 4 dropped tail slots freed.
        self.assertEqual(allocator.available_size(0), avail_before + 4)

    def test_skip_insert_on_zero_unfinished(self):
        # Unfinished skip mirrors the disabled-cache path: no tree key, cleanup
        # runs, and the still-running request keeps its KV and prefix
        # bookkeeping untouched (it has not finished generating).
        pool, allocator, cache = self._create_stack()
        token_ids = list(range(1, 13))
        kv_indices = allocator.alloc(len(token_ids), dp_rank=0)
        pool.write((0, slice(0, len(token_ids))), kv_indices)
        prior_prefix = np.empty((0,), dtype=np.int32)
        req = MockRequest(
            req_pool_idx=0,
            origin_input_ids=list(token_ids),
            output_ids=[],
            fill_ids=list(token_ids),
            prefix_indices=prior_prefix,
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

        # No node was materialized; the request still points at root.
        self.assertEqual(cache.total_size(), 0)
        self.assertIs(req.last_node, cache.root_node)
        # cleanup still ran.
        self.assertTrue(cleanup_calls)
        # The running request keeps its KV and prefix bookkeeping intact.
        self.assertEqual(allocator.available_size(0), avail_before)
        self.assertIs(req.prefix_indices, prior_prefix)
        self.assertEqual(req.cache_protected_len, 0)

    def test_skip_insert_on_zero_unfinished_preserves_protected_prefix(self):
        # With a protected prefix, the unfinished skip path still frees nothing
        # and preserves the protected prefix bookkeeping.
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
        req.cache_protected_len = 4  # first 4 KV slots are tree-protected

        avail_before = allocator.available_size(0)
        self._cap_to(cache, 0)
        cache.cache_unfinished_req(req)

        self.assertEqual(cache.total_size(), 0)
        # Nothing freed; the protected [0:4] prefix and the running tail persist.
        self.assertEqual(allocator.available_size(0), avail_before)
        self.assertEqual(req.cache_protected_len, 4)

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
        # Entire committed range freed (old_prefix_len=0 upward).
        self.assertEqual(allocator.available_size(0), avail_before + 12)


class _CowReq:
    """Minimal Req surrogate for recurrent CoW match recording."""

    def __init__(self, dp_rank=0):
        self.dp_rank = dp_rank
        self.recurrent_cow_src_index = None


class _ReleaseReq:
    """Req surrogate carrying a request-owned recurrent slot, enough to drive
    the production release entry point ``release_kv_cache`` (the scheduler's
    abort and retract paths both funnel through it)."""

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
        # No matched prefix: the whole committed range is request-owned.
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
    """Req surrogate for a running recurrent request mid chunked-prefill: it owns
    a running recurrent slot (``recurrent_pool_idx`` set) and carries the cache
    bookkeeping ``cache_unfinished_req`` reads/writes."""

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
    """Cache-level recurrent component: commit / match-CoW / evict / lock /
    slot ledger, all at page_size=1 (PR#1 scope)."""

    def setUp(self):
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192
        self.recurrent_size = 8  # global recurrent slots
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
        """Simulate a finished-request recurrent commit: pop a running slot and
        donate it to the tree via insert (mirrors cache_finished_req)."""
        slot = pool.alloc_recurrent_slot(0)
        params = InsertParams(
            key=RadixKey(key, None, 0),
            value=value,
            recurrent_value=pool.recurrent_value_from_slot(slot),
        )
        result = cache.insert(params)
        return slot, result

    def _admit(self, cache, allocator, pool, key, dp_rank=0):
        """Admit a running recurrent request: grab a req slot + a request-owned
        recurrent running slot + KV indices, and wire them into the pools the way
        prefill does, so the production release path can free them."""
        req = _ReleaseReq(
            req_pool_idx=None, recurrent_pool_idx=None, origin_input_ids=key, dp_rank=dp_rank
        )
        pool.alloc([req])  # assigns req_pool_idx + a recurrent running slot
        kv_indices = allocator.alloc(len(key), dp_rank=dp_rank)
        self.assertIsNotNone(kv_indices)
        pool.write((req.req_pool_idx, slice(0, len(key))), kv_indices)
        # No prefix was matched/locked: the lock release walks from root (no-op).
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

        # The tree took ownership: committed leaf holds the slot.
        self.assertTrue(result.recurrent_committed)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        # active=0 (no live request), tree_owned=1, free=slots-1.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        # A prefix hit records the src slot for CoW (no allocation at match).
        req = _CowReq(dp_rank=0)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(key, None, 0), cow_recurrent=True, req=req)
        )
        self.assertTrue(np.array_equal(match.device_indices, value))
        self.assertEqual(req.recurrent_cow_src_index, slot)

        # Eviction frees the recurrent slot and reports it.
        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(pool.recurrent_available_size(0), slots)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_abort_and_retract_release_recurrent_slots(self):
        """Aborting a request mid-decode and retracting another must return their
        request-owned recurrent slots to the free list, leaving no leak: the
        ledger ``active + tree_owned + free == slots_per_rank`` recovers to full
        (active == 0). Both production release paths funnel through
        ``release_kv_cache``: abort uses ``is_insert=True`` (a duplicate of an
        existing leaf is not re-committed, so the slot is freed), retract uses
        ``is_insert=False`` (no insert at all).
        """
        state_pool, pool, allocator, cache = self._create_recurrent_setup()
        slots = pool.slots_per_rank

        # A committed tree leaf so the aborted request's sequence is a duplicate
        # (recurrent_exist -> not re-committed -> request slot is freed).
        committed_key = [1, 2, 3, 4]
        committed_val = np.arange(100, 104, dtype=np.int32)
        _, commit_res = self._commit(cache, pool, committed_key, committed_val)
        self.assertTrue(commit_res.recurrent_committed)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)  # tree owns 1

        # Admit two running requests; each takes a request-owned recurrent slot.
        abort_req = self._admit(cache, allocator, pool, committed_key)  # duplicate seq
        retract_req = self._admit(cache, allocator, pool, [9, 10, 11])  # distinct seq
        self.assertIsNotNone(abort_req.recurrent_pool_idx)
        self.assertIsNotNone(retract_req.recurrent_pool_idx)
        # tree_owned=1, active=2 -> free = slots - 3.
        self.assertEqual(pool.recurrent_available_size(0), slots - 3)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 2)

        # Abort: finished release with insert; the duplicate leaf is not
        # re-committed, so the running slot is freed (not donated).
        release_kv_cache(abort_req, cache, is_insert=True)
        self.assertIsNone(abort_req.recurrent_pool_idx)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 1)

        # Retract: release without insert; the running slot is freed directly.
        release_kv_cache(retract_req, cache, is_insert=False)
        self.assertIsNone(retract_req.recurrent_pool_idx)

        # No request-owned slots remain; only the tree leaf is still held.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        # Evicting the tree leaf returns the ledger to fully free.
        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), slots)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_continuation_match_hits_committed_leaf(self):
        """Serving reuse path: a follow-up whose prompt extends a committed
        sequence (multi-turn / P -> P+suffix) walks past and stops on the
        committed leaf, so recurrent hits the full committed length and records
        the CoW src. This is the case cross-request recurrent reuse is built for.
        """
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
        """page=1 regression floor for the PR#1 Step 5.4 full_only rematch
        (schedule_batch.init_next_round_input). A RUNNING recurrent request owns
        its recurrent state in its running slot, so during chunked-prefill its
        OWN unfinished FULL prefix lives in the tree as recurrent-LESS interior
        nodes. The scheduler's next-round re-match must use full_only=True /
        cow_recurrent=False, else the leaf-only recurrent validator rejects those
        nodes and the request's whole FULL prefix collapses to root (re-prefilled
        from scratch / KV double-counted).

        This drives the REAL UnifiedRadixCache (the param-only TestFullOnlyRematch
        asserts the flags; here the prefix preservation is genuinely exercised):
          (a) full_only=True (the fix's params) preserves the entire FULL prefix
              and does NOT set a recurrent_cow_src_index (no re-clone);
          (b) the WRONG params full_only=False / cow_recurrent=True truncate that
              same prefix to 0 -- proving the flag is load-bearing, not cosmetic;
          (c) a second chunked-continuation cache_unfinished_req grows and keeps
              cache_protected_len, with the slot ledger balanced throughout.
        """
        state_pool, pool, allocator, cache = self._create_recurrent_setup()

        # --- Admit a running recurrent request (req slot + a running slot). ---
        chunk1 = list(range(1, 9))  # 8-token first chunk
        req = _RunningRecurrentReq(req_pool_idx=None, recurrent_pool_idx=None, fill_ids=chunk1)
        pool.alloc([req])  # assigns req_pool_idx + a running recurrent slot
        self.assertIsNotNone(req.recurrent_pool_idx)
        running_slot = req.recurrent_pool_idx
        # active=1 (this running req), tree_owned=0, free=slots-1.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 1)

        # --- Chunk 1: write KV, cache the unfinished prefix into the tree. ---
        kv1 = allocator.alloc(len(chunk1), dp_rank=0)
        self.assertIsNotNone(kv1)
        pool.write((req.req_pool_idx, slice(0, len(chunk1))), kv1)
        req.last_node = cache.root_node
        cache.cache_unfinished_req(req)

        # Unfinished page=1 donates no recurrent value: the running slot stays
        # request-owned, the cached FULL nodes hold NO recurrent value.
        self.assertEqual(req.recurrent_pool_idx, running_slot)
        self.assertEqual(req.cache_protected_len, len(chunk1))
        self.assertEqual(len(req.prefix_indices), len(chunk1))
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 1)

        prefix_key = RadixKey(chunk1, None, 0)

        # --- (a) The fix's params: full_only=True / cow_recurrent=False. ---
        cow_probe = _CowReq(dp_rank=0)
        fixed = cache.match_prefix(
            MatchPrefixParams(key=prefix_key, full_only=True, cow_recurrent=False, req=cow_probe)
        )
        # The whole FULL prefix survives -- it is NOT gated on recurrent state.
        self.assertEqual(len(fixed.device_indices), len(chunk1))
        self.assertTrue(np.array_equal(fixed.device_indices, np.asarray(req.prefix_indices)))
        # No re-clone: a running req owns its slot, so no CoW src is recorded.
        self.assertIsNone(cow_probe.recurrent_cow_src_index)

        # --- (b) The WRONG params a new-prefill would use on this same key. ---
        # full_only=False + cow_recurrent=True makes the leaf-only recurrent
        # validator reject the recurrent-less FULL nodes, collapsing the match
        # to root -- this is the regression the fix prevents.
        wrong_probe = _CowReq(dp_rank=0)
        wrong = cache.match_prefix(
            MatchPrefixParams(key=prefix_key, full_only=False, cow_recurrent=True, req=wrong_probe)
        )
        self.assertEqual(len(wrong.device_indices), 0)
        self.assertIsNone(wrong_probe.recurrent_cow_src_index)

        # --- (c) Second chunked-continuation: prefix preserved + ledger sound. ---
        chunk2 = list(range(9, 13))  # 4 more tokens
        req.fill_ids = chunk1 + chunk2
        kv2 = allocator.alloc(len(chunk2), dp_rank=0)
        self.assertIsNotNone(kv2)
        pool.write((req.req_pool_idx, slice(len(chunk1), len(chunk1) + len(chunk2))), kv2)
        protected_before = req.cache_protected_len
        cache.cache_unfinished_req(req)

        # The continuation extends the protected prefix (never truncates it) and
        # still does not clone or donate a recurrent slot.
        self.assertEqual(req.cache_protected_len, len(chunk1) + len(chunk2))
        self.assertGreater(req.cache_protected_len, protected_before)
        self.assertEqual(len(req.prefix_indices), len(chunk1) + len(chunk2))
        self.assertEqual(req.recurrent_pool_idx, running_slot)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 1)

        # The extended FULL prefix is still fully matchable full_only.
        full_key = RadixKey(chunk1 + chunk2, None, 0)
        again = cache.match_prefix(MatchPrefixParams(key=full_key, full_only=True))
        self.assertEqual(len(again.device_indices), len(chunk1) + len(chunk2))

    def test_shorter_match_misses_recurrent_leaf(self):
        """A match stopping one token short of the committed leaf (the universal
        adjust_max_prefix_ids = input_len - 1, i.e. an identical-prompt resubmit)
        lands on a recurrent-less split parent. Match requires every component's
        validator to accept the same node, so the whole match -- FULL KV included
        -- falls back to root. This is why repeated-identical prompts report
        cached-token 0; matchable intermediate snapshots are PR#2 scope.
        """
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
        """Caching AB after ABCD lands on internal node B: recurrent is leaf-only,
        so it is not committed there (no orphan); the running slot is freed by
        the caller (recurrent_committed=False)."""
        state_pool, pool, _, cache = self._create_recurrent_setup()

        long_key = [1, 2, 3, 4]
        long_val = np.arange(10, 14, dtype=np.int32)
        long_slot, long_res = self._commit(cache, pool, long_key, long_val)
        self.assertTrue(long_res.recurrent_committed)

        short_key = [1, 2]
        short_val = np.arange(20, 22, dtype=np.int32)
        short_slot, short_res = self._commit(cache, pool, short_key, short_val)

        # Internal target → not committed; the longer leaf keeps its recurrent.
        self.assertFalse(short_res.recurrent_committed)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        # The short slot is still request-owned (caller frees it).
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

        # Recurrent value moved evictable → protected.
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 1)

        # Locked leaf survives eviction.
        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 0)

        # Unlock restores, then eviction frees.
        cache.dec_lock_ref(leaf, lock_result.to_dec_params())
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 0)
        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), pool.slots_per_rank)

    def test_longer_prefix_after_shorter_frees_parent(self):
        """Caching AB (leaf) then ABCD makes AB internal via _add_new_node; its
        recurrent value must be dropped (not stranded on the unevictable internal
        node), so eviction can reclaim every slot (regression for B1)."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        slots = pool.slots_per_rank

        ab_slot, ab_res = self._commit(cache, pool, [1, 2], np.arange(10, 12, dtype=np.int32))
        self.assertTrue(ab_res.recurrent_committed)
        _, abcd_res = self._commit(cache, pool, [1, 2, 3, 4], np.arange(20, 24, dtype=np.int32))
        self.assertTrue(abcd_res.recurrent_committed)

        # AB became internal → its slot was freed; only ABCD's leaf is tree-owned.
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

        evict_result = cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertEqual(evict_result.recurrent_num_evicted, 1)
        self.assertEqual(pool.recurrent_available_size(0), slots)

    def test_internal_transition_while_locked_frees_on_unlock(self):
        """If a node becomes internal while its recurrent value is locked (a CoW
        src this round), the free is deferred to the final unlock (B1)."""
        state_pool, pool, _, cache = self._create_recurrent_setup()
        self._commit(cache, pool, [1, 2], np.arange(10, 12, dtype=np.int32))
        ab = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2], None, 0))).last_device_node

        lock = cache.inc_lock_ref(ab)  # AB recurrent: evictable → protected
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 1)

        # Cache ABCD → AB gains a child while locked; recurrent free is deferred.
        self._commit(cache, pool, [1, 2, 3, 4], np.arange(20, 24, dtype=np.int32))
        self.assertIsNotNone(ab.component_data[ComponentType.RECURRENT].value)
        free_before = pool.recurrent_available_size(0)

        cache.dec_lock_ref(ab, lock.to_dec_params())  # final unlock frees it
        self.assertIsNone(ab.component_data[ComponentType.RECURRENT].value)
        self.assertEqual(pool.recurrent_available_size(0), free_before + 1)
        self.assertEqual(cache.component_protected_size_[ComponentType.RECURRENT][0], 0)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0), 0)

    def test_copy_slots_bitwise_clone_all_layers(self):
        """copy_slots clones src→dst bitwise across all recurrent + conv layers;
        src==0 rows are no-ops. This is the page=1 building block of CoW
        equivalence (full KL==0 model parity is the TPU test)."""
        state_pool, _, _, _ = self._create_recurrent_setup()
        src, dst = 2, 5

        # Seed distinct per-layer data into the src slot; leave dst zeroed.
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

        # Indices arrive P("data")-sharded in production (forward metadata);
        # mirror that here so copy_slots' shard_map in_specs match. copy_slots
        # runs under jit in production (jitted_run_model) so the in-kernel
        # gather/scatter enter shard_map's manual mode.
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
            # src is untouched (the tree's value is preserved).
            np.testing.assert_array_equal(rec[src], src_rec)

        # src==0 → no clone: dst keeps its (zero) content.
        zero_idx = jax.device_put(np.array([0], dtype=np.int32), data_sh)
        noop_rec, _ = copy(zero_idx, dst_idx)
        for layer in range(state_pool.num_linear_recurrent_layers):
            dst_before = np.asarray(state_pool.recurrent_buffers[layer])[dst]
            np.testing.assert_array_equal(np.asarray(noop_rec[layer])[dst], dst_before)

    def test_copy_slots_multi_dp_locality(self):
        """copy_slots under dp_size=2: src/dst are per-rank-LOCAL slot indices, so
        the clone must stay within each DP shard (no cross-rank corruption).
        (plan §6 multi-DP / reviewer S1)"""
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
            # Seed global row i = i, so each rank's local slot has distinct data.
            seed = np.broadcast_to(
                np.arange(buf.shape[0], dtype=np.float32).reshape(-1, 1, 1, 1), buf.shape
            )
            state_pool.recurrent_buffers[0] = jax.device_put(
                seed.astype(np.asarray(buf).dtype), state_pool.recurrent_sharding
            )

            data_sh = jax.sharding.NamedSharding(dp_mesh, jax.sharding.PartitionSpec("data"))
            # Both ranks clone LOCAL slot 2 → LOCAL slot 3 (global 2→3 on rank0,
            # global 7→8 on rank1).
            src = jax.device_put(np.array([2, 2], dtype=np.int32), data_sh)
            dst = jax.device_put(np.array([3, 3], dtype=np.int32), data_sh)
            new_rec, _ = jax.jit(state_pool.copy_slots)(src, dst)
            rec = np.asarray(new_rec[0])

        # rank0 local-3 (global 3) == its local-2 (global 2) == 2.0;
        # rank1 local-3 (global 8) == its local-2 (global 7) == 7.0.
        # Cross-rank leakage would make global 3 == 7.0 (or vice versa).
        self.assertEqual(float(rec[3].reshape(-1)[0]), 2.0)
        self.assertEqual(float(rec[8].reshape(-1)[0]), 7.0)


class _ExtraBufferReq:
    """Req surrogate carrying the extra-buffer ping-pong fields, enough to drive
    RecurrentComponent.prepare/cleanup directly (PR#2 page>=128 path)."""

    def __init__(self, dp_rank=0):
        self.dp_rank = dp_rank
        self.req_pool_idx = None
        self.recurrent_pool_idx = None
        self.recurrent_ping_pong_track_buffer = None
        self.recurrent_next_track_idx = None
        self.recurrent_last_track_seqlen = None
        self.recurrent_cow_src_index = None


class _AllocReq:
    """Minimal Req surrogate for ScheduleBatch.alloc_req_slots demand counting
    (carries the fields HybridReqToTokenPool.alloc reads)."""

    def __init__(self, dp_rank=0, recurrent_pool_idx=None):
        self.dp_rank = dp_rank
        self.recurrent_pool_idx = recurrent_pool_idx
        self.req_pool_idx = None
        self.is_chunked = 0
        self.kv_committed_len = 0
        self.recurrent_ping_pong_track_buffer = None
        self.recurrent_next_track_idx = None
        self.recurrent_last_track_seqlen = None


class TestUnifiedRadixCacheRecurrentExtraBuffer(CustomTestCase):
    """Extra-buffer (page>=128) recurrent commit/donation: prepare caps the key at
    the materialized boundary; cleanup commits the KEEP ping-pong slot (finished)
    or donates it with a secured replacement (unfinished). Ledger balanced after
    every transition. Step 10 slot-shortfall eviction is 3x new reqs."""

    def setUp(self):
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.recurrent_size = 24  # global recurrent slots (3 per req leaves room)
        self.rec_num_heads = 8
        self.rec_head_dim = 16
        self.conv_kernel_size = 4
        self.track_interval = 128

    def _create_setup(self, dp_size=1, recurrent_size=None):
        recurrent_size = recurrent_size or self.recurrent_size
        state_pool = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            size=recurrent_size,
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
            enable_recurrent_extra_buffer=True,
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
            enable_recurrent_extra_buffer=True,
            recurrent_track_interval=self.track_interval,
        )
        return state_pool, hybrid_pool, allocator, cache

    @staticmethod
    def _admit(pool, dp_rank=0):
        """Allocate a req (running + 2 track slots under extra-buffer)."""
        req = _ExtraBufferReq(dp_rank=dp_rank)
        pool.alloc([req])
        return req

    @staticmethod
    def _simulate_boundary_scatter(pool, req, final_seq_len):
        """Mirror _recurrent_track_entry: the scatter writes the slot at
        next_track_idx, sets the watermark, then flips next_track_idx. The keep
        slot afterward is the one just written."""
        idx = req.recurrent_next_track_idx
        req.recurrent_last_track_seqlen = final_seq_len
        req.recurrent_next_track_idx = pool.get_recurrent_ping_pong_other_idx(idx)
        return req.recurrent_ping_pong_track_buffer[idx]

    def _component(self, cache):
        return cache.components[ComponentType.RECURRENT]

    # --- Case 1: prepare caps the key at the materialized boundary ---

    def test_prepare_returns_watermark_caps_inserted_key(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        req = self._admit(pool)
        self._simulate_boundary_scatter(pool, req, final_seq_len=256)

        params = InsertParams()
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=300, is_finished=True)
        self.assertEqual(cap, 256)
        self.assertIsNotNone(params.recurrent_value)

    def test_prepare_returns_zero_when_no_boundary(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        req = self._admit(pool)  # recurrent_last_track_seqlen stays None
        self.assertIsNone(req.recurrent_last_track_seqlen)

        params = InsertParams()
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=100, is_finished=True)
        self.assertEqual(cap, 0)
        self.assertIsNone(params.recurrent_value)

    # --- Case 2: finished commit uses the KEEP slot; cleanup balances ---

    def test_finished_commit_uses_keep_slot_then_frees_running_and_other(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        slots = pool.slots_per_rank
        req = self._admit(pool)
        running = req.recurrent_pool_idx
        keep_slot = self._simulate_boundary_scatter(pool, req, final_seq_len=128)
        # 3 request-owned slots (running + 2 track), no tree-owned yet.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 3)

        params = InsertParams(
            key=RadixKey([1, 2, 3, 4], None, 0),
            value=np.arange(10, 14, dtype=np.int32),
        )
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=4, is_finished=True)
        self.assertEqual(cap, 128)
        # The donated value is the KEEP slot, NOT the running slot.
        self.assertEqual(int(params.recurrent_value[0]), keep_slot)
        self.assertNotEqual(int(params.recurrent_value[0]), running)

        result = cache.insert(params)
        self.assertTrue(result.recurrent_committed)
        comp.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )

        # Tree owns the keep slot; running + other track slot freed. Net: 1 tree + 2 free.
        self.assertIsNone(req.recurrent_pool_idx)
        self.assertIsNone(req.recurrent_ping_pong_track_buffer)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 0)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

    def test_finished_not_committed_frees_running_and_both_track(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        slots = pool.slots_per_rank
        # Pre-commit a leaf so the same key is a duplicate (recurrent_exist).
        seed = self._admit(pool)
        self._simulate_boundary_scatter(pool, seed, final_seq_len=128)
        seed_params = InsertParams(
            key=RadixKey([1, 2, 3, 4], None, 0), value=np.arange(10, 14, dtype=np.int32)
        )
        comp.prepare_for_caching_req(seed, seed_params, token_ids_len=4, is_finished=True)
        seed_res = cache.insert(seed_params)
        comp.cleanup_after_caching_req(
            seed, is_finished=True, insert_result=seed_res, insert_params=seed_params
        )
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

        req = self._admit(pool)
        self._simulate_boundary_scatter(pool, req, final_seq_len=128)
        params = InsertParams(
            key=RadixKey([1, 2, 3, 4], None, 0), value=np.arange(10, 14, dtype=np.int32)
        )
        comp.prepare_for_caching_req(req, params, token_ids_len=4, is_finished=True)
        result = cache.insert(params)
        # Duplicate of the existing leaf: not re-committed.
        self.assertFalse(result.recurrent_committed)
        comp.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )

        # All 3 request-owned slots freed; only the seed leaf remains tree-owned.
        self.assertIsNone(req.recurrent_ping_pong_track_buffer)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 0)
        self.assertEqual(pool.recurrent_available_size(0), slots - 1)

    # --- Case 3: unfinished donation keeps running live, allocates replacement ---

    def test_unfinished_donation_secures_replacement_and_clears_watermark(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        slots = pool.slots_per_rank
        req = self._admit(pool)
        running = req.recurrent_pool_idx
        keep_slot = self._simulate_boundary_scatter(pool, req, final_seq_len=128)
        other_slot = next(s for s in req.recurrent_ping_pong_track_buffer if s != keep_slot)
        free_before = pool.recurrent_available_size(0)

        params = InsertParams(
            key=RadixKey([5, 6, 7, 8], None, 0), value=np.arange(20, 24, dtype=np.int32)
        )
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=4, is_finished=False)
        self.assertEqual(cap, 128)
        # Donated value is the keep slot; a fresh replacement took its buffer place.
        self.assertEqual(int(params.recurrent_value[0]), keep_slot)
        self.assertNotIn(keep_slot, req.recurrent_ping_pong_track_buffer)
        self.assertIn(other_slot, req.recurrent_ping_pong_track_buffer)
        # Replacement consumed one free slot; running slot untouched.
        self.assertEqual(pool.recurrent_available_size(0), free_before - 1)
        self.assertEqual(req.recurrent_pool_idx, running)

        result = cache.insert(params)
        self.assertTrue(result.recurrent_committed)
        comp.cleanup_after_caching_req(
            req, is_finished=False, insert_result=result, insert_params=params
        )

        # Tree owns the donated slot; the request keeps running + 2 track slots.
        self.assertIsNone(req.recurrent_last_track_seqlen)
        self.assertEqual(req.recurrent_pool_idx, running)
        self.assertEqual(len([s for s in req.recurrent_ping_pong_track_buffer if s]), 2)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 1)
        # active = running(1) + 2 track = 3; tree_owned=1; free = slots - 4.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 3)
        self.assertEqual(pool.recurrent_available_size(0), slots - 4)

    # --- Case 4: unfinished donation NOT committed → free orphan, clear watermark ---

    def test_unfinished_donation_not_committed_frees_orphan(self):
        _, pool, _, cache = self._create_setup()
        comp = self._component(cache)
        slots = pool.slots_per_rank
        # Pre-commit a leaf so the unfinished re-donation of the same key is a dup.
        seed = self._admit(pool)
        self._simulate_boundary_scatter(pool, seed, final_seq_len=128)
        seed_params = InsertParams(
            key=RadixKey([5, 6, 7, 8], None, 0), value=np.arange(20, 24, dtype=np.int32)
        )
        comp.prepare_for_caching_req(seed, seed_params, token_ids_len=4, is_finished=True)
        seed_res = cache.insert(seed_params)
        comp.cleanup_after_caching_req(
            seed, is_finished=True, insert_result=seed_res, insert_params=seed_params
        )

        req = self._admit(pool)
        running = req.recurrent_pool_idx
        keep_slot = self._simulate_boundary_scatter(pool, req, final_seq_len=128)
        params = InsertParams(
            key=RadixKey([5, 6, 7, 8], None, 0), value=np.arange(20, 24, dtype=np.int32)
        )
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=4, is_finished=False)
        self.assertEqual(cap, 128)
        donated = int(params.recurrent_value[0])
        self.assertEqual(donated, keep_slot)
        result = cache.insert(params)
        self.assertFalse(result.recurrent_committed)  # duplicate of the seed leaf
        free_before = pool.recurrent_available_size(0)
        comp.cleanup_after_caching_req(
            req, is_finished=False, insert_result=result, insert_params=params
        )

        # The orphaned donated slot is freed; watermark cleared; request keeps
        # running + 2 (replacement + other) track slots.
        self.assertEqual(pool.recurrent_available_size(0), free_before + 1)
        self.assertIn(donated, pool.recurrent_free_slots[0])
        self.assertIsNone(req.recurrent_last_track_seqlen)
        self.assertEqual(req.recurrent_pool_idx, running)
        self.assertEqual(len([s for s in req.recurrent_ping_pong_track_buffer if s]), 2)
        # active = running + 2 track = 3; only the seed leaf is tree-owned.
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 3)
        _ = slots  # capacity sanity already covered by the ledger assertion

    # --- Case 5: replacement alloc fails (all locked) → donation skipped ---

    def test_unfinished_replacement_alloc_fails_skips_donation(self):
        # Size 3: exactly one req's worth of slots, so after admission the free
        # list is empty and there is no tree leaf to evict for a replacement.
        _, pool, _, cache = self._create_setup(recurrent_size=3)
        comp = self._component(cache)
        req = self._admit(pool)
        keep_slot = self._simulate_boundary_scatter(pool, req, final_seq_len=128)
        self.assertEqual(pool.recurrent_available_size(0), 0)
        buffer_before = list(req.recurrent_ping_pong_track_buffer)

        params = InsertParams(
            key=RadixKey([9, 10, 11, 12], None, 0), value=np.arange(30, 34, dtype=np.int32)
        )
        cap = comp.prepare_for_caching_req(req, params, token_ids_len=4, is_finished=False)
        # Replacement unavailable → donation skipped: returns 0, buffer + watermark intact.
        self.assertEqual(cap, 0)
        self.assertIsNone(params.recurrent_value)
        self.assertEqual(req.recurrent_ping_pong_track_buffer, buffer_before)
        self.assertIn(keep_slot, req.recurrent_ping_pong_track_buffer)
        self.assertEqual(req.recurrent_last_track_seqlen, 128)

        # Cleanup with no commit + no donation: frees nothing, leaves req intact.
        comp.cleanup_after_caching_req(
            req, is_finished=False, insert_result=None, insert_params=params
        )
        self.assertEqual(req.recurrent_last_track_seqlen, 128)
        self.assertEqual(req.recurrent_ping_pong_track_buffer, buffer_before)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=[req]), 3)

    # --- Case 6: Step 10 alloc_req_slots evicts the exact 3x shortfall ---

    def _seed_tree_leaves(self, comp, cache, pool, keys, dp_rank=0):
        """Commit one finished extra-buffer req per key, leaving each as a
        tree-owned leaf the eviction can reclaim."""
        for i, key in enumerate(keys):
            req = self._admit(pool, dp_rank=dp_rank)
            self._simulate_boundary_scatter(pool, req, final_seq_len=128)
            params = InsertParams(
                key=RadixKey(list(key), None, dp_rank),
                value=np.arange(100 + i * 10, 100 + i * 10 + len(key), dtype=np.int32),
            )
            comp.prepare_for_caching_req(req, params, token_ids_len=len(key), is_finished=True)
            res = cache.insert(params)
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=res, insert_params=params
            )

    def _make_batch(self, cache, pool, allocator, reqs):
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        return ScheduleBatch(
            reqs_info=[],
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            is_hybrid_recurrent=True,
        )

    def test_alloc_req_slots_evicts_exact_three_x_shortfall_dp1(self):
        _, pool, allocator, cache = self._create_setup(recurrent_size=12)  # 12 slots/rank
        comp = self._component(cache)
        slots = pool.slots_per_rank
        # Fill the free list with tree leaves: 4 leaves use 4 slots, leaving 8 free.
        self._seed_tree_leaves(comp, cache, pool, [[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(pool.recurrent_available_size(0), slots - 4)  # 8 free
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 4)

        # 3 new reqs → demand = 9 > 8 free → evict exactly 1 leaf.
        batch = self._make_batch(cache, pool, allocator, None)
        reqs = [_AllocReq(dp_rank=0) for _ in range(3)]
        batch.alloc_req_slots(reqs)

        # Exactly one leaf evicted; the 3 reqs each got running + 2 track (9 slots).
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], 3)
        for r in reqs:
            self.assertIsNotNone(r.recurrent_pool_idx)
            self.assertEqual(len(r.recurrent_ping_pong_track_buffer), 2)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=reqs), 9)

    def test_alloc_req_slots_evicts_exact_three_x_shortfall_dp2(self):
        _, pool, allocator, cache = self._create_setup(dp_size=2, recurrent_size=24)
        comp = self._component(cache)
        slots = pool.slots_per_rank  # 12 per rank
        # Seed 4 leaves on rank 1 only (rank 0 stays empty).
        self._seed_tree_leaves(comp, cache, pool, [[1, 2], [3, 4], [5, 6], [7, 8]], dp_rank=1)
        self.assertEqual(pool.recurrent_available_size(1), slots - 4)
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][1], 4)

        batch = self._make_batch(cache, pool, allocator, None)
        reqs = [_AllocReq(dp_rank=1) for _ in range(3)]  # demand 9 > 8 free → evict 1
        batch.alloc_req_slots(reqs)

        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][1], 3)
        self.assertEqual(cache.assert_recurrent_slot_ledger(1, live_reqs=reqs), 9)
        # Rank 0 untouched.
        self.assertEqual(pool.recurrent_available_size(0), slots)

    def test_alloc_req_slots_no_evict_when_enough_free(self):
        _, pool, allocator, cache = self._create_setup(recurrent_size=12)
        comp = self._component(cache)
        self._seed_tree_leaves(comp, cache, pool, [[1, 2]])  # 1 leaf, 11 free
        leaves_before = cache.component_evictable_size_[ComponentType.RECURRENT][0]

        batch = self._make_batch(cache, pool, allocator, None)
        reqs = [_AllocReq(dp_rank=0) for _ in range(3)]  # demand 9 <= 11 free
        batch.alloc_req_slots(reqs)

        # No eviction: enough free slots already.
        self.assertEqual(cache.component_evictable_size_[ComponentType.RECURRENT][0], leaves_before)
        self.assertEqual(cache.assert_recurrent_slot_ledger(0, live_reqs=reqs), 9)


if __name__ == "__main__":
    unittest.main()
