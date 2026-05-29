# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_swa_radix_cache.py -q

import os

# Simulate multi-device on CPU to satisfy JAX Mesh creation
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.test.test_utils import CustomTestCase


class TestSWARadixCache(CustomTestCase):
    def setUp(self):
        # Keep KV sizes small to make tests light-weight
        self.devices = jax.devices()
        self.mesh = Mesh(
            np.array(self.devices[:1]).reshape(1, 1),
            axis_names=("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )

        # Small buffers to avoid heavy allocations
        self.kv_head_num = 1
        self.head_dim = 1
        self.dtype = jax.numpy.bfloat16

        # Token pool sizes (FULL and SWA)
        self.full_size = 4096
        self.swa_size = 4096

        # Request→token mapping pool (host-side)
        self.req_pool = ReqToTokenPool(size=64, max_context_len=512)

        # Hybrid KV pool (FULL + SWA)
        self.kv_pool = SWAKVPool(
            size=self.full_size,
            size_swa=self.swa_size,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            token_to_kv_pool_class=MHATokenToKVPool,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            mesh=self.mesh,
        )

        # Allocator over the hybrid KV pool
        self.allocator = SWATokenToKVPoolAllocator(
            size=self.full_size,
            size_swa=self.swa_size,
            kvcache=self.kv_pool,
        )

        # SWA radix tree (page_size=1, small sliding window)
        self.cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=64,
            page_size=1,
            disable=False,
        )

    def _create_swa_allocator(self, dp_size: int):
        kv_pool = SWAKVPool(
            size=128,
            size_swa=128,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            token_to_kv_pool_class=MHATokenToKVPool,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            mesh=self.mesh,
        )
        return SWATokenToKVPoolAllocator(
            size=128,
            size_swa=128,
            kvcache=kv_pool,
            dp_size=dp_size,
        )

    def _alloc_indices(self, n: int) -> np.ndarray:
        idx = self.allocator.alloc(n)
        self.assertIsNotNone(idx)
        self.assertEqual(len(idx), n)
        return idx

    # ------------------------------------------------------------------ #
    #  Helpers for tombstone / size-consistency tests                      #
    # ------------------------------------------------------------------ #

    def _find_node_by_key_prefix(self, start_token):
        """Walk tree from root to find the node whose key starts with start_token."""
        root = self.cache.root_node
        child_key = self.cache.get_child_key_fn(RadixKey([start_token], None))
        if child_key in root.children:
            return root.children[child_key]
        return None

    def _make_small_cache(self, sliding_window_size=4):
        """Create a cache with a small sliding window for tombstone tests."""
        return SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=sliding_window_size,
            page_size=1,
            disable=False,
        )

    def _verify_size_consistency_for(self, cache, msg: str = ""):
        """Walk the tree and verify that tracked sizes match actual tree content."""
        full_total = 0
        swa_total = 0
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            if node != cache.root_node and node.value is not None:
                full_total += len(node.value)
                if not node.swa_tombstone:
                    swa_total += len(node.value)
            stack.extend(node.children.values())

        actual_full = sum(cache.full_evictable_size_.values()) + sum(
            cache.full_protected_size_.values()
        )
        actual_swa = sum(cache.swa_evictable_size_.values()) + sum(
            cache.swa_protected_size_.values()
        )
        self.assertEqual(
            full_total,
            actual_full,
            f"full size mismatch: tree={full_total}, tracked={actual_full}. {msg}",
        )
        self.assertEqual(
            swa_total,
            actual_swa,
            f"swa size mismatch: tree={swa_total}, tracked={actual_swa}. {msg}",
        )

    def _make_tree_with_tombstone(self):
        """
        Helper: insert two sequences that share a prefix into a small-window cache,
        then SWA-evict the shared prefix so it becomes a tombstone.

        Tree structure after setup:
          root → [0..9] (internal, tombstone after evict)
                    ├→ [10..14] (leaf A suffix)
                    └→ [20..24] (leaf B suffix)

        Returns (cache, key_a, val_a, key_b, val_b).
        """
        cache = self._make_small_cache(sliding_window_size=4)

        shared = list(range(0, 10))
        key_a = shared + list(range(10, 15))
        key_b = shared + list(range(20, 25))
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        cache.insert(key_a, value=val_a, prev_prefix_len=0)
        cache.insert(key_b, value=val_b, prev_prefix_len=0)

        # SWA-evict: evict the shared internal node (LRU, oldest access)
        cache.evict(full_num_tokens=0, swa_num_tokens=len(shared))

        return cache, key_a, val_a, key_b, val_b

    # ------------------------------------------------------------------ #
    #  Original tests (preserved from local)                               #
    # ------------------------------------------------------------------ #

    def test_insert_and_match_exact(self):
        """Insert a single sequence and match the same key."""
        key = list(range(100, 110))
        value = self._alloc_indices(len(key))

        prefix_len = self.cache.insert(key, value=value, prev_prefix_len=0)
        self.assertEqual(prefix_len, 0)

        match = self.cache.match_prefix(key)
        np.testing.assert_array_equal(np.asarray(match.device_indices), value)
        self.assertEqual(len(match.device_indices), len(key))
        self.assertIs(match.last_device_node, match.last_host_node)

    def test_insert_and_match_shorter_prefix(self):
        """Insert a sequence and match a shorter prefix."""
        key = list(range(200, 200 + 8))
        value = self._alloc_indices(len(key))
        self.cache.insert(key, value=value, prev_prefix_len=0)

        prefix_key = key[:5]
        match = self.cache.match_prefix(prefix_key)
        np.testing.assert_array_equal(np.asarray(match.device_indices), value[: len(prefix_key)])
        self.assertEqual(len(match.device_indices), len(prefix_key))

    def test_evict_and_reset_basic(self):
        """Evict some tokens and then reset the cache."""
        key1 = list(range(300, 300 + 16))
        key2 = list(range(400, 400 + 16))
        val1 = self._alloc_indices(len(key1))
        val2 = self._alloc_indices(len(key2))

        self.cache.insert(key1, value=val1, prev_prefix_len=0)
        self.cache.insert(key2, value=val2, prev_prefix_len=0)

        total_before, swa_before = self.cache.total_size()
        self.assertGreater(total_before, 0)
        self.assertEqual(total_before, swa_before)

        self.cache.evict(full_num_tokens=8, swa_num_tokens=0)
        total_after, swa_after = self.cache.total_size()
        self.assertLess(total_after, total_before)
        self.assertLessEqual(swa_after, total_after)

        self.cache.reset()
        total_reset, swa_reset = self.cache.total_size()
        self.assertEqual(total_reset, 0)
        self.assertEqual(swa_reset, 0)

    def test_two_sequences_with_shared_prefix(self):
        """Insert two sequences sharing a prefix; match both."""
        # First sequence A
        key_A = list(range(1000, 1000 + 6))  # [1000..1005]
        val_A = self._alloc_indices(len(key_A))
        self.cache.insert(key_A, value=val_A, prev_prefix_len=0)

        # Second sequence B shares first 4 tokens with A, then diverges
        shared = key_A[:4]
        suffix_B = list(range(2000, 2000 + 4))
        key_B = shared + suffix_B  # length 8
        val_B = self._alloc_indices(len(key_B))
        self.cache.insert(key_B, value=val_B, prev_prefix_len=0)

        # Matching A should return exactly val_A
        match_A = self.cache.match_prefix(key_A)
        np.testing.assert_array_equal(np.asarray(match_A.device_indices), val_A)

        # Matching B should reuse A's prefix indices and use B's suffix indices
        match_B = self.cache.match_prefix(key_B)
        matched_B = np.asarray(match_B.device_indices)
        self.assertEqual(len(matched_B), len(key_B))
        np.testing.assert_array_equal(matched_B[: len(shared)], val_A[: len(shared)])
        np.testing.assert_array_equal(matched_B[len(shared) :], val_B[len(shared) :])

    def test_lock_prevents_eviction_until_unlocked(self):
        """Lock a leaf node; eviction should skip it until unlocked."""
        key = list(range(500, 500 + 8))
        value = self._alloc_indices(len(key))
        self.cache.insert(key, value=value, prev_prefix_len=0)

        total_before, _ = self.cache.total_size()
        self.assertEqual(total_before, len(key))
        self.assertEqual(self.cache.full_evictable_size(), len(key))

        # Lock the leaf node
        match = self.cache.match_prefix(key)
        leaf = match.last_device_node
        swa_uuid = self.cache.inc_lock_ref(leaf)
        self.assertEqual(self.cache.full_protected_size(), len(key))
        self.assertEqual(self.cache.full_evictable_size(), 0)

        # Eviction should have no effect while locked
        self.cache.evict(full_num_tokens=len(key), swa_num_tokens=0)
        total_after_lock_evict, _ = self.cache.total_size()
        self.assertEqual(total_after_lock_evict, total_before)

        # Unlock and then evict; tokens should be removed
        self.cache.dec_lock_ref(leaf, swa_uuid)
        self.assertEqual(self.cache.full_protected_size(), 0)
        self.assertEqual(self.cache.full_evictable_size(), len(key))

        self.cache.evict(full_num_tokens=len(key), swa_num_tokens=0)
        total_after_unlock_evict, _ = self.cache.total_size()
        self.assertEqual(total_after_unlock_evict, 0)

    def test_paged_match_aligns_to_page_size(self):
        """For page_size>1, match_prefix aligns keys to page boundary."""
        # Create a separate cache with page_size=4
        paged_cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=64,
            page_size=4,
            disable=False,
        )

        key = list(range(600, 600 + 8))  # length 8
        value = self._alloc_indices(len(key))
        paged_cache.insert(key, value=value, prev_prefix_len=0)

        # Exact match still returns full value
        match_full = paged_cache.match_prefix(key)
        np.testing.assert_array_equal(np.asarray(match_full.device_indices), value)

        # Shorter key of length 6 should be truncated to 4 (page-aligned)
        partial_key = key[:6]
        match_partial = paged_cache.match_prefix(partial_key)
        matched_partial = np.asarray(match_partial.device_indices)
        self.assertEqual(len(matched_partial), 4)
        np.testing.assert_array_equal(matched_partial, value[:4])

    def test_disabled_cache_basic(self):
        """When disabled, cache should behave as a no-op."""
        disabled_cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=64,
            page_size=1,
            disable=True,
        )

        key = [1, 2, 3, 4]
        value = self._alloc_indices(len(key))

        # Insert should report zero prefix and not change size
        prefix_len = disabled_cache.insert(key, value=value, prev_prefix_len=0)
        self.assertEqual(prefix_len, 0)
        total_size, swa_size = disabled_cache.total_size()
        self.assertEqual(total_size, 0)
        self.assertEqual(swa_size, 0)

        # Match should always return empty
        match = disabled_cache.match_prefix(key)
        self.assertEqual(len(match.device_indices), 0)

    def test_extra_key_namespace_isolation(self):
        """Test that same tokens with different extra_keys don't share cache in SWA"""
        key = [1, 2, 3, 4, 5]

        # Insert with extra_key="adapter_a"
        value_a = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "adapter_a"), value=value_a, prev_prefix_len=0)

        # Insert with extra_key="adapter_b"
        value_b = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "adapter_b"), value=value_b, prev_prefix_len=0)

        # Match with "adapter_a" should return value_a
        match_a = self.cache.match_prefix(RadixKey(key, "adapter_a"))
        np.testing.assert_array_equal(np.asarray(match_a.device_indices), value_a)

        # Match with "adapter_b" should return value_b
        match_b = self.cache.match_prefix(RadixKey(key, "adapter_b"))
        np.testing.assert_array_equal(np.asarray(match_b.device_indices), value_b)

        # Verify they don't share cache (different values)
        self.assertFalse(np.array_equal(value_a, value_b))

    def test_extra_key_same_namespace_sharing(self):
        """Test that same tokens with same extra_key share cache in SWA"""
        # Insert short sequence with extra_key
        key_short = [10, 20, 30]
        value_short = self._alloc_indices(len(key_short))
        self.cache.insert(RadixKey(key_short, "shared_key"), value=value_short, prev_prefix_len=0)

        # Insert longer sequence with same extra_key should reuse prefix
        key_long = [10, 20, 30, 40, 50]
        value_long = self._alloc_indices(len(key_long))
        prefix_len = self.cache.insert(
            RadixKey(key_long, "shared_key"), value=value_long, prev_prefix_len=0
        )

        # Should have matched the first 3 tokens from cache
        self.assertGreater(prefix_len, 0)

        # Matching the short key should still work
        match_short = self.cache.match_prefix(RadixKey(key_short, "shared_key"))
        self.assertEqual(len(match_short.device_indices), len(key_short))

    def test_extra_key_none_vs_string(self):
        """Test that None extra_key is different from a string extra_key"""
        key = [100, 200, 300]

        # Insert with extra_key=None
        value_none = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None), value=value_none, prev_prefix_len=0)

        # Insert with extra_key="test"
        value_test = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "test"), value=value_test, prev_prefix_len=0)

        # Match with None should return value_none
        match_none = self.cache.match_prefix(RadixKey(key, None))
        np.testing.assert_array_equal(np.asarray(match_none.device_indices), value_none)

        # Match with "test" should return value_test
        match_test = self.cache.match_prefix(RadixKey(key, "test"))
        np.testing.assert_array_equal(np.asarray(match_test.device_indices), value_test)

        # They should be different
        self.assertFalse(np.array_equal(value_none, value_test))

    def test_backward_compatibility_plain_list(self):
        """Test that plain list still works (defaults to extra_key=None)"""
        key = [7, 8, 9]
        value = self._alloc_indices(len(key))

        # Insert with plain list
        prefix_len = self.cache.insert(key, value=value, prev_prefix_len=0)
        self.assertEqual(prefix_len, 0)

        # Match with plain list
        match_plain = self.cache.match_prefix(key)
        np.testing.assert_array_equal(np.asarray(match_plain.device_indices), value)

        # Match with RadixKey(key, None) should give same result
        match_radix = self.cache.match_prefix(RadixKey(key, None))
        np.testing.assert_array_equal(
            np.asarray(match_plain.device_indices), np.asarray(match_radix.device_indices)
        )

    def test_dp_rank_namespace_isolation(self):
        """Test that same tokens with different dp_ranks don't share cache"""
        key = [1, 2, 3, 4, 5]

        # Insert with dp_rank=0
        value_rank0 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None, 0), value=value_rank0, prev_prefix_len=0)

        # Insert with dp_rank=1
        value_rank1 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None, 1), value=value_rank1, prev_prefix_len=0)

        # Match with dp_rank=0 should return value_rank0
        match_rank0 = self.cache.match_prefix(RadixKey(key, None, 0))
        np.testing.assert_array_equal(np.asarray(match_rank0.device_indices), value_rank0)

        # Match with dp_rank=1 should return value_rank1
        match_rank1 = self.cache.match_prefix(RadixKey(key, None, 1))
        np.testing.assert_array_equal(np.asarray(match_rank1.device_indices), value_rank1)

        # Verify values are different (different cache namespaces)
        self.assertFalse(np.array_equal(value_rank0, value_rank1))

    def test_dp_rank_none_shared_namespace(self):
        """Test that dp_rank=None creates a shared namespace"""
        key = [10, 20, 30]

        # Insert with dp_rank=None
        value_none = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None, None), value=value_none, prev_prefix_len=0)

        # Match with dp_rank=None should hit cache
        match = self.cache.match_prefix(RadixKey(key, None, None))
        np.testing.assert_array_equal(np.asarray(match.device_indices), value_none)

        # Insert longer sequence with dp_rank=None should reuse prefix
        longer_key = [10, 20, 30, 40, 50]
        longer_value = self._alloc_indices(len(longer_key))
        prefix_len = self.cache.insert(
            RadixKey(longer_key, None, None), value=longer_value, prev_prefix_len=0
        )
        self.assertEqual(prefix_len, 3)  # Reused 3 tokens from cache

    def test_dp_rank_none_vs_explicit_rank(self):
        """Test that dp_rank=None and dp_rank=0 are different namespaces"""
        key = [100, 200, 300]

        # Insert with dp_rank=None
        value_none = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None, None), value=value_none, prev_prefix_len=0)

        # Insert with dp_rank=0
        value_rank0 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, None, 0), value=value_rank0, prev_prefix_len=0)

        # Match with None should return value_none
        match_none = self.cache.match_prefix(RadixKey(key, None, None))
        np.testing.assert_array_equal(np.asarray(match_none.device_indices), value_none)

        # Match with 0 should return value_rank0
        match_rank0 = self.cache.match_prefix(RadixKey(key, None, 0))
        np.testing.assert_array_equal(np.asarray(match_rank0.device_indices), value_rank0)

        # Verify they are different
        self.assertFalse(np.array_equal(value_none, value_rank0))

    def test_combined_extra_key_and_dp_rank(self):
        """Test that extra_key and dp_rank work together for dual isolation"""
        key = [7, 8, 9]

        # Create 4 different cache namespaces
        value_lora_a_rank0 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "lora_a", 0), value=value_lora_a_rank0, prev_prefix_len=0)

        value_lora_a_rank1 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "lora_a", 1), value=value_lora_a_rank1, prev_prefix_len=0)

        value_lora_b_rank0 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "lora_b", 0), value=value_lora_b_rank0, prev_prefix_len=0)

        value_lora_b_rank1 = self._alloc_indices(len(key))
        self.cache.insert(RadixKey(key, "lora_b", 1), value=value_lora_b_rank1, prev_prefix_len=0)

        # Verify each combination returns its own value
        match = self.cache.match_prefix(RadixKey(key, "lora_a", 0))
        np.testing.assert_array_equal(np.asarray(match.device_indices), value_lora_a_rank0)

        match = self.cache.match_prefix(RadixKey(key, "lora_a", 1))
        np.testing.assert_array_equal(np.asarray(match.device_indices), value_lora_a_rank1)

        match = self.cache.match_prefix(RadixKey(key, "lora_b", 0))
        np.testing.assert_array_equal(np.asarray(match.device_indices), value_lora_b_rank0)

        match = self.cache.match_prefix(RadixKey(key, "lora_b", 1))
        np.testing.assert_array_equal(np.asarray(match.device_indices), value_lora_b_rank1)

        # Verify all values are different
        values = [value_lora_a_rank0, value_lora_a_rank1, value_lora_b_rank0, value_lora_b_rank1]
        for i, v1 in enumerate(values):
            for v2 in values[i + 1 :]:
                self.assertFalse(np.array_equal(v1, v2))

    def test_dp_rank_preserves_on_slicing(self):
        """Test that RadixKey slicing preserves both extra_key and dp_rank"""
        key = RadixKey([1, 2, 3, 4, 5], "test_key", 2)

        # Test slicing
        sliced = key[2:]
        self.assertEqual(sliced.token_ids, [3, 4, 5])
        self.assertEqual(sliced.extra_key, "test_key")
        self.assertEqual(sliced.dp_rank, 2)

        # Test single index access returns RadixKey with dp_rank
        single = key[0]
        self.assertEqual(single.token_ids, [1])
        self.assertEqual(single.extra_key, "test_key")
        self.assertEqual(single.dp_rank, 2)

    def test_swa_free_group_batching_multi_rank(self):
        """Test SWA allocator free_group batching for multiple DP ranks."""
        dp_size = 2
        allocator = self._create_swa_allocator(dp_size=dp_size)

        initial_full = [allocator.full_available_size(dp_rank=r) for r in range(dp_size)]
        initial_swa = [allocator.swa_available_size(dp_rank=r) for r in range(dp_size)]

        alloc_size = 4
        alloc_r0 = allocator.alloc(alloc_size, dp_rank=0)
        alloc_r1 = allocator.alloc(alloc_size, dp_rank=1)
        self.assertIsNotNone(alloc_r0)
        self.assertIsNotNone(alloc_r1)

        allocator.free_group_begin()
        self.assertIsInstance(allocator.free_group, list)
        self.assertEqual(len(allocator.free_group), dp_size)
        for rank in range(dp_size):
            self.assertEqual(allocator.free_group[rank], [])

        allocator.free(alloc_r0, dp_rank=0)
        allocator.free(alloc_r1, dp_rank=1)

        self.assertEqual(allocator.full_available_size(dp_rank=0), initial_full[0] - alloc_size)
        self.assertEqual(allocator.full_available_size(dp_rank=1), initial_full[1] - alloc_size)
        self.assertEqual(allocator.swa_available_size(dp_rank=0), initial_swa[0] - alloc_size)
        self.assertEqual(allocator.swa_available_size(dp_rank=1), initial_swa[1] - alloc_size)

        allocator.free_group_end()
        self.assertEqual(allocator.full_available_size(dp_rank=0), initial_full[0])
        self.assertEqual(allocator.full_available_size(dp_rank=1), initial_full[1])
        self.assertEqual(allocator.swa_available_size(dp_rank=0), initial_swa[0])
        self.assertEqual(allocator.swa_available_size(dp_rank=1), initial_swa[1])

    def test_evictable_size_returns_min(self):
        cache = self.cache
        indices = self._alloc_indices(10)
        cache.insert(RadixKey(list(range(10)), None), indices)
        self.assertEqual(cache.evictable_size(), 10)

    # ------------------------------------------------------------------ #
    #  T1-T12: Tombstone / protected-prefix correctness tests             #
    # ------------------------------------------------------------------ #

    def test_tombstone_creation(self):
        """T1: SWA evict creates tombstone on internal node, swa_evictable_size_ decreases."""
        cache = self._make_small_cache(sliding_window_size=4)

        shared = list(range(0, 10))
        key_a = shared + list(range(10, 15))
        key_b = shared + list(range(20, 25))
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        cache.insert(key_a, value=val_a, prev_prefix_len=0)
        cache.insert(key_b, value=val_b, prev_prefix_len=0)

        full_before, swa_before = cache.total_size()
        self.assertEqual(full_before, swa_before)  # no tombstones yet
        swa_evictable_before = cache.swa_evictable_size()

        # SWA evict the shared prefix
        cache.evict(full_num_tokens=0, swa_num_tokens=len(shared))

        full_after, swa_after = cache.total_size()
        # Full tokens unchanged (tombstone keeps full), SWA tokens decreased
        self.assertEqual(full_after, full_before)
        self.assertLess(swa_after, swa_before)

        # swa_evictable_size_ should have decreased
        swa_evictable_after = cache.swa_evictable_size()
        self.assertLess(swa_evictable_after, swa_evictable_before)

        # Walk the tree to find the shared internal node and verify tombstone
        child_key = cache.get_child_key_fn(RadixKey([0], None))
        internal_node = cache.root_node.children[child_key]
        self.assertTrue(internal_node.swa_tombstone)
        self.assertEqual(len(internal_node.value), 10)

        self._verify_size_consistency_for(cache, "after tombstone creation")

    def test_tombstone_prefix_match(self):
        """T2: Tombstone node does not affect prefix matching results when sliding window is satisfied."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        # With sliding_window_size=4 and suffix length=5 (>4), matching should still work
        match_a = cache.match_prefix(key_a)
        self.assertEqual(len(match_a.device_indices), len(key_a))

        match_b = cache.match_prefix(key_b)
        self.assertEqual(len(match_b.device_indices), len(key_b))

    def test_tombstone_healing_branch1(self):
        """T3: swa_evicted_seqlen <= node_start → full revive of tombstone."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        # Verify shared prefix is tombstone by walking tree
        child_key = cache.get_child_key_fn(RadixKey([0], None))
        tombstone_node = cache.root_node.children[child_key]
        self.assertTrue(tombstone_node.swa_tombstone)

        # Re-insert key_a with swa_evicted_seqlen=0 (Branch 1: not evicted)
        new_val = self._alloc_indices(len(key_a))
        cache.insert(key_a, value=new_val, prev_prefix_len=0, swa_evicted_seqlen=0)

        # The tombstone should be healed (revived)
        healed_node = cache.root_node.children[child_key]
        self.assertFalse(healed_node.swa_tombstone)

        self._verify_size_consistency_for(cache, "after branch 1 healing")

    def test_tombstone_healing_branch2(self):
        """T4: swa_evicted_seqlen in middle of tombstone node → split and partial revive."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        # Verify shared prefix is tombstone
        child_key = cache.get_child_key_fn(RadixKey([0], None))
        tombstone_node = cache.root_node.children[child_key]
        self.assertTrue(tombstone_node.swa_tombstone)

        # Re-insert with swa_evicted_seqlen=5 (in the middle of shared[0:10])
        # Branch 2: split at position 5, revive [5:10], keep [0:5] as tombstone
        new_val = self._alloc_indices(len(key_a))
        cache.insert(key_a, value=new_val, prev_prefix_len=0, swa_evicted_seqlen=5)

        # Walk tree: root → [0:5] (tombstone) → [5:10] (revived) → {[10:15], [20:24]}
        front_node = cache.root_node.children[child_key]
        self.assertTrue(front_node.swa_tombstone)
        self.assertEqual(len(front_node.value), 5)

        # The back node [5:10] should have been revived
        # It should be a child of front_node
        self.assertGreater(len(front_node.children), 0)
        # Find the child whose key starts with token 5
        back_child_key = cache.get_child_key_fn(RadixKey([5], None))
        self.assertIn(back_child_key, front_node.children)
        back_node = front_node.children[back_child_key]
        self.assertFalse(back_node.swa_tombstone)
        self.assertEqual(len(back_node.value), 5)

        self._verify_size_consistency_for(cache, "after branch 2 healing")

    def test_tombstone_healing_branch3(self):
        """T5: swa_evicted_seqlen >= node_end → keep tombstone."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        # Verify shared prefix is tombstone
        child_key = cache.get_child_key_fn(RadixKey([0], None))
        self.assertTrue(cache.root_node.children[child_key].swa_tombstone)

        # Re-insert with swa_evicted_seqlen=15 (>= node_end=10)
        # Branch 3: entire node already evicted → keep tombstone
        new_val = self._alloc_indices(len(key_a))
        cache.insert(key_a, value=new_val, prev_prefix_len=0, swa_evicted_seqlen=15)

        # The shared prefix should still be tombstone
        self.assertTrue(cache.root_node.children[child_key].swa_tombstone)

        self._verify_size_consistency_for(cache, "after branch 3 (keep tombstone)")

    def test_cascade_delete_tombstone(self):
        """T6: Leaf delete cascades to tombstone parent."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        full_before, _ = cache.total_size()

        # Evict enough full tokens to delete all leaves, which cascade to tombstone parent
        cache.evict(full_num_tokens=full_before, swa_num_tokens=0)

        full_after, swa_after = cache.total_size()
        self.assertEqual(full_after, 0)
        self.assertEqual(swa_after, 0)

        self._verify_size_consistency_for(cache, "after cascade delete")

    def test_size_tracking_consistency(self):
        """T7: Insert/evict/delete cycle — sizes match actual tree at every step."""
        cache = self._make_small_cache(sliding_window_size=4)
        keys = [list(range(i * 100, i * 100 + 20)) for i in range(5)]
        vals = [self._alloc_indices(20) for _ in range(5)]

        # Insert all
        for k, v in zip(keys, vals):
            cache.insert(k, value=v, prev_prefix_len=0)
            self._verify_size_consistency_for(cache, f"after insert {k[0]}")

        # SWA evict some
        cache.evict(full_num_tokens=0, swa_num_tokens=20)
        self._verify_size_consistency_for(cache, "after swa evict 20")

        # Full evict some
        cache.evict(full_num_tokens=20, swa_num_tokens=0)
        self._verify_size_consistency_for(cache, "after full evict 20")

        # Insert more (reuse prefix)
        new_key = keys[0] + [999, 998, 997]
        new_val = self._alloc_indices(len(new_key))
        cache.insert(new_key, value=new_val, prev_prefix_len=0)
        self._verify_size_consistency_for(cache, "after re-insert with extension")

        # Evict everything
        full_total, _ = cache.total_size()
        cache.evict(full_num_tokens=full_total, swa_num_tokens=0)
        self._verify_size_consistency_for(cache, "after evict all")

    def test_protected_prefix_basic(self):
        """T8: The protected tree prefix prevents swa_evicted_seqlen from corrupting tree slots during insert."""
        cache = self._make_small_cache(sliding_window_size=4)
        key = list(range(0, 20))
        val_first = self._alloc_indices(len(key))

        # First insert: puts everything into tree
        cache.insert(key, value=val_first, prev_prefix_len=0)

        # Match to confirm
        match = cache.match_prefix(key)
        self.assertEqual(len(match.device_indices), 20)

        # SWA-evict to create tombstone
        cache.evict(full_num_tokens=0, swa_num_tokens=20)

        # Second insert with protected_prefix_len=10 (prev_prefix_len) and swa_evicted_seqlen=5
        # Branch 2 applies: swa_evicted_seqlen(5) < node_end(20), split at 5, revive [5:20]
        val_second = self._alloc_indices(len(key))
        cache.insert(key, value=val_second, prev_prefix_len=10, swa_evicted_seqlen=5)

        # Match should still return 20 tokens (sliding_window=4, suffix > 4)
        match2 = cache.match_prefix(key)
        self.assertEqual(len(match2.device_indices), 20)

        self._verify_size_consistency_for(cache, "after protected prefix insert")

    def test_new_node_tombstone_split(self):
        """T9: New node crossing swa_evicted_seqlen boundary → correct split."""
        cache = self._make_small_cache(sliding_window_size=4)
        key = list(range(0, 20))
        val = self._alloc_indices(len(key))

        # Insert with swa_evicted_seqlen=10 → new node should be split:
        # [0:10] as tombstone + [10:20] as non-tombstone
        cache.insert(key, value=val, prev_prefix_len=0, swa_evicted_seqlen=10)

        full_total, swa_total = cache.total_size()
        self.assertEqual(full_total, 20)
        self.assertEqual(swa_total, 10)  # only [10:20] has SWA

        # Verify tree structure: root should have one child (tombstone [0:10])
        # which has one child (non-tombstone [10:20])
        root = cache.root_node
        self.assertEqual(len(root.children), 1)
        child_key = list(root.children.keys())[0]
        tombstone_node = root.children[child_key]
        self.assertTrue(tombstone_node.swa_tombstone)
        self.assertEqual(len(tombstone_node.value), 10)

        self.assertEqual(len(tombstone_node.children), 1)
        leaf_key = list(tombstone_node.children.keys())[0]
        leaf_node = tombstone_node.children[leaf_key]
        self.assertFalse(leaf_node.swa_tombstone)
        self.assertEqual(len(leaf_node.value), 10)

        # Match should return all 20 tokens (sliding_window=4, non-tombstone suffix=10 > 4)
        match = cache.match_prefix(key)
        self.assertEqual(len(match.device_indices), 20)

        self._verify_size_consistency_for(cache, "after new node tombstone split")

    def test_evict_both_pools(self):
        """T10: Phase 1 (leaf) + Phase 2 (tombstone) interleaved eviction."""
        cache = self._make_small_cache(sliding_window_size=4)
        shared = list(range(0, 8))
        key_a = shared + list(range(10, 18))
        key_b = shared + list(range(20, 28))
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        cache.insert(key_a, value=val_a, prev_prefix_len=0)
        cache.insert(key_b, value=val_b, prev_prefix_len=0)

        full_before, swa_before = cache.total_size()
        self.assertEqual(full_before, swa_before)

        # Phase 2 SWA evict: should tombstone the shared prefix
        cache.evict(full_num_tokens=0, swa_num_tokens=len(shared))
        full_mid, swa_mid = cache.total_size()
        self.assertEqual(full_mid, full_before)  # full unchanged
        self.assertLess(swa_mid, swa_before)  # SWA decreased
        self._verify_size_consistency_for(cache, "after phase 2 evict")

        # Phase 1 full evict: should delete leaf + cascade tombstone
        cache.evict(full_num_tokens=8, swa_num_tokens=0)
        full_after, swa_after = cache.total_size()
        self.assertLess(full_after, full_mid)
        self._verify_size_consistency_for(cache, "after phase 1 evict")

    def test_insert_after_tombstone_evict_cycle(self):
        """T11: Multiple insert→evict→insert cycles — no size leaks."""
        cache = self._make_small_cache(sliding_window_size=4)

        for cycle in range(5):
            key = list(range(cycle * 100, cycle * 100 + 16))
            val = self._alloc_indices(len(key))
            cache.insert(key, value=val, prev_prefix_len=0)
            self._verify_size_consistency_for(cache, f"cycle {cycle} insert")

            # SWA evict some
            cache.evict(full_num_tokens=0, swa_num_tokens=8)
            self._verify_size_consistency_for(cache, f"cycle {cycle} swa evict")

            # Re-insert with new values (simulating healing)
            new_val = self._alloc_indices(len(key))
            cache.insert(key, value=new_val, prev_prefix_len=0, swa_evicted_seqlen=0)
            self._verify_size_consistency_for(cache, f"cycle {cycle} re-insert")

        # Final full evict
        full_total, _ = cache.total_size()
        cache.evict(full_num_tokens=full_total + 100, swa_num_tokens=0)
        full_final, swa_final = cache.total_size()
        self.assertEqual(full_final, 0, "full size should be 0 after evict all")
        self.assertEqual(swa_final, 0, "swa size should be 0 after evict all")
        self._verify_size_consistency_for(cache, "after final evict all")

    def test_size_tracking_fuzz(self):
        """T12: Random operations, verify size consistency at each step."""
        import random

        rng = random.Random(42)
        cache = self._make_small_cache(sliding_window_size=4)

        for step in range(50):
            op = rng.choice(["insert", "evict_swa", "evict_full"])
            if op == "insert":
                length = rng.randint(4, 32)
                start = rng.randint(0, 500)
                key = list(range(start, start + length))
                val = self._alloc_indices(length)
                swa_evicted = rng.choice([0, 0, 0, rng.randint(0, length)])
                cache.insert(
                    key,
                    value=val,
                    prev_prefix_len=0,
                    swa_evicted_seqlen=swa_evicted,
                )
            elif op == "evict_swa":
                amount = rng.randint(1, 16)
                cache.evict(full_num_tokens=0, swa_num_tokens=amount)
            else:
                amount = rng.randint(1, 16)
                cache.evict(full_num_tokens=amount, swa_num_tokens=0)

            self._verify_size_consistency_for(cache, f"fuzz step {step} op={op}")

    # ------------------------------------------------------------------ #
    #  T13-T22: Gap fix verification tests                                #
    # ------------------------------------------------------------------ #

    def test_tombstone_reset_unconditional(self):
        """T13: _match_prefix_helper resets match_len_since_tombstone even when
        sliding window check fails.

        Scenario: Two consecutive tombstone nodes on the path. Before the fix,
        the second tombstone would NOT reset match_len_since_tombstone because
        the first tombstone already set it to 0 (< sliding_window_size), so the
        condition was False and the reset was skipped. After the fix, the reset
        is unconditional.
        """
        cache = self._make_small_cache(sliding_window_size=4)

        # Build a tree: root → [0..4] → [5..9] → [10..14] (leaf)
        # We insert two sequences to force a split so the shared prefixes become
        # internal nodes that we can tombstone independently.
        shared1 = list(range(0, 5))
        shared2 = list(range(5, 10))
        suffix_a = list(range(10, 15))
        suffix_b = list(range(20, 25))

        key_a = shared1 + shared2 + suffix_a
        key_b = shared1 + shared2 + suffix_b
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        cache.insert(key_a, value=val_a, prev_prefix_len=0)
        cache.insert(key_b, value=val_b, prev_prefix_len=0)

        # SWA-evict both shared prefixes: first [0..4], then [5..9]
        # This creates two consecutive tombstone nodes
        cache.evict(full_num_tokens=0, swa_num_tokens=10)

        # Walk tree to verify two tombstones
        child_key_0 = cache.get_child_key_fn(RadixKey([0], None))
        node_0_4 = cache.root_node.children[child_key_0]
        self.assertTrue(node_0_4.swa_tombstone, "First shared prefix should be tombstone")

        # Match should still work: suffix (5 tokens) > sliding_window (4)
        match_a = cache.match_prefix(key_a)
        # Even with two consecutive tombstones, the suffix after the last tombstone
        # should be enough to match if it exceeds sliding_window_size
        self.assertGreater(len(match_a.device_indices), 0)

        self._verify_size_consistency_for(cache, "after double tombstone match")

    def test_new_leaf_never_tombstone(self):
        """T14: When swa_evicted_seqlen >= total_prefix_length + len(key),
        all remaining tokens are SWA-evicted. No node is created (can't be
        non-tombstone because SWA is gone, can't be tombstone leaf because
        of the leaf-must-not-be-tombstone invariant). Value is freed and
        insert returns early.
        """
        cache = self._make_small_cache(sliding_window_size=4)
        key = list(range(0, 10))
        val = self._alloc_indices(len(key))

        # Insert with swa_evicted_seqlen=20 (>> total_prefix_length + len(key) = 10)
        # Upstream behavior: free value, return early, no node created
        cache.insert(key, value=val, prev_prefix_len=0, swa_evicted_seqlen=20)

        full_total, swa_total = cache.total_size()
        # No node created → tree is empty
        self.assertEqual(full_total, 0)
        self.assertEqual(swa_total, 0)

        # Root should have no children
        self.assertEqual(len(cache.root_node.children), 0)

        self._verify_size_consistency_for(cache, "after all-evicted insert")

    def test_new_leaf_never_tombstone_exact_boundary(self):
        """T15: swa_evicted_seqlen == total_prefix_length + len(key) boundary case.

        When swa_evicted_seqlen exactly equals total_prefix_length + len(key),
        all tokens are SWA-evicted. No node is created, value is freed.
        """
        cache = self._make_small_cache(sliding_window_size=4)
        key = list(range(0, 10))
        val = self._alloc_indices(len(key))

        # swa_evicted_seqlen == total_prefix_length(0) + len(key)(10) = 10
        cache.insert(key, value=val, prev_prefix_len=0, swa_evicted_seqlen=10)

        full_total, swa_total = cache.total_size()
        # No node created → tree is empty
        self.assertEqual(full_total, 0)
        self.assertEqual(swa_total, 0)

        # Root should have no children
        self.assertEqual(len(cache.root_node.children), 0)

        self._verify_size_consistency_for(cache, "after exact boundary insert")

    def test_delete_leaf_uses_child_key_fn(self):
        """T17: _delete_leaf correctly removes nodes using get_child_key_fn lookup."""
        cache = self._make_small_cache(sliding_window_size=4)

        # Insert two sequences sharing a prefix to create internal + leaf structure
        shared = list(range(0, 5))
        key_a = shared + list(range(10, 15))
        key_b = shared + list(range(20, 25))
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        cache.insert(key_a, value=val_a, prev_prefix_len=0)
        cache.insert(key_b, value=val_b, prev_prefix_len=0)

        full_before, swa_before = cache.total_size()
        self.assertEqual(full_before, 15)  # 5 shared + 5 suffix_a + 5 suffix_b

        # The tree has: root → [0..4] (internal) → {[10..14] (leaf A), [20..24] (leaf B)}
        # Evict one leaf by full eviction
        cache.evict(full_num_tokens=5, swa_num_tokens=0)

        full_after, swa_after = cache.total_size()
        self.assertLess(full_after, full_before)

        # Verify tree is still consistent
        self._verify_size_consistency_for(cache, "after leaf delete via child_key_fn")

    def test_delete_tombstone_leaf_uses_child_key_fn(self):
        """T18: _delete_tombstone_leaf correctly removes tombstone nodes using get_child_key_fn."""
        cache, key_a, val_a, key_b, val_b = self._make_tree_with_tombstone()

        # At this point: shared prefix is tombstone, two leaves exist
        full_before, _ = cache.total_size()

        # Full evict everything — should cascade delete tombstone parent via _delete_tombstone_leaf
        cache.evict(full_num_tokens=full_before, swa_num_tokens=0)

        full_after, swa_after = cache.total_size()
        self.assertEqual(full_after, 0)
        self.assertEqual(swa_after, 0)

        # Root should have no children
        self.assertEqual(len(cache.root_node.children), 0)
        self._verify_size_consistency_for(cache, "after tombstone cascade delete via child_key_fn")

    def test_evict_req_swa_uses_locked_tree_boundary(self):
        """T19: Request SWA eviction derives its protected boundary from the locked tree path."""
        cache = self._make_small_cache(sliding_window_size=4)
        key = list(range(0, 20))
        tree_indices = self._alloc_indices(len(key))
        cache.insert(key, value=tree_indices, prev_prefix_len=0)
        match = cache.match_prefix(key)

        tail_indices = self._alloc_indices(10)
        req_indices = np.concatenate([tree_indices, tail_indices])
        self.req_pool.req_to_token[0, :30] = req_indices

        class MockReq:
            def __init__(self, last_node):
                self.swa_evicted_seqlen = 3
                self.req_pool_idx = 0
                self.last_node = last_node

        req = MockReq(match.last_device_node)
        swa_before = self.allocator.swa_available_size()

        cache.evict_req_swa(req, pre_len=30)

        # Protected prefix is 20, so only uncached tail slots [20:25) are reclaimed.
        self.assertEqual(req.swa_evicted_seqlen, 25)
        self.assertEqual(self.allocator.swa_available_size(), swa_before + 5)
        self.assertEqual(self.allocator.count_swa_mapped(tree_indices), len(tree_indices))
        self.assertEqual(self.allocator.count_swa_mapped(tail_indices[:5]), 0)
        self.assertEqual(self.allocator.count_swa_mapped(tail_indices[5:]), 5)

    def test_evict_req_swa_page_alignment_uses_tree_boundary(self):
        """T20: Request SWA eviction aligns by page size using the tree-derived protected boundary."""
        paged_cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=4,
            page_size=4,
            disable=False,
        )

        key = list(range(0, 8))
        tree_indices = self._alloc_indices(len(key))
        paged_cache.insert(key, value=tree_indices, prev_prefix_len=0)
        match = paged_cache.match_prefix(key)

        tail_indices = self._alloc_indices(12)
        req_indices = np.concatenate([tree_indices, tail_indices])
        self.req_pool.req_to_token[0, :20] = req_indices

        class MockReq:
            def __init__(self, last_node):
                self.swa_evicted_seqlen = 0
                self.req_pool_idx = 0
                self.last_node = last_node

        req = MockReq(match.last_device_node)

        paged_cache.evict_req_swa(req, pre_len=20)

        # Protected prefix is 8. The eviction frontier is 20 - 4 - 4 = 12, already aligned.
        self.assertEqual(req.swa_evicted_seqlen, 12)
        self.assertEqual(self.allocator.count_swa_mapped(tree_indices), len(tree_indices))
        self.assertEqual(self.allocator.count_swa_mapped(tail_indices[:4]), 0)
        self.assertEqual(self.allocator.count_swa_mapped(tail_indices[4:]), 8)

    def test_swa_evicted_seqlen_page_alignment_assertion(self):
        """T21: _insert_helper asserts swa_evicted_seqlen % page_size == 0 when handling tombstones."""
        # Create a paged cache with page_size=4
        paged_cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=64,
            page_size=4,
            disable=False,
        )

        # Insert then tombstone: create a sequence, evict its SWA to make tombstone
        shared = list(range(0, 8))
        key_a = shared + list(range(10, 18))
        key_b = shared + list(range(20, 28))
        val_a = self._alloc_indices(len(key_a))
        val_b = self._alloc_indices(len(key_b))

        paged_cache.insert(key_a, value=val_a, prev_prefix_len=0)
        paged_cache.insert(key_b, value=val_b, prev_prefix_len=0)
        paged_cache.evict(full_num_tokens=0, swa_num_tokens=8)

        # Re-inserting with non-page-aligned swa_evicted_seqlen should assert
        new_val = self._alloc_indices(len(key_a))
        with self.assertRaises(AssertionError):
            paged_cache.insert(key_a, value=new_val, prev_prefix_len=0, swa_evicted_seqlen=3)

    def test_cache_unfinished_req_writeback_range(self):
        """T22: cache_unfinished_req writes back from last_matched_prefix_len, not len(prefix_indices)."""
        cache = self._make_small_cache(sliding_window_size=64)

        # Simulate first insert (initial request)
        key_part1 = list(range(0, 10))
        val1 = self._alloc_indices(len(key_part1))
        cache.insert(key_part1, value=val1, prev_prefix_len=0)

        # After first insert, match should return all 10 tokens
        match1 = cache.match_prefix(key_part1)
        self.assertEqual(len(match1.device_indices), 10)

        # Insert extended sequence (simulating cache_unfinished_req)
        key_full = list(range(0, 20))
        val2 = self._alloc_indices(len(key_full))
        cache.insert(key_full, value=val2, prev_prefix_len=10, swa_evicted_seqlen=0)

        # Match the full key
        match2 = cache.match_prefix(key_full)
        self.assertEqual(len(match2.device_indices), 20)

        # Verify old cached indices are preserved (not overwritten)
        np.testing.assert_array_equal(
            np.asarray(match2.device_indices[:10]),
            np.asarray(match1.device_indices),
        )

        self._verify_size_consistency_for(cache, "after simulated cache_unfinished_req")


class TestSchedulerCacheInit(CustomTestCase):
    """Tests for scheduler cache type selection with hybrid models (#202)."""

    def _select_cache_type(self, is_hybrid, chunked_prefill_size, disable_radix_cache):
        """Mirror the condition logic in scheduler.py init_memory_pool_and_cache."""
        if is_hybrid:
            return "SWARadixCache"
        elif chunked_prefill_size is not None and disable_radix_cache:
            return "ChunkCache"
        else:
            return "RadixCache"

    def test_hybrid_with_disable_radix_cache_gets_swa_radix_cache(self):
        """When is_hybrid=True and disable_radix_cache=True, must use SWARadixCache, not ChunkCache."""
        self.assertEqual(
            self._select_cache_type(
                is_hybrid=True, chunked_prefill_size=8192, disable_radix_cache=True
            ),
            "SWARadixCache",
        )

    def test_hybrid_with_radix_cache_enabled_gets_swa_radix_cache(self):
        """When is_hybrid=True and disable_radix_cache=False, must use SWARadixCache."""
        self.assertEqual(
            self._select_cache_type(
                is_hybrid=True, chunked_prefill_size=8192, disable_radix_cache=False
            ),
            "SWARadixCache",
        )

    def test_non_hybrid_with_disable_radix_cache_gets_chunk_cache(self):
        """Non-hybrid with disable_radix_cache should still use ChunkCache."""
        self.assertEqual(
            self._select_cache_type(
                is_hybrid=False, chunked_prefill_size=8192, disable_radix_cache=True
            ),
            "ChunkCache",
        )


if __name__ == "__main__":
    unittest.main()
