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


class TestSWARadixCache(unittest.TestCase):
    def setUp(self):
        # Keep KV sizes small to make tests light-weight
        self.devices = jax.devices()
        self.mesh = Mesh([self.devices[0]], axis_names=("tensor",))

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


if __name__ == "__main__":
    unittest.main()
