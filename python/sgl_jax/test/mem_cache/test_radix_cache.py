# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/test_radix_cache.py -v
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
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import (
    RadixCache,
    TreeNode,
    _key_match_page_size1,
    _key_match_paged,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestRadixCache(unittest.TestCase):
    def setUp(self):
        self.devices = jax.devices()
        self.kv_head_num = 32
        self.head_dim = 128
        self.layer_num = 24
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    def _create_auto_device_setup(self):
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
        )

        # create allocator
        allocator = TokenToKVPoolAllocator(
            # size=self.pool_size, dtype=self.dtype, kvcache=kv_cache
            size=self.pool_size,
            kvcache=kv_cache,
        )

        return req_pool, allocator

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

    def test_tree_node_basic(self):
        node = TreeNode()
        self.assertIsNotNone(node.id)
        self.assertEqual(node.lock_ref, 0)
        self.assertTrue(node.evicted)  # value is None
        self.assertFalse(node.backuped)  # host_value is None

        # test comparison operation
        node2 = TreeNode()
        # node created earlier, so should be less than node2
        self.assertTrue(node < node2)

    def test_key_match_functions(self):
        # test key matching function
        # test page_size=1 matching
        key1 = [1, 2, 3, 4, 5]
        key2 = [1, 2, 6, 7, 8]
        result = _key_match_page_size1(key1, key2)
        self.assertEqual(result, 2)  # first two elements match

        # test paged matching
        key1 = [1, 2, 3, 4, 5, 6]
        key2 = [1, 2, 3, 4, 7, 8]
        result = _key_match_paged(key1, key2, page_size=2)
        self.assertEqual(result, 4)  # first two pages match

    def test_disabled_cache(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator, disable=True)

        # test disabled cache behavior
        key = [1, 2, 3, 4, 5]
        match_result = cache.match_prefix(key)
        self.assertEqual(len(match_result.device_indices), 0)

        insert_result = cache.insert(key)
        self.assertEqual(insert_result, 0)

    def test_basic_insert_and_match(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator)

        # test insert
        key1 = [1, 2, 3, 4, 5]
        prefix_len = cache.insert(key1)
        self.assertEqual(prefix_len, 0)  # new inserted, no prefix

        # test match
        match_result = cache.match_prefix(key1)
        self.assertEqual(len(match_result.device_indices), len(key1))

        # test partial match
        key2 = [1, 2, 3]
        match_result = cache.match_prefix(key2)
        self.assertEqual(len(match_result.device_indices), len(key2))

        # test no match
        key3 = [6, 7, 8]
        match_result = cache.match_prefix(key3)
        self.assertEqual(len(match_result.device_indices), 0)

    def test_basic_insert_with_value(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator)

        key = [1, 2, 3, 4, 5]
        value = [9, 8, 7, 6, 5]
        prefix_len = cache.insert(key, value)
        self.assertEqual(prefix_len, 0)

        key2 = [1, 2, 3]
        match_result = cache.match_prefix(key2)
        self.assertEqual(len(match_result.device_indices), len(key2))
        value2 = match_result.device_indices
        self.assertEqual(value2.tolist(), value[: len(key2)])

    def test_prefix_extension(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator)

        # insert short sequence
        key1 = [1, 2, 3]
        cache.insert(key1)

        # insert long sequence (contains previous prefix)
        key2 = [1, 2, 3, 4, 5]
        prefix_len = cache.insert(key2)
        self.assertEqual(prefix_len, 3)  # matched 3 tokens

        # verify both sequences can be correctly matched
        match_result1 = cache.match_prefix(key1)
        self.assertEqual(len(match_result1.device_indices), len(key1))

        match_result2 = cache.match_prefix(key2)
        self.assertEqual(len(match_result2.device_indices), len(key2))

    def test_lock_reference_counting(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator)

        # insert data
        key = [1, 2, 3, 4, 5]
        cache.insert(key)

        # get leaf node
        match_result = cache.match_prefix(key)
        last_node = match_result.last_device_node

        # test increase lock reference
        initial_protected = cache.protected_size()
        initial_evictable = cache.evictable_size()

        cache.inc_lock_ref(last_node)

        # verify size change
        self.assertGreaterEqual(cache.protected_size(), initial_protected)
        self.assertLessEqual(cache.evictable_size(), initial_evictable)

        # test decrease lock reference
        cache.dec_lock_ref(last_node)

        # verify restored to initial state
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_paged_cache(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator, page_size=4)

        # test page aligned sequence
        key1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens, aligned to 8
        cache.insert(key1)

        match_result = cache.match_prefix(key1)
        self.assertEqual(len(match_result.device_indices), 8)

        # test non-page aligned sequence (should be truncated)
        key2 = [1, 2, 3, 4, 5, 6, 7]  # 7 tokens, should be truncated to 4
        match_result = cache.match_prefix(key2)
        self.assertEqual(len(match_result.device_indices), 4)

    def test_eviction(self):
        req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(req_pool, allocator)

        # insert multiple sequences
        keys = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        for key in keys:
            cache.insert(key)

        initial_size = cache.total_size()
        self.assertGreater(initial_size, 0)

        # execute eviction
        cache.evict(5)  # evict 5 tokens

        # verify size reduced (possibly not reduced to 5, because of protected nodes)
        final_size = cache.total_size()
        self.assertLessEqual(final_size, initial_size)

    def test_empty_key_handling(self):
        req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(req_pool, allocator)

        # test empty key
        empty_key = []
        match_result = cache.match_prefix(empty_key)
        self.assertEqual(len(match_result.device_indices), 0)

        insert_result = cache.insert(empty_key)
        self.assertEqual(insert_result, 0)

    def test_pretty_print(self):
        req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(req_pool, allocator)

        # insert some data
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])

        # test print (should not throw exception)
        try:
            cache.pretty_print()
        except Exception as e:
            self.fail(f"pretty_print() raised an exception: {e}")

    def test_reset_functionality(self):
        req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(req_pool, allocator)

        # insert data
        cache.insert([1, 2, 3])
        cache.insert([4, 5, 6])

        self.assertGreater(cache.total_size(), 0)

        # reset
        cache.reset()

        # verify reset state
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.total_size(), 0)

    def test_empty_match_consistency(self):
        req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(req_pool, allocator)

        # test empty key matching
        empty_key = []
        match_result = cache.match_prefix(empty_key)
        device_indices = match_result.device_indices

        # verify empty array metadata
        self.assertIsInstance(device_indices, np.ndarray)
        self.assertEqual(device_indices.dtype, np.int32)
        self.assertEqual(len(device_indices), 0)

        # test no match
        no_match_key = [999, 888, 777]
        match_result = cache.match_prefix(no_match_key)
        device_indices = match_result.device_indices

        self.assertIsInstance(device_indices, np.ndarray)
        self.assertEqual(device_indices.dtype, np.int32)
        self.assertEqual(len(device_indices), 0)


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
    ):
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = origin_input_ids
        self.output_ids = output_ids
        self.fill_ids = fill_ids
        self.prefix_indices = prefix_indices
        self.last_node = last_node


class TestRadixCacheWithRequests(unittest.TestCase):
    """test RadixCache with request related functionality"""

    def setUp(self):
        """set up test environment"""
        self.devices = jax.devices()
        self.kv_head_num = 32
        self.head_dim = 128
        self.layer_num = 24
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

        self.req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
        )

        # use tensor axis for single device (but not actually sharded)
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            # use default kv_partition_axis="tensor"
        )

        self.allocator = TokenToKVPoolAllocator(
            # size=self.pool_size, dtype=self.dtype, kvcache=kv_cache
            size=self.pool_size,
            kvcache=kv_cache,
        )

        self.cache = RadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )

    def test_cache_finished_req_disabled(self):
        """test cache finished request disabled"""
        # create disabled cache
        disabled_cache = RadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            disable=True,
            page_size=1,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )

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

    def test_cache_unfinished_req_disabled(self):
        disabled_cache = RadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            disable=True,
            page_size=1,
            kv_head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )

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


if __name__ == "__main__":
    unittest.main()
