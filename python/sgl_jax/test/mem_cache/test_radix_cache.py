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
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import (
    RadixCache,
    TreeNode,
    _key_match_page_size1,
    _key_match_paged,
)


class TestRadixCache(unittest.TestCase):
    def setUp(self):
        self.devices = jax.devices()
        self.kv_head_num = 32
        self.head_dim = 128
        self.layer_num = 24
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    def _create_single_device_setup(self):
        mesh = Mesh([self.devices[0]], axis_names=("tensor",))

        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            mesh=mesh,
            token_partition_axis="tensor",
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
            # size=self.pool_size, dtype=self.dtype, kvcache=kv_cache
            size=self.pool_size,
            kvcache=kv_cache,
        )

        return mesh, req_pool, allocator

    def _create_multi_device_setup(self):
        if len(self.devices) < 2:
            self.skipTest("need at least 2 devices for multi-device test")

        # use multi-device mesh, need to reshape devices to 2D array
        if len(self.devices) >= 8:
            # use 8 devices, reshape to (2, 4) shape
            devices = np.array(self.devices[:8]).reshape(2, 4)
            mesh = Mesh(devices, axis_names=("data", "tensor"))
        elif len(self.devices) >= 4:
            # use 4 devices, reshape to (2, 2) shape
            devices = np.array(self.devices[:4]).reshape(2, 2)
            mesh = Mesh(devices, axis_names=("data", "tensor"))
        else:
            # use 2 devices, reshape to (1, 2) shape
            devices = np.array(self.devices[:2]).reshape(1, 2)
            mesh = Mesh(devices, axis_names=("data", "tensor"))

        # create memory pool
        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            mesh=mesh,
            token_partition_axis="data",
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
            kv_partition_axis="tensor",
        )

        # create allocator
        allocator = TokenToKVPoolAllocator(
            # size=self.pool_size, dtype=self.dtype, kvcache=kv_cache
            size=self.pool_size,
            kvcache=kv_cache,
        )

        return mesh, req_pool, allocator

    def _create_auto_device_setup(self):
        if len(self.devices) > 1:
            return self._create_multi_device_setup()
        else:
            return self._create_single_device_setup()

    def _create_radix_cache(self, mesh, req_pool, allocator, **kwargs):
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

    def _print_cache_sharding_info(self, cache, mesh, req_pool, allocator):
        print("\n" + "=" * 60)
        print(f"[MESH INFO] device number: {len(self.devices)}, Mesh axis: {mesh.axis_names}")
        print(f"[MESH INFO] Mesh device layout: {mesh.devices.shape}")
        print(f"[MESH INFO] Mesh: {mesh}")

        def print_sharding(obj, name, prefix=""):
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(obj, "sharding") and obj.sharding is not None:
                print(f"[SHARDING] {full_name}: sharding={obj.sharding}")
                if hasattr(obj, "addressable_shards"):
                    for i, shard in enumerate(obj.addressable_shards):
                        print(
                            f"    [SHARD] idx={i}, device={shard.device}, index={getattr(shard, 'index', None)}, shape={getattr(shard.data, 'shape', None)}"
                        )
            elif hasattr(obj, "shape"):
                print(
                    f"[SHARDING] {full_name}: Unsharded, shape={obj.shape}, device={getattr(obj, 'device', 'unknown')}"
                )
            else:
                print(f"[SHARDING] {full_name}: Non-JAX array, type={type(obj)}")

        # print RadixCache sharding information
        if hasattr(cache, "kv_cache_sharding"):
            print(f"[CACHE] KV cache sharding strategy: {cache.kv_cache_sharding}")
        if hasattr(cache, "token_sharding"):
            print(f"[CACHE] Token sharding strategy: {cache.token_sharding}")

        # print ReqToTokenPool sharding information
        print_sharding(req_pool.req_to_token, "req_to_token_pool.req_to_token")

        # print KV Cache sharding information
        kv_cache = allocator.get_kvcache()
        if hasattr(kv_cache, "k_buffer") and kv_cache.k_buffer:
            print_sharding(kv_cache.k_buffer[0], "kv_cache.k_buffer[0]", "allocator")
        if hasattr(kv_cache, "v_buffer") and kv_cache.v_buffer:
            print_sharding(kv_cache.v_buffer[0], "kv_cache.v_buffer[0]", "allocator")
        if hasattr(kv_cache, "kv_buffer") and kv_cache.kv_buffer:
            print_sharding(kv_cache.kv_buffer[0], "kv_cache.kv_buffer[0]", "allocator")

        print("=" * 60)

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

    def test_single_device_init(self):
        mesh, req_pool, allocator = self._create_single_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)
        print(f"\n[single device test] device number: {len(self.devices)}")
        self._print_cache_sharding_info(cache, mesh, req_pool, allocator)

        self.assertIsNotNone(cache.root_node)
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.total_size(), 0)

    def test_multi_device_init(self):
        mesh, req_pool, allocator = self._create_multi_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)
        print(f"\n[multi device test] device number: {len(self.devices)}")
        self._print_cache_sharding_info(cache, mesh, req_pool, allocator)

        self.assertIsNotNone(cache.root_node)
        self.assertEqual(cache.root_node.lock_ref, 1)

    def test_disabled_cache(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator, disable=True)

        # test disabled cache behavior
        key = [1, 2, 3, 4, 5]
        match_result = cache.match_prefix(key)
        self.assertEqual(len(match_result.device_indices), 0)

        insert_result = cache.insert(key)
        self.assertEqual(insert_result, 0)

    def test_basic_insert_and_match(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

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
        mesh, req_pool, allocator = self._create_auto_device_setup()
        cache = self._create_radix_cache(mesh, req_pool, allocator)

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
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

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
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

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
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator, page_size=4)

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
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

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

    def test_get_cached_kv(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # first allocate some KV cache space
        key = [1, 2, 3, 4, 5]
        kv_indices = allocator.alloc(len(key))
        self.assertIsNotNone(kv_indices, "should be able to allocate KV cache space")

        # write some test data to the allocated KV cache position
        kv_cache = allocator.get_kvcache()
        test_k_data = jnp.arange(
            len(key) * self.kv_head_num * self.head_dim, dtype=self.dtype
        ).reshape(len(key), self.kv_head_num, self.head_dim)
        test_v_data = test_k_data * 2  # V data is 2 times of K data

        # write data to first layer as test
        kv_cache.set_kv_buffer(0, kv_indices, test_k_data, test_v_data)

        # insert data to RadixCache, pass in real KV indices
        cache.insert(key, kv_indices)

        # get KV data
        kv_data, matched_len = cache.get_cached_kv(key)

        # verify shape
        expected_shape = (self.layer_num, matched_len, self.kv_head_num, self.head_dim)
        self.assertEqual(kv_data.shape, expected_shape)
        self.assertEqual(matched_len, len(key))

        # verify data correctness - check if the data in the first layer matches the written data
        # note: get_cached_kv returns K data
        first_layer_k_data = kv_data[0]  # data in the first layer

        # move data to CPU for comparison
        first_layer_k_data_cpu = jax.device_get(first_layer_k_data)
        test_k_data_cpu = jax.device_get(test_k_data)

        # verify data content
        self.assertTrue(
            jnp.array_equal(first_layer_k_data_cpu, test_k_data_cpu),
            "data in the first layer should match the written test data",
        )

        # clean up allocated resources
        allocator.free(kv_indices)

    def test_get_cached_kv_without_value(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # insert token sequence, without passing value
        key = [1, 2, 3, 4, 5]
        cache.insert(key)  # no value passed

        # get KV data
        kv_data, matched_len = cache.get_cached_kv(key)

        # when only inserting token sequence, match_prefix can find matching token sequence
        # so matched_len should be the length of the token sequence
        self.assertEqual(
            matched_len,
            len(key),
            "when inserting token sequence, matched_len should be equal to the length of the token sequence",
        )

        # verify shape - the returned data shape should be (layer_num, matched_len, kv_head_num, head_dim)
        expected_shape = (self.layer_num, matched_len, self.kv_head_num, self.head_dim)
        self.assertEqual(kv_data.shape, expected_shape)

        # since there is no actual KV data, get_cpu_copy will use token values as indices
        # this will return data at the corresponding position in the KV cache (usually zero values, because the cache is initialized to zero)
        # verify that the returned data is not empty, but contains zero value data
        self.assertEqual(kv_data.shape[1], matched_len, "returned KV data length should match")

        # verify data content - should be all zero (because KV cache is initialized to zero)
        kv_data_cpu = jax.device_get(kv_data)
        self.assertTrue(
            jnp.allclose(kv_data_cpu, 0),
            "When there is no actual KV data, the returned data should be all zeros",
        )

    def test_empty_key_handling(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # test empty key
        empty_key = []
        match_result = cache.match_prefix(empty_key)
        self.assertEqual(len(match_result.device_indices), 0)

        insert_result = cache.insert(empty_key)
        self.assertEqual(insert_result, 0)

    def test_kv_cache_events(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator, enable_kv_cache_events=True)

        # test event queue
        events = cache.take_events()
        self.assertEqual(len(events), 0)  # initially empty

        # disable event cache
        cache_no_events = self._create_radix_cache(
            mesh, req_pool, allocator, enable_kv_cache_events=False
        )

        events = cache_no_events.take_events()
        self.assertEqual(len(events), 0)

    def test_pretty_print(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # insert some data
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])

        # test print (should not throw exception)
        try:
            cache.pretty_print()
        except Exception as e:
            self.fail(f"pretty_print() raised an exception: {e}")

    def test_reset_functionality(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

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

    def test_device_consistency(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # insert data
        key = [1, 2, 3, 4, 5]
        cache.insert(key)

        # query data
        match_result = cache.match_prefix(key)
        device_indices = match_result.device_indices

        # check if the returned array is on CPU
        self.assertTrue(hasattr(device_indices, "device"))
        print(f"device_indices device: {device_indices.device}")

        # verify device type (should be CPU)
        if hasattr(device_indices, "device"):
            device_str = str(device_indices.device)
            self.assertIn("cpu", device_str.lower(), f"Expected CPU device, got: {device_str}")

        # check array content correctness
        self.assertEqual(len(device_indices), len(key))

        # convert to Python list and verify content
        indices_list = device_indices.tolist()
        self.assertEqual(indices_list, key)

    def test_cross_device_operations(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # insert multiple sequences
        keys = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        for key in keys:
            cache.insert(key)

        # test prefix matching
        prefix_key = [1, 2]
        match_result = cache.match_prefix(prefix_key)

        # verify device consistency
        device_indices = match_result.device_indices
        if hasattr(device_indices, "device"):
            device_str = str(device_indices.device)
            self.assertIn("cpu", device_str.lower(), f"Expected CPU device, got: {device_str}")

        # verify content
        self.assertEqual(len(device_indices), len(prefix_key))
        self.assertEqual(device_indices.tolist(), prefix_key)

    def test_empty_match_device_consistency(self):
        mesh, req_pool, allocator = self._create_auto_device_setup()

        cache = self._create_radix_cache(mesh, req_pool, allocator)

        # test empty key matching
        empty_key = []
        match_result = cache.match_prefix(empty_key)
        device_indices = match_result.device_indices

        # verify empty array is on the correct device
        if hasattr(device_indices, "device"):
            device_str = str(device_indices.device)
            self.assertIn(
                "cpu",
                device_str.lower(),
                f"Expected CPU device for empty array, got: {device_str}",
            )

        # test no match
        no_match_key = [999, 888, 777]
        match_result = cache.match_prefix(no_match_key)
        device_indices = match_result.device_indices

        # verify no match result is on the correct device
        if hasattr(device_indices, "device"):
            device_str = str(device_indices.device)
            self.assertIn(
                "cpu",
                device_str.lower(),
                f"Expected CPU device for no-match array, got: {device_str}",
            )

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

        # create single device environment
        mesh = Mesh([self.devices[0]], axis_names=("tensor",))

        self.req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            mesh=mesh,
            token_partition_axis="tensor",
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

        print(f"\n[request test class] device count: {len(self.devices)}")
        self._print_cache_sharding_info(self.cache, mesh, self.req_pool, self.allocator)

    def _print_cache_sharding_info(self, cache, mesh, req_pool, allocator):
        """print cache related sharding information"""
        print("\n" + "=" * 60)
        print(f"[MESH INFO] device count: {len(self.devices)}, Mesh axis: {mesh.axis_names}")
        print(f"[MESH INFO] Mesh device layout: {mesh.devices.shape}")
        print(f"[MESH INFO] Mesh: {mesh}")

        def print_sharding(obj, name, prefix=""):
            """recursive print object sharding information"""
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(obj, "sharding") and obj.sharding is not None:
                print(f"[SHARDING] {full_name}: sharding={obj.sharding}")
                if hasattr(obj, "addressable_shards"):
                    for i, shard in enumerate(obj.addressable_shards):
                        print(
                            f"    [SHARD] idx={i}, device={shard.device}, index={getattr(shard, 'index', None)}, shape={getattr(shard.data, 'shape', None)}"
                        )
            elif hasattr(obj, "shape"):
                print(
                    f"[SHARDING] {full_name}: Unsharded, shape={obj.shape}, device={getattr(obj, 'device', 'unknown')}"
                )
            else:
                print(f"[SHARDING] {full_name}: Non-JAX array, type={type(obj)}")

        # print RadixCache sharding information
        if hasattr(cache, "kv_cache_sharding"):
            print(f"[CACHE] KV cache sharding strategy: {cache.kv_cache_sharding}")
        if hasattr(cache, "token_sharding"):
            print(f"[CACHE] Token sharding strategy: {cache.token_sharding}")

        # print ReqToTokenPool sharding information
        print_sharding(req_pool.req_to_token, "req_to_token_pool.req_to_token")

        # print KV Cache sharding information
        kv_cache = allocator.get_kvcache()
        if hasattr(kv_cache, "k_buffer") and kv_cache.k_buffer:
            print_sharding(kv_cache.k_buffer[0], "kv_cache.k_buffer[0]", "allocator")
        if hasattr(kv_cache, "v_buffer") and kv_cache.v_buffer:
            print_sharding(kv_cache.v_buffer[0], "kv_cache.v_buffer[0]", "allocator")
        if hasattr(kv_cache, "kv_buffer") and kv_cache.kv_buffer:
            print_sharding(kv_cache.kv_buffer[0], "kv_cache.kv_buffer[0]", "allocator")

        print("=" * 60)

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
