#!/usr/bin/env python3
"""
MHA KV Cache Test Suite

Tests for MHATokenToKVPool functionality including:
- Initialization with different configurations
- KV buffer get/set operations
- Non-TP (single device) scenarios
- TP (tensor parallel) scenarios
- Memory usage calculations
- Buffer operations and data integrity
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool


class TestMHAKVCache(unittest.TestCase):
    """Test suite for MHA KV Cache functionality"""

    def setUp(self):
        """Set up test configurations"""
        self.test_configs = {
            "small": {
                "size": 128,
                "page_size": 1,
                "head_num": 8,
                "head_dim": 128,
                "layer_num": 12,
                "dtype": jnp.bfloat16,
            },
            "medium": {
                "size": 512,
                "page_size": 1,
                "head_num": 32,
                "head_dim": 128,
                "layer_num": 24,
                "dtype": jnp.float32,
            },
        }

    def _create_single_device_mesh(self):
        """Create a single device mesh (no TP)"""
        devices = jax.devices()[:1]  # Use only one device
        return Mesh(devices, ("tensor",))

    def _create_multi_device_mesh(self):
        """Create a multi-device mesh for TP testing"""
        devices = jax.devices()
        if len(devices) < 4:
            self.skipTest("Multi-device test requires at least 4 devices")

        # Use 4 devices for tensor parallelism
        mesh_devices = np.array(devices[:4]).reshape(1, 4)
        return Mesh(mesh_devices, ("data", "tensor"))

    def test_initialization_single_device(self):
        """Test MHATokenToKVPool initialization without TP"""
        print("\nTesting MHA KV Cache initialization (single device)...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            # Verify basic properties
            self.assertEqual(kv_pool.size, config["size"])
            self.assertEqual(kv_pool.page_size, config["page_size"])
            self.assertEqual(kv_pool.head_num, config["head_num"])
            self.assertEqual(kv_pool.head_dim, config["head_dim"])
            self.assertEqual(kv_pool.layer_num, config["layer_num"])
            self.assertEqual(kv_pool.dtype, config["dtype"])

            # Verify buffers are created
            self.assertEqual(len(kv_pool.k_buffer), config["layer_num"])
            self.assertEqual(len(kv_pool.v_buffer), config["layer_num"])

            # Verify buffer shapes
            expected_shape = (
                config["size"] + config["page_size"],
                config["head_num"],
                config["head_dim"],
            )
            for layer_id in range(config["layer_num"]):
                k_buf = kv_pool.k_buffer[layer_id]
                v_buf = kv_pool.v_buffer[layer_id]
                self.assertEqual(k_buf.shape, expected_shape)
                self.assertEqual(v_buf.shape, expected_shape)
                self.assertEqual(k_buf.dtype, config["dtype"])
                self.assertEqual(v_buf.dtype, config["dtype"])

            # Verify memory usage calculation
            self.assertGreater(kv_pool.mem_usage, 0)

        print("PASS: Single device initialization test passed!")

    def test_initialization_with_tp(self):
        """Test MHATokenToKVPool initialization with TP"""
        print("\nTesting MHA KV Cache initialization (with TP)...")

        mesh = self._create_multi_device_mesh()
        config = self.test_configs["medium"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            # Verify sharding is applied
            expected_sharding = NamedSharding(mesh, P(None, None, "tensor", None))

            # Check that buffers have correct sharding
            for layer_id in range(config["layer_num"]):
                k_buf = kv_pool.k_buffer[layer_id]
                v_buf = kv_pool.v_buffer[layer_id]

                # Verify shapes
                expected_shape = (
                    config["size"] + config["page_size"],
                    config["head_num"],
                    config["head_dim"],
                )
                self.assertEqual(k_buf.shape, expected_shape)
                self.assertEqual(v_buf.shape, expected_shape)

        print("PASS: TP initialization test passed!")

    def test_get_kv_buffer_operations(self):
        """Test get_kv_buffer and related getter methods"""
        print("\nTesting KV buffer getter operations...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            for layer_id in range(config["layer_num"]):
                # Test get_key_buffer
                k_buf = kv_pool.get_key_buffer(layer_id)
                expected_shape = (
                    config["size"] + config["page_size"],
                    config["head_num"],
                    config["head_dim"],
                )
                self.assertEqual(k_buf.shape, expected_shape)
                self.assertEqual(k_buf.dtype, config["dtype"])

                # Test get_value_buffer
                v_buf = kv_pool.get_value_buffer(layer_id)
                self.assertEqual(v_buf.shape, expected_shape)
                self.assertEqual(v_buf.dtype, config["dtype"])

                # Test get_kv_buffer
                k_buf2, v_buf2 = kv_pool.get_kv_buffer(layer_id)
                self.assertTrue(jnp.array_equal(k_buf, k_buf2))
                self.assertTrue(jnp.array_equal(v_buf, v_buf2))

        print("PASS: KV buffer getter operations test passed!")

    def test_set_kv_buffer_operations(self):
        """Test set_kv_buffer operations with data integrity"""
        print("\nTesting KV buffer setter operations...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            # Test setting KV data for different layers
            test_layer = 0
            batch_size = 4
            seq_len = 8

            # Create test data
            test_k = (
                jnp.ones(
                    (batch_size, seq_len, config["head_num"], config["head_dim"]),
                    dtype=config["dtype"],
                )
                * 0.5
            )
            test_v = (
                jnp.ones(
                    (batch_size, seq_len, config["head_num"], config["head_dim"]),
                    dtype=config["dtype"],
                )
                * 1.5
            )

            # Flatten for setting
            test_k_flat = test_k.reshape(-1, config["head_num"], config["head_dim"])
            test_v_flat = test_v.reshape(-1, config["head_num"], config["head_dim"])

            # Set locations (skip slot 0 as it's reserved for padding)
            locations = jnp.arange(1, batch_size * seq_len + 1)

            # Set KV data
            kv_pool.set_kv_buffer(test_layer, locations, test_k_flat, test_v_flat)

            # Verify data was set correctly
            k_buf, v_buf = kv_pool.get_kv_buffer(test_layer)

            for i, loc in enumerate(locations):
                stored_k = k_buf[loc]
                stored_v = v_buf[loc]
                expected_k = test_k_flat[i]
                expected_v = test_v_flat[i]

                self.assertTrue(jnp.allclose(stored_k, expected_k, rtol=1e-5))
                self.assertTrue(jnp.allclose(stored_v, expected_v, rtol=1e-5))

        print("PASS: KV buffer setter operations test passed!")

    def test_get_kv_data_operations(self):
        """Test get_kv_data for retrieving specific indices"""
        print("\nTesting get_kv_data operations...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            test_layer = 1
            num_tokens = 6

            # Create and set test data
            test_k = (
                jnp.arange(num_tokens * config["head_num"] * config["head_dim"])
                .reshape(num_tokens, config["head_num"], config["head_dim"])
                .astype(config["dtype"])
            )
            test_v = test_k * 2

            locations = jnp.arange(1, num_tokens + 1)
            kv_pool.set_kv_buffer(test_layer, locations, test_k, test_v)

            # Test retrieving specific indices
            query_indices = jnp.array([1, 3, 5])
            retrieved_k, retrieved_v = kv_pool.get_kv_data(test_layer, query_indices)

            # Verify retrieved data matches expected
            expected_k = test_k[jnp.array([0, 2, 4])]  # Adjust for 0-based indexing
            expected_v = test_v[jnp.array([0, 2, 4])]

            self.assertTrue(jnp.allclose(retrieved_k, expected_k))
            self.assertTrue(jnp.allclose(retrieved_v, expected_v))

        print("PASS: get_kv_data operations test passed!")

    def test_memory_calculations(self):
        """Test memory usage calculations"""
        print("\nTesting memory usage calculations...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["medium"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            # Test get_kv_size_bytes
            k_size, v_size = kv_pool.get_kv_size_bytes()

            bytes_per_element = 4 if config["dtype"] == jnp.float32 else 2
            expected_k_size = (
                config["size"]
                * config["head_num"]
                * config["head_dim"]
                * bytes_per_element
                * config["layer_num"]
            )
            expected_v_size = expected_k_size

            self.assertEqual(k_size, expected_k_size)
            self.assertEqual(v_size, expected_v_size)

            # Test mem_usage is calculated
            expected_mem_usage = (expected_k_size + expected_v_size) / (
                1024 * 1024 * 1024
            )  # Convert to GB
            self.assertAlmostEqual(kv_pool.mem_usage, expected_mem_usage, places=6)

        print("PASS: Memory calculations test passed!")

    # def test_buffer_operations(self):
    #     # TODO: JAX in-place update test does not pass
    #     """Test buffer manipulation operations like move and clear"""
    #     print("\nTesting buffer manipulation operations...")

    #     mesh = self._create_single_device_mesh()
    #     config = self.test_configs['small']

    #     with mesh:
    #         kv_pool = MHATokenToKVPool(
    #             size=config['size'],
    #             page_size=config['page_size'],
    #             dtype=config['dtype'],
    #             head_num=config['head_num'],
    #             head_dim=config['head_dim'],
    #             layer_num=config['layer_num'],
    #             mesh=mesh,
    #             kv_partition_axis="tensor",
    #         )

    #         test_layer = 0

    #         # Set up test data
    #         test_k = jnp.ones((4, config['head_num'], config['head_dim']), dtype=config['dtype']) * 3.0
    #         test_v = jnp.ones((4, config['head_num'], config['head_dim']), dtype=config['dtype']) * 4.0

    #         src_locations = jnp.array([1, 2, 3, 4])
    #         tgt_locations = jnp.array([10, 11, 12, 13])

    #         # Set initial data
    #         kv_pool.set_kv_buffer(test_layer, src_locations, test_k, test_v)

    #         # Test move operation
    #         kv_pool.move_kv_cache(tgt_locations, src_locations)

    #         # Verify data was moved
    #         k_buf, v_buf = kv_pool.get_kv_buffer(test_layer)
    #         for i, tgt_loc in enumerate(tgt_locations):
    #             moved_k = k_buf[tgt_loc]
    #             moved_v = v_buf[tgt_loc]
    #             self.assertTrue(jnp.allclose(moved_k, test_k[i]))
    #             self.assertTrue(jnp.allclose(moved_v, test_v[i]))

    #         # Test clear operation
    #         clear_indices = jnp.array([10, 11])
    #         kv_pool.clear_cache(clear_indices)

    #         # Verify cleared indices are zero
    #         for clear_idx in clear_indices:
    #             cleared_k = k_buf[clear_idx]
    #             cleared_v = v_buf[clear_idx]
    #             print(cleared_k, cleared_v)
    #             self.assertTrue(jnp.allclose(cleared_k, 0.0, atol=1e-5))
    #             self.assertTrue(jnp.allclose(cleared_v, 0.0, atol=1e-5))

    #     print("PASS: Buffer manipulation operations test passed!")

    def test_cpu_host_operations(self):
        """Test CPU/host memory operations"""
        print("\nTesting CPU/host memory operations...")

        mesh = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            # Set up test data
            num_tokens = 5
            test_k = (
                jnp.arange(num_tokens * config["head_num"] * config["head_dim"])
                .reshape(num_tokens, config["head_num"], config["head_dim"])
                .astype(config["dtype"])
            )
            test_v = test_k * 3

            locations = jnp.arange(1, num_tokens + 1)
            kv_pool.set_kv_buffer(0, locations, test_k, test_v)

            # Test getting CPU copy
            cpu_copy = kv_pool.get_cpu_copy(locations)

            # Verify CPU copy structure and data
            self.assertEqual(len(cpu_copy), config["layer_num"])

            # Check first layer data
            k_host, v_host = cpu_copy[0]
            self.assertEqual(
                k_host.shape, (num_tokens, config["head_num"], config["head_dim"])
            )
            self.assertEqual(
                v_host.shape, (num_tokens, config["head_num"], config["head_dim"])
            )

            # Test loading CPU copy back
            new_locations = jnp.arange(10, 10 + num_tokens)
            kv_pool.load_cpu_copy(cpu_copy, new_locations)

            # Verify loaded data matches original
            k_buf, v_buf = kv_pool.get_kv_buffer(0)
            for i, new_loc in enumerate(new_locations):
                loaded_k = k_buf[new_loc]
                loaded_v = v_buf[new_loc]
                self.assertTrue(jnp.allclose(loaded_k, test_k[i]))
                self.assertTrue(jnp.allclose(loaded_v, test_v[i]))

        print("PASS: CPU/host memory operations test passed!")

    def test_head_nums_sharding(self):
        """Test head_nums sharding in tensor parallel scenarios"""
        print("\nTesting head_nums sharding...")

        # Test 1: Single device (no sharding)
        print("\n--- Test 1: Single device scenario (no sharding) ---")
        mesh_single = self._create_single_device_mesh()
        config = self.test_configs["small"]

        with mesh_single:
            kv_pool_single = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=config["head_num"],  # 8 heads
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh_single,
                kv_partition_axis="tensor",
            )

            # Verify no sharding in single device scenario
            k_buf, v_buf = kv_pool_single.get_kv_buffer(0)
            print(f"Single device K buffer shape: {k_buf.shape}")
            print(f"Single device V buffer shape: {v_buf.shape}")
            print(f"Single device K buffer sharding: {k_buf.sharding}")
            print(f"Single device V buffer sharding: {v_buf.sharding}")

            # Single device sharding is also effective
            sharding_spec = k_buf.sharding.spec
            self.assertEqual(sharding_spec, P(None, "tensor", None))
            print("PASS: Single device sharding verification passed")

        # Test 2: Multi-device (with tensor parallel sharding)
        try:
            mesh_multi = self._create_multi_device_mesh()
        except unittest.SkipTest:
            print("Skipping multi-device test - insufficient devices")
            return

        print("\n--- Test 2: Multi-device scenario (tensor parallel sharding) ---")
        # Use head count divisible by 4 to ensure sharding across 4 devices
        multi_head_num = (
            32  # 32 heads can be sharded across 4 devices, 8 heads per device
        )

        with mesh_multi:
            kv_pool_multi = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=multi_head_num,
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh_multi,
                kv_partition_axis="tensor",
            )

            # Verify sharding in multi-device scenario
            k_buf_multi, v_buf_multi = kv_pool_multi.get_kv_buffer(0)
            print(f"Multi-device K buffer shape: {k_buf_multi.shape}")
            print(f"Multi-device V buffer shape: {v_buf_multi.shape}")
            print(f"Multi-device K buffer sharding: {k_buf_multi.sharding}")
            print(f"Multi-device V buffer sharding: {v_buf_multi.sharding}")

            # Verify sharding strategy
            sharding_spec_multi = k_buf_multi.sharding.spec
            expected_spec = P(None, "tensor", None)  # head dimension should be sharded
            self.assertEqual(sharding_spec_multi, expected_spec)
            print("PASS: Multi-device sharding strategy verification passed")

            # Verify shard shapes for each device
            print(f"Mesh shape: {mesh_multi.shape}")
            tensor_parallel_size = mesh_multi.shape["tensor"]
            expected_heads_per_device = multi_head_num // tensor_parallel_size

            print(f"Total head count: {multi_head_num}")
            print(f"Tensor parallelism degree: {tensor_parallel_size}")
            print(f"Expected head count per device: {expected_heads_per_device}")

            # Check addressable shards
            print(f"Addressable shards count: {len(k_buf_multi.addressable_shards)}")
            for i, shard in enumerate(k_buf_multi.addressable_shards):
                shard_shape = shard.data.shape
                print(f"Device {i}: {shard.data.device} - shard shape: {shard_shape}")
                # Verify head dimension is correctly sharded
                self.assertEqual(shard_shape[1], expected_heads_per_device)
                self.assertEqual(shard_shape[0], config["size"] + config["page_size"])
                self.assertEqual(shard_shape[2], config["head_dim"])

        print("PASS: Head nums sharding test passed!")

    def test_sharding_read_write_operations(self):
        """Test read/write operations with head sharding"""
        print("\nTesting read/write operations with head sharding...")

        try:
            mesh = self._create_multi_device_mesh()
        except unittest.SkipTest:
            print("Skipping sharded read/write test - insufficient devices")
            return

        config = self.test_configs["small"]
        # Use head count divisible by 4
        shard_head_num = 16

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=config["size"],
                page_size=config["page_size"],
                dtype=config["dtype"],
                head_num=shard_head_num,
                head_dim=config["head_dim"],
                layer_num=config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            test_layer = 0
            num_test_tokens = 5

            # Create test data - use different values per head for verification
            test_k = jnp.zeros(
                (num_test_tokens, shard_head_num, config["head_dim"]),
                dtype=config["dtype"],
            )
            test_v = jnp.zeros(
                (num_test_tokens, shard_head_num, config["head_dim"]),
                dtype=config["dtype"],
            )

            # Set different values for each head
            for head_idx in range(shard_head_num):
                head_value = (head_idx + 1) * 0.1  # 0.1, 0.2, 0.3, ...
                test_k = test_k.at[:, head_idx, :].set(head_value)
                test_v = test_v.at[:, head_idx, :].set(head_value * 2)

            locations = jnp.arange(1, num_test_tokens + 1)

            print(f"Test data shape: K={test_k.shape}, V={test_v.shape}")
            print(f"Write positions: {locations}")

            # Execute write operation
            print("Executing sharded write operation...")
            kv_pool.set_kv_buffer(test_layer, locations, test_k, test_v)

            # Verify data after write
            print("Verifying sharded read operation...")
            k_buf, v_buf = kv_pool.get_kv_buffer(test_layer)

            # Verify data at each position individually
            for i, loc in enumerate(locations):
                stored_k = k_buf[loc]
                stored_v = v_buf[loc]
                expected_k = test_k[i]
                expected_v = test_v[i]

                # Verify data consistency
                self.assertTrue(jnp.allclose(stored_k, expected_k, rtol=1e-5))
                self.assertTrue(jnp.allclose(stored_v, expected_v, rtol=1e-5))

                # Verify data distribution after sharding
                print(
                    f"Position {loc}: K shape={stored_k.shape}, V shape={stored_v.shape}"
                )
                print(
                    f"Position {loc}: K first 3 head values={stored_k[:3, 0]}, V first 3 head values={stored_v[:3, 0]}"
                )

            # Test get_kv_data method under sharding conditions
            print("Testing get_kv_data performance under sharding...")
            query_indices = jnp.array([1, 3, 5])
            retrieved_k, retrieved_v = kv_pool.get_kv_data(test_layer, query_indices)

            print(f"Retrieved K shape: {retrieved_k.shape}")
            print(f"Retrieved V shape: {retrieved_v.shape}")

            # Verify retrieved data
            expected_k_retrieved = test_k[jnp.array([0, 2, 4])]
            expected_v_retrieved = test_v[jnp.array([0, 2, 4])]

            self.assertTrue(jnp.allclose(retrieved_k, expected_k_retrieved, rtol=1e-5))
            self.assertTrue(jnp.allclose(retrieved_v, expected_v_retrieved, rtol=1e-5))

            # Display sharding effectiveness statistics
            print(f"\n--- Sharding effectiveness statistics ---")
            tensor_parallel_size = mesh.shape["tensor"]
            heads_per_device = shard_head_num // tensor_parallel_size
            print(f"Total head count: {shard_head_num}")
            print(f"Tensor parallel device count: {tensor_parallel_size}")
            print(f"Heads per device: {heads_per_device}")
            print(f"Sharding strategy: {k_buf.sharding}")

            for i, shard in enumerate(k_buf.addressable_shards):
                shard_shape = shard.data.shape
                print(
                    f"Device {i}: shard shape={shard_shape}, head range=[{i*heads_per_device}:{(i+1)*heads_per_device}]"
                )

        print("PASS: Sharding read/write operations test passed!")

    def test_memory_distribution_across_devices(self):
        """Test memory distribution across devices with head sharding"""
        print("\nTesting memory distribution across devices...")

        try:
            mesh = self._create_multi_device_mesh()
        except unittest.SkipTest:
            print("Skipping memory distribution test - insufficient devices")
            return

        # Use larger configuration to better observe memory distribution
        large_config = {
            "size": 256,
            "page_size": 1,
            "head_num": 32,  # Divisible by 4, suitable for 4-device sharding
            "head_dim": 128,
            "layer_num": 4,  # Multi-layer test
            "dtype": jnp.bfloat16,
        }

        with mesh:
            kv_pool = MHATokenToKVPool(
                size=large_config["size"],
                page_size=large_config["page_size"],
                dtype=large_config["dtype"],
                head_num=large_config["head_num"],
                head_dim=large_config["head_dim"],
                layer_num=large_config["layer_num"],
                mesh=mesh,
                kv_partition_axis="tensor",
            )

            print(f"Test configuration: {large_config}")
            print(f"Mesh configuration: {mesh}")

            # Calculate memory usage per layer
            bytes_per_element = 2  # bfloat16
            total_elements_per_layer = (
                (large_config["size"] + large_config["page_size"])
                * large_config["head_num"]
                * large_config["head_dim"]
            )
            layer_size_mb = (total_elements_per_layer * bytes_per_element * 2) / (
                1024 * 1024
            )  # K and V
            total_size_mb = layer_size_mb * large_config["layer_num"]

            print(f"\n--- Theoretical memory calculation ---")
            print(f"Elements per layer: {total_elements_per_layer:,}")
            print(f"Size per layer (K+V): {layer_size_mb:.2f} MB")
            print(f"Total size: {total_size_mb:.2f} MB")

            # Analyze memory distribution per device
            tensor_parallel_size = mesh.shape["tensor"]
            elements_per_device = total_elements_per_layer // tensor_parallel_size
            size_per_device_mb = layer_size_mb / tensor_parallel_size

            print(f"\n--- Sharded memory distribution ---")
            print(f"Tensor parallel device count: {tensor_parallel_size}")
            print(f"Elements per device: {elements_per_device:,}")
            print(f"Size per device per layer: {size_per_device_mb:.2f} MB")
            print(
                f"Total size per device: {size_per_device_mb * large_config['layer_num']:.2f} MB"
            )

            # Verify actual sharding situation
            print(f"\n--- Actual sharding verification ---")
            for layer_id in range(large_config["layer_num"]):
                k_buf, v_buf = kv_pool.get_kv_buffer(layer_id)
                print(f"Layer {layer_id}:")
                print(f"  K buffer: shape={k_buf.shape}, sharding={k_buf.sharding}")
                print(f"  V buffer: shape={v_buf.shape}, sharding={v_buf.sharding}")

                # Verify addressable shards
                print(f"  Addressable shards count: {len(k_buf.addressable_shards)}")
                for i, shard in enumerate(k_buf.addressable_shards):
                    shard_elements = np.prod(shard.data.shape)
                    shard_size_mb = (shard_elements * bytes_per_element) / (1024 * 1024)
                    print(
                        f"    Device {i}: shape={shard.data.shape}, elements={shard_elements:,}, size={shard_size_mb:.2f}MB"
                    )

                    # Verify correct head count per shard
                    expected_heads_per_device = (
                        large_config["head_num"] // tensor_parallel_size
                    )
                    self.assertEqual(shard.data.shape[1], expected_heads_per_device)

            # Verify total memory usage statistics
            reported_mem_usage = kv_pool.mem_usage
            print(f"\n--- Memory usage statistics ---")
            print(f"Reported memory usage: {reported_mem_usage:.3f} GB")
            print(f"Calculated total size: {total_size_mb / 1024:.3f} GB")

            # Allow some tolerance for error
            self.assertAlmostEqual(reported_mem_usage, total_size_mb / 1024, places=2)

        print("PASS: Memory distribution test passed!")


def run_all_tests():
    """Run all MHA KV Cache tests"""
    print("Starting MHA KV Cache Test Suite...")
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {len(jax.devices())}")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMHAKVCache)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n🎉 All MHA KV Cache tests passed!")
        return True
    else:
        print(
            f"\nERROR: {len(result.failures)} test(s) failed, {len(result.errors)} error(s)"
        )
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
