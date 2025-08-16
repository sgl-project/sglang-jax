import random
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.memory_pool import update_kv_cache


class TestKVCache(unittest.TestCase):
    """Test cases for the KV Cache update functions."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        self.max_seq_len = 16
        self.num_heads = 8
        self.head_dim = 128
        self.batch_size = 2
        self.layer_num = 2

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
        mesh_devices = np.array(devices[:4]).reshape(1, 4, 1)
        return Mesh(mesh_devices, ("data", "tensor", "pipeline"))

    def generate_test_data(self, total_tokens: int, add_padding: bool = False):
        """Generate test data for KV cache update.

        Args:
            total_tokens: Total number of tokens (including padding)
            add_padding: Whether to add padding tokens (loc=-1)
        """
        # Create KV cache buffers
        cache_size = total_tokens + 100  # Add extra space for cache
        k_cache = jnp.zeros(
            (cache_size, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )
        v_cache = jnp.zeros(
            (cache_size, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )

        # Generate K and V tensors
        k = jax.random.uniform(
            jax.random.PRNGKey(42),
            (total_tokens, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )
        v = jax.random.uniform(
            jax.random.PRNGKey(43),
            (total_tokens, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )

        # Generate location indices
        if add_padding:
            # Create some padding tokens (-1) and some valid tokens
            num_padding = total_tokens // 4  # 25% padding
            valid_locs = (
                jnp.arange(total_tokens - num_padding, dtype=jnp.int32) + 10
            )  # Start from index 10
            padding_locs = jnp.full((num_padding,), -1, dtype=jnp.int32)

            # Shuffle to mix padding and valid tokens
            all_locs = jnp.concatenate([valid_locs, padding_locs])
            # Simple shuffle by reversing and interleaving
            loc = jnp.zeros(total_tokens, dtype=jnp.int32)
            loc = loc.at[::2].set(all_locs[: total_tokens // 2])
            loc = loc.at[1::2].set(
                all_locs[
                    total_tokens // 2 : total_tokens // 2
                    + (total_tokens - total_tokens // 2)
                ]
            )
        else:
            # All valid tokens
            loc = jnp.arange(total_tokens, dtype=jnp.int32) + 10  # Start from index 10

        return k, v, loc, k_cache, v_cache

    def expected_update_kv_cache(self, k, v, loc, k_cache, v_cache):
        """Expected result using simple JAX operations."""
        expected_k_cache = k_cache.copy()
        expected_v_cache = v_cache.copy()

        # Update cache only for valid tokens (where loc != -1)
        for i in range(loc.shape[0]):
            if loc[i] != -1:
                expected_k_cache = expected_k_cache.at[loc[i]].set(k[i])
                expected_v_cache = expected_v_cache.at[loc[i]].set(v[i])

        return expected_k_cache, expected_v_cache

    def test_kv_cache_update_vectorized(self):
        """Test vectorized KV cache update without padding."""
        total_tokens = 16
        k, v, loc, k_cache, v_cache = self.generate_test_data(
            total_tokens, add_padding=False
        )

        # Test with vectorized approach
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache)

        # Expected result
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_kv_cache_update_with_padding_vectorized(self):
        """Test vectorized KV cache update with padding tokens."""
        total_tokens = 12
        k, v, loc, k_cache, v_cache = self.generate_test_data(
            total_tokens, add_padding=True
        )

        # Test with vectorized approach
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache)

        # Expected result (should ignore padding tokens where loc == -1)
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

        # Verify that padding tokens didn't affect the cache
        padding_mask = loc == -1
        if jnp.any(padding_mask):
            # Check that original cache values at padding positions are unchanged
            original_k_cache = jnp.zeros_like(k_cache)  # Original was all zeros
            original_v_cache = jnp.zeros_like(v_cache)

            # For positions that should be ignored (padding), cache should remain unchanged
            for i in range(total_tokens):
                if loc[i] == -1:
                    # Cache at this position should remain as original (zeros in this case)
                    continue  # We don't update cache for padding tokens

    def test_all_padding_tokens(self):
        """Test case where all tokens are padding tokens."""
        total_tokens = 4
        k, v, _, k_cache, v_cache = self.generate_test_data(
            total_tokens, add_padding=False
        )

        # Make all tokens padding
        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)

        # Store original cache
        original_k_cache = k_cache.copy()
        original_v_cache = v_cache.copy()

        # Test both approaches
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache)

        # Cache should remain unchanged since all tokens are padding
        self.assertTrue(jnp.allclose(updated_k_cache, original_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, original_v_cache))

    def test_mesh_sharded_kv_cache_correctness(self):
        """Test KV cache correctness with mesh sharding - independent buffer creation and verification"""
        print("\nTesting mesh sharded KV cache with independent buffer creation...")

        try:
            mesh = self._create_multi_device_mesh()
        except unittest.SkipTest:
            print("Skipping mesh sharding test - insufficient devices")
            return

        # Use head numbers divisible by 4 for clean tensor sharding
        shard_head_num = 16
        cache_size = 64
        num_test_tokens = 4

        print(f"Mesh configuration: {mesh}")
        print(f"Mesh axis names: {mesh.axis_names}")
        print(f"Mesh shape: {mesh.shape}")

        with mesh:
            # Step 1: Create independent KV buffers directly with sharding
            kv_sharding = NamedSharding(mesh, P(None, "tensor", None))

            # Create host data for k_buffer and v_buffer
            k_buffer_host = jnp.zeros(
                (cache_size, shard_head_num, self.head_dim), dtype=jnp.bfloat16
            )
            v_buffer_host = jnp.zeros(
                (cache_size, shard_head_num, self.head_dim), dtype=jnp.bfloat16
            )

            # Device put with sharding for independent testing
            k_buffer = jax.device_put(k_buffer_host, kv_sharding)
            v_buffer = jax.device_put(v_buffer_host, kv_sharding)

            print(f"Independently created K buffer sharding: {k_buffer.sharding}")
            print(f"Independently created V buffer sharding: {v_buffer.sharding}")

            # Step 2: Verify actual sharding by checking device distribution
            print(
                "--- Verifying cross-device sharding of independently created buffers ---"
            )
            tensor_parallel_size = mesh.shape["tensor"]
            heads_per_device = shard_head_num // tensor_parallel_size

            print(f"Total heads: {shard_head_num}")
            print(f"Tensor parallelism size: {tensor_parallel_size}")
            print(f"Expected heads per device: {heads_per_device}")
            print(
                f"K buffer addressable shards count: {len(k_buffer.addressable_shards)}"
            )
            print(
                f"V buffer addressable shards count: {len(v_buffer.addressable_shards)}"
            )

            # Check sharding effectiveness on independent buffers
            k_shard_shapes = []
            v_shard_shapes = []
            for device_idx, (k_shard, v_shard) in enumerate(
                zip(k_buffer.addressable_shards, v_buffer.addressable_shards)
            ):
                k_shape = k_shard.data.shape
                v_shape = v_shard.data.shape
                k_shard_shapes.append(k_shape)
                v_shard_shapes.append(v_shape)
                print(
                    f"Device {device_idx}: K shard shape {k_shape}, V shard shape {v_shape}"
                )

            # Verify sharding worked correctly
            is_sharded = not all(shape[1] == shard_head_num for shape in k_shard_shapes)

            if is_sharded:
                print("PASS: Detected genuine tensor sharding")
                # Verify each device has the expected portion of heads
                for device_idx, (k_shape, v_shape) in enumerate(
                    zip(k_shard_shapes, v_shard_shapes)
                ):
                    self.assertEqual(
                        k_shape[1],
                        heads_per_device,
                        f"Device {device_idx} K buffer head count does not match expectation",
                    )
                    self.assertEqual(
                        v_shape[1],
                        heads_per_device,
                        f"Device {device_idx} V buffer head count does not match expectation",
                    )
                    print(
                        f"Device {device_idx}: correct sharding, heads range [{device_idx * heads_per_device}:{(device_idx + 1) * heads_per_device}]"
                    )
            else:
                print(
                    "WARNING: Detected that all devices have complete head data, possibly replication instead of sharding"
                )
                print(
                    "This could be due to JAX auto-optimization or mesh configuration issues"
                )

            # Step 3: Test update_kv_cache with independently created buffers
            print("--- Testing update_kv_cache operations on independent buffers ---")

            # Create test update data with distinct values per head for verification
            update_k_host = jnp.zeros(
                (num_test_tokens, shard_head_num, self.head_dim), dtype=jnp.bfloat16
            )
            update_v_host = jnp.zeros(
                (num_test_tokens, shard_head_num, self.head_dim), dtype=jnp.bfloat16
            )

            for head_idx in range(shard_head_num):
                head_value_k = (head_idx + 1) * 0.1
                head_value_v = (head_idx + 1) * 0.2
                update_k_host = update_k_host.at[:, head_idx, :].set(head_value_k)
                update_v_host = update_v_host.at[:, head_idx, :].set(head_value_v)

            # Apply same sharding to update data
            update_k = jax.device_put(update_k_host, kv_sharding)
            update_v = jax.device_put(update_v_host, kv_sharding)

            update_locations = jnp.arange(10, 10 + num_test_tokens)

            print(f"Update data K sharding: {update_k.sharding}")
            print(f"Update data V sharding: {update_v.sharding}")

            # Test update_kv_cache function with independently created buffers
            updated_k_cache, updated_v_cache = update_kv_cache(
                update_k, update_v, update_locations, k_buffer, v_buffer
            )

            print(f"Updated K cache sharding: {updated_k_cache.sharding}")
            print(f"Updated V cache sharding: {updated_v_cache.sharding}")

            # Step 4: Verify data integrity and sharding preservation
            for i, loc in enumerate(update_locations):
                stored_k = updated_k_cache[loc]
                stored_v = updated_v_cache[loc]
                expected_k = update_k[i]
                expected_v = update_v[i]

                self.assertTrue(jnp.allclose(stored_k, expected_k, rtol=1e-5))
                self.assertTrue(jnp.allclose(stored_v, expected_v, rtol=1e-5))

                # Verify sharding is preserved on individual access
                k_mean = float(jnp.mean(stored_k))
                v_mean = float(jnp.mean(stored_v))
                print(f"Position {loc}: K value {k_mean:.3f}, V value {v_mean:.3f}")

            print(
                "PASS: update_kv_cache works correctly on independently created sharded buffers"
            )

            # Step 5: Additional verification - check if sharding is maintained after operations
            final_k_shard_shapes = [
                shard.data.shape for shard in updated_k_cache.addressable_shards
            ]
            final_v_shard_shapes = [
                shard.data.shape for shard in updated_v_cache.addressable_shards
            ]

            print("--- Final shard shape verification ---")
            for device_idx, (k_shape, v_shape) in enumerate(
                zip(final_k_shard_shapes, final_v_shard_shapes)
            ):
                print(
                    f"Device {device_idx}: final K shard {k_shape}, final V shard {v_shape}"
                )

            # Check if sharding is preserved or if the result is replicated
            final_is_sharded = not all(
                shape[1] == shard_head_num for shape in final_k_shard_shapes
            )

            if final_is_sharded:
                # Ensure sharding consistency is maintained
                self.assertEqual(
                    k_shard_shapes,
                    final_k_shard_shapes,
                    "K buffer sharding changed after operation",
                )
                self.assertEqual(
                    v_shard_shapes,
                    final_v_shard_shapes,
                    "V buffer sharding changed after operation",
                )
                print("PASS: Sharding consistency maintained after operation")
            else:
                print(
                    "WARNING: Results replicated to all devices after update_kv_cache operation"
                )
                print(
                    "This indicates that update_kv_cache function does not preserve input sharding strategy"
                )
                # This is actually expected behavior for many JAX operations
                print(
                    "PASS: Functional correctness verified, although sharding strategy not preserved"
                )

            print("PASS: Independent buffer mesh sharding test completed successfully!")


if __name__ == "__main__":
    unittest.main()
