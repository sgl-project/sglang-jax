import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.memory_pool import (
    update_kv_cache_vectorized as update_kv_cache,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


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
                all_locs[total_tokens // 2 : total_tokens // 2 + (total_tokens - total_tokens // 2)]
            )
        else:
            # All valid tokens
            loc = jnp.arange(total_tokens, dtype=jnp.int32) + 10  # Start from index 10

        # Set up sharding with data partition for DP support
        kv_sharding = NamedSharding(mesh, P("data", "tensor", None))
        loc_sharding = NamedSharding(mesh, P("data"))

        k = jax.device_put(k, kv_sharding)
        v = jax.device_put(v, kv_sharding)
        k_cache = jax.device_put(k_cache, kv_sharding)
        v_cache = jax.device_put(v_cache, kv_sharding)
        loc = jax.device_put(loc, loc_sharding)

        return k, v, loc, k_cache, v_cache

    def expected_update_kv_cache(self, k, v, loc, k_cache, v_cache):
        """Expected result using simple JAX operations."""
        # Convert sharded arrays to numpy for reference computation
        k_np = np.array(k)
        v_np = np.array(v)
        loc_np = np.array(loc)
        k_cache_np = np.array(k_cache)
        v_cache_np = np.array(v_cache)

        # Update cache only for valid tokens (where loc != -1)
        for i in range(loc_np.shape[0]):
            if loc_np[i] != -1:
                k_cache_np[loc_np[i]] = k_np[i]
                v_cache_np[loc_np[i]] = v_np[i]

        # Convert back to JAX arrays with same sharding as input
        expected_k_cache = jax.device_put(jnp.array(k_cache_np), k_cache.sharding)
        expected_v_cache = jax.device_put(jnp.array(v_cache_np), v_cache.sharding)

        return expected_k_cache, expected_v_cache

    def test_kv_cache_update_page_size_1(self):
        """Test KV cache update with page_size=1."""
        total_tokens = 16
        k, v, loc, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=False)

        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=1)

        # Expected result
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_kv_cache_update_page_size_1_with_padding(self):
        """Test KV cache update with page_size=1 and padding tokens."""
        total_tokens = 12
        k, v, loc, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=True)

        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=1)

        # Expected result (should ignore padding tokens where loc == -1)
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_kv_cache_update_page_size_4(self):
        """Test KV cache update with page_size=4."""
        total_tokens = 16
        k, v, loc, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=False)

        # Test with page_size=4
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=4)

        # Expected result
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )
        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_kv_cache_update_page_size_4_with_padding(self):
        """Test KV cache update with page_size=4 and padding tokens."""
        total_tokens = 12
        k, v, loc, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=True)

        # Test with page_size=4
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=4)

        # Expected result (should ignore padding tokens where loc == -1)
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_kv_cache_update_page_size_8_contiguous(self):
        """Test KV cache update with page_size=8 and contiguous locations."""
        total_tokens = 16
        k, v, loc, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=False)

        # Test with page_size=8
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=8)

        # Expected result
        expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
            k, v, loc, k_cache, v_cache
        )

        self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

    def test_all_padding_tokens(self):
        """Test case where all tokens are padding tokens."""
        total_tokens = 4
        k, v, _, k_cache, v_cache = self.generate_test_data(total_tokens, add_padding=False)

        # Make all tokens padding
        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)
        loc_sharding = NamedSharding(mesh, P("data"))
        loc = jax.device_put(loc, loc_sharding)

        # Store original cache
        original_k_cache = k_cache.copy()
        original_v_cache = v_cache.copy()

        # Test both approaches
        updated_k_cache, updated_v_cache = update_kv_cache(k, v, loc, k_cache, v_cache, page_size=8)

        # Cache should remain unchanged since all tokens are padding
        self.assertTrue(jnp.allclose(updated_k_cache, original_k_cache))
        self.assertTrue(jnp.allclose(updated_v_cache, original_v_cache))

    def test_kv_cache_update_multiple_segments_with_padding(self):
        """Test KV cache update with multiple contiguous segments of different lengths and padding."""
        # Corner case: multiple segments with varying lengths and padding
        # Segments: [11-17] (7 tokens), [22-25] (4 tokens), [30-39] (10 tokens), then padding
        total_tokens = 25

        # Create location array with multiple segments and padding
        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)

        # Segment 1: positions 0-6 -> cache locations 11-17 (7 tokens)
        loc = loc.at[0:7].set(jnp.arange(11, 18))

        # Segment 2: positions 7-10 -> cache locations 22-25 (4 tokens)
        loc = loc.at[7:11].set(jnp.arange(22, 26))

        # Segment 3: positions 11-20 -> cache locations 30-39 (10 tokens)
        loc = loc.at[11:21].set(jnp.arange(30, 40))

        # Positions 21-24 remain as padding (-1)

        # Generate test data
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

        cache_size = total_tokens + 50
        k_cache = jnp.zeros(
            (cache_size, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )
        v_cache = jnp.zeros(
            (cache_size, self.num_heads, self.head_dim),
            dtype=jnp.bfloat16,
        )

        # Set up sharding with data partition for DP support
        kv_sharding = NamedSharding(mesh, P("data", "tensor", None))
        loc_sharding = NamedSharding(mesh, P("data"))

        k = jax.device_put(k, kv_sharding)
        v = jax.device_put(v, kv_sharding)
        k_cache = jax.device_put(k_cache, kv_sharding)
        v_cache = jax.device_put(v_cache, kv_sharding)
        loc = jax.device_put(loc, loc_sharding)

        # Test with different page sizes
        for page_size in [1, 2, 4, 8]:
            with self.subTest(page_size=page_size):
                updated_k_cache, updated_v_cache = update_kv_cache(
                    k, v, loc, k_cache, v_cache, page_size=page_size
                )

                # Expected result
                expected_k_cache, expected_v_cache = self.expected_update_kv_cache(
                    k, v, loc, k_cache, v_cache
                )

                self.assertTrue(jnp.allclose(updated_k_cache, expected_k_cache))
                self.assertTrue(jnp.allclose(updated_v_cache, expected_v_cache))

                # Verify specific segments are updated correctly
                updated_k_cache = jax.sharding.reshard(updated_k_cache, P())
                k = jax.sharding.reshard(k, P())

                # Segment 1: cache locations 11-17
                for i in range(7):
                    cache_pos = 11 + i
                    input_pos = i
                    self.assertTrue(
                        jnp.allclose(updated_k_cache[cache_pos], k[input_pos], rtol=1e-5)
                    )

                # Segment 2: cache locations 22-25
                for i in range(4):
                    cache_pos = 22 + i
                    input_pos = 7 + i
                    self.assertTrue(
                        jnp.allclose(updated_k_cache[cache_pos], k[input_pos], rtol=1e-5)
                    )

                # Segment 3: cache locations 30-39
                for i in range(10):
                    cache_pos = 30 + i
                    input_pos = 11 + i
                    self.assertTrue(
                        jnp.allclose(updated_k_cache[cache_pos], k[input_pos], rtol=1e-5)
                    )


if __name__ == "__main__":
    unittest.main()
