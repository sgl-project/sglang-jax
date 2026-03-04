# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_split_kv_cache.py -v

import os

# Set up multi-device simulation for tensor parallelism
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)

class TestSplitKVCache(unittest.TestCase):
    def setUp(self):
        self.kv_head_num = 32
        self.layer_num = 2
        self.pool_size = 100
        self.page_size = 1
        self.dtype = jnp.bfloat16
        
        # Split case
        self.head_dim = 128
        self.v_head_dim = 64
        
        # Fused case
        self.fused_head_dim = 128
        self.fused_v_head_dim = 128

    def test_initialization_split(self):
        """Test initialization of split KV cache"""
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            v_head_dim=self.v_head_dim
        )
        
        self.assertTrue(kv_cache.is_split)
        self.assertIsNone(kv_cache.kv_buffer)
        self.assertIsNotNone(kv_cache.k_buffer)
        self.assertIsNotNone(kv_cache.v_buffer)
        self.assertEqual(len(kv_cache.k_buffer), self.layer_num)
        self.assertEqual(len(kv_cache.v_buffer), self.layer_num)
        
        # Check shapes
        # k_buffer: [size+page_size, head_num, head_dim]
        expected_k_shape = (self.pool_size + self.page_size, self.kv_head_num, self.head_dim)
        expected_v_shape = (self.pool_size + self.page_size, self.kv_head_num, self.v_head_dim)
        
        self.assertEqual(kv_cache.k_buffer[0].shape, expected_k_shape)
        self.assertEqual(kv_cache.v_buffer[0].shape, expected_v_shape)

    def test_initialization_fused(self):
        """Test initialization of fused KV cache (backward compatibility)"""
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.fused_head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            v_head_dim=self.fused_v_head_dim
        )
        
        self.assertFalse(kv_cache.is_split)
        self.assertIsNotNone(kv_cache.kv_buffer)
        self.assertIsNone(kv_cache.k_buffer)
        self.assertIsNone(kv_cache.v_buffer)
        
        # Fused shape: [size+page, head_num*2, head_dim]
        expected_shape = (self.pool_size + self.page_size, self.kv_head_num * 2, self.fused_head_dim)
        self.assertEqual(kv_cache.kv_buffer[0].shape, expected_shape)

    @mock.patch("sgl_jax.srt.mem_cache.memory_pool.update_kv_cache_vectorized")
    def test_set_and_get_kv_buffer_split(self, mock_update):
        """Test setting and getting data in split KV cache"""
        def update_side_effect(k, v, loc, k_cache, v_cache, page_size, kv_partition_axis, mesh=None):
            # Simple simulation of update
            safe_loc = jnp.where(loc == -1, 0, loc)
            k_updated = k_cache.at[safe_loc].set(k)
            v_updated = v_cache.at[safe_loc].set(v)
            return k_updated, v_updated

        mock_update.side_effect = update_side_effect

        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            v_head_dim=self.v_head_dim
        )
        
        # Prepare data
        num_tokens = 10
        layer_id = 0
        loc = jnp.arange(num_tokens, dtype=jnp.int32)
        
        k_data = jax.random.normal(jax.random.PRNGKey(0), (num_tokens, self.kv_head_num, self.head_dim)).astype(self.dtype)
        v_data = jax.random.normal(jax.random.PRNGKey(1), (num_tokens, self.kv_head_num, self.v_head_dim)).astype(self.dtype)
        
        # Set buffer
        kv_cache.set_kv_buffer(layer_id, loc, k_data, v_data)
        
        # Verify mock called
        self.assertTrue(mock_update.called)
        
        # Get buffer
        k_ret, v_ret = kv_cache.get_kv_buffer(layer_id)
        
        # Verify separate buffers are updated
        k_slice = k_ret[loc]
        v_slice = v_ret[loc]
        
        np.testing.assert_allclose(k_slice, k_data, atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(v_slice, v_data, atol=1e-2, rtol=1e-2)

    def test_get_fused_kv_buffer_error(self):
        """Test that get_fused_kv_buffer raises error for split cache"""
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            v_head_dim=self.v_head_dim
        )
        
        with self.assertRaises(NotImplementedError):
            kv_cache.get_fused_kv_buffer(0)

    def test_memory_usage_calculation(self):
        """Test memory usage calculation"""
        # Split
        kv_cache_split = MHATokenToKVPool(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            v_head_dim=self.v_head_dim
        )
        k_size, v_size = kv_cache_split.get_kv_size_bytes()
        
        itemsize = jnp.dtype(self.dtype).itemsize
        expected_k_size = (self.pool_size + self.page_size) * self.kv_head_num * self.head_dim * itemsize * self.layer_num
        expected_v_size = (self.pool_size + self.page_size) * self.kv_head_num * self.v_head_dim * itemsize * self.layer_num
        
        self.assertEqual(k_size, expected_k_size)
        self.assertEqual(v_size, expected_v_size)

if __name__ == "__main__":
    unittest.main()
