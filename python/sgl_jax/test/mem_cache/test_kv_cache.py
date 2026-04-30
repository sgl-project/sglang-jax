import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.util import align_to, get_dtype_packing
from sgl_jax.srt.mem_cache.memory_pool import merge_kv
from sgl_jax.srt.mem_cache.memory_pool import (
    update_fused_kv_cache_vectorized as update_fused_kv_cache,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


def _make_fused_cache(cache_size, num_heads, head_dim, page_size, dtype=jnp.bfloat16):
    """Create a 5D fused KV cache buffer filled with zeros."""
    packing = get_dtype_packing(dtype)
    head_dim_aligned = align_to(head_dim, 128)
    num_pages = (cache_size + page_size - 1) // page_size + 1  # +1 sentinel page
    shape = (num_pages, page_size, num_heads * 2 // packing, packing, head_dim_aligned)
    cache = jnp.zeros(shape, dtype=dtype)
    return jax.device_put(cache, P(None, None, "tensor", None, None))


def _extract_kv_from_fused(fused_cache):
    """Extract separate 3D k and v from a 5D fused cache for verification.

    Returns k, v each of shape [total_tokens, num_kv_heads, head_dim].
    """
    num_pages, page_size, heads_x2_per_pack, packing, head_dim = fused_cache.shape
    total_tokens = num_pages * page_size
    flat = jax.lax.reshape(
        fused_cache,
        (total_tokens, heads_x2_per_pack * packing, head_dim),
        out_sharding=P(None, "tensor", None),
    )
    kv_sharding = NamedSharding(mesh, P(None, "tensor", None))
    k = flat.at[:, ::2, :].get(out_sharding=kv_sharding)
    v = flat.at[:, 1::2, :].get(out_sharding=kv_sharding)
    return k, v


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

    def generate_test_data(self, total_tokens: int, page_size: int, add_padding: bool = False):
        """Generate test data for fused KV cache update.

        Returns:
            fused_kv: 5D [tokens, 1, heads*2//packing, packing, head_dim_aligned]
            loc: [total_tokens] int32
            kv_cache: 5D [num_pages, page_size, heads*2//packing, packing, head_dim_aligned]
            k: original 3D k [total_tokens, num_heads, head_dim] for verification
            v: original 3D v [total_tokens, num_heads, head_dim] for verification
        """
        cache_size = total_tokens + 100

        # Generate K and V tensors (3D)
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
            num_padding = total_tokens // 4
            valid_locs = jnp.arange(total_tokens - num_padding, dtype=jnp.int32) + 10
            padding_locs = jnp.full((num_padding,), -1, dtype=jnp.int32)
            all_locs = jnp.concatenate([valid_locs, padding_locs])
            loc = jnp.zeros(total_tokens, dtype=jnp.int32)
            loc = loc.at[::2].set(all_locs[: total_tokens // 2])
            loc = loc.at[1::2].set(
                all_locs[total_tokens // 2 : total_tokens // 2 + (total_tokens - total_tokens // 2)]
            )
        else:
            loc = jnp.arange(total_tokens, dtype=jnp.int32) + 10

        # Merge k/v into 5D fused format
        fused_kv = merge_kv(k, v)
        fused_kv = jax.device_put(fused_kv, P(None, None, "tensor", None, None))

        # Create 5D fused cache
        kv_cache = _make_fused_cache(cache_size, self.num_heads, self.head_dim, page_size)

        loc = jax.device_put(loc, P(None))

        return fused_kv, loc, kv_cache, k, v

    def expected_update_kv_cache(self, k, v, loc, cache_size):
        """Expected result using simple JAX operations on 3D k/v.

        Returns expected_k, expected_v each [cache_size, num_heads, head_dim].
        """
        expected_k = jnp.zeros((cache_size, self.num_heads, self.head_dim), dtype=k.dtype)
        expected_v = jnp.zeros((cache_size, self.num_heads, self.head_dim), dtype=v.dtype)

        for i in range(loc.shape[0]):
            if loc[i] != -1:
                expected_k = expected_k.at[loc[i]].set(k[i])
                expected_v = expected_v.at[loc[i]].set(v[i])

        return expected_k, expected_v

    def _run_and_verify(self, total_tokens, page_size, add_padding):
        """Run fused KV cache update and verify against reference."""
        fused_kv, loc, kv_cache, k, v = self.generate_test_data(
            total_tokens, page_size, add_padding
        )

        updated_cache = update_fused_kv_cache(fused_kv, loc, kv_cache, page_size=page_size)

        # Extract k/v from updated fused cache
        updated_k, updated_v = _extract_kv_from_fused(updated_cache)
        cache_tokens = updated_k.shape[0]

        # Expected result
        expected_k, expected_v = self.expected_update_kv_cache(k, v, loc, cache_tokens)

        # head_dim may be aligned (padded with zeros), so slice to original head_dim for comparison
        self.assertTrue(
            jnp.allclose(updated_k[:, :, : self.head_dim], expected_k),
            f"K mismatch: max diff = {jnp.max(jnp.abs(updated_k[:, :, :self.head_dim] - expected_k))}",
        )
        self.assertTrue(
            jnp.allclose(updated_v[:, :, : self.head_dim], expected_v),
            f"V mismatch: max diff = {jnp.max(jnp.abs(updated_v[:, :, :self.head_dim] - expected_v))}",
        )

        return updated_k, updated_v, k, v, loc

    def test_kv_cache_update_page_size_1(self):
        """Test KV cache update with page_size=1."""
        self._run_and_verify(16, page_size=1, add_padding=False)

    def test_kv_cache_update_page_size_1_with_padding(self):
        """Test KV cache update with page_size=1 and padding tokens."""
        self._run_and_verify(12, page_size=1, add_padding=True)

    def test_kv_cache_update_page_size_4(self):
        """Test KV cache update with page_size=4."""
        self._run_and_verify(16, page_size=4, add_padding=False)

    def test_kv_cache_update_page_size_4_with_padding(self):
        """Test KV cache update with page_size=4 and padding tokens."""
        self._run_and_verify(12, page_size=4, add_padding=True)

    def test_kv_cache_update_page_size_8_contiguous(self):
        """Test KV cache update with page_size=8 and contiguous locations."""
        self._run_and_verify(16, page_size=8, add_padding=False)

    def test_all_padding_tokens(self):
        """Test case where all tokens are padding tokens."""
        total_tokens = 4
        page_size = 8
        fused_kv, _, kv_cache, k, v = self.generate_test_data(
            total_tokens, page_size, add_padding=False
        )

        # Make all tokens padding
        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)

        original_cache = kv_cache.copy()

        updated_cache = update_fused_kv_cache(fused_kv, loc, kv_cache, page_size=page_size)

        # Cache should remain unchanged since all tokens are padding
        self.assertTrue(jnp.allclose(updated_cache, original_cache))

    def test_kv_cache_update_multiple_segments_with_padding(self):
        """Test KV cache update with multiple contiguous segments of different lengths and padding."""
        total_tokens = 25

        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)
        loc = loc.at[0:7].set(jnp.arange(11, 18))
        loc = loc.at[7:11].set(jnp.arange(22, 26))
        loc = loc.at[11:21].set(jnp.arange(30, 40))

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
        fused_kv = merge_kv(k, v)
        fused_kv = jax.device_put(fused_kv, P(None, None, "tensor", None, None))

        for page_size in [1, 2, 4, 8]:
            with self.subTest(page_size=page_size):
                kv_cache = _make_fused_cache(cache_size, self.num_heads, self.head_dim, page_size)

                updated_cache = update_fused_kv_cache(fused_kv, loc, kv_cache, page_size=page_size)

                updated_k, updated_v = _extract_kv_from_fused(updated_cache)
                cache_tokens = updated_k.shape[0]

                expected_k, expected_v = self.expected_update_kv_cache(k, v, loc, cache_tokens)

                self.assertTrue(
                    jnp.allclose(updated_k[:, :, : self.head_dim], expected_k),
                    f"K mismatch at page_size={page_size}",
                )
                self.assertTrue(
                    jnp.allclose(updated_v[:, :, : self.head_dim], expected_v),
                    f"V mismatch at page_size={page_size}",
                )

                # Verify specific segments
                for i in range(7):
                    self.assertTrue(
                        jnp.allclose(updated_k[11 + i, :, : self.head_dim], k[i], rtol=1e-5)
                    )
                for i in range(4):
                    self.assertTrue(
                        jnp.allclose(updated_k[22 + i, :, : self.head_dim], k[7 + i], rtol=1e-5)
                    )
                for i in range(10):
                    self.assertTrue(
                        jnp.allclose(updated_k[30 + i, :, : self.head_dim], k[11 + i], rtol=1e-5)
                    )

                print(f"  ✓ page_size={page_size} passed")


if __name__ == "__main__":
    unittest.main()
