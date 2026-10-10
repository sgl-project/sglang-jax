import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.util import align_to, get_dtype_packing
from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    VMEM_HEADROOM_BYTES,
    VMEM_SIZE,
    get_num_slices_per_block,
    get_slot_mapping,
)
from sgl_jax.srt.mem_cache.memory_pool import merge_kv
from sgl_jax.srt.mem_cache.memory_pool import (
    update_fused_kv_cache_vectorized as update_fused_kv_cache,
)
from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer
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
    return jax.device_put(cache, P("data", None, "tensor", None, None))


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


class TestKVCacheBlockBudget(unittest.TestCase):
    def test_scratch_leaves_compiler_headroom(self):
        # Local shard shapes: Gemma FULL/SWA, then MiMo FULL/SWA. MiMo's
        # 192-wide K and 128-wide V are padded to a shared head_dim of 256.
        for page_size, heads, head_dim, expected_block in [
            (128, 1, 512, 255),
            (128, 2, 256, 255),
            (256, 1, 256, 255),
            (256, 2, 256, 127),
        ]:
            with self.subTest(page_size=page_size, heads=heads, head_dim=head_dim):
                kv = jax.ShapeDtypeStruct((4096, 1, heads, 2, head_dim), jnp.bfloat16)
                cache = jax.ShapeDtypeStruct((77, page_size, heads, 2, head_dim), jnp.bfloat16)
                block = get_num_slices_per_block(kv, cache, page_size)
                bytes_per_slice = page_size * heads * 2 * head_dim * 2
                scratch_bytes = block * bytes_per_slice
                # Use the upstream margin and the largest tile that fits it.
                self.assertEqual(VMEM_SIZE, 64 * 1024 * 1024)
                self.assertEqual(VMEM_HEADROOM_BYTES, 128 * 1024)
                self.assertEqual(block, expected_block)
                self.assertLessEqual(scratch_bytes, VMEM_SIZE - VMEM_HEADROOM_BYTES)
                self.assertGreater(
                    scratch_bytes + bytes_per_slice, VMEM_SIZE - VMEM_HEADROOM_BYTES
                )
                small = jax.ShapeDtypeStruct((17, 1, heads, 2, head_dim), jnp.bfloat16)
                self.assertEqual(get_num_slices_per_block(small, cache, page_size), 17)

    def test_block_tail_padding_preserves_every_real_slice(self):
        for page_size, heads, head_dim, max_block in [(128, 1, 512, 255), (256, 2, 256, 127)]:
            cache = jax.ShapeDtypeStruct((77, page_size, heads, 2, head_dim), jnp.bfloat16)
            for tokens in (
                max_block - 1,
                max_block,
                max_block + 1,
                2 * max_block - 1,
                2 * max_block,
                2 * max_block + 1,
                512,
                1536,
            ):
                with self.subTest(page_size=page_size, tokens=tokens):
                    kv = jax.ShapeDtypeStruct((tokens, 1, heads, 2, head_dim), jnp.bfloat16)
                    block = get_num_slices_per_block(kv, cache, page_size)
                    self.assertEqual(block, min(tokens, max_block))
                    sources = jnp.arange(tokens, dtype=jnp.int32)
                    destinations = sources[::-1] * 2 + page_size
                    lengths = jnp.where(sources % 11 == 0, 0, 1).at[-1].set(1)
                    mapping = np.asarray(get_slot_mapping(block, destinations, sources, lengths))
                    self.assertEqual(mapping.shape, (3, ((tokens + block - 1) // block) * block))
                    np.testing.assert_array_equal(
                        mapping[:, :tokens], np.stack([destinations, sources, lengths])
                    )
                    np.testing.assert_array_equal(mapping[:, tokens:], 0)

    def test_float32_budget_and_small_page_writes(self):
        for page_size, expected_block in ((1, 32704), (128, 255), (256, 127)):
            with self.subTest(page_size=page_size):
                kv = jax.ShapeDtypeStruct((65536, 1, 2, 1, 256), jnp.float32)
                cache = jax.ShapeDtypeStruct((77, page_size, 2, 1, 256), jnp.float32)
                block = get_num_slices_per_block(kv, cache, page_size)
                self.assertEqual(block, expected_block)
                scratch_bytes = block * page_size * 2 * 256 * 4
                self.assertLessEqual(scratch_bytes, VMEM_SIZE - VMEM_HEADROOM_BYTES)
                self.assertGreater(
                    scratch_bytes + page_size * 2 * 256 * 4, VMEM_SIZE - VMEM_HEADROOM_BYTES
                )


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
        fused_kv = jax.device_put(fused_kv, P("data", None, "tensor", None, None))

        # Create 5D fused cache
        kv_cache = _make_fused_cache(cache_size, self.num_heads, self.head_dim, page_size)

        loc = jax.device_put(loc, P("data"))

        return fused_kv, loc, kv_cache, k, v

    def expected_update_kv_cache(self, k, v, loc, cache_size):
        """Expected result using host arrays, independently of device sharding.

        Returns expected_k, expected_v each [cache_size, num_heads, head_dim].
        """
        k, v, loc = np.asarray(k), np.asarray(v), np.asarray(loc)
        expected_k = np.zeros((cache_size, self.num_heads, self.head_dim), dtype=k.dtype)
        expected_v = np.zeros((cache_size, self.num_heads, self.head_dim), dtype=v.dtype)

        for i in range(loc.shape[0]):
            if loc[i] != -1:
                expected_k[loc[i]] = k[i]
                expected_v[loc[i]] = v[i]

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
        loc = jax.device_put(loc, P("data"))

        original_cache = kv_cache.copy()

        updated_cache = update_fused_kv_cache(fused_kv, loc, kv_cache, page_size=page_size)

        # Cache should remain unchanged since all tokens are padding
        self.assertTrue(jnp.allclose(updated_cache, original_cache))

    def test_large_writes_preserve_full_cache_across_block_tails(self):
        # Exercise the shared ordinary-write and HiCache/FULL-only write paths.
        # CPU uses conftest's scatter shim; TPU executes the real Pallas kernel.
        tensor_size = mesh.shape["tensor"]
        for page_size, local_heads, head_dim, pages, expected_block in [
            (128, 1, 512, 77, 255),
            (128, 2, 256, 62, 255),
            (256, 1, 256, 20, 255),
            (256, 2, 256, 20, 127),
        ]:
            local_cache = jax.ShapeDtypeStruct(
                (pages, page_size, local_heads, 2, head_dim), jnp.bfloat16
            )
            large = jax.ShapeDtypeStruct((4096, 1, local_heads, 2, head_dim), jnp.bfloat16)
            block = get_num_slices_per_block(large, local_cache, page_size)
            self.assertEqual(block, expected_block)
            cases = [
                (n, False)
                for n in (
                    block - 1,
                    block,
                    block + 1,
                    2 * block - 1,
                    2 * block,
                    2 * block + 1,
                    512,
                    1536,
                    2048,
                )
            ]
            cases.append((2 * block + 1, True))
            for tokens, all_padding in cases:
                with self.subTest(
                    page_size=page_size,
                    heads=local_heads,
                    head_dim=head_dim,
                    tokens=tokens,
                    all_padding=all_padding,
                ):
                    tail = (local_heads * tensor_size, 2, head_dim)
                    rng = np.random.default_rng(42)
                    # Exact BF16 integers distinguish tokens, heads, K/V and
                    # channels. A nonzero initial pool detects stray writes.
                    initial = rng.integers(-128, 128, (pages, page_size, *tail)).astype(np.float32)
                    values = rng.integers(-128, 128, (tokens, 1, *tail)).astype(np.float32)
                    locations = np.arange(tokens, dtype=np.int32)[::-1] * 2 + page_size
                    locations[::11] = -1
                    locations[block - 1 : block + 1] = -1
                    # Keep the last real slice active even for a one-slice tail.
                    locations[-1] = page_size
                    if all_padding:
                        locations[:] = -1
                    expected = initial.copy().reshape((-1, *tail))
                    valid = locations >= 0
                    expected[locations[valid]] = values[valid, 0]
                    spec = NamedSharding(mesh, P("data", None, "tensor", None, None))
                    cache = jax.device_put(jnp.asarray(initial, dtype=jnp.bfloat16), spec)
                    kv = jax.device_put(jnp.asarray(values, dtype=jnp.bfloat16), spec)
                    loc = jax.device_put(locations, NamedSharding(mesh, P("data")))
                    if tokens == 512:
                        # Exact failing Gemma FULL H2D batch size; this wrapper
                        # also exercises donated input/output buffer aliasing.
                        result = write_kv_layer(kv, loc, cache, page_size, "tensor", "data", mesh)
                    else:
                        result = update_fused_kv_cache(kv, loc, cache, page_size=page_size)
                    actual = np.asarray(jax.block_until_ready(result), dtype=np.float32)
                    np.testing.assert_array_equal(actual.reshape(expected.shape), expected)

    def test_float32_page_one_full_cache(self):
        heads = mesh.shape["tensor"]
        k = jnp.arange(17 * heads * 128, dtype=jnp.float32).reshape(17, heads, 128)
        fused = merge_kv(k, -k)
        spec = NamedSharding(mesh, P("data", None, "tensor", None, None))
        fused = jax.device_put(fused, spec)
        cache = jax.device_put(jnp.full((64, 1, 2 * heads, 1, 128), -99, jnp.float32), spec)
        locations = np.arange(17, dtype=np.int32)[::-1] * 2 + 1
        locations[::3] = -1
        expected = np.asarray(cache).copy()
        valid = locations >= 0
        expected[locations[valid]] = np.asarray(fused)[valid]
        result = write_kv_layer(
            fused,
            jax.device_put(locations, NamedSharding(mesh, P("data"))),
            cache,
            1,
            "tensor",
            "data",
            mesh,
        )
        np.testing.assert_array_equal(np.asarray(jax.block_until_ready(result)), expected)

    def test_kv_cache_update_multiple_segments_with_padding(self):
        """Test KV cache update with multiple contiguous segments of different lengths and padding."""
        total_tokens = 25

        loc = jnp.full((total_tokens,), -1, dtype=jnp.int32)
        loc = loc.at[0:7].set(jnp.arange(11, 18))
        loc = loc.at[7:11].set(jnp.arange(22, 26))
        loc = loc.at[11:21].set(jnp.arange(30, 40))
        loc = jax.device_put(loc, P("data"))

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
        fused_kv = jax.device_put(fused_kv, P("data", None, "tensor", None, None))

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
