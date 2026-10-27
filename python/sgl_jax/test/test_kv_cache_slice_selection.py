import itertools
import unittest

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    VMEM_HEADROOM_BYTES,
    VMEM_SIZE,
    get_num_slices_per_block,
)


class TestKVCacheSliceSelection(unittest.TestCase):
    def test_token_blocks_bound_unrolled_dmas_and_scratch(self):
        for tokens, heads, dim, dtype in itertools.product(
            (1, 16, 127, 128, 129, 1024), (1, 8, 32, 512), (128, 256), (jnp.bfloat16, jnp.float32)
        ):
            with self.subTest(tokens=tokens, heads=heads, dim=dim, dtype=dtype):
                packing = 4 // jnp.dtype(dtype).itemsize
                new = jax.ShapeDtypeStruct((tokens, 1, heads * 2 // packing, packing, dim), dtype)
                cache = jax.ShapeDtypeStruct((64, 128, heads * 2 // packing, packing, dim), dtype)
                slices = get_num_slices_per_block(new, cache, page_size=1)
                self.assertGreater(slices, 0)
                self.assertLessEqual(slices, min(tokens, 128))
                self.assertLessEqual(
                    slices * heads * 2 * dim * jnp.dtype(dtype).itemsize,
                    VMEM_SIZE - VMEM_HEADROOM_BYTES,
                )

    def test_page_slices_still_reserve_their_full_copy_length(self):
        new = jax.ShapeDtypeStruct((1024, 1, 32, 2, 256), jnp.bfloat16)
        for page_size in (64, 128, 256):
            with self.subTest(page_size=page_size):
                cache = jax.ShapeDtypeStruct((64, page_size, 32, 2, 256), jnp.bfloat16)
                slices = get_num_slices_per_block(new, cache, page_size)
                self.assertGreater(slices, 0)
                self.assertLessEqual(
                    slices * page_size * 64 * 256 * 2,
                    VMEM_SIZE - VMEM_HEADROOM_BYTES,
                )


if __name__ == "__main__":
    unittest.main()
