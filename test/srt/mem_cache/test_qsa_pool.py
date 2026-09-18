"""CPU tests for QSA's compressed indexer-key cache.

One page table addresses both the token cache and the compressed cache, which
holds only as long as the compressed cache is allocated at a page size of
``page_size // compress_ratio``. That is what is checked here: the two
construction gates that make the geometry legal, and the shapes that come out
of it. The page walk itself is checked in ``test/srt/kernels/qsa/test_paging.py``.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool
from sgl_jax.test.test_utils import CustomTestCase

PAGE_SIZE = 128
RATIO = 4
IDX_DIM = 128


def _mesh():
    import numpy as np

    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))


def _pool(**kw):
    args = dict(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=jnp.bfloat16,
        head_num=1,
        head_dim=128,
        layer_num=2,
        mesh=_mesh(),
        indexer_key_dim=IDX_DIM,
        num_indexer_layers=2,
        compress_ratio=RATIO,
        max_reqs=8,
    )
    args.update(kw)
    return QSATokenToKVPool(**args)


class TestQSAPoolBuffers(CustomTestCase):
    def test_geometry_gates(self):
        """Both conditions are refused at construction: a ratio that does not divide
        page_size, and a compressed page that is not a whole number of sublanes."""
        with self.assertRaisesRegex(ValueError, "must divide page_size"):
            _pool(compress_ratio=5)
        # page_size // ratio == 1 is odd, so get_kv_cache_shape would round the
        # compressed page up to 2 and the two caches would stop sharing a page.
        with self.assertRaisesRegex(ValueError, "multiple of the dtype packing"):
            _pool(page_size=4, size=4 * 8, compress_ratio=4)

    def test_buffer_shapes(self):
        """The compressed page holds page_size // ratio entries -- the divisor the
        shared page walk assumes -- and the inherited GQA cache is untouched."""
        pool = _pool()
        pages = (pool.size + PAGE_SIZE * pool.dp_size) // PAGE_SIZE
        self.assertEqual(
            pool.get_compressed_key_buffer(0).shape,
            (pages, (PAGE_SIZE // RATIO) // 2, 2, IDX_DIM),
        )
        self.assertEqual(pool.get_open_group_buffer(0).shape, (8, RATIO, IDX_DIM))
        self.assertEqual(len(pool.compressed_key_buffer), 2)
        self.assertEqual(len(pool.open_group_buffer), 2)
        # The main GQA cache is untouched by the subclass.
        self.assertEqual(pool.get_fused_kv_buffer(0).shape[0], pages)

    def test_no_indexer_allocates_nothing(self):
        """Without indexer parameters the subclass costs nothing over the plain
        GQA pool."""
        pool = _pool(indexer_key_dim=0, num_indexer_layers=0)
        self.assertEqual(pool.compressed_key_buffer, [])
        self.assertEqual(pool.open_group_buffer, [])
        self.assertEqual(pool.get_indexer_size_bytes(), 0)

    def test_indexer_bytes_match_the_buffers(self):
        """The reported indexer size is the buffers actually allocated -- the
        number the memory budget is computed from."""
        pool = _pool()
        counted = sum(
            b.size * b.dtype.itemsize for b in pool.compressed_key_buffer + pool.open_group_buffer
        )
        self.assertEqual(pool.get_indexer_size_bytes(), counted)

    def test_pytree_round_trip_preserves_the_indexer_state(self):
        """Flatten/unflatten keeps both new buffers and the geometry fields, so the
        pool survives crossing a jit boundary."""
        pool = _pool()
        pool.compressed_key_buffer[0] = pool.compressed_key_buffer[0].at[0, 0, 0, 0].set(7)
        pool.open_group_buffer[1] = pool.open_group_buffer[1].at[3, 2, 5].set(9)

        children, aux = pool.tree_flatten()
        back = QSATokenToKVPool.tree_unflatten(aux, children)

        self.assertEqual(back.compress_ratio, RATIO)
        self.assertEqual(back.max_reqs, 8)
        self.assertEqual(back.indexer_key_dim, IDX_DIM)
        self.assertEqual(back.num_indexer_layers, 2)
        self.assertEqual(back.head_num, pool.head_num)
        self.assertEqual(float(back.get_compressed_key_buffer(0)[0, 0, 0, 0]), 7.0)
        self.assertEqual(float(back.get_open_group_buffer(1)[3, 2, 5]), 9.0)

    def test_replace_buffer_accepts_the_triple(self):
        """Write-back takes either the plain KV list or the (kv, compressed, ring)
        triple, and the list form leaves the indexer state alone."""
        pool = _pool()
        new_compressed = [jnp.full_like(b, 2) for b in pool.compressed_key_buffer]
        new_ring = [jnp.full_like(b, 3) for b in pool.open_group_buffer]
        pool.replace_buffer((list(pool.kv_buffer), new_compressed, new_ring))
        self.assertEqual(float(pool.get_compressed_key_buffer(1)[0, 0, 0, 0]), 2.0)
        self.assertEqual(float(pool.get_open_group_buffer(0)[0, 0, 0]), 3.0)

        # The plain list form still works, and leaves the indexer state alone.
        pool.replace_buffer(list(pool.kv_buffer))
        self.assertEqual(float(pool.get_compressed_key_buffer(1)[0, 0, 0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
