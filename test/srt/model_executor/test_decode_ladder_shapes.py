"""Decode KV-ladder precompile shapes (registered default ladder, see environ.py).

Every bs bucket yields its max-capacity shape plus one shape per ladder rung below that
capacity, for each decode page bucket; the page table traced for a rung is bounded by the
rung's own page count, so a 64-page bucket at a 2048-token rung is compiled with 16 pages
(the #1696 review case: page_size 128, bs 1, rung 2048, bucket 64)."""

import unittest
from unittest import mock

from sgl_jax.srt.managers.schedule_batch import decode_kv_ladder_steps
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _manager(page_size, bs_buckets, cache_loc_buckets, page_buckets):
    m = CompilationManager.__new__(CompilationManager)
    m.page_size = page_size
    m.bs_buckets = list(bs_buckets)
    m.cache_loc_buckets = list(cache_loc_buckets)
    m.decode_page_buckets = page_buckets
    m.token_buckets = [page_size]
    m.max_padded_batch_size = max(bs_buckets)
    return m


def _ladder(value):
    decode_kv_ladder_steps.cache_clear()  # the parsed ladder is cached per process
    return mock.patch.dict("os.environ", {"SGLANG_JAX_DECODE_KV_LADDER": value})


class DecodeLadderShapesTest(unittest.TestCase):
    def test_rung_shapes_and_page_bound(self):
        with _ladder("2048"):
            m = _manager(128, [1], [135168], [64])
            shapes = list(m.iter_model_shapes(ForwardMode.DECODE))
        self.assertEqual(shapes, [(1, 1, 135168, 64), (1, 1, 2048, 64)])
        bs, _, cache_loc, pages = shapes[1]
        # the rung's page table is bounded by the rung, not by the 64-page bucket
        self.assertEqual(m.decode_page_count(bs, cache_loc, pages), 16)
        self.assertEqual(m.decode_page_count(1, 135168, 64), 64)

    def test_rungs_below_capacity_only_and_page_aligned(self):
        with _ladder("4096,16384,65536,200000"):
            m = _manager(128, [8], [8 * 135168], [16, 64])
            shapes = list(m.iter_model_shapes(ForwardMode.DECODE))
        # 200000 -> page-aligned 200064, 8 * 200064 >= 8 * 135168 so it is dropped
        self.assertEqual(
            shapes,
            [
                (8, 8, 8 * 135168, 16),
                (8, 8, 8 * 4096, 16),
                (8, 8, 8 * 16384, 16),
                (8, 8, 8 * 65536, 16),
                (8, 8, 8 * 135168, 64),
                (8, 8, 8 * 4096, 64),
                (8, 8, 8 * 16384, 64),
                (8, 8, 8 * 65536, 64),
            ],
        )

    def test_empty_ladder_keeps_single_bucket(self):
        with _ladder(""):
            m = _manager(128, [1, 8], [135168, 8 * 135168], None)
            shapes = list(m.iter_model_shapes(ForwardMode.DECODE))
        self.assertEqual(shapes, [(1, 1, 135168, None), (8, 8, 8 * 135168, None)])


if __name__ == "__main__":
    unittest.main()
