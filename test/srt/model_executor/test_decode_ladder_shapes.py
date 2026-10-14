"""Decode KV-ladder precompile shapes: the page bucket traced for a ladder shape must not exceed the
ladder's per-request page capacity (review on #1696: bs=1, page_size=128, step=2048, pages=64 would
otherwise warm up seq_len 8191 against a 16-entry page table)."""

import os
import unittest

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


class DecodeLadderShapesTest(unittest.TestCase):
    def test_ladder_pages_capped_to_step_capacity(self):
        os.environ["SGLANG_JAX_DECODE_KV_LADDER"] = "2048"
        try:
            m = _manager(128, [1], [135168], [64])
            shapes = list(m.iter_model_shapes(ForwardMode.DECODE))
        finally:
            del os.environ["SGLANG_JAX_DECODE_KV_LADDER"]
        self.assertEqual(shapes, [(1, 1, 135168, 64), (1, 1, 2048, 16)])
        bs, _, cache_loc, pages = shapes[1]
        # warmup seq_len must fit the ladder's page table for that shape
        self.assertLess(m.decode_dummy_seq_len(pages), cache_loc)
        self.assertEqual(m.decode_page_count(bs, cache_loc, pages), 16)

    def test_ladder_without_page_buckets_and_dedupe(self):
        os.environ["SGLANG_JAX_DECODE_KV_LADDER"] = "4096,16384"
        try:
            m = _manager(128, [8], [8 * 135168], [16, 64])
            shapes = list(m.iter_model_shapes(ForwardMode.DECODE))
        finally:
            del os.environ["SGLANG_JAX_DECODE_KV_LADDER"]
        # two page buckets x (max + 2 ladder steps), ladder pages capped at step // page_size
        # (4096 -> 32 pages: bucket 16 kept, bucket 64 capped to 32 -> no duplicate of bucket 16)
        self.assertEqual(
            shapes,
            [
                (8, 8, 8 * 135168, 16),
                (8, 8, 8 * 4096, 16),
                (8, 8, 8 * 16384, 16),
                (8, 8, 8 * 135168, 64),
                (8, 8, 8 * 4096, 32),
                (8, 8, 8 * 16384, 64),
            ],
        )


if __name__ == "__main__":
    unittest.main()
