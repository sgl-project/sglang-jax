"""Speculative extend batches take the smallest cache_loc padding that holds the
packed layout instead of the largest one (max_running_requests x context)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

from sgl_jax.srt.managers.schedule_batch import (
    fit_cache_loc_padding,
    spec_cache_loc_needs,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

CTX = 135168
PADDINGS = [CTX * b for b in (1, 2, 4, 8, 16, 32, 64)]


class FitCacheLocPaddingTest(unittest.TestCase):
    def test_smallest_padding_that_holds_the_packed_length(self):
        # cc64 1k/1k: bs padded 64 x longest 2176 aligned tokens = 139264 > 1 x CTX
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [64 * 2176], 1), 2 * CTX)
        # 64 x 2048 fits the first bucket exactly
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [64 * 2048], 1), CTX)
        # bs1 verify: bs padded 64 x 1280
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [64 * 1280], 1), CTX)

    def test_need_covers_the_fixed_stride_view(self):
        # one long request among short ones: the packed sum fits 1 x CTX but the
        # fixed-stride view (bs_padded x longest) needs the next bucket.
        bs_padded, longest, packed = 64, 2176, 20 * 2176
        need = max(packed, bs_padded * longest)
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [need], 1), 2 * CTX)
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [packed], 1), CTX)

    def test_dp_segments_use_the_largest_rank(self):
        # two ranks, the busier one decides; padding must split evenly across dp
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [1000, CTX + 1], 2), 4 * CTX)
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [CTX, CTX], 2), 2 * CTX)

    def test_falls_back_to_the_largest_padding(self):
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [PADDINGS[-1] + 1], 1), PADDINGS[-1])
        self.assertEqual(fit_cache_loc_padding(PADDINGS, [], 1), CTX)

    def test_spec_cache_loc_needs_per_rank(self):
        # rank 0: 3 requests, padded bs 4 -> max(sum aligned, 4 x longest)
        needs = spec_cache_loc_needs([[1000, 2050, 130], None, []], 4, 128)
        self.assertEqual(needs, [max(1024 + 2176 + 256, 4 * 2176), 0, 0])

    def test_spec_extend_predicate(self):
        self.assertTrue(ForwardMode.TARGET_VERIFY.is_spec_extend())
        self.assertTrue(ForwardMode.DRAFT_EXTEND.is_spec_extend())
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.MIXED):
            self.assertFalse(mode.is_spec_extend())


if __name__ == "__main__":
    unittest.main()
