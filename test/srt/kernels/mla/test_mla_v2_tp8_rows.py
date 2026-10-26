"""CPU: GLM-5.2 tp8 (8 heads, page 128) MLA v2 rows resolve on TPU v7 and neighbours still fall back."""

import unittest

from sgl_jax.srt.kernels.mla.v2 import tuned_block_sizes as tb


class TestMlaV2Tp8Rows(unittest.TestCase):
    def test_rows_present(self):
        t = tb.TUNED_BLOCK_SIZES_MLA["TPU v7"]
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, 8)], (16, 1, 2))
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, 64)], (16, 1, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 8, 512, 64, 128, 8)], (16, 8))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 8, 512, 64, 128, 64)], (16, 64))

    def test_dp_attend_64_head_mixed_rows(self):
        t = tb.TUNED_BLOCK_SIZES_MLA["TPU v7"]
        for mnt, val in ((1, (16, 1)), (2, (16, 1)), (4, (16, 4)), (8, (16, 8))):
            self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 64, 512, 64, 128, mnt)], val)
            self.assertEqual(
                t[("decode", "bfloat16", "bfloat16", 64, 512, 64, 128, mnt)], (16, 1, 2)
            )

    def test_v9_mtp_verify_rows(self):
        t = tb.TUNED_BLOCK_SIZES_MLA["TPU v7"]
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, 4)], (8, 1, 2))
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, 16)], (8, 1, 4))
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, 32)], (8, 1, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 8, 512, 64, 128, 4)], (8, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 8, 512, 64, 128, 16)], (8, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 8, 512, 64, 128, 32)], (8, 4))
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 64, 512, 64, 128, 16)], (8, 1, 4))
        self.assertEqual(t[("decode", "bfloat16", "bfloat16", 64, 512, 64, 128, 32)], (8, 1, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 64, 512, 64, 128, 16)], (8, 4))
        self.assertEqual(t[("mixed", "bfloat16", "bfloat16", 64, 512, 64, 128, 32)], (8, 4))

    def test_unmeasured_buckets_not_borrowed(self):
        t = tb.TUNED_BLOCK_SIZES_MLA["TPU v7"]
        for mnt in (1, 2):
            self.assertNotIn(("decode", "bfloat16", "bfloat16", 8, 512, 64, 128, mnt), t)


if __name__ == "__main__":
    unittest.main()
