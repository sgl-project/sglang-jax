"""CPU: the blockwise qmm tile resolver keeps tuned out blocks that divide N exactly.

Regression for the GLM-5.2 v7x rows: ``_nearest_power_of_two_multiple`` used to turn
768 -> 512 (N=768) and 1536 -> 1024 (N=1536), so the swept values never reached the
kernel and the wrapper padded the weight on every call.
"""

import unittest

import jax.numpy as jnp

import sgl_jax.srt.kernels.quantized_matmul.blockwise_utils as bwu
from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import (
    tuned_block_sizes as tb,
)


class TestExactDivisorTiles(unittest.TestCase):
    def setUp(self):
        self._v = bwu._get_current_tpu_version
        self._t = tb.get_tpu_version
        bwu._get_current_tpu_version = lambda: 7
        tb.get_tpu_version = lambda: 7

    def tearDown(self):
        bwu._get_current_tpu_version = self._v
        tb.get_tpu_version = self._t

    def _res(self, m, n_out, n_in):
        v = bwu.get_safe_blockwise_tuned_value(m, n_out, n_in, jnp.bfloat16, jnp.float8_e4m3fn, 128)
        return tuple(v)[:3]

    def test_tp8_prefill_rows_resolve_as_written(self):
        self.assertEqual(self._res(8192, 1536, 6144), (1024, 1536, 1024))
        self.assertEqual(self._res(1024, 1536, 6144), (256, 1536, 2048))
        self.assertEqual(self._res(8192, 6144, 1536), (1024, 2048, 1536))
        self.assertEqual(self._res(1024, 6144, 256), (256, 2048, 256))
        self.assertEqual(self._res(1, 6144, 256), (1, 2048, 256))
        self.assertEqual(self._res(64, 6144, 2048), (64, 2048, 2048))

    def test_existing_tp16_768_row_is_honoured(self):
        self.assertEqual(self._res(8192, 768, 6144), (256, 768, 6144))

    def test_non_divisor_still_snaps(self):
        # 1536 does not divide 2048: snap to a power-of-two multiple of 256 within N.
        v = bwu._nearest_power_of_two_multiple(1536, 256, 2048)
        self.assertIn(v, (1024, 2048))


if __name__ == "__main__":
    unittest.main()
