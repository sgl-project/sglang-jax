"""CPU-only tests for gmm_v2's activation-quant dtype selection.

The kernel picks the lhs (activation) quant dtype from the rhs (weight) dtype
and the MXU capabilities of the target chip. These tests pin that table down
without a TPU by building ``TpuInfo`` for each chip generation.
"""

import os
import unittest
from unittest import mock

import jax.numpy as jnp
from jax._src.tpu_info import ChipVersion
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import select_lhs_quant_dtype

F8 = jnp.dtype(jnp.float8_e4m3fn)
I8 = jnp.dtype(jnp.int8)
I4 = jnp.dtype(jnp.int4)


def _info(chip: ChipVersion):
    return pltpu.get_tpu_info_for_chip(chip, 1)


class SelectLhsQuantDtypeTest(unittest.TestCase):
    def setUp(self):
        # Each test sees the default (unset) escape hatch unless it sets it.
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop("SGLANG_JAX_GMM_INT4_A8", None)

    def tearDown(self):
        self._env.stop()

    def test_fp8_rhs_uses_fp8_mxu(self):
        self.assertEqual(select_lhs_quant_dtype(F8, _info(ChipVersion.TPU_7X)), F8)
        self.assertEqual(select_lhs_quant_dtype(F8, _info(ChipVersion.TPU_V6E)), F8)
        # v5e has no fp8 MXU: activations stay unquantized (W8A16).
        self.assertIsNone(select_lhs_quant_dtype(F8, _info(ChipVersion.TPU_V5E)))

    def test_int8_rhs_uses_int8_mxu_only(self):
        self.assertEqual(select_lhs_quant_dtype(I8, _info(ChipVersion.TPU_V6E)), I8)
        self.assertEqual(select_lhs_quant_dtype(I8, _info(ChipVersion.TPU_V5E)), I8)
        # int8 weights are not exactly representable in e4m3: no W8A8 on v7x.
        self.assertIsNone(select_lhs_quant_dtype(I8, _info(ChipVersion.TPU_7X)))

    def test_int4_rhs_prefers_int8_mxu(self):
        self.assertEqual(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_V6E)), I8)
        self.assertEqual(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_V5E)), I8)

    def test_int4_rhs_falls_back_to_fp8_mxu_without_int8(self):
        # W4A8 on v7x: int4 values are exact in e4m3, so the kernel upcasts the
        # weight tile and quantizes activations to e4m3.
        self.assertEqual(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_7X)), F8)
        self.assertEqual(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_7)), F8)

    def test_int4_a8_escape_hatch(self):
        os.environ["SGLANG_JAX_GMM_INT4_A8"] = "0"
        # Escape hatch: keep bf16 activations (W4A16) on fp8-only chips ...
        self.assertIsNone(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_7X)))
        # ... but it never disables a native int8 MXU path.
        self.assertEqual(select_lhs_quant_dtype(I4, _info(ChipVersion.TPU_V6E)), I8)

    def test_no_mxu_support_leaves_lhs_unquantized(self):
        v4 = _info(ChipVersion.TPU_V4)
        self.assertIsNone(select_lhs_quant_dtype(F8, v4))
        self.assertIsNone(select_lhs_quant_dtype(I8, v4))
        self.assertIsNone(select_lhs_quant_dtype(I4, v4))


if __name__ == "__main__":
    unittest.main()
