"""SGL_JAX_AOT_DENSE_EXACT=1 lets the AOT input path mark <= DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT
extend shapes exact-dense, mirroring ForwardBatch.init_new, so CPU AOT gates compile the same
attention arm the live server runs for short prompts."""

import os
import types
import unittest
from unittest import mock

from sgl_jax.srt.model_executor.aot_inputs import apply_aot_dense_exact


def _batch(mode="EXTEND"):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(name=mode), dsa_dense_exact=False
    )


class AotDenseExactKnobTest(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(
            os.environ, {"DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT": "2048"}, clear=False
        ):
            os.environ.pop("SGL_JAX_AOT_DENSE_EXACT", None)
            b = _batch()
            self.assertFalse(apply_aot_dense_exact(b, 1024))
            self.assertFalse(b.dsa_dense_exact)

    def test_fires_for_short_extend_only(self):
        env = {"SGL_JAX_AOT_DENSE_EXACT": "1", "DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT": "2048"}
        with mock.patch.dict(os.environ, env, clear=False):
            b = _batch()
            self.assertTrue(apply_aot_dense_exact(b, 1024))
            self.assertTrue(b.dsa_dense_exact)
            self.assertFalse(apply_aot_dense_exact(_batch(), 4096))
            self.assertFalse(apply_aot_dense_exact(_batch("DECODE"), 64))
            self.assertFalse(apply_aot_dense_exact(_batch(), None))

    def test_needs_live_bound(self):
        env = {"SGL_JAX_AOT_DENSE_EXACT": "1", "DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT": "0"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertFalse(apply_aot_dense_exact(_batch(), 1024))


if __name__ == "__main__":
    unittest.main()
