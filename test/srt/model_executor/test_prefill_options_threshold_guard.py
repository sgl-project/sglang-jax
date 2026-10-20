"""Scheduling-group micro-batch executables must carry the prefill XLA options."""

import importlib
import os
import unittest

from sgl_jax.srt.model_executor import aot_dispatch


def _reload(**env):
    for k in (
        "SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS",
        "DSA_PREFILL_MICROBATCH",
        "DSA_PREFILL_MICROBATCH_MIN_TOKENS",
    ):
        os.environ.pop(k, None)
    os.environ.update(env)
    importlib.reload(aot_dispatch)
    return aot_dispatch.prefill_xla_options_min_tokens()


class PrefillOptionsThresholdGuardTest(unittest.TestCase):
    def tearDown(self):
        _reload()

    def test_clamped_when_microbatched_and_threshold_above_microbatch(self):
        self.assertEqual(
            _reload(
                SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS="2048",
                DSA_PREFILL_MICROBATCH="2",
                DSA_PREFILL_MICROBATCH_MIN_TOKENS="512",
            ),
            512,
        )

    def test_untouched_when_consistent_or_no_groups(self):
        self.assertEqual(
            _reload(
                SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS="512",
                DSA_PREFILL_MICROBATCH="2",
                DSA_PREFILL_MICROBATCH_MIN_TOKENS="512",
            ),
            512,
        )
        self.assertEqual(
            _reload(
                SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS="2048",
                DSA_PREFILL_MICROBATCH="1",
                DSA_PREFILL_MICROBATCH_MIN_TOKENS="512",
            ),
            2048,
        )
        self.assertEqual(
            _reload(), 2048
        )  # registered defaults: options threshold 2048, micro-batch 2 x 2048


if __name__ == "__main__":
    unittest.main()
