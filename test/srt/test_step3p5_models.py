"""Step 3.5 Flash real-weight accuracy e2e (TPU, full BF16 checkpoint).

This is the ONLY Step 3.5 test that needs the real 398GB checkpoint — accuracy
cannot be measured with random weights. It closes the real-weight correctness
leg (gsm8k / mmlu thresholds) on top of the microscale flash==naive / I3 / serving
self-consistency already verified.

DEVIATION from the repo's usual setUpClass-launches-the-server pattern (e.g.
test_deepseek_v2_lite_models.py): the 196B model is multi-host (v6e-16), which
``popen_launch_server`` (single process) cannot orchestrate. So this test runs
against an ALREADY-RUNNING server. Launch the multi-host Step-3.5 server
separately, then run with:

    STEP35_MODEL_PATH=<hf-id-or-local-path>   # model name for the eval client
    STEP35_BASE_URL=http://127.0.0.1:30000    # optional; defaults to the test URL
    python -m pytest test/srt/test_step3p5_models.py -v

Without STEP35_MODEL_PATH the whole class skips, so it is CI-safe.

The thresholds below are conservative "not fundamentally broken" floors; the
first measured run reports the actual scores (write_github_step_summary), after
which they should be tightened to the real numbers — exactly how the existing
gsm8k thresholds in this repo were calibrated.
"""

import os
import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

_MODEL = os.environ.get("STEP35_MODEL_PATH")
_BASE_URL = os.environ.get("STEP35_BASE_URL", DEFAULT_URL_FOR_TEST)

# Floors from measured v7x runs (max_tokens=8192 — see _eval; Step-3.5 is a
# reasoning model, so a small generation budget truncates the long CoT before the
# final answer and tanks MMLU: 2048 gave a false 0.727, 8192 gives 0.859 ≈ the
# official 85.8). The MMLU subset is seeded (fixed 128), so run-to-run variance is
# only concurrency bf16 flips, not example sampling; floors are measured-minus a
# (heuristic, not variance-swept) margin:
#   gsm8k = 0.905 (200 ex) -> floor 0.88
#   mmlu  = 0.859 (128 ex) -> floor 0.80 (wider margin: single 8192 measurement)
_GSM8K_FLOOR = 0.88
_MMLU_FLOOR = 0.80


@unittest.skipUnless(
    _MODEL,
    "real-weight accuracy: set STEP35_MODEL_PATH (and launch a Step-3.5 server at "
    "STEP35_BASE_URL) to run; skipped otherwise.",
)
class TestStep3p5FlashAccuracy(CustomTestCase):
    """gsm8k / mmlu accuracy against a running real-weight Step 3.5 server."""

    def _eval(self, eval_name, num_examples):
        args = SimpleNamespace(
            base_url=_BASE_URL,
            model=_MODEL,
            eval_name=eval_name,
            num_examples=num_examples,
            num_threads=32,
            # Step-3.5 is a reasoning model: a tight budget truncates the CoT before
            # the final answer (MMLU dropped to 0.727 at 2048; 0.859 at 8192). The
            # server must be launched with --context-length >= prompt + 8192.
            max_tokens=8192,
        )
        score = run_eval(args)["score"]
        # Print to stdout so the score survives in the pytest log (run with -s) even
        # when the per-rank /tmp/*.json is destroyed on container exit.
        print(f"\n[step3p5-accuracy] {eval_name} score = {score:.4f}\n", flush=True)
        return score

    def test_gsm8k(self):
        score = self._eval("gsm8k", 200)
        if is_in_ci():
            write_github_step_summary(f"### step3p5 test_gsm8k\n{score=:.4f}\n")
        self.assertGreater(score, _GSM8K_FLOOR, f"gsm8k {score:.4f} below floor {_GSM8K_FLOOR}")

    def test_mmlu(self):
        score = self._eval("mmlu", 128)
        if is_in_ci():
            write_github_step_summary(f"### step3p5 test_mmlu\n{score=:.4f}\n")
        self.assertGreater(score, _MMLU_FLOOR, f"mmlu {score:.4f} below floor {_MMLU_FLOOR}")


if __name__ == "__main__":
    unittest.main()
