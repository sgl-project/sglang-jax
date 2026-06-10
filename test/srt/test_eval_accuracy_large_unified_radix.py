"""MMLU sanity for --enable-unified-radix-tree (HiCache Stage 1, S1c / #1337).

Mirrors test_eval_accuracy_large.py with the unified-radix-tree flag on, to
confirm the UnifiedRadixCache serving path does not regress accuracy.

NOTE: deliberately NOT registered in test/srt/run_suite.py. CI suite placement
(accuracy-test-tpu-v6e-1 vs a nightly suite) is deferred pending team
discussion; run it manually:

    cd test/srt && python -m unittest \
        test_eval_accuracy_large_unified_radix.TestEvalAccuracyLargeUnifiedRadix.test_mmlu

Caveat: `python run_suite.py --suite all` globs test_*.py and WOULD pick this
up; that is expected -- the deferral is about the named per-PR/nightly suites.
"""

import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestEvalAccuracyLargeUnifiedRadix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--nnodes",
                "1",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.8",
                "--max-prefill-tokens",
                "8192",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--precompile-bs-paddings",
                "64",
                "--precompile-token-paddings",
                "8192",
                "--chunked-prefill-size",
                "-1",
                "--max-running-requests",
                "64",
                "--page-size",
                "64",
                "--enable-unified-radix-tree",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=200,
            num_threads=32,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f'### test_mmlu\n{metrics["score"]=:.4f}\n')
        print("mmlu metrics", metrics)

        self.assertGreater(metrics["score"], 0.698)


if __name__ == "__main__":
    unittest.main()
