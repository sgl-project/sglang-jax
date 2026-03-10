"""
Usage:
python -m unittest test_moe_quantized_eval_accuracy.TestMoEInt8QuantizedEvalTP2EP2.test_gpqa
"""

import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_MOE_30B,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

GPQA_SCORE_THRESHOLD = 0.39


class TestMoEInt8QuantizedEvalTP2EP2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_MOE_30B
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--skip-server-warmup",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
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
                "128",
                "--quantization-config-path",
                "int8_qwen3_30b_a3b.yaml",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gpqa(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gpqa",
            num_examples=None,
            num_threads=32,
            temperature=0.0,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], GPQA_SCORE_THRESHOLD)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gpqa (int8 quantized tp2_ep2)\n" f'{metrics["score"]=:.4f}\n'
            )


if __name__ == "__main__":
    unittest.main()
