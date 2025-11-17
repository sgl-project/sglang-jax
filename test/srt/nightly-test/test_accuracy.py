import os
import sys
import unittest
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN_7B,
    CustomTestCase,
    popen_launch_server,
)


class TestModelAccuracy(CustomTestCase):
    def test_qwen_7b(self):
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            QWEN_7B,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.36)

        ## kill process
        kill_process_tree(process.pid)

    def test_qwen_7b_tp_4(self):
        pass

    def test_qwen3_8b(self):
        pass

    def test_qwen3_8b_tp_4(self):
        pass


if __name__ == "__main__":
    unittest.main()
