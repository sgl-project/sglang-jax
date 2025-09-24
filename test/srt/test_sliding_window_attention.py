import unittest
from types import SimpleNamespace

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.run_eval import run_eval
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA2_2B_IT,
    CustomTestCase,
    popen_launch_server,
)


class TestSlidingWindowAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GEMMA2_2B_IT
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.8",
                "--download-dir",
                "/tmp/",
                "--max-running-requests",
                "256",
                "--precompile-bs-paddings",
                "32",
                "--precompile-token-paddings",
                "4096",
                "--context-length",
                "4096",
                "--swa-full-tokens-ratio",
                "0.2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=512,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
