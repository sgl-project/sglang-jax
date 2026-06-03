import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA4_31B_IT,
    CustomTestCase,
    popen_launch_server,
)


class TestGemma4Model(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GEMMA4_31B_IT
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
                "--tp-size",
                "4",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--page-size",
                "64",
                "--max-running-requests",
                "32",
                "--chunked-prefill-size",
                "2048",
                "--swa-full-tokens-ratio",
                "0.1",
                "--disable-radix-cache",
                "--precompile-bs-paddings",
                "32",
                "--precompile-token-paddings",
                "2048",
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
            num_examples=64,
            num_threads=32,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.70)


if __name__ == "__main__":
    unittest.main()
