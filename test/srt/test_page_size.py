import unittest
from types import SimpleNamespace

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.run_eval import run_eval
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPageSizeGreaterThanOne(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
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
                "--mem-fraction-static",
                "0.65",
                "--max-prefill-tokens",
                "8192",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--jax-precompile-prefill-token-paddings",
                "16384",
                "--jax-precompile-decode-bs-paddings",
                "64",
                "--page-size",
                "64",
                "--max-running-requests",
                "64",
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
            num_examples=128,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
