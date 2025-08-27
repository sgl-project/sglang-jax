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


class TestBaseTp(CustomTestCase):
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
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--nnodes",
                "1",
                "--tp-size",
                "4",
                "--random-seed",
                "3",
                "--device",
                "tpu",
                "--node-rank",
                "0",
                "--mem-fraction-static",
                "0.4",
                "--download-dir",
                "/tmp",
                "--dtype",
                "bfloat16",
                "--jax-precompile-decode-bs-paddings",
                "32",
                "--max-running-requests",
                "32",
                "--skip-server-warmup",
                "--attention-backend",
                "fa",
                "--jax-precompile-prefill-token-paddings",
                "4096",
                "--page-size",
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
            num_examples=64,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.2)


if __name__ == "__main__":
    unittest.main()
