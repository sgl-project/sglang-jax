import unittest
from types import SimpleNamespace

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.run_curl import run_curl
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestCacheMiss(CustomTestCase):
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
                "0.2",
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
                "8",
                "--page-size",
                "64",
                "--max-running-requests",
                "8",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_cache_miss(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=6,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)


if __name__ == "__main__":
    unittest.main()
