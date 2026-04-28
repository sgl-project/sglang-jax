"""Integration tests for the retract / release_kv_cache path.

SGLANG_TEST_RETRACT=1 forces retract on batch_size > 10. Pass criterion:
the worker stays alive (process.poll() is None) -- a leak would trip
scheduler.check_memory() and SIGQUIT.
"""

import time
import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class _BaseRetractDecode(CustomTestCase):
    """Abstract base; concrete subclasses set ``other_args`` to vary
    page_size / radix on/off."""

    other_args: list[str] = []

    @classmethod
    def setUpClass(cls):
        if cls is _BaseRetractDecode:
            raise unittest.SkipTest("base class")
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--random-seed",
            "3",
            "--tp",
            "4",
            "--mem-fraction-static",
            "0.65",
            "--chunked-prefill-size",
            "128",
            "--max-prefill-tokens",
            "8192",
            "--max-running-requests",
            "64",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fa",
            "--precompile-token-paddings",
            "16384",
            "--precompile-bs-paddings",
            "64",
        ] + cls.other_args
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=launch_args,
            env={
                "SGLANG_TEST_RETRACT": "1",
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
            num_threads=16,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.5)
        time.sleep(1)  # let scheduler.check_memory() run on idle
        assert self.process.poll() is None, "Server crashed during retract test"


class TestRetractDecodePaged(_BaseRetractDecode):
    """page_size=16, radix on -- exercises paged RadixCache.cache_finished_req."""

    other_args = ["--page-size", "16"]


class TestRetractDecodeChunkCache(_BaseRetractDecode):
    """page_size=1, radix off -- exercises ChunkCache.cache_finished_req."""

    other_args = ["--disable-radix-cache"]


class TestRetractDecodeChunkCachePaged(_BaseRetractDecode):
    """page_size=16, radix off -- exercises paged ChunkCache + alignment."""

    other_args = ["--disable-radix-cache", "--page-size", "16"]


if __name__ == "__main__":
    unittest.main()
