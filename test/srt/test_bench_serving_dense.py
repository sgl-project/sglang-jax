import asyncio
import itertools
import unittest
from random import random, uniform
from types import SimpleNamespace

import requests

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_MOE_30B,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestBenchServing(CustomTestCase):
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
                "--device",
                "tpu",
                "--random-seed",
                "3",
                "--chunked-prefill-size",
                "2048",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "256",
                "--page-size",
                "128",
                "--disable-radix-cache",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_throughput_default(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            num_prompts=500,
            request_rate=float("inf"),
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 500

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 28299)

    def test_output_throughput_default(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            num_prompts=500,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 500

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2345)

    def test_ttft_default(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            num_prompts=1,
            request_rate=float("inf"),
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 1

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_ttft_ms"], 52)

    def test_itl_default(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            num_prompts=1,
            request_rate=float("inf"),
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 1

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default\n" f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_itl_ms"], 16)


if __name__ == "__main__":
    unittest.main()
