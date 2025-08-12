import asyncio
import itertools
import unittest
from random import random, uniform

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)


class TestBenchServing(CustomTestCase):
    def test_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "4096",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.1",
            ],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 3800)

    def test_offline_throughput_non_stream_small_batch_size(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "4096",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "10",
                "--mem-fraction-static",
                "0.1",
            ],
            dataset_name="sharegpt",
            random_input_len=None,
            random_output_len=None,
            disable_stream=True,
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_non_stream_small_batch_size\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 1050)

    def test_offline_throughput_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "4096",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.1",
            ],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_without_radix_cache\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 3800)

    def test_online_latency_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=100,
            request_rate=1,
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "4096",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.1",
            ],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_latency_default\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            self.assertLess(res["median_ttft_ms"], 86)
            self.assertLess(res["median_itl_ms"], 10)


if __name__ == "__main__":
    unittest.main()
