import asyncio
import itertools
import unittest
from random import random, uniform

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    QWEN3_MOE_30B,
    CustomTestCase,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)


class TestBenchServing(CustomTestCase):
    def test_input_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["input_throughput"], 20000)

    def test_output_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
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
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["output_throughput"], 1000)

    def test_input_throughput_default_tp_4(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default_tp_4\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["input_throughput"], 60000)

    def test_output_throughput_default_tp_4(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
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
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default_tp_4\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["output_throughput"], 10000)

    def test_moe_input_throughput_default(self):
        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_input_throughput_default\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["input_throughput"], 1400)

    def test_moe_output_throughput_default(self):
        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
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
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_output_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            # self.assertGreater(res["output_throughput"], 1000)

    def test_ttft_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_ttft_ms"], 110)

    def test_itl_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
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
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default\n" f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_itl_ms"], 20)

    def test_ttft_default_tp_4(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default_tp_4\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_ttft_ms"], 35)

    def test_itl_default_tp_4(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default_tp_4\n"
                f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_itl_ms"], 10)

    def test_moe_ttft_default(self):
        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_ttft_ms"], 400)

    def test_moe_itl_default(self):
        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--tp-size",
                "4",
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
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_online_itl_default\n"
                f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            # self.assertLess(res["median_itl_ms"], 20)


if __name__ == "__main__":
    unittest.main()
