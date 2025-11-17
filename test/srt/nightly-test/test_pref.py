import asyncio
import itertools
import unittest
from random import random, uniform

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    QWEN3_MOE_30B,
    QWEN_7B,
    CustomTestCase,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)


class TestModePerf(CustomTestCase):
    def test_input_throughput_qwen_7b_con_8_1k_1(self):
        res = run_bench_serving(
            model=QWEN_7B,
            num_prompts=24,
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
                "8",
                "--page-size",
                "128",
                "--disable-radix-cache",
            ],
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=8,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_qwen_7b\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 28299)

    def test_input_throughput_qwen_7b_tp_4_con_8_1k_1(self):
        pass

    def test_output_throughput_qwen_7b_con_8_1k_1(self):
        res = run_bench_serving(
            model=QWEN_7B,
            num_prompts=24,
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
                "8",
                "--page-size",
                "128",
                "--disable-radix-cache",
            ],
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=8,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_qwen_7b\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2345)

    def test_output_throughput_qwen_7b_tp_4_con_8_1k_1(self):
        pass

    def test_ttft_qwen_7b_con_8_1k_1(self):
        res = run_bench_serving(
            model=QWEN_7B,
            num_prompts=24,
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
                "8",
                "--page-size",
                "128",
                "--disable-radix-cache",
            ],
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=8,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_ttft_ms"], 52)

    def test_ttft_qwen_7b_tp_4_con_8_1k_1(self):
        pass

    def test_itl_qwen_7b_con_8_1k_1(self):
        res = run_bench_serving(
            model=QWEN_7B,
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
                "8",
                "--page-size",
                "128",
                "--disable-radix-cache",
            ],
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=8,
            random_range_ratio=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default\n" f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_itl_ms"], 16)

    def test_itl_qwen_7b_tp_4_con_8_1k_1(self):
        pass


if __name__ == "__main__":
    unittest.main()
