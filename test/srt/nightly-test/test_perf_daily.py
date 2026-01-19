import argparse
import asyncio
import itertools
import os
import time
import unittest
from random import random, uniform

import requests

from sgl_jax.bench_serving import SHAREGPT_URL, download_and_cache_file, run_benchmark
from sgl_jax.test.test_utils import (
    BAILING_MOE,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA2_2B_IT,
    QWEN2_5_7B_INSTRUCT,
    QWEN3_8B,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    QWEN3_MOE_30B,
    QWEN_7B,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)

MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")


class TestModelPerf(CustomTestCase):
    sharegpt_dataset_path = None
    BASIC_SERVER_ARGS = [
        "--trust-remote-code",
        "--skip-server-warmup",
        "--random-seed",
        "3",
        "--download-dir",
        "/dev/shm/",
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        "256",
        "--attention-backend",
        "fa",
        "--page-size",
        "128",
        "--chunked-prefill-size",
        "2048",
        "--mem-fraction-static",
        "0.8",
        "--disable-radix-cache",
    ]

    @classmethod
    def setUpClass(cls):
        local_dataset_path = os.path.join(
            MOUNT_ROOT, "dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
        )

        if os.path.exists(local_dataset_path):
            print(f"Using path: {local_dataset_path}")
            cls.sharegpt_dataset_path = local_dataset_path
        else:
            print(f"Local dataset not found at '{local_dataset_path}'.")
            cls.sharegpt_dataset_path = download_and_cache_file(SHAREGPT_URL)

        print(f"Dataset is ready at location: {cls.sharegpt_dataset_path}")

    def test_qwen_7b_performance_tp_1_daily(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN_7B
        model_dir_name = "QWEN_7B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)

        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id

        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels

        # input length levels (1k, 4k, 8k)
        input_lengths = [1024]
        concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
            "--context-length",  #  because the seq_length in the config.json is 8192
            "10240",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        results_summary = []

        try:
            static_config = {
                "dataset_name": "random",
                "dataset_path": self.sharegpt_dataset_path,
                "warmup_requests": 0,
            }

            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = concurrency * 3

                        args = get_benchmark_args(
                            base_url=base_url,
                            tokenizer=model_path_for_server,
                            num_prompts=current_num_prompts,
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            seed=42,
                        )

                        vars(args).update(static_config)

                        metrics = run_benchmark(args)

                        results_summary.append(
                            {
                                "concurrency": concurrency,
                                "input": in_len,
                                "output": out_len,
                                "ttft_ms": metrics.get("median_ttft_ms", 0),
                                "itl_ms": metrics.get("median_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": raw_model_id,
                                "tpu_size": 1,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PERF_OUTPUT_DIR", "./test/nightly_test_output/perf/local_run")
            os.makedirs(output_dir, exist_ok=True)
            concurrency_str = "_".join(map(str, concurrency_levels))
            filename = f"performance_results_{model_dir_name}_tp_1.csv"
            output_filename = os.path.join(output_dir, filename)
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

        print("\n" + "=" * 100)
        print(
            f"{'Con':<5} | {'Input':<6} | {'Output':<6} | {'TTFT (ms)':<10} | {'ITL (ms)':<10} | {'Out TPS':<10} | {'In TPS':<10}| {'Model':<20}"
        )
        print("-" * 100)
        for r in results_summary:
            print(
                f"{r['concurrency']:<5} | {r['input']:<6} | {r['output']:<6} | {r['ttft_ms']:<10.2f} | {r['itl_ms']:<10.2f} | {r['out_tps']:<10.2f} | {r['in_tps']:<10.2f}| {r['model_name']:<20}"
            )
        print("=" * 100)


if __name__ == "__main__":
    unittest.main()
