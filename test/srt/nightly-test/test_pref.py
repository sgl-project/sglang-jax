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
    is_in_ci,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)

MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")

trace_output_dir = os.path.abspath(
    os.path.join(
        os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run"), "model_traces"
    )
)
os.makedirs(trace_output_dir, exist_ok=True)
print(f"[CI Info] Precision Tracer Output Dir: {trace_output_dir}")


class TestModePerf(CustomTestCase):
    sharegpt_dataset_path = None
    BASIC_SERVER_ARGS = [
        "--trust-remote-code",
        "--random-seed",
        "3",
        "--max-prefill-tokens",
        "16384",
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
        "0.7",
        "--enable-precision-tracer",
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

    def test_qwen_7b_performance_tp_1(self):
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]
        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            num_steps=10,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
            print("[CI Info] Waiting for traces to flush...")
            time.sleep(5)

            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results.csv")
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

    def test_qwen_7b_performance_tp_4(self):
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []
        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_4.csv")
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

    def test_qwen3_8b_performance_tp_1(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_8B
        model_dir_name = "QWEN3_8B"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results.csv")
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

    def test_qwen3_8b_performance_tp_4(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_8B
        model_dir_name = "QWEN3_8B"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_4.csv")
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

    def test_QWEN3_MOE_30B_performance_tp_2_ep_2(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_MOE_30B
        model_dir_name = "QWEN3_MOE_30B"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "2",
            "--ep-size",
            "2",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_2_ep_2.csv")
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

    def test_GEMMA2_2B_IT_performance_tp_1(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = GEMMA2_2B_IT
        model_dir_name = "GEMMA2_2B_IT"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]
        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
            "--disable-hybrid-swa-memory",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results.csv")
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

    def test_GEMMA2_2B_IT_performance_tp_4(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = GEMMA2_2B_IT
        model_dir_name = "GEMMA2_2B_IT"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
            "--disable-hybrid-swa-memory",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
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
                "--max-prefill-tokens",
                "16384",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
                "--disable-hybrid-swa-memory",
                "--mem-fraction-static",
                "0.8",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_4.csv")
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

    def test_bailing_moe_performance_tp_2_ep_2(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = BAILING_MOE
        model_dir_name = "BAILING_MOE"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "2",
            "--ep-size",
            "2",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_2_ep_2.csv")
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

    def test_QWEN2_5_7B_INSTRUCT_performance_tp_1(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN2_5_7B_INSTRUCT
        model_dir_name = "QWEN2_5_7B_INSTRUCT"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]
        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results.csv")
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

    def test_QWEN2_5_7B_INSTRUCT_performance_tp_4(self):
        import os

        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN2_5_7B_INSTRUCT
        model_dir_name = "QWEN2_5_7B_INSTRUCT"
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
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        # concurrency_levels = [8]
        # # input length levels (1k, 4k, 8k)
        # input_lengths = [1024]

        output_lengths = [1, 1024]
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
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
                "SGLANG_JAX_PROFILER_DIR": trace_output_dir,
            },
        )

        results_summary = []

        try:
            for concurrency in concurrency_levels:
                for in_len in input_lengths:
                    for out_len in output_lengths:
                        print(f"\n{'#'*60}")
                        print(
                            f"Testing Scenario: Concurrency={concurrency} | Input={in_len} | Output={out_len}"
                        )
                        print(f"{'#'*60}")

                        current_num_prompts = max(concurrency * 3, 50)

                        args = argparse.Namespace(
                            max_concurrency=concurrency,
                            random_input_len=in_len,
                            random_output_len=out_len,
                            num_prompts=current_num_prompts,
                            backend="sgl-jax",
                            base_url=base_url,
                            host="0.0.0.0",
                            port=int(base_url.split(":")[-1]),
                            model=model_path_for_server,
                            tokenizer=model_path_for_server,
                            dataset_name="sharegpt",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path=self.sharegpt_dataset_path,
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=True,
                            pd_separated=False,
                            tokenize_prompt=False,
                            adjust_prompt_max_retry=10,
                            # ShareGPT
                            sharegpt_output_len=None,
                            sharegpt_context_len=None,
                            apply_chat_template=False,
                            prompt_suffix="",
                            # Generated-Shared-Prefix (gsp)
                            gsp_num_groups=64,
                            gsp_prompts_per_group=16,
                            gsp_system_prompt_len=2048,
                            gsp_question_len=128,
                            gsp_output_len=256,
                        )

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
                                "tpu_size": 4,
                            }
                        )

                        time.sleep(1)

        finally:
            process.terminate()
            process.wait()

        if results_summary:
            import csv
            import os

            output_dir = os.getenv("PREF_OUTPUT_DIR", "./test/nightly_test_output/pref/local_run")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, "performance_results_tp_4.csv")
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)


if __name__ == "__main__":
    unittest.main()
