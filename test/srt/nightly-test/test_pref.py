import argparse
import asyncio
import itertools
import time
import unittest
from random import random, uniform

import requests

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.test.test_utils import (
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
    bailing_moe,
    is_in_ci,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)


class TestModePerf(CustomTestCase):
    def test_qwen_7b_performance_tp_1(self):
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        # concurrency_levels = [8, 16, 32, 64, 128, 256]

        concurrency_levels = [8, 16, 32]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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

    def test_QWEN3_CODER_30B_A3B_INSTRUCT_performance_tp_2_ep_2(self):
        model = QWEN3_CODER_30B_A3B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
                "--disable-hybrid-swa-memory",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
                "--disable-hybrid-swa-memory",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
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
        model = bailing_moe
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

    def test_QWEN2_5_7B_INSTRUCT_performance_tp_1(self):
        model = QWEN2_5_7B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

    def test_QWEN2_5_7B_INSTRUCT_performance_tp_4(self):
        model = QWEN2_5_7B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST

        # define test parameters
        # concurrency levels
        concurrency_levels = [8, 16, 32, 64, 128, 256]
        # input length levels (1k, 4k, 8k)
        input_lengths = [1024, 4096, 8192]

        output_lengths = [1, 1024]

        # launch server
        process = popen_launch_server(
            model,
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
                "65536",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
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
                            model=model,
                            tokenizer=model,
                            dataset_name="random",
                            random_range_ratio=1.0,
                            request_rate=float("inf"),
                            warmup_requests=1,
                            flush_cache=True,
                            dataset_path="",
                            output_file=None,
                            output_details=False,
                            disable_tqdm=False,
                            disable_stream=False,
                            disable_ignore_eos=False,
                            return_logprob=False,
                            seed=42,
                            extra_request_body=None,
                            lora_name=None,
                            profile=False,
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
                                "ttft_ms": metrics.get("mean_ttft_ms", 0),
                                "itl_ms": metrics.get("mean_itl_ms", 0),
                                "in_tps": metrics.get("input_throughput", 0),
                                "out_tps": metrics.get("output_throughput", 0),
                                "model_name": model,
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

            output_filename = "./test/nightly_test_output/performance_results.csv"
            file_exists = os.path.exists(output_filename)
            with open(output_filename, "a", newline="", encoding="utf-8") as csvfile:

                headers = results_summary[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(results_summary)

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
                "--grammar-backend",
                "none",
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
                "--grammar-backend",
                "none",
                "--tensor-parallel-size",
                "4",
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
                "--grammar-backend",
                "none",
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
                "--grammar-backend",
                "none",
                "--tensor-parallel-size",
                "4",
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
                "--grammar-backend",
                "none",
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
                "--grammar-backend",
                "none",
                "--tensor-parallel-size",
                "4",
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
                "--grammar-backend",
                "none",
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
                "--grammar-backend",
                "none",
                "--tensor-parallel-size",
                "4",
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


if __name__ == "__main__":
    unittest.main()
