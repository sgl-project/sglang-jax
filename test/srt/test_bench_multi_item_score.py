# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Score API multi-item performance benchmark.

This module compares the performance of single-item vs multi-item scoring
under specific load conditions:
- Prompt length: 2000 tokens
- Number of candidates: 500
- Tokens per candidate: 20
"""

import os
import statistics
import time
import unittest
from dataclasses import dataclass
from typing import List

import jax
import numpy as np
from transformers import AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci, write_github_step_summary

# =============================================================================
# Benchmark Configuration
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from a score benchmark run."""

    name: str
    total_time_sec: float
    latency_per_item_ms: float
    throughput_items_sec: float
    num_items: int
    prompt_len: int
    candidate_len: int


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        return default
    return values


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _pick_benchmark_model_path() -> str:
    local_path = "/models/Qwen/Qwen3-0.6B"
    if os.path.isdir(local_path):
        return local_path
    return "Qwen/Qwen3-0.6B"


def _pick_benchmark_download_dir() -> str:
    override = os.getenv("MULTI_ITEM_BENCH_DOWNLOAD_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override

    candidates = [
        "/data/huggingface_models",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        "/tmp/huggingface_models",
    ]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                return path
        except OSError:
            continue
    return candidates[-1]


# =============================================================================
# Test Class
# =============================================================================


class TestMultiItemScorePerformance(CustomTestCase):
    """
    Benchmarks comparing single-item scoring with multi-item scoring.
    """

    model_name = _pick_benchmark_model_path()
    download_dir = _pick_benchmark_download_dir()
    LOG_LEVEL = os.getenv("MULTI_ITEM_BENCH_LOG_LEVEL", "error")
    engine = None
    tokenizer = None
    label_token_ids = [198]  # '\n' token or similar common token

    # Target scenario
    PROMPT_LEN = 2000
    NUM_CANDIDATES = 500
    CANDIDATE_LEN = 20

    # Multi-item specific
    STATIC_PREFIX_LEN = 100
    DYNAMIC_SUFFIX_LEN = 1900  # 100 + 1900 = 2000
    PREFILL_EXTEND_BATCH_SIZE = int(os.getenv("MULTI_ITEM_EXTEND_BATCH_SIZE", "32"))
    WARMUP_RUNS = int(os.getenv("MULTI_ITEM_BENCH_WARMUP_RUNS", "1"))
    TIMED_RUNS = int(os.getenv("MULTI_ITEM_BENCH_TIMED_RUNS", "4"))

    @classmethod
    def setUpClass(cls):
        """Initialize engine with multi-item support."""
        print(f"[Benchmark] Loading model: {cls.model_name}")

        cls.engine = Engine(
            model_path=cls.model_name,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.7,  # Leave room for compilation
            max_prefill_tokens=32768,  # Support large packed sequences
            chunked_prefill_size=-1,
            download_dir=cls.download_dir,
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024, 4096, 16384],
            page_size=64,
            log_requests=False,
            disable_radix_cache=True,
            log_level=cls.LOG_LEVEL,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)
        print("[Benchmark] Engine initialized")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_benchmark_single_item_sequential(self):
        """
        Scenario: 500 candidates scored one-by-one (or in small batches).
        Each candidate has the same 2000 token prompt prefix.
        """
        print(
            f"\n[Benchmark] Starting Single-Item Sequential (Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN})"
        )

        # Generate dummy data (using token IDs to be precise)
        query_tokens = [1] * self.PROMPT_LEN
        candidate_tokens_list = [[2] * self.CANDIDATE_LEN for _ in range(self.NUM_CANDIDATES)]

        # Warmup (1 request)
        self.engine.score(
            query=query_tokens,
            items=candidate_tokens_list[:1],
            label_token_ids=self.label_token_ids,
        )

        start_time = time.perf_counter()

        # We simulate the user sending them in batches of 32 (max recommended for single-item)
        BATCH_SIZE = 32
        for i in range(0, self.NUM_CANDIDATES, BATCH_SIZE):
            chunk = candidate_tokens_list[i : i + BATCH_SIZE]
            self.engine.score(
                query=query_tokens,
                items=chunk,
                label_token_ids=self.label_token_ids,
            )

        total_time = time.perf_counter() - start_time

        result = BenchmarkResult(
            name="Single-Item Sequential",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=self.PROMPT_LEN,
            candidate_len=self.CANDIDATE_LEN,
        )
        self._report_result(result)

    def test_benchmark_multi_item_packed(self):
        """
        Scenario: 500 candidates scored in a single packed sequence.
        Uses 100 token static prefix + 1900 token dynamic suffix + 500 * 20 token items.
        """
        print(
            f"\n[Benchmark] Starting Multi-Item Packed (Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN})"
        )

        # Generate dummy data
        # Prefix = 100 (static) + 1900 (dynamic) = 2000
        query_tokens = [1] * (self.STATIC_PREFIX_LEN + self.DYNAMIC_SUFFIX_LEN)
        candidate_tokens_list = [[2] * self.CANDIDATE_LEN for _ in range(self.NUM_CANDIDATES)]

        # Warm up with the same shape used by timed runs to avoid counting
        # first-hit compilation in steady-state throughput.
        for _ in range(self.WARMUP_RUNS):
            self.engine.score(
                query=query_tokens,
                items=candidate_tokens_list,
                label_token_ids=self.label_token_ids,
            )

        run_times = []
        for run_idx in range(self.TIMED_RUNS):
            start_time = time.perf_counter()
            self.engine.score(
                query=query_tokens,
                items=candidate_tokens_list,
                label_token_ids=self.label_token_ids,
            )
            run_times.append(time.perf_counter() - start_time)
            print(
                f"  Run {run_idx + 1}/{self.TIMED_RUNS}: "
                f"{run_times[-1]:.2f} sec ({self.NUM_CANDIDATES / run_times[-1]:.2f} items/sec)"
            )

        total_time = statistics.mean(run_times)

        result = BenchmarkResult(
            name="Multi-Item Packed",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=self.PROMPT_LEN,
            candidate_len=self.CANDIDATE_LEN,
        )
        self._report_result(result)

    def test_benchmark_scenario_1(self):
        """
        Scenario 1:
        - 500 candidate items per request
        - 20 tokens per candidate
        - 2000-token static prefix
        - 20 token dynamic suffix
        """
        print("\n[Benchmark] Starting Scenario 1")
        static_prefix = [1] * 2000
        dynamic_suffix = [2] * 20
        query_tokens = static_prefix + dynamic_suffix
        items = [[3] * 20 for _ in range(500)]

        # Warmup
        self.engine.score(query=query_tokens, items=items[:1], label_token_ids=self.label_token_ids)

        start_time = time.perf_counter()
        self.engine.score(query=query_tokens, items=items, label_token_ids=self.label_token_ids)
        total_time = time.perf_counter() - start_time

        result = BenchmarkResult(
            name="Scenario 1",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / 500,
            throughput_items_sec=500 / total_time,
            num_items=500,
            prompt_len=2020,
            candidate_len=20,
        )
        self._report_result(result)

    def test_benchmark_scenario_2(self):
        """
        Scenario 2:
        - 500 candidate items per request
        - 10 tokens per candidate
        - 1900-token static prefix
        - 10 token dynamic suffix
        """
        print("\n[Benchmark] Starting Scenario 2")
        static_prefix = [1] * 1900
        dynamic_suffix = [2] * 10
        query_tokens = static_prefix + dynamic_suffix
        items = [[3] * 10 for _ in range(500)]

        # Warmup
        self.engine.score(query=query_tokens, items=items[:1], label_token_ids=self.label_token_ids)

        start_time = time.perf_counter()
        self.engine.score(query=query_tokens, items=items, label_token_ids=self.label_token_ids)
        total_time = time.perf_counter() - start_time

        result = BenchmarkResult(
            name="Scenario 2",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / 500,
            throughput_items_sec=500 / total_time,
            num_items=500,
            prompt_len=1910,
            candidate_len=10,
        )
        self._report_result(result)

    def test_benchmark_multi_item_chunk_scaling(self):
        """
        Test multi-item performance across different chunk sizes.
        Updated for 500 items.
        """
        chunk_sizes = [32, 64, 128]
        results = []

        print(
            f"\n[Benchmark] Starting Multi-Item Chunk Scaling (Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN})"
        )

        query_tokens = [1] * (self.STATIC_PREFIX_LEN + self.DYNAMIC_SUFFIX_LEN)
        candidate_tokens_list = [[2] * self.CANDIDATE_LEN for _ in range(self.NUM_CANDIDATES)]

        for cs in chunk_sizes:
            print(f"  Testing chunk_size={cs}...")

            # Note: We need a new engine or to change server_args at runtime if supported.
            # Since Engine currently uses a fixed process pool, we'll simulate the impact
            # by manually chunking the request if the engine doesn't support runtime change,
            # BUT the goal is to test the server-side chunking.
            # To truly test server-side, we would need to restart the engine.
            # For this benchmark, we'll assume we want to see the effect of the server's
            # configured chunk_size.

            # Since we can't easily restart the Engine mid-test-class, we'll perform
            # a manual chunking benchmark to simulate the server behavior for different sizes.

            start_time = time.perf_counter()
            # Manual chunking simulation
            for i in range(0, self.NUM_CANDIDATES, cs):
                chunk = candidate_tokens_list[i : i + cs]
                self.engine.score(
                    query=query_tokens, items=chunk, label_token_ids=self.label_token_ids
                )

            total_time = time.perf_counter() - start_time

            result = BenchmarkResult(
                name=f"Multi-Item (Chunk={cs})",
                total_time_sec=total_time,
                latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
                throughput_items_sec=self.NUM_CANDIDATES / total_time,
                num_items=self.NUM_CANDIDATES,
                prompt_len=self.PROMPT_LEN,
                candidate_len=self.CANDIDATE_LEN,
            )
            self._report_result(result)
            results.append((cs, result))

        print("\n[Benchmark] Chunk Size Scaling Summary")
        print("  Chunk Size | Throughput (items/s) | Latency/Item (ms)")
        print("  -----------|---------------------|------------------")
        for cs, res in results:
            print(f"  {cs:10} | {res.throughput_items_sec:19.2f} | {res.latency_per_item_ms:16.2f}")

    def _report_result(self, result: BenchmarkResult):
        report = (
            f"  Throughput: {result.throughput_items_sec:.2f} items/sec\n"
            f"  Latency per item: {result.latency_per_item_ms:.2f} ms\n"
            f"  Total time for {result.num_items} items: {result.total_time_sec:.2f} sec\n"
        )
        print(report)

        if is_in_ci():
            write_github_step_summary(
                f"### {result.name} (Prompt={result.prompt_len}, Items={result.num_items})\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Throughput | {result.throughput_items_sec:.2f} items/sec |\n"
                f"| Latency/Item | {result.latency_per_item_ms:.2f} ms |\n"
                f"| Total Time | {result.total_time_sec:.2f} s |\n"
            )


class TestMultiItemPrefillExtendPerformance(CustomTestCase):
    """
    Benchmarks for multi-item scoring using prefill+extend mode.
    """

    model_name = _pick_benchmark_model_path()
    download_dir = _pick_benchmark_download_dir()
    LOG_LEVEL = os.getenv("MULTI_ITEM_BENCH_LOG_LEVEL", "error")
    engine = None
    label_token_ids = [198]

    PROMPT_LEN = 2000
    NUM_CANDIDATES = 500
    CANDIDATE_LEN = 20
    PREFILL_EXTEND_BATCH_SIZE = int(os.getenv("MULTI_ITEM_EXTEND_BATCH_SIZE", "12"))
    PREFILL_EXTEND_MAX_RUNNING_REQUESTS = int(
        os.getenv("MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS", "12")
    )
    PREFILL_EXTEND_PRECOMPILE_BS_PADDINGS = _parse_int_list_env(
        "MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    ENABLE_SCORE_FROM_CACHE_V2 = _parse_bool_env("MULTI_ITEM_ENABLE_SCORE_FROM_CACHE_V2", False)
    DISABLE_OVERLAP_SCHEDULE = _parse_bool_env(
        "MULTI_ITEM_DISABLE_OVERLAP_SCHEDULE", ENABLE_SCORE_FROM_CACHE_V2
    )
    SCORE_FROM_CACHE_V2_ITEMS_PER_STEP = int(
        os.getenv("MULTI_ITEM_SCORE_FROM_CACHE_V2_ITEMS_PER_STEP", "64")
    )
    SCORE_FASTPATH_LOG_METRICS = _parse_bool_env("MULTI_ITEM_SCORE_FASTPATH_LOG_METRICS", False)
    SCORE_LABEL_ONLY_LOGPROB = _parse_bool_env("MULTI_ITEM_SCORE_LABEL_ONLY_LOGPROB", False)
    WARMUP_RUNS = int(os.getenv("MULTI_ITEM_BENCH_WARMUP_RUNS", "3"))
    TIMED_RUNS = int(os.getenv("MULTI_ITEM_BENCH_TIMED_RUNS", "4"))

    @classmethod
    def setUpClass(cls):
        print(f"[Benchmark] Loading model (prefill+extend): {cls.model_name}")
        cls.engine = Engine(
            model_path=cls.model_name,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.7,
            max_prefill_tokens=32768,
            chunked_prefill_size=-1,
            download_dir=cls.download_dir,
            dtype="bfloat16",
            precompile_bs_paddings=cls.PREFILL_EXTEND_PRECOMPILE_BS_PADDINGS,
            max_running_requests=cls.PREFILL_EXTEND_MAX_RUNNING_REQUESTS,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024, 4096, 16384],
            page_size=64,
            log_requests=False,
            disable_radix_cache=False,
            enable_scoring_cache=True,
            multi_item_enable_prefill_extend=True,
            multi_item_extend_batch_size=cls.PREFILL_EXTEND_BATCH_SIZE,
            disable_overlap_schedule=cls.DISABLE_OVERLAP_SCHEDULE,
            multi_item_enable_score_from_cache_v2=cls.ENABLE_SCORE_FROM_CACHE_V2,
            multi_item_score_from_cache_v2_items_per_step=cls.SCORE_FROM_CACHE_V2_ITEMS_PER_STEP,
            multi_item_score_fastpath_log_metrics=cls.SCORE_FASTPATH_LOG_METRICS,
            multi_item_score_label_only_logprob=cls.SCORE_LABEL_ONLY_LOGPROB,
            log_level=cls.LOG_LEVEL,
        )
        print("[Benchmark] Prefill+extend engine initialized")

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_benchmark_multi_item_prefill_extend(self):
        print(
            f"\n[Benchmark] Starting Multi-Item Prefill+Extend "
            f"(Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN}, ExtendBatch={self.PREFILL_EXTEND_BATCH_SIZE})"
        )
        query_tokens = [1] * self.PROMPT_LEN
        candidate_tokens_list = [[2] * self.CANDIDATE_LEN for _ in range(self.NUM_CANDIDATES)]

        # Warm up with the exact shape used in timed runs.
        # For scoring-cache mode, two warmups ensure both cold prefill and
        # cache-hit extend paths are compiled before timing.
        for _ in range(self.WARMUP_RUNS):
            self.engine.score(
                query=query_tokens,
                items=candidate_tokens_list,
                label_token_ids=self.label_token_ids,
            )

        run_times = []
        for run_idx in range(self.TIMED_RUNS):
            start_time = time.perf_counter()
            self.engine.score(
                query=query_tokens,
                items=candidate_tokens_list,
                label_token_ids=self.label_token_ids,
            )
            run_times.append(time.perf_counter() - start_time)
            print(
                f"  Run {run_idx + 1}/{self.TIMED_RUNS}: "
                f"{run_times[-1]:.2f} sec ({self.NUM_CANDIDATES / run_times[-1]:.2f} items/sec)"
            )

        total_time = statistics.mean(run_times)

        result = BenchmarkResult(
            name="Multi-Item Prefill+Extend",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=self.PROMPT_LEN,
            candidate_len=self.CANDIDATE_LEN,
        )
        report = (
            f"  Throughput: {result.throughput_items_sec:.2f} items/sec\n"
            f"  Latency per item: {result.latency_per_item_ms:.2f} ms\n"
            f"  Total time for {result.num_items} items: {result.total_time_sec:.2f} sec\n"
        )
        print(report)

        if is_in_ci():
            write_github_step_summary(
                f"### {result.name} (Prompt={result.prompt_len}, Items={result.num_items})\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Throughput | {result.throughput_items_sec:.2f} items/sec |\n"
                f"| Latency/Item | {result.latency_per_item_ms:.2f} ms |\n"
                f"| Total Time | {result.total_time_sec:.2f} s |\n"
            )


if __name__ == "__main__":
    unittest.main()
