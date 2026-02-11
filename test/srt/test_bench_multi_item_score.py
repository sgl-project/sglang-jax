"""
Score API multi-item performance benchmark.

This module compares the performance of single-item vs multi-item scoring
under specific load conditions:
- Prompt length: 2000 tokens
- Number of candidates: 500
- Tokens per candidate: 20
"""

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


# =============================================================================
# Test Class
# =============================================================================


class TestMultiItemScorePerformance(CustomTestCase):
    """
    Benchmarks comparing single-item scoring with multi-item scoring.
    """

    model_name = "/models/Qwen/Qwen3-0.6B"
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
    DELIMITER_TOKEN_ID = 128001  # Specific delimiter for multi-item

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
            download_dir="/data/huggingface_models",  # Use GCS mount
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024, 4096, 16384],
            page_size=64,
            log_requests=False,
            # Enable multi-item delimiter at engine level
            multi_item_scoring_delimiter=cls.DELIMITER_TOKEN_ID,
            disable_radix_cache=True,
            max_multi_item_seq_len=32768,
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

        # Warmup
        # Note: In multi-item mode, the Engine.score implementation handles packing and chunking.
        self.engine.score(
            query=query_tokens,
            items=candidate_tokens_list[:10],  # Small warmup
            label_token_ids=self.label_token_ids,
        )

        start_time = time.perf_counter()

        # The Engine.score handles the 500 items.
        # Internally it will chunk them based on max_multi_item_seq_len (default 32k)
        self.engine.score(
            query=query_tokens,
            items=candidate_tokens_list,
            label_token_ids=self.label_token_ids,
        )

        total_time = time.perf_counter() - start_time

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

            """

            chunk_sizes = [32, 64, 128]

            results = []

            

            print(f"\n[Benchmark] Starting Multi-Item Chunk Scaling (Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN})")

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


if __name__ == "__main__":
    unittest.main()
