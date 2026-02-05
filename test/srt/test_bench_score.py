"""
Score API performance benchmark with regression detection.

This module provides performance benchmarks for the `/v1/score` API endpoint,
validating latency and throughput thresholds to catch performance regressions.

The Score API is a prefill-only operation that computes probability scores
for specified tokens given a query and items. This makes it much faster than
generation workloads, but we still need to ensure performance doesn't regress.

Design Document: sglang-jax-dev-scripts/rfcs/002-cicd-tpu-testing.md

Test Coverage:
- Single item latency (p50, p99 thresholds)
- Batch throughput (items/second threshold)
- Large batch latency (20 items)
- Warmup handling

Usage:
    # Run all benchmarks
    python3 -m unittest test.srt.test_bench_score

    # Run individual benchmark
    python3 -m unittest test.srt.test_bench_score.TestScoreAPIPerformance.test_score_latency_single_item

Requirements:
    - TPU access (tests use device="tpu")
    - Model: Qwen/Qwen3-1.7B (small model for fast benchmarks)

Performance Notes:
    - Engine initialization: ~30-60s
    - Each benchmark: ~30-60s
    - Total runtime: ~3 minutes
"""

import statistics
import time
import unittest
from dataclasses import dataclass

import jax
from transformers import AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

# =============================================================================
# Performance Thresholds
# =============================================================================
# These thresholds are based on TPU v6e baseline measurements.
# Adjust based on actual baseline runs if needed.
#
# The Score API uses prefill-only execution (max_new_tokens=0), so it's
# significantly faster than generation benchmarks.

# Latency thresholds (in milliseconds)
SCORE_LATENCY_P50_THRESHOLD_MS = 50.0  # p50 latency must be under 50ms
SCORE_LATENCY_P99_THRESHOLD_MS = 150.0  # p99 latency must be under 150ms

# Throughput threshold (items scored per second)
SCORE_THROUGHPUT_THRESHOLD_IPS = 100.0  # Must achieve at least 100 items/sec

# Large batch latency multiplier (p99 threshold = base * multiplier)
LARGE_BATCH_LATENCY_MULTIPLIER = 3.0


# =============================================================================
# Benchmark Configuration
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from a score benchmark run."""

    throughput_ips: float  # Items scored per second
    latency_p50_ms: float  # 50th percentile latency
    latency_p95_ms: float  # 95th percentile latency
    latency_p99_ms: float  # 99th percentile latency
    latency_mean_ms: float  # Mean latency
    latency_min_ms: float  # Minimum latency
    latency_max_ms: float  # Maximum latency
    num_requests: int  # Number of requests made
    total_items: int  # Total items scored
    total_time_sec: float  # Total benchmark time


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    batch_size: int  # Number of items per request
    num_requests: int  # Number of requests to run
    warmup_requests: int  # Number of warmup requests (not measured)
    name: str  # Descriptive name for reporting


# =============================================================================
# Test Class
# =============================================================================


class TestScoreAPIPerformance(CustomTestCase):
    """
    Score API performance benchmarks with regression thresholds.

    These tests validate that Score API performance meets minimum
    requirements and catches regressions before they reach production.

    Benchmarks measure:
    - Latency: Time to complete a single score request (p50, p95, p99)
    - Throughput: Items scored per second across multiple requests

    All benchmarks use the same engine instance to minimize initialization
    overhead and provide consistent measurements.
    """

    model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    engine = None
    tokenizer = None
    label_token_ids = None

    @classmethod
    def setUpClass(cls):
        """Initialize engine and prepare test data."""
        print(f"[Benchmark] Loading model: {cls.model_name}")

        cls.engine = Engine(
            model_path=cls.model_name,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            chunked_prefill_size=1024,
            download_dir="/dev/shm",
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)

        # Get label token IDs for common classification tokens
        # Using tokens that should exist in most vocabularies
        cls.label_tokens = [" yes", " no", " maybe"]
        cls.label_token_ids = []
        for token in cls.label_tokens:
            encoding = cls.tokenizer.encode_plus(token, add_special_tokens=False)
            if encoding["input_ids"]:
                cls.label_token_ids.append(encoding["input_ids"][0])

        print("[Benchmark] Model loaded")
        print(f"[Benchmark] Label tokens: {cls.label_tokens}")
        print(f"[Benchmark] Label token IDs: {cls.label_token_ids}")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def run_score_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a score benchmark with the given configuration.

        Args:
            config: Benchmark configuration (batch size, num requests, etc.)

        Returns:
            BenchmarkResult with throughput and latency metrics.
        """
        # Prepare test data
        query = (
            "Is the following statement true or false? Answer with yes, no, or maybe. Statement:"
        )
        items = [
            f" This is test item number {i} for performance benchmarking."
            for i in range(config.batch_size)
        ]

        # Warmup runs (not measured)
        for _ in range(config.warmup_requests):
            self.engine.score(
                query=query,
                items=items,
                label_token_ids=self.label_token_ids,
                apply_softmax=True,
            )

        # Benchmark runs
        latencies_ms = []
        total_start = time.perf_counter()

        for _ in range(config.num_requests):
            start = time.perf_counter()
            scores = self.engine.score(
                query=query,
                items=items,
                label_token_ids=self.label_token_ids,
                apply_softmax=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

            # Validate output structure
            assert (
                len(scores) == config.batch_size
            ), f"Expected {config.batch_size} score lists, got {len(scores)}"

        total_time_sec = time.perf_counter() - total_start

        # Compute metrics
        latencies_ms.sort()
        total_items = config.batch_size * config.num_requests

        def percentile(data, p):
            """Compute percentile of sorted data."""
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            return data[idx]

        return BenchmarkResult(
            throughput_ips=total_items / total_time_sec,
            latency_p50_ms=percentile(latencies_ms, 50),
            latency_p95_ms=percentile(latencies_ms, 95),
            latency_p99_ms=percentile(latencies_ms, 99),
            latency_mean_ms=statistics.mean(latencies_ms),
            latency_min_ms=min(latencies_ms),
            latency_max_ms=max(latencies_ms),
            num_requests=config.num_requests,
            total_items=total_items,
            total_time_sec=total_time_sec,
        )

    def _report_result(self, name: str, result: BenchmarkResult):
        """Print benchmark results and write to GitHub step summary if in CI."""
        report = (
            f"\n[Benchmark] {name}\n"
            f"  Throughput: {result.throughput_ips:.1f} items/sec\n"
            f"  Latency p50: {result.latency_p50_ms:.1f} ms\n"
            f"  Latency p95: {result.latency_p95_ms:.1f} ms\n"
            f"  Latency p99: {result.latency_p99_ms:.1f} ms\n"
            f"  Latency mean: {result.latency_mean_ms:.1f} ms\n"
            f"  Latency min/max: {result.latency_min_ms:.1f} / {result.latency_max_ms:.1f} ms\n"
            f"  Total items: {result.total_items}\n"
            f"  Total time: {result.total_time_sec:.2f} sec\n"
        )
        print(report)

        if is_in_ci():
            write_github_step_summary(
                f"### {name}\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Throughput | {result.throughput_ips:.1f} items/sec |\n"
                f"| Latency p50 | {result.latency_p50_ms:.1f} ms |\n"
                f"| Latency p95 | {result.latency_p95_ms:.1f} ms |\n"
                f"| Latency p99 | {result.latency_p99_ms:.1f} ms |\n"
                f"| Total items | {result.total_items} |\n"
            )

    # =========================================================================
    # Benchmark Tests
    # =========================================================================

    def test_score_latency_single_item(self):
        """
        Test Score API latency with single item per request.

        This benchmark measures the baseline latency for scoring a single item.
        It validates p50 and p99 latency thresholds to catch regressions.

        Failure indicates:
        - Model forward pass is slower than expected
        - Logprob extraction has performance issues
        - Prefill-only optimization may not be working
        """
        config = BenchmarkConfig(
            batch_size=1,
            num_requests=50,
            warmup_requests=5,
            name="Single Item Latency",
        )

        result = self.run_score_benchmark(config)
        self._report_result(config.name, result)

        # Validate thresholds
        self.assertLess(
            result.latency_p50_ms,
            SCORE_LATENCY_P50_THRESHOLD_MS,
            f"p50 latency {result.latency_p50_ms:.1f}ms exceeds threshold "
            f"{SCORE_LATENCY_P50_THRESHOLD_MS}ms",
        )

        self.assertLess(
            result.latency_p99_ms,
            SCORE_LATENCY_P99_THRESHOLD_MS,
            f"p99 latency {result.latency_p99_ms:.1f}ms exceeds threshold "
            f"{SCORE_LATENCY_P99_THRESHOLD_MS}ms",
        )

    def test_score_throughput_batch(self):
        """
        Test Score API throughput with batched items.

        This benchmark measures throughput when scoring multiple items
        in a single request. Batching should improve throughput compared
        to single-item requests.

        Validates minimum throughput threshold (items/second).

        Failure indicates:
        - Batch processing has performance issues
        - Memory bandwidth limitations
        - Suboptimal batch handling
        """
        config = BenchmarkConfig(
            batch_size=8,
            num_requests=30,
            warmup_requests=5,
            name="Batch Throughput (8 items)",
        )

        result = self.run_score_benchmark(config)
        self._report_result(config.name, result)

        # Validate throughput threshold
        self.assertGreater(
            result.throughput_ips,
            SCORE_THROUGHPUT_THRESHOLD_IPS,
            f"Throughput {result.throughput_ips:.1f} items/sec below threshold "
            f"{SCORE_THROUGHPUT_THRESHOLD_IPS} items/sec",
        )

    def test_score_latency_large_batch(self):
        """
        Test Score API latency with large batch (20 items).

        This benchmark ensures large batches don't cause excessive latency.
        Large batches are common in ranking/reranking use cases where
        many candidates need to be scored simultaneously.

        Uses a relaxed latency threshold (3x base p99 threshold) since
        large batches naturally take longer.

        Failure indicates:
        - Memory issues with large batches
        - Non-linear scaling with batch size
        - Potential OOM or timeout issues
        """
        config = BenchmarkConfig(
            batch_size=20,
            num_requests=20,
            warmup_requests=3,
            name="Large Batch Latency (20 items)",
        )

        result = self.run_score_benchmark(config)
        self._report_result(config.name, result)

        # Large batch has relaxed latency threshold
        large_batch_threshold = SCORE_LATENCY_P99_THRESHOLD_MS * LARGE_BATCH_LATENCY_MULTIPLIER

        self.assertLess(
            result.latency_p99_ms,
            large_batch_threshold,
            f"Large batch p99 latency {result.latency_p99_ms:.1f}ms exceeds threshold "
            f"{large_batch_threshold:.1f}ms",
        )

        # Also check throughput - large batches should still be efficient
        self.assertGreater(
            result.throughput_ips,
            SCORE_THROUGHPUT_THRESHOLD_IPS * 0.5,  # Allow 50% reduction for large batches
            f"Large batch throughput {result.throughput_ips:.1f} items/sec is too low",
        )

    def test_score_scaling_with_batch_size(self):
        """
        Test Score API scaling across different batch sizes.

        This benchmark validates that performance scales reasonably
        with batch size. It runs benchmarks at multiple batch sizes
        and reports the scaling characteristics.

        This is an informational benchmark - it reports metrics but
        doesn't enforce strict thresholds, as scaling depends on
        hardware and model size.
        """
        batch_sizes = [1, 4, 8, 16]
        results = []

        for batch_size in batch_sizes:
            config = BenchmarkConfig(
                batch_size=batch_size,
                num_requests=20,
                warmup_requests=3,
                name=f"Batch Size {batch_size}",
            )
            result = self.run_score_benchmark(config)
            results.append((batch_size, result))

        # Report scaling summary
        print("\n[Benchmark] Scaling Summary")
        print("  Batch Size | Throughput (items/s) | Latency p50 (ms)")
        print("  -----------|---------------------|------------------")
        for batch_size, result in results:
            print(
                f"  {batch_size:10} | {result.throughput_ips:19.1f} | {result.latency_p50_ms:16.1f}"
            )

        if is_in_ci():
            rows = "\n".join(
                [f"| {bs} | {r.throughput_ips:.1f} | {r.latency_p50_ms:.1f} |" for bs, r in results]
            )
            write_github_step_summary(
                f"### Score API Scaling\n"
                f"| Batch Size | Throughput (items/s) | Latency p50 (ms) |\n"
                f"|------------|---------------------|------------------|\n"
                f"{rows}\n"
            )

        # Basic sanity check: throughput should increase with batch size
        # (at least up to a point)
        throughputs = [r.throughput_ips for _, r in results]
        self.assertGreater(
            throughputs[-1], throughputs[0], "Throughput should increase with batch size"
        )


if __name__ == "__main__":
    unittest.main()
