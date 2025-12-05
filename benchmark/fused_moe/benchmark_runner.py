"""Core benchmark execution for MoE implementations."""

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from benchmark.fused_moe.config_utils import MoEBenchmarkConfig
from benchmark.fused_moe.synthetic_data import (
    compute_imbalance_metrics,
    create_hidden_states,
    generate_router_logits,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    implementation: str  # "fused" or "epmoe"
    scenario: str  # "random", "balanced", "imbalanced"
    num_tokens: int
    ep_size: int
    tp_size: int
    num_experts: int
    num_experts_per_tok: int

    # Latency metrics (in milliseconds)
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Load imbalance metrics
    max_load: int
    min_load: int
    avg_load: float
    max_imbalance: float

    # Throughput
    throughput: float  # tokens/sec


class MoEBenchmarkRunner:
    """Orchestrates benchmark execution for both MoE implementations."""

    def __init__(
        self,
        config: MoEBenchmarkConfig,
        mesh: Mesh,
        warmup_iters: int = 1,
        benchmark_iters: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            mesh: JAX mesh with (expert, tensor) axes
            warmup_iters: Number of warmup iterations (default: 1 for JAX JIT)
            benchmark_iters: Number of benchmark iterations
            verbose: Enable verbose logging
        """
        self.config = config
        self.mesh = mesh
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.verbose = verbose

        # Create dummy config for layer initialization
        self.dummy_config = self._create_dummy_config()

        # Will be initialized later
        self.fused_moe = None
        self.epmoe_topk = None
        self.epmoe = None

    def _create_dummy_config(self):
        """Create a minimal config object for MoE layer initialization."""
        return SimpleNamespace(
            hidden_size=self.config.hidden_size,
            ep_size=self.config.ep_size,
        )

    def initialize_layers(self, fused_weights: dict, epmoe_weights: dict):
        """
        Initialize both MoE implementations with synthetic weights.

        Args:
            fused_weights: Weights for FusedEPMoE (w1, w2)
            epmoe_weights: Weights for EPMoE (wi_0, wi_1, wo)
        """

        from flax import nnx

        from sgl_jax.srt.layers.fused_moe import FusedEPMoE
        from sgl_jax.srt.layers.moe import EPMoE, TopK

        dtype = jnp.bfloat16 if self.config.dtype == "bfloat16" else jnp.float32

        # Initialize FusedEPMoE
        if self.verbose:
            print("Initializing FusedEPMoE...")

        self.fused_moe = nnx.eval_shape(
            lambda: FusedEPMoE(
                config=self.dummy_config,
                num_experts=self.config.num_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
                ep_size=self.config.ep_size,
                mesh=self.mesh,
                intermediate_dim=self.config.intermediate_size,
                weight_dtype=dtype,
                dtype=dtype,
                activation=self.config.activation,
                renormalize_topk_logits=self.config.renormalize_topk_logits,
            )
        )

        # Overwrite weights with synthetic values
        self.fused_moe.w1.value = fused_weights["w1"]
        self.fused_moe.w2.value = fused_weights["w2"]

        # Initialize EPMoE components
        if self.verbose:
            print("Initializing EPMoE...")

        self.epmoe_topk = TopK(
            topk=self.config.num_experts_per_tok,
            renormalize=self.config.renormalize_topk_logits,
        )

        self.epmoe = nnx.eval_shape(
            lambda: EPMoE(
                config=self.dummy_config,
                num_experts=self.config.num_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
                ep_size=self.config.ep_size,
                mesh=self.mesh,
                intermediate_dim=self.config.intermediate_size,
                weight_dtype=dtype,
                dtype=dtype,
                activation=self.config.activation,
            )
        )

        # Overwrite weights
        self.epmoe.wi_0.value = epmoe_weights["wi_0"]
        self.epmoe.wi_1.value = epmoe_weights["wi_1"]
        self.epmoe.wo.value = epmoe_weights["wo"]

        if self.verbose:
            print("Layer initialization complete.")

    def run_fused_moe(
        self,
        hidden_states: jax.Array,
        router_logits: jax.Array,
    ) -> Tuple[jax.Array, List[float]]:
        """
        Run FusedEPMoE forward pass and measure latency.

        Args:
            hidden_states: Input tokens (num_tokens, hidden_size)
            router_logits: Router logits (num_tokens, num_experts)

        Returns:
            output: MoE output
            latencies: List of latencies in milliseconds
        """
        # Ensure inputs are on device
        hidden_states = jax.device_put(hidden_states)
        router_logits = jax.device_put(router_logits)

        # JIT compile
        @jax.jit
        def fused_forward(hidden_states, router_logits):
            return self.fused_moe(hidden_states, router_logits)

        # Warmup (trigger JIT compilation)
        if self.verbose:
            print(f"  Warmup: {self.warmup_iters} iteration(s)...")
        for _ in range(self.warmup_iters):
            output = fused_forward(hidden_states, router_logits)
            jax.block_until_ready(output)

        # Benchmark
        if self.verbose:
            print(f"  Benchmark: {self.benchmark_iters} iterations...")
        latencies = []
        for _ in range(self.benchmark_iters):
            start = time.perf_counter()
            output = fused_forward(hidden_states, router_logits)
            jax.block_until_ready(output)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

        return output, latencies

    def run_epmoe(
        self,
        hidden_states: jax.Array,
        router_logits: jax.Array,
    ) -> Tuple[jax.Array, List[float], jax.Array]:
        """
        Run EPMoE forward pass and measure latency.

        Args:
            hidden_states: Input tokens (num_tokens, hidden_size)
            router_logits: Router logits (num_tokens, num_experts)

        Returns:
            output: MoE output
            latencies: List of latencies in milliseconds
            topk_ids: Expert assignments for imbalance calculation
        """
        hidden_states = jax.device_put(hidden_states)
        router_logits = jax.device_put(router_logits)

        # JIT compile (TopK + EPMoE together)
        @jax.jit
        def epmoe_forward(hidden_states, router_logits):
            topk_weights, topk_ids = self.epmoe_topk(router_logits)
            output = self.epmoe(hidden_states, topk_weights, topk_ids)
            return output, topk_ids

        # Warmup
        if self.verbose:
            print(f"  Warmup: {self.warmup_iters} iteration(s)...")
        for _ in range(self.warmup_iters):
            output, topk_ids = epmoe_forward(hidden_states, router_logits)
            jax.block_until_ready(output)

        # Benchmark
        if self.verbose:
            print(f"  Benchmark: {self.benchmark_iters} iterations...")
        latencies = []
        for _ in range(self.benchmark_iters):
            start = time.perf_counter()
            output, topk_ids = epmoe_forward(hidden_states, router_logits)
            jax.block_until_ready(output)
            latencies.append((time.perf_counter() - start) * 1000)

        return output, latencies, topk_ids

    def benchmark_scenario(
        self,
        scenario: str,
        num_tokens: int,
        imbalance_factor: float = 3.0,
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Run benchmark for a single scenario.

        Args:
            scenario: "random", "balanced", or "imbalanced"
            num_tokens: Number of tokens to test
            imbalance_factor: Target imbalance for "imbalanced" scenario

        Returns:
            fused_result: Results for FusedEPMoE
            epmoe_result: Results for EPMoE
        """
        if self.verbose:
            print(f"\nBenchmarking scenario={scenario}, num_tokens={num_tokens}")

        # Generate data
        hidden_states = create_hidden_states(
            num_tokens,
            self.config.hidden_size,
            dtype=jnp.bfloat16 if self.config.dtype == "bfloat16" else jnp.float32,
        )

        router_logits = generate_router_logits(
            num_tokens,
            self.config.num_experts,
            scenario,
            num_experts_per_tok=self.config.num_experts_per_tok,
            imbalance_factor=imbalance_factor,
        )

        # Run FusedEPMoE
        if self.verbose:
            print("Running FusedEPMoE...")
        fused_output, fused_latencies = self.run_fused_moe(hidden_states, router_logits)

        # Run EPMoE
        if self.verbose:
            print("Running EPMoE...")
        epmoe_output, epmoe_latencies, topk_ids = self.run_epmoe(hidden_states, router_logits)

        # Compute imbalance metrics (same for both since they use same router logits)
        imbalance = compute_imbalance_metrics(topk_ids, self.config.num_experts)

        if self.verbose:
            print(f"  Max imbalance: {imbalance['max_imbalance']:.2f}x")

        # Create results
        fused_result = self._create_result(
            "fused", scenario, num_tokens, fused_latencies, imbalance
        )
        epmoe_result = self._create_result(
            "epmoe", scenario, num_tokens, epmoe_latencies, imbalance
        )

        return fused_result, epmoe_result

    def _create_result(
        self,
        implementation: str,
        scenario: str,
        num_tokens: int,
        latencies: List[float],
        imbalance: dict,
    ) -> BenchmarkResult:
        """Create BenchmarkResult from latency measurements."""
        latencies_array = np.array(latencies)

        return BenchmarkResult(
            implementation=implementation,
            scenario=scenario,
            num_tokens=num_tokens,
            ep_size=self.config.ep_size,
            tp_size=self.config.tp_size,
            num_experts=self.config.num_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            latency_mean=float(np.mean(latencies_array)),
            latency_std=float(np.std(latencies_array)),
            latency_p50=float(np.percentile(latencies_array, 50)),
            latency_p95=float(np.percentile(latencies_array, 95)),
            latency_p99=float(np.percentile(latencies_array, 99)),
            latency_min=float(np.min(latencies_array)),
            latency_max=float(np.max(latencies_array)),
            max_load=imbalance["max_load"],
            min_load=imbalance["min_load"],
            avg_load=imbalance["avg_load"],
            max_imbalance=imbalance["max_imbalance"],
            throughput=num_tokens / (np.mean(latencies_array) / 1000),  # tokens/sec
        )
