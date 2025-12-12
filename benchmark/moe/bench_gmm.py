"""
Benchmark megablox gmm with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_gmm [--scenario random|balanced|imbalanced]
"""

from __future__ import annotations

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.moe.utils import (
    compute_gmm_tiling,
    format_load_info,
    multiple_iteration_timeit_from_trace,
    prepare_gmm_inputs,
    select_cases,
)
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm


def run_all(
    scenario: str,
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> None:
    cases = list(select_cases())

    print(f"Running megablox gmm benchmarks with scenario='{scenario}', dtype={dtype}")
    for case in cases:
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}"
        )

        data = prepare_gmm_inputs(case, scenario, dtype=dtype)
        load_info = format_load_info(data["group_sizes"])
        print(f"  router load -> {load_info}")

        @functools.partial(
            jax.jit,
            static_argnames=("preferred_element_type", "tiling"),
        )
        def jitted(lhs, rhs, group_sizes, preferred_element_type, tiling):
            return gmm(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type=preferred_element_type,
                tiling=tiling,
            )

        gmm_fn = functools.partial(
            jitted,
            data["gmm_lhs"],
            data["gmm_rhs"],
            data["group_sizes"],
            dtype,
            compute_gmm_tiling(
                data["gmm_lhs"].shape[0],
                data["gmm_rhs"].shape[1],
                data["gmm_rhs"].shape[2],
            ),
        )

        # Warmup compile
        jax.block_until_ready(gmm_fn())

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: gmm_fn(),
            data_generator=lambda: (),
            task=f"gmm_{case.name}",
            tries=iters,
        )

        mean_ms = float(np.mean(times)) if times else float("nan")
        print(f"  megablox_gmm: {mean_ms:.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark megablox gmm.")
    parser.add_argument(
        "--scenario",
        choices=["random", "balanced", "imbalanced"],
        default="random",
        help="Router logits distribution pattern.",
    )
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.scenario, args.iters)
