"""
Benchmark ep_moe (reference path) for comparison with fused_moe.

Usage:
    python -m benchmark.moe.bench_ep_moe [--scenario random|balanced|imbalanced]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    build_group_sizes,
    format_load_info,
    generate_router_logits,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.layers.moe import EPMoE, TopK


def build_mesh(ep_size: int = 1, tp_size: int = 1):
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    return create_device_mesh(
        ici_parallelism=[1, tp_size * ep_size],
        dcn_parallelism=[1, 1],
        mesh_axes=("data", "tensor"),
    )


def prepare_ep_moe_inputs(
    case: MoEBenchmarkCase,
    scenario: str,
    *,
    dtype: jnp.dtype,
) -> dict[str, jax.Array]:
    """Prepare inputs for ep_moe path (TopK + EPMoE).

    We only create tokens/router logits/group sizes here. Weights are created
    inside the mesh with explicit sharding to avoid huge random initialization.
    """
    tokens = jnp.empty((case.num_tokens, case.hidden_size), dtype=dtype)
    router_logits = generate_router_logits(
        case.num_tokens,
        case.num_experts,
        scenario,
        num_experts_per_tok=case.top_k,
    ).astype(dtype)
    group_sizes, topk_ids = build_group_sizes(router_logits, case.top_k, case.num_experts)
    return {
        "tokens": tokens,
        "router_logits": router_logits,
        "group_sizes": group_sizes,
        "topk_ids": topk_ids,
    }


def run_all(
    scenario: str,
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> None:
    cases = list(select_cases())

    print(f"Running ep_moe benchmarks with scenario='{scenario}', dtype={dtype}")
    for case in cases:
        assert case.ep_size is not None
        assert case.tp_size is not None
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}"
        )

        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
        data = prepare_ep_moe_inputs(case, scenario, dtype=dtype)
        load_info = format_load_info(data["group_sizes"])
        print(f"  router load -> {load_info}")

        with jax.set_mesh(mesh):
            topk_layer = TopK(
                topk=case.top_k,
                renormalize=case.renormalize_topk_logits,
                num_expert_group=case.num_expert_group,
                topk_group=case.topk_group,
                routed_scaling_factor=case.routed_scaling_factor,
            )

            ep_moe_layer = EPMoE(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=dtype,
                dtype=dtype,
                activation=case.activation,
                layer_id=0,
            )

            @jax.jit
            def ep_moe_fn(hidden_states, router_logits):
                topk_weights, topk_ids = topk_layer(router_logits)
                return ep_moe_layer(hidden_states, topk_weights, topk_ids)

            # warmup
            start = time.perf_counter()
            jax.block_until_ready(ep_moe_fn(data["tokens"], data["router_logits"]))
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"warmup in {elapsed_ms} ms")

            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: ep_moe_fn(data["tokens"], data["router_logits"]),
                data_generator=lambda: (),
                task=f"ep_moe_{case.name}",
                tries=iters,
            )

            mean_ms = float(np.mean(times)) if times else float("nan")
            print(f"  ep_moe: {mean_ms:.3f} ms (trace) | samples={times}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ep_moe reference path.")
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
