"""
Benchmark ep_moe (reference path) for comparison with fused_moe.

Usage:
    python -m benchmark.moe.bench_fused_moe --use-shared-expert  --use-grouped-topk --num-tokens 128 --imbalance-mode sparse_hotspot --hotspot-ratio 1 --hotspot-count 48
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as _compilation_cache

from benchmark.moe.utils import (
    BAILING_BASE,
    MoEBenchmarkCase,
    build_group_sizes,
    build_mesh,
    format_load_info,
    generate_router_logits,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.layers.moe import EPMoE, TopK


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
    *,
    num_tokens: list[int] | None = None,
) -> None:
    raw_cases: list[MoEBenchmarkCase] | None = None
    if num_tokens is not None:
        raw_cases = [
            MoEBenchmarkCase(
                name=(
                    f"custom_nt{n}_ne{BAILING_BASE['num_experts']}_tk{BAILING_BASE['top_k']}"
                    f"_h{BAILING_BASE['hidden_size']}_i{BAILING_BASE['intermediate_size']}"
                ),
                num_tokens=n,
                **BAILING_BASE,
            )
            for n in num_tokens
        ]
    cases = list(select_cases(raw_cases))

    print(f"Running ep_moe benchmarks with scenario='{scenario}', dtype={dtype}")
    for case in cases:
        assert case.ep_size is not None
        assert case.tp_size is not None
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}"
        )
        print(
            f"  mesh: ep_size={case.ep_size}, tp_size={case.tp_size}, "
            f"devices_used={case.ep_size * case.tp_size}/{len(jax.devices())}"
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

            # Avoid capturing massive expert weights as XLA constants: split NNx modules
            # into (def, state) and pass the state leaves as explicit jitted inputs.
            topk_def, topk_state = nnx.split(topk_layer)
            topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)
            moe_def, moe_state = nnx.split(ep_moe_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

            @jax.jit(static_argnames=("topk_state_def", "moe_state_def"))
            def ep_moe_fn(
                hidden_states,
                router_logits,
                *,
                topk_state_def,
                topk_state_leaves,
                moe_state_def,
                moe_state_leaves,
            ):
                topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
                topk = nnx.merge(topk_def, topk_state)
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)

                topk_weights, topk_ids = topk(router_logits)
                return moe(hidden_states, topk_weights, topk_ids)

            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: ep_moe_fn(
                    data["tokens"],
                    data["router_logits"],
                    topk_state_def=topk_state_def,
                    topk_state_leaves=topk_state_leaves,
                    moe_state_def=moe_state_def,
                    moe_state_leaves=moe_state_leaves,
                ),
                data_generator=lambda: (),
                task=f"ep_moe_{case.name}",
                tries=iters,
            )
            if len(times) > 1:
                times = times[1:]
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
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Override benchmark cases with custom num_tokens list (e.g. --num-tokens 8 16 256 4096).",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=str,
        default=None,
        help="Optional JAX compilation cache directory (persists compiled executables across runs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)
    run_all(args.scenario, args.iters, num_tokens=args.num_tokens)
