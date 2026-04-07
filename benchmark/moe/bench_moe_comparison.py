"""
Compare EPMoE (psum/all-reduce) vs FusedEPMoE (fused Pallas kernel + A2A)
on MiMoV2Flash MoE dimensions across different EP/TP configurations.

Designed for 16 v6e chips (4 hosts x 4 chips, all ICI).

Usage (single-host, 4 chips):
    python -m benchmark.moe.bench_moe_comparison

Usage (multi-host, 16 chips):
    python -m benchmark.moe.bench_moe_comparison --num-tokens 512 1024 2048
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as _compilation_cache

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    generate_router_logits,
    prepare_fused_moe_inputs,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.layers.moe import EPMoE, FusedEPMoE, TopK

# MiMoV2Flash MoE dimensions
MIMO_NUM_EXPERTS = 256
MIMO_TOP_K = 8
MIMO_HIDDEN_SIZE = 4096
MIMO_INTERMEDIATE_SIZE = 2048
MIMO_ACTIVATION = "silu"

DEFAULT_TOKEN_COUNTS = (128, 256, 512, 1024, 2048, 4096, 8192)


def _parallelism_configs(num_devices: int) -> list[tuple[int, int]]:
    """Generate (ep_size, tp_size) pairs where ep * tp == num_devices."""
    configs = []
    ep = 1
    while ep <= num_devices:
        if num_devices % ep == 0 and MIMO_NUM_EXPERTS % ep == 0:
            configs.append((ep, num_devices // ep))
        ep *= 2
    return configs


def _run_epmoe(
    case: MoEBenchmarkCase,
    ep_size: int,
    tp_size: int,
    iters: int,
    scenario: str,
) -> float | None:
    """Benchmark EPMoE with given EP/TP config. Returns mean latency in ms."""
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)

    tokens = jnp.empty((case.num_tokens, case.hidden_size), dtype=jnp.bfloat16)
    router_logits = generate_router_logits(
        case.num_tokens,
        case.num_experts,
        scenario,
        num_experts_per_tok=case.top_k,
    ).astype(jnp.bfloat16)

    with jax.set_mesh(mesh):
        topk_layer = TopK(topk=case.top_k, renormalize=True)
        moe_layer = EPMoE(
            hidden_size=case.hidden_size,
            num_experts=case.num_experts,
            num_experts_per_tok=case.top_k,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=case.intermediate_size,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation=case.activation,
            layer_id=0,
        )

        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(moe_layer)
        moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_treedef", "moe_treedef"))
        def fn(hidden, logits, *, topk_treedef, topk_leaves, moe_treedef, moe_leaves):
            topk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_treedef, topk_leaves))
            moe = nnx.merge(moe_def, jax.tree_util.tree_unflatten(moe_treedef, moe_leaves))
            w, ids = topk(logits)
            return moe(hidden, w, ids)

        try:
            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: fn(
                    tokens,
                    router_logits,
                    topk_treedef=topk_treedef,
                    topk_leaves=topk_leaves,
                    moe_treedef=moe_treedef,
                    moe_leaves=moe_leaves,
                ),
                data_generator=lambda: (),
                task=f"epmoe_ep{ep_size}_tp{tp_size}_nt{case.num_tokens}",
                tries=iters,
            )
        except Exception as e:
            print(f"    ERROR EPMoE EP={ep_size} TP={tp_size}: {e}")
            traceback.print_exc()
            return None

    if len(times) > 1:
        times = times[1:]
    return float(np.mean(times)) if times else None


def _run_fused_epmoe(
    case: MoEBenchmarkCase,
    ep_size: int,
    iters: int,
    imbalance_mode: str,
) -> float | None:
    """Benchmark FusedEPMoE (ep_size == num_devices, tp_size must be 1)."""
    mesh = build_mesh(ep_size=ep_size, tp_size=1)

    data = prepare_fused_moe_inputs(
        case,
        weight_dtype=jnp.bfloat16,
        mesh=mesh,
        include_weights=False,
    )
    target_counts = MoEImbalanceSimulator.generate_counts(
        case.num_tokens,
        case.top_k,
        case.num_experts,
        mode=imbalance_mode,
    )
    custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
        case.num_tokens,
        case.num_experts,
        case.top_k,
        target_counts,
    )
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    data["router_logits"] = jax.device_put(
        custom_logits,
        NamedSharding(mesh, P("tensor", None)),
    )

    with jax.set_mesh(mesh):
        fused_layer = FusedEPMoE(
            hidden_size=case.hidden_size,
            num_experts=case.num_experts,
            num_experts_per_tok=case.top_k,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=case.intermediate_size,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation=case.activation,
            layer_id=0,
            renormalize_topk_logits=True,
        )
        topk_layer = TopK(topk=case.top_k, renormalize=True)

        moe_def, moe_state = nnx.split(fused_layer)
        moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)
        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)

        @jax.jit(static_argnames=("moe_treedef", "topk_treedef"))
        def fn(tokens, logits, *, moe_treedef, moe_leaves, topk_treedef, topk_leaves):
            moe = nnx.merge(moe_def, jax.tree_util.tree_unflatten(moe_treedef, moe_leaves))
            topk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_treedef, topk_leaves))
            w, ids = topk(logits)
            return moe(tokens, w, ids)

        try:
            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: fn(
                    data["tokens"],
                    data["router_logits"],
                    moe_treedef=moe_treedef,
                    moe_leaves=moe_leaves,
                    topk_treedef=topk_treedef,
                    topk_leaves=topk_leaves,
                ),
                data_generator=lambda: (),
                task=f"fused_ep{ep_size}_nt{case.num_tokens}",
                tries=iters,
            )
        except Exception as e:
            print(f"    ERROR FusedEPMoE EP={ep_size}: {e}")
            traceback.print_exc()
            return None

    if len(times) > 1:
        times = times[1:]
    return float(np.mean(times)) if times else None


def run_comparison(
    num_tokens: list[int] | None = None,
    iters: int = 3,
    scenario: str = "balanced",
) -> None:
    num_devices = len(jax.devices())
    token_list = list(num_tokens or DEFAULT_TOKEN_COUNTS)
    par_configs = _parallelism_configs(num_devices)

    print("MoE Comparison Benchmark (MiMoV2Flash dimensions)")
    print(f"  devices: {num_devices} x {jax.devices()[0].device_kind}")
    print(
        f"  experts={MIMO_NUM_EXPERTS}, top_k={MIMO_TOP_K}, "
        f"hidden={MIMO_HIDDEN_SIZE}, intermediate={MIMO_INTERMEDIATE_SIZE}"
    )
    print(f"  EP/TP configs (EPMoE): {par_configs}")
    print(f"  token counts: {token_list}")
    print(f"  scenario: {scenario}, iters: {iters}")
    print()

    # Collect results: (backend, ep, tp, num_tokens) -> latency_ms
    results: dict[tuple[str, int, int, int], float | None] = {}

    for nt in token_list:
        # Skip if num_tokens not divisible by largest EP
        if nt % num_devices != 0:
            print(f"[SKIP] num_tokens={nt} not divisible by {num_devices}, skipping")
            continue

        case = MoEBenchmarkCase(
            name=f"mimo_nt{nt}",
            num_tokens=nt,
            num_experts=MIMO_NUM_EXPERTS,
            top_k=MIMO_TOP_K,
            hidden_size=MIMO_HIDDEN_SIZE,
            intermediate_size=MIMO_INTERMEDIATE_SIZE,
            activation=MIMO_ACTIVATION,
        )

        print(f"=== num_tokens={nt} ===")

        # EPMoE with various EP/TP configs
        for ep, tp in par_configs:
            if nt % ep != 0:
                print(f"  EPMoE  EP={ep:>2} TP={tp:>2}: SKIP (tokens not divisible by EP)")
                results[("EPMoE", ep, tp, nt)] = None
                continue
            print(f"  EPMoE  EP={ep:>2} TP={tp:>2}: ", end="", flush=True)
            ms = _run_epmoe(case, ep, tp, iters, scenario)
            results[("EPMoE", ep, tp, nt)] = ms
            if ms is not None:
                print(f"{ms:.3f} ms")
            else:
                print("FAILED")

        # FusedEPMoE (always EP=num_devices, TP=1)
        print(f"  Fused  EP={num_devices:>2} TP= 1: ", end="", flush=True)
        ms = _run_fused_epmoe(case, num_devices, iters, scenario)
        results[("FusedEPMoE", num_devices, 1, nt)] = ms
        if ms is not None:
            print(f"{ms:.3f} ms")
        else:
            print("FAILED")
        print()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY: MoE latency (ms) — MiMoV2Flash dimensions")
    print("=" * 90)

    # Header
    configs_header = []
    for ep, tp in par_configs:
        configs_header.append(f"EP{ep}TP{tp}")
    configs_header.append(f"Fused(EP{num_devices})")

    header = f"{'tokens':>8} | " + " | ".join(f"{h:>12}" for h in configs_header)
    print(header)
    print("-" * len(header))

    for nt in token_list:
        if nt % num_devices != 0:
            continue
        row = f"{nt:>8} | "
        cells = []
        for ep, tp in par_configs:
            ms = results.get(("EPMoE", ep, tp, nt))
            cells.append(f"{ms:>12.3f}" if ms else f"{'—':>12}")
        ms = results.get(("FusedEPMoE", num_devices, 1, nt))
        cells.append(f"{ms:>12.3f}" if ms else f"{'—':>12}")
        row += " | ".join(cells)
        print(row)

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EPMoE vs FusedEPMoE on MiMoV2Flash MoE dimensions.",
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Token counts to benchmark. Default: 128..8192",
    )
    parser.add_argument(
        "--scenario",
        choices=["random", "balanced", "imbalanced"],
        default="balanced",
        help="Router distribution for EPMoE path.",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=str,
        default=None,
        help="JAX compilation cache directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        faulthandler.enable(file=sys.stdout, all_threads=True)
    except Exception:
        pass

    args = parse_args()
    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)

    run_comparison(
        num_tokens=args.num_tokens,
        iters=args.iters,
        scenario=args.scenario,
    )
