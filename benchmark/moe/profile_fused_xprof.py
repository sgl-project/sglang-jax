"""Profile FusedEPMoE with tuned configs — capture xprof trace for roofline analysis."""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    prepare_fused_moe_inputs,
)
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
EP_SIZE = 16

WARMUP = 3
PROFILE_STEPS = 5


def _make_jit_fn(topk_layer, moe_layer):
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

    kwargs = dict(
        topk_treedef=topk_treedef,
        topk_leaves=topk_leaves,
        moe_treedef=moe_treedef,
        moe_leaves=moe_leaves,
    )
    return fn, kwargs


def profile_scenario(num_tokens, trace_dir):
    """Profile a single scenario with xprof trace capture."""
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    case = MoEBenchmarkCase(
        name=f"profile_nt{num_tokens}",
        num_tokens=num_tokens,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )
    data = prepare_fused_moe_inputs(
        case, weight_dtype=jnp.bfloat16, mesh=mesh, include_weights=False
    )
    target_counts = MoEImbalanceSimulator.generate_counts(
        num_tokens, TOP_K, NUM_EXPERTS, mode="balanced"
    )
    custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
        num_tokens, NUM_EXPERTS, TOP_K, target_counts
    )
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    data["router_logits"] = jax.device_put(custom_logits, NamedSharding(mesh, P("tensor", None)))

    with jax.set_mesh(mesh):
        topk_layer = TopK(topk=TOP_K, renormalize=True)
        fused_layer = FusedEPMoE(
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            ep_size=EP_SIZE,
            mesh=mesh,
            intermediate_dim=INTERMEDIATE_SIZE,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            renormalize_topk_logits=True,
        )
        fn, kwargs = _make_jit_fn(topk_layer, fused_layer)

        # Warmup
        for _ in range(WARMUP):
            out = fn(data["tokens"], data["router_logits"], **kwargs)
            jax.block_until_ready(out)

        # Wall-clock timing
        times = []
        for _ in range(PROFILE_STEPS):
            t0 = time.perf_counter()
            out = fn(data["tokens"], data["router_logits"], **kwargs)
            jax.block_until_ready(out)
            times.append((time.perf_counter() - t0) * 1000)

        median = sorted(times)[len(times) // 2]
        print(f"  Wall-clock: median={median:.3f}ms  times={[f'{t:.3f}' for t in times]}")

        # Capture xprof trace
        scenario_trace_dir = os.path.join(trace_dir, f"nt{num_tokens}")
        os.makedirs(scenario_trace_dir, exist_ok=True)
        print(f"  Capturing xprof trace to {scenario_trace_dir} ...")

        with jax.profiler.trace(scenario_trace_dir):
            for i in range(PROFILE_STEPS):
                with jax.profiler.StepTraceAnnotation(f"fused_moe_nt{num_tokens}", step_num=i):
                    out = fn(data["tokens"], data["router_logits"], **kwargs)
                    jax.block_until_ready(out)

        print("  Trace saved.")
        return median


def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    num_devices = len(jax.devices())
    trace_dir = os.environ.get("PROFILE_DIR", "/gcs/prayer/moe_profiling/20260408_xprof")

    print("=" * 80)
    print("FusedEPMoE Profiling (xprof trace capture)")
    print("=" * 80)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print(f"  EP: {EP_SIZE}, warmup: {WARMUP}, profile_steps: {PROFILE_STEPS}")
    print(f"  Trace dir: {trace_dir}")
    print("=" * 80)

    for nt in [256, 1024, 16384]:
        print(f"\n--- num_tokens={nt} ---")
        profile_scenario(nt, trace_dir)

    print("\nAll traces captured. Use xprof to analyze.")


if __name__ == "__main__":
    main()
