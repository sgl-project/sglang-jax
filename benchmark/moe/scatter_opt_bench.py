"""EPMoE vs FusedEPMoE comparison benchmark + scatter optimization experiments.

Compares:
  1. EPMoE baseline (indexed_gmm + gate-up fusion + adaptive tiling)
  2. FusedEPMoE (Pallas all-to-all kernel)
  3. EPMoE with int8 pre-gather (halves gather bandwidth)

All configs use MiMoV2Flash shapes: 256 experts, top_k=8, H=4096, I=2048.
Wall-clock timing: 3 warmup + 10 bench iterations.
"""

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
    generate_router_logits,
    prepare_fused_moe_inputs,
)
from sgl_jax.srt.layers.moe import EPMoE, FusedEPMoE, TopK

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048

WARMUP = 3
BENCH_STEPS = 10


def _make_jit_fn(topk_layer, moe_layer):
    """Create a JIT-compiled function for MoE inference."""
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


def bench_epmoe(ep_size, tp_size, num_tokens, **extra_kwargs):
    """Benchmark EPMoE with given config. Returns list of step times in ms."""
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)
    tokens = jnp.empty((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16)
    router_logits = generate_router_logits(
        num_tokens, NUM_EXPERTS, "balanced", num_experts_per_tok=TOP_K
    ).astype(jnp.bfloat16)

    with jax.set_mesh(mesh):
        topk_layer = TopK(topk=TOP_K, renormalize=True)
        moe_layer = EPMoE(
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=INTERMEDIATE_SIZE,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            **extra_kwargs,
        )
        fn, kwargs = _make_jit_fn(topk_layer, moe_layer)

        for _ in range(WARMUP):
            out = fn(tokens, router_logits, **kwargs)
            jax.block_until_ready(out)

        times = []
        for _ in range(BENCH_STEPS):
            t0 = time.perf_counter()
            out = fn(tokens, router_logits, **kwargs)
            jax.block_until_ready(out)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_fused_epmoe(ep_size, num_tokens):
    """Benchmark FusedEPMoE with given config. Returns list of step times in ms."""
    tp_size = 1  # FusedEPMoE shards experts across all devices
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)

    case = MoEBenchmarkCase(
        name="fused_bench",
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
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=INTERMEDIATE_SIZE,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            renormalize_topk_logits=True,
        )
        fn, kwargs = _make_jit_fn(topk_layer, fused_layer)

        for _ in range(WARMUP):
            out = fn(data["tokens"], data["router_logits"], **kwargs)
            jax.block_until_ready(out)

        times = []
        for _ in range(BENCH_STEPS):
            t0 = time.perf_counter()
            out = fn(data["tokens"], data["router_logits"], **kwargs)
            jax.block_until_ready(out)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def report_times(name, times, baseline_median=None):
    """Print timing stats for a benchmark run."""
    median = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    t_min, t_max = min(times), max(times)
    speedup = ""
    if baseline_median is not None and baseline_median > 0:
        ratio = (baseline_median - median) / baseline_median * 100
        speedup = f"{ratio:+.1f}%"
    print(
        f"  {name:<45} | {median:>9.3f}ms {mean:>9.3f}ms {t_min:>9.3f}ms {t_max:>9.3f}ms | {speedup:>8}"
    )
    return median


def main():
    num_devices = len(jax.devices())
    print("=" * 110)
    print("EPMoE Scatter Optimization Experiments")
    print("=" * 110)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print(f"  warmup: {WARMUP}, bench: {BENCH_STEPS}")
    print("=" * 110)

    output_dir = os.environ.get("PROFILE_DIR", "/tmp/scatter_opt")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "results.txt")
    all_results = []

    # Scenarios to test
    scenarios = [
        ("ep16_tp1_nt16384", 16, 1, 16384),
        ("ep16_tp1_nt1024", 16, 1, 1024),
    ]

    for scenario_name, ep_size, tp_size, num_tokens in scenarios:
        if ep_size * tp_size > num_devices:
            print(
                f"\n[SKIP] {scenario_name}: needs {ep_size * tp_size} devices, have {num_devices}"
            )
            continue

        print(f"\n{'─' * 110}")
        print(f"Scenario: {scenario_name} (ep={ep_size}, tp={tp_size}, tokens={num_tokens})")
        print(f"{'─' * 110}")
        header = f"  {'config':<45} | {'median':>10} {'mean':>10} {'min':>10} {'max':>10} | {'vs base':>8}"
        print(header)
        print(f"  {'-' * 105}")

        baseline_median = None

        # 1. EPMoE baseline
        try:
            times = bench_epmoe(ep_size, tp_size, num_tokens)
            baseline_median = report_times("EPMoE (baseline)", times)
            all_results.append((scenario_name, "epmoe_baseline", baseline_median))
        except Exception as e:
            print(f"  EPMoE baseline ERROR: {e}")

        # 2. FusedEPMoE
        try:
            fused_ep = ep_size * tp_size  # FusedEPMoE uses all devices as EP
            times = bench_fused_epmoe(fused_ep, num_tokens)
            med = report_times(f"FusedEPMoE (ep={fused_ep})", times, baseline_median)
            all_results.append((scenario_name, "fused_epmoe", med))
        except Exception as e:
            print(f"  FusedEPMoE ERROR: {e}")

        # 3. EPMoE with int8 pre-gather quantization
        try:
            times = bench_epmoe(ep_size, tp_size, num_tokens, pre_gather_quant_dtype=jnp.int8)
            med = report_times("EPMoE + int8 pre-gather", times, baseline_median)
            all_results.append((scenario_name, "epmoe_int8_pregather", med))
        except Exception as e:
            print(f"  EPMoE int8 pre-gather ERROR: {e}")

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    for scenario, config, median in all_results:
        print(f"  {scenario:<30} {config:<30} {median:.3f}ms")

    # Write results
    with open(result_file, "w") as f:
        f.write("scenario,config,median_ms\n")
        for scenario, config, median in all_results:
            f.write(f"{scenario},{config},{median:.3f}\n")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
