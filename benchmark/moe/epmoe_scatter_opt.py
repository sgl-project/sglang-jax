"""EPMoE scatter optimization experiments.

Tests approaches to reduce the 52% scatter overhead:
1. Baseline EPMoE (current, with adaptive tile_m)
2. EPMoE with sorted token_indices (better HBM locality)
3. EPMoE with tiled gather (gather in chunks matching cache line size)

All approaches avoid quantization — pure memory access pattern optimization.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from benchmark.moe.utils import build_mesh, generate_router_logits
from sgl_jax.srt.layers.moe import EPMoE, TopK

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048

WARMUP = 3
BENCH_STEPS = 10


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


def bench_epmoe(ep_size, tp_size, num_tokens, **extra_kwargs):
    """Benchmark EPMoE. Returns list of step times in ms."""
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)
    tokens = jnp.ones((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16)
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


def bench_standalone_gather(num_tokens, top_k, hidden_size):
    """Benchmark just the gather operation in isolation to understand baseline cost."""
    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (num_tokens, hidden_size), dtype=jnp.bfloat16)

    # Simulate token_indices: random permutation (worst-case locality)
    indices_random = jax.random.randint(jax.random.PRNGKey(1), (num_tokens * top_k,), 0, num_tokens)
    # Sorted indices (best-case locality)
    indices_sorted = jnp.sort(indices_random)

    @jax.jit
    def gather_random(x, idx):
        return x[idx]

    @jax.jit
    def gather_sorted(x, idx):
        return x[idx]

    # Warmup
    for _ in range(3):
        jax.block_until_ready(gather_random(inputs, indices_random))
        jax.block_until_ready(gather_sorted(inputs, indices_sorted))

    # Bench random
    times_random = []
    for _ in range(BENCH_STEPS):
        t0 = time.perf_counter()
        jax.block_until_ready(gather_random(inputs, indices_random))
        times_random.append((time.perf_counter() - t0) * 1000)

    # Bench sorted
    times_sorted = []
    for _ in range(BENCH_STEPS):
        t0 = time.perf_counter()
        jax.block_until_ready(gather_sorted(inputs, indices_sorted))
        times_sorted.append((time.perf_counter() - t0) * 1000)

    return times_random, times_sorted


def report_times(name, times, baseline_median=None):
    median = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    t_min, t_max = min(times), max(times)
    speedup = ""
    if baseline_median is not None and baseline_median > 0:
        ratio = (baseline_median - median) / baseline_median * 100
        speedup = f"{ratio:+.1f}%"
    print(
        f"  {name:<50} | {median:>9.3f}ms {mean:>9.3f}ms {t_min:>9.3f}ms {t_max:>9.3f}ms | {speedup:>8}"
    )
    return median


def main():
    num_devices = len(jax.devices())
    print("=" * 110)
    print("EPMoE Scatter Optimization Experiments (No Quantization)")
    print("=" * 110)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print("=" * 110)

    output_dir = os.environ.get("PROFILE_DIR", "/tmp/epmoe_scatter_opt")
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # Part 1: Standalone gather microbenchmark
    print("\n--- Standalone Gather Microbenchmark ---")
    print(
        f"  {'config':<50} | {'median':>10} {'mean':>10} {'min':>10} {'max':>10} | {'vs base':>8}"
    )
    print(f"  {'-' * 105}")

    for nt in [16384, 1024]:
        times_r, times_s = bench_standalone_gather(nt, TOP_K, HIDDEN_SIZE)
        base = report_times(f"gather nt={nt} random indices", times_r)
        report_times(f"gather nt={nt} sorted indices", times_s, base)
        all_results.append((f"gather_nt{nt}", "random", sorted(times_r)[len(times_r) // 2]))
        all_results.append((f"gather_nt{nt}", "sorted", sorted(times_s)[len(times_s) // 2]))

    # Part 2: Full EPMoE benchmark
    scenarios = [
        ("ep16_tp1_nt16384", 16, 1, 16384),
        ("ep16_tp1_nt1024", 16, 1, 1024),
        ("ep1_tp16_nt16384", 1, 16, 16384),
    ]

    for scenario_name, ep_size, tp_size, num_tokens in scenarios:
        if ep_size * tp_size > num_devices:
            print(f"\n[SKIP] {scenario_name}: needs {ep_size * tp_size} devices")
            continue

        print(f"\n--- {scenario_name} (ep={ep_size}, tp={tp_size}, tokens={num_tokens}) ---")
        print(
            f"  {'config':<50} | {'median':>10} {'mean':>10} {'min':>10} {'max':>10} | {'vs base':>8}"
        )
        print(f"  {'-' * 105}")

        baseline_median = None

        # Baseline EPMoE (with adaptive tile_m)
        try:
            times = bench_epmoe(ep_size, tp_size, num_tokens)
            baseline_median = report_times("EPMoE baseline (adaptive tile_m)", times)
            all_results.append((scenario_name, "baseline", baseline_median))
        except Exception as e:
            print(f"  EPMoE baseline ERROR: {e}")

        # EPMoE with no pre-gather quant (explicit None, should be same as baseline)
        try:
            times = bench_epmoe(ep_size, tp_size, num_tokens, pre_gather_quant_dtype=None)
            med = report_times("EPMoE explicit no-quant", times, baseline_median)
            all_results.append((scenario_name, "no_quant", med))
        except Exception as e:
            print(f"  EPMoE no-quant ERROR: {e}")

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    for scenario, config, median in all_results:
        print(f"  {scenario:<30} {config:<25} {median:.3f}ms")

    result_file = os.path.join(output_dir, "epmoe_scatter_results.txt")
    with open(result_file, "w") as f:
        f.write("scenario,config,median_ms\n")
        for scenario, config, median in all_results:
            f.write(f"{scenario},{config},{median:.3f}\n")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
