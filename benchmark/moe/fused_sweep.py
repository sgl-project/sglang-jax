"""FusedEPMoE block config sweep — find the best bt/bf/bd1 for v6e-16.

Tests multiple block configurations to optimize FusedEPMoE performance.
Key hypothesis: larger bt reduces barrier/metadata overhead per token.
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
    prepare_fused_moe_inputs,
)
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048

WARMUP = 3
BENCH_STEPS = 10


def _make_jit_fn(topk_layer, moe_layer, block_config=None):
    topk_def, topk_state = nnx.split(topk_layer)
    topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)
    moe_def, moe_state = nnx.split(moe_layer)
    moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)

    @jax.jit(static_argnames=("topk_treedef", "moe_treedef", "block_config"))
    def fn(hidden, logits, *, topk_treedef, topk_leaves, moe_treedef, moe_leaves, block_config):
        topk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_treedef, topk_leaves))
        moe = nnx.merge(moe_def, jax.tree_util.tree_unflatten(moe_treedef, moe_leaves))
        w, ids = topk(logits)
        return moe(hidden, w, ids, block_config=block_config)

    kwargs = dict(
        topk_treedef=topk_treedef,
        topk_leaves=topk_leaves,
        moe_treedef=moe_treedef,
        moe_leaves=moe_leaves,
        block_config=block_config,
    )
    return fn, kwargs


def bench_fused_epmoe_with_config(ep_size, num_tokens, block_config):
    """Benchmark FusedEPMoE with a specific block config."""
    tp_size = 1
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)

    case = MoEBenchmarkCase(
        name="fused_sweep",
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
        fn, kwargs = _make_jit_fn(topk_layer, fused_layer, block_config=block_config)

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
    median = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    t_min, t_max = min(times), max(times)
    speedup = ""
    if baseline_median is not None and baseline_median > 0:
        ratio = (baseline_median - median) / baseline_median * 100
        speedup = f"{ratio:+.1f}%"
    print(
        f"  {name:<55} | {median:>9.3f}ms {mean:>9.3f}ms {t_min:>9.3f}ms {t_max:>9.3f}ms | {speedup:>8}"
    )
    return median


def main():
    num_devices = len(jax.devices())
    ep_size = 16
    print("=" * 120)
    print("FusedEPMoE Block Config Sweep")
    print("=" * 120)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print(f"  EP: {ep_size}, warmup: {WARMUP}, bench: {BENCH_STEPS}")
    print("=" * 120)

    output_dir = os.environ.get("PROFILE_DIR", "/tmp/fused_sweep")
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # Configs to sweep: larger bt = fewer outer loops = fewer barriers
    configs = [
        # Default fallback (baseline)
        (
            "default_bt32",
            FusedMoEBlockConfig(
                bt=32, bf=512, bd1=1024, bd2=1024, btc=32, bfc=512, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        # Larger bt
        (
            "bt64",
            FusedMoEBlockConfig(
                bt=64, bf=512, bd1=1024, bd2=1024, btc=64, bfc=512, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        (
            "bt128",
            FusedMoEBlockConfig(
                bt=128, bf=512, bd1=1024, bd2=1024, btc=128, bfc=512, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        (
            "bt256",
            FusedMoEBlockConfig(
                bt=256, bf=512, bd1=1024, bd2=1024, btc=256, bfc=512, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        # Larger bf (intermediate tile)
        (
            "bt64_bf1024",
            FusedMoEBlockConfig(
                bt=64, bf=1024, bd1=1024, bd2=1024, btc=64, bfc=1024, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        (
            "bt128_bf1024",
            FusedMoEBlockConfig(
                bt=128,
                bf=1024,
                bd1=1024,
                bd2=1024,
                btc=128,
                bfc=1024,
                bd1c=1024,
                bd2c=1024,
                bse=512,
            ),
        ),
        # bf=2048 (full intermediate dim in one tile — avoids activation recomputation)
        (
            "bt64_bf2048",
            FusedMoEBlockConfig(
                bt=64, bf=2048, bd1=1024, bd2=1024, btc=64, bfc=1024, bd1c=1024, bd2c=1024, bse=512
            ),
        ),
        (
            "bt128_bf2048",
            FusedMoEBlockConfig(
                bt=128,
                bf=2048,
                bd1=1024,
                bd2=1024,
                btc=128,
                bfc=1024,
                bd1c=1024,
                bd2c=1024,
                bse=512,
            ),
        ),
        # Auto-tuned (None = use tuned_block_configs table, fallback to default)
        ("auto", None),
    ]

    for scenario_name, num_tokens in [("nt16384", 16384), ("nt1024", 1024)]:
        if ep_size > num_devices:
            print(f"\n[SKIP] needs {ep_size} devices, have {num_devices}")
            continue

        print(f"\n{'─' * 120}")
        print(f"Scenario: ep{ep_size}_tp1_{scenario_name} (tokens={num_tokens})")
        print(f"{'─' * 120}")
        header = f"  {'config':<55} | {'median':>10} {'mean':>10} {'min':>10} {'max':>10} | {'vs base':>8}"
        print(header)
        print(f"  {'-' * 115}")

        baseline_median = None

        for config_name, block_config in configs:
            try:
                times = bench_fused_epmoe_with_config(ep_size, num_tokens, block_config)
                med = report_times(f"FusedEPMoE {config_name}", times, baseline_median)
                if baseline_median is None:
                    baseline_median = med
                all_results.append((scenario_name, config_name, med))
            except Exception as e:
                print(f"  FusedEPMoE {config_name} ERROR: {e}")
                all_results.append((scenario_name, config_name, -1))

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    for scenario, config, median in all_results:
        status = f"{median:.3f}ms" if median > 0 else "ERROR"
        print(f"  {scenario:<20} {config:<30} {status}")

    result_file = os.path.join(output_dir, "fused_sweep_results.txt")
    with open(result_file, "w") as f:
        f.write("scenario,config,median_ms\n")
        for scenario, config, median in all_results:
            f.write(f"{scenario},{config},{median:.3f}\n")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
