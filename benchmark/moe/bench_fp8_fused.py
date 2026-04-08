"""Benchmark FusedEPMoE with fp8 weights vs bf16 weights on v6e."""

from __future__ import annotations

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


def bench_one(num_tokens, weight_dtype, quantize_to_fp8=False):
    """Benchmark a single config. Returns list of step times in ms."""
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    case = MoEBenchmarkCase(
        name="fp8_bench",
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
            weight_dtype=weight_dtype,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            renormalize_topk_logits=True,
        )

        if quantize_to_fp8:
            # Manually quantize weights to fp8 with sub-channel scales
            fused_layer.quantized_dtype = jnp.float8_e4m3fn
            fused_layer.quantize_weights()
            print(
                f"    Quantized weights to fp8. w1 dtype={fused_layer.w1.value.dtype}, "
                f"w1_scale shape={fused_layer.w1_scale.value.shape}"
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


def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    num_devices = len(jax.devices())
    print("=" * 80)
    print("FusedEPMoE: bf16 vs fp8 weight benchmark")
    print("=" * 80)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print(f"  EP: {EP_SIZE}, warmup: {WARMUP}, bench: {BENCH_STEPS}")
    print("=" * 80)

    token_sizes = [256, 1024, 4096, 16384]

    # Phase 1: bf16 baseline
    print("\n### Phase 1: bf16 weights (baseline) ###")
    bf16_results = {}
    for nt in token_sizes:
        print(f"\n  nt={nt} bf16...")
        times = bench_one(nt, weight_dtype=jnp.bfloat16, quantize_to_fp8=False)
        median = sorted(times)[len(times) // 2]
        bf16_results[nt] = median
        print(f"    median={median:.3f}ms  times={[f'{t:.3f}' for t in times]}")

    # Phase 2: fp8 weights (quantized from bf16)
    print("\n### Phase 2: fp8 weights (quantized) ###")
    fp8_results = {}
    for nt in token_sizes:
        print(f"\n  nt={nt} fp8...")
        times = bench_one(nt, weight_dtype=jnp.bfloat16, quantize_to_fp8=True)
        median = sorted(times)[len(times) // 2]
        fp8_results[nt] = median
        print(f"    median={median:.3f}ms  times={[f'{t:.3f}' for t in times]}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: bf16 vs fp8 weight")
    print("=" * 80)
    print(f"{'nt':>8s}  {'bf16(ms)':>10s}  {'fp8(ms)':>10s}  {'speedup':>10s}")
    print("-" * 45)
    for nt in token_sizes:
        bf16_ms = bf16_results[nt]
        fp8_ms = fp8_results[nt]
        speedup = bf16_ms / fp8_ms
        print(f"{nt:8d}  {bf16_ms:10.3f}  {fp8_ms:10.3f}  {speedup:9.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
