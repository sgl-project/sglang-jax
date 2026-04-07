"""Verify tuned block configs are picked up and perform as expected."""

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


def bench(num_tokens):
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    case = MoEBenchmarkCase(
        name="verify",
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
        # block_config=None -> uses tuned_block_configs table lookup
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
    print("FusedEPMoE Tuned Config Verification")
    print("=" * 80)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, H: {HIDDEN_SIZE}, I: {INTERMEDIATE_SIZE}")
    print(f"  EP: {EP_SIZE}, warmup: {WARMUP}, bench: {BENCH_STEPS}")
    print("=" * 80)

    for nt, expected_ms in [(1024, 1.4), (16384, 9.0)]:
        print(f"\n--- num_tokens={nt} (expected < {expected_ms}ms) ---")
        times = bench(nt)
        median = sorted(times)[len(times) // 2]
        mean = sum(times) / len(times)
        t_min, t_max = min(times), max(times)
        status = "PASS" if median < expected_ms else "REGRESSED"
        print(
            f"  median={median:.3f}ms  mean={mean:.3f}ms  min={t_min:.3f}ms  max={t_max:.3f}ms  [{status}]"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
