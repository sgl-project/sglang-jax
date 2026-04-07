"""
Capture xprof traces for EPMoE vs FusedEPMoE on MiMoV2Flash MoE dimensions.

Saves traces to /tmp/moe_profiles/ for extraction via kubectl cp or GCS upload.
"""

from __future__ import annotations

import os

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
PROFILE_DIR = os.environ.get("PROFILE_DIR", "/gcs/moe_profiles")

# Which configs to profile: (backend, ep_size, tp_size, num_tokens)
# num_tokens: 16384 = prefill (16k input), 1024 = decode (1k output)
PROFILE_CASES = [
    ("epmoe", 1, 16, 1024),
    ("epmoe", 8, 2, 1024),
    ("epmoe", 16, 1, 1024),
    ("fused", 16, 1, 1024),
    ("epmoe", 1, 16, 16384),
    ("epmoe", 16, 1, 16384),
    ("fused", 16, 1, 16384),
]


def profile_epmoe(ep_size: int, tp_size: int, num_tokens: int, trace_dir: str) -> None:
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)
    tokens = jnp.empty((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16)
    router_logits = generate_router_logits(
        num_tokens,
        NUM_EXPERTS,
        "balanced",
        num_experts_per_tok=TOP_K,
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

        kwargs = dict(
            topk_treedef=topk_treedef,
            topk_leaves=topk_leaves,
            moe_treedef=moe_treedef,
            moe_leaves=moe_leaves,
        )

        # Warmup (compile)
        out = fn(tokens, router_logits, **kwargs)
        jax.block_until_ready(out)
        print("    warmup done")

        # Profile
        with jax.profiler.trace(trace_dir):
            for i in range(5):
                out = fn(tokens, router_logits, **kwargs)
                jax.block_until_ready(out)
        print(f"    trace saved to {trace_dir}")


def profile_fused(ep_size: int, num_tokens: int, trace_dir: str) -> None:
    case = MoEBenchmarkCase(
        name="fused_profile",
        num_tokens=num_tokens,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )
    mesh = build_mesh(ep_size=ep_size, tp_size=1)

    data = prepare_fused_moe_inputs(
        case,
        weight_dtype=jnp.bfloat16,
        mesh=mesh,
        include_weights=False,
    )
    target_counts = MoEImbalanceSimulator.generate_counts(
        num_tokens,
        TOP_K,
        NUM_EXPERTS,
        mode="balanced",
    )
    custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
        num_tokens,
        NUM_EXPERTS,
        TOP_K,
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
        topk_layer = TopK(topk=TOP_K, renormalize=True)

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

        kwargs = dict(
            moe_treedef=moe_treedef,
            moe_leaves=moe_leaves,
            topk_treedef=topk_treedef,
            topk_leaves=topk_leaves,
        )

        # Warmup
        out = fn(data["tokens"], data["router_logits"], **kwargs)
        jax.block_until_ready(out)
        print("    warmup done")

        # Profile
        with jax.profiler.trace(trace_dir):
            for i in range(5):
                out = fn(data["tokens"], data["router_logits"], **kwargs)
                jax.block_until_ready(out)
        print(f"    trace saved to {trace_dir}")


def main():
    num_devices = len(jax.devices())
    print(f"MoE xprof profiling: {num_devices} x {jax.devices()[0].device_kind}")
    os.makedirs(PROFILE_DIR, exist_ok=True)

    for backend, ep, tp, nt in PROFILE_CASES:
        tag = f"{backend}_ep{ep}_tp{tp}_nt{nt}"
        trace_dir = os.path.join(PROFILE_DIR, tag)
        print(f"\n[{tag}]")

        if backend == "epmoe":
            profile_epmoe(ep, tp, nt, trace_dir)
        else:
            profile_fused(ep, nt, trace_dir)

    # List output
    print("\n=== Profiles saved ===")
    for d in sorted(os.listdir(PROFILE_DIR)):
        full = os.path.join(PROFILE_DIR, d)
        if os.path.isdir(full):
            files = []
            for root, _, fnames in os.walk(full):
                files.extend(fnames)
            total_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, fnames in os.walk(full)
                for f in fnames
            ) / (1024 * 1024)
            print(f"  {d}: {len(files)} files, {total_mb:.1f} MB")

    print(f"\n=== Done. Traces saved to {PROFILE_DIR} ===")


if __name__ == "__main__":
    main()
