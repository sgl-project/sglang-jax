"""
Capture xprof traces for FusedEPMoE on MiMoV2Flash MoE dimensions.

Supports bf16, FP8 sub-channel, and FP8 2D block-wise quantization.
Saves traces to PROFILE_DIR for analysis via ant-profiler.
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
    prepare_fused_moe_inputs,
)
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK
from sgl_jax.srt.utils.quantization.quantization_config import QuantizationConfig

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
PROFILE_DIR = os.environ.get("PROFILE_DIR", "/gcs/moe_profiles")

# (tag, ep_size, num_tokens, weight_dtype, subc_quant_wsz, quant_block_n)
PROFILE_CASES = [
    # bf16 baseline
    ("fused_bf16", 16, 64, None, None, None),
    ("fused_bf16", 16, 8192, None, None, None),
    # FP8 sub-channel (wsz=128)
    ("fused_fp8_subchan128", 16, 64, jnp.float8_e4m3fn, 128, None),
    ("fused_fp8_subchan128", 16, 8192, jnp.float8_e4m3fn, 128, None),
    # FP8 2D block-wise (block_k=128, block_n=128)
    ("fused_fp8_block128", 16, 64, jnp.float8_e4m3fn, 128, 128),
    ("fused_fp8_block128", 16, 8192, jnp.float8_e4m3fn, 128, 128),
]


def profile_fused(
    ep_size: int,
    num_tokens: int,
    trace_dir: str,
    weight_dtype=None,
    subc_quant_wsz: int | None = None,
    quant_block_n: int | None = None,
) -> None:
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

    quantization_config = None
    if weight_dtype is not None:
        quantization_config = QuantizationConfig(
            moe_weight_dtype=weight_dtype,
            moe_activation_dtype=None,
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
            quantization_config=quantization_config,
        )

        if quantization_config is not None:
            if subc_quant_wsz is not None:
                fused_layer.subc_quant_wsz = subc_quant_wsz
            if quant_block_n is not None:
                fused_layer.quant_block_n = quant_block_n
            fused_layer.quantize_weights()

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

    for tag, ep, nt, w_dtype, wsz, block_n in PROFILE_CASES:
        full_tag = f"{tag}_ep{ep}_nt{nt}"
        trace_dir = os.path.join(PROFILE_DIR, full_tag)
        quant_info = ""
        if w_dtype is not None:
            if block_n is not None:
                quant_info = f", 2D block-wise (bk={wsz}, bn={block_n})"
            else:
                quant_info = f", sub-channel (wsz={wsz})"
        print(f"\n[{full_tag}] tokens={nt}, ep={ep}{quant_info}")

        profile_fused(
            ep, nt, trace_dir, weight_dtype=w_dtype, subc_quant_wsz=wsz, quant_block_n=block_n
        )

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
