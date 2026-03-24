"""
Benchmark tpu-inference's fused_ep_moe kernel for comparison with sglang-jax.

Usage (via launcher on TPU pod):
    python3 -u /tmp/launcher.py scripts/gke_tpu7x/bench_tpu_inference_moe.py \
        --num-experts 8 --top-k 2 --hidden-size 2048 --intermediate-size 512 \
        --num-tokens 64 128 256 --iters 5 --warmup-iters 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark.utils import multiple_iteration_timeit_from_trace
from scripts.gke_tpu7x.tpu_inference_fused_moe.kernel import fused_ep_moe

proc = jax.process_index()
is_main = proc == 0


def log(msg):
    if is_main:
        print(msg, flush=True)


def build_mesh_for_tpu_inference(ep_size: int) -> Mesh:
    """Build a 2D mesh with axes ('data', 'model') for tpu-inference.

    The kernel hardcodes axis_index("data") for DP and uses ep_axis_name="model" for EP.
    """
    devices = jax.devices()[:ep_size]
    mesh_shape = (1, ep_size)
    device_array = np.array(devices).reshape(mesh_shape)
    return Mesh(device_array, axis_names=("data", "model"))


def prepare_tpu_inference_inputs(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    mesh: Mesh,
    ep_axis_name: str = "model",
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Create inputs in tpu-inference format.

    Key differences from sglang-jax:
    - w1: (E, 2, D, F) — gate and up projections fused along dim 1
    - w2: (E, F, D)
    - gating_output: (num_tokens, num_experts)
    """
    ep_size = mesh.shape[ep_axis_name]

    tokens_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    logits_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    # tpu-inference expects w1 sharded on expert axis
    w1_sharding = NamedSharding(mesh, P(ep_axis_name, None, None, None))
    w2_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))

    tokens = jax.jit(
        lambda: jnp.zeros((num_tokens, hidden_size), dtype=dtype),
        out_shardings=tokens_sharding,
    )()

    # w1: (num_experts, 2, hidden_size, intermediate_size) — gate+up fused
    w1 = jax.jit(
        lambda: jnp.zeros(
            (num_experts, 2, hidden_size, intermediate_size), dtype=dtype
        ),
        out_shardings=w1_sharding,
    )()

    # w2: (num_experts, intermediate_size, hidden_size)
    w2 = jax.jit(
        lambda: jnp.zeros(
            (num_experts, intermediate_size, hidden_size), dtype=dtype
        ),
        out_shardings=w2_sharding,
    )()

    # gating_output: (num_tokens, num_experts)
    gating_output = jax.jit(
        lambda: jnp.zeros((num_tokens, num_experts), dtype=dtype),
        out_shardings=logits_sharding,
    )()

    return tokens, w1, w2, gating_output


def run_all(
    iters: int,
    warmup_iters: int = 2,
    num_tokens_list: list[int] | None = None,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    act_fn: str = "silu",
):
    if num_tokens_list is None:
        num_tokens_list = [64, 128, 256, 512, 1024]

    num_devices = len(jax.devices())
    ep_size = min(num_devices, num_experts)
    # Find largest valid ep_size
    for ep in range(ep_size, 0, -1):
        if num_experts % ep == 0:
            ep_size = ep
            break

    log(f"=== tpu-inference fused_ep_moe benchmark ===")
    log(f"  devices={num_devices}, ep_size={ep_size}")
    log(
        f"  num_experts={num_experts}, top_k={top_k}, "
        f"hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
        f"act_fn={act_fn}"
    )
    log(f"  iters={iters}, warmup={warmup_iters}")

    mesh = build_mesh_for_tpu_inference(ep_size)
    ep_axis_name = "model"

    results: list[tuple[int, float]] = []

    for num_tokens in num_tokens_list:
        if num_tokens % ep_size != 0:
            log(f"\n[SKIP] num_tokens={num_tokens} not divisible by ep_size={ep_size}")
            continue

        log(f"\n[case] num_tokens={num_tokens}")

        tokens, w1, w2, gating_output = prepare_tpu_inference_inputs(
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh=mesh,
            ep_axis_name=ep_axis_name,
        )

        @partial(jax.jit)
        def compute(tokens, w1, w2, gating_output):
            return fused_ep_moe(
                mesh=mesh,
                tokens=tokens,
                w1=w1,
                w2=w2,
                gating_output=gating_output,
                top_k=top_k,
                renormalize_topk_logits=True,
                act_fn=act_fn,
                scoring_fn="softmax",
                ep_axis_name=ep_axis_name,
            )

        task = "tpu-inference-fused-moe"

        try:
            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: compute(tokens, w1, w2, gating_output),
                data_generator=lambda: (),
                task=task,
                tries=iters,
                warmup=warmup_iters,
            )
        except Exception as e:
            log(f"  ERROR: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            continue

        if len(times) > 1:
            times = times[1:]
        mean_ms = float(np.mean(times)) if times else float("nan")
        log(f"  tpu-inference fused_ep_moe: {mean_ms:.3f} ms | samples={times}")
        results.append((num_tokens, mean_ms))

    log(f"\n=== Summary ===")
    log(f"{'num_tokens':>12} | {'mean_ms':>10}")
    log(f"{'-'*12}-+-{'-'*10}")
    for nt, ms in results:
        log(f"{nt:>12} | {ms:>10.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark tpu-inference fused_ep_moe")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--act-fn", type=str, default="silu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        num_tokens_list=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        act_fn=args.act_fn,
    )
