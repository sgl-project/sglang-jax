"""Profile the two matrix products used by the DSA decode score kernel.

The first microkernel computes one ``[H, D] @ [K, D].T`` score tile.
The second consumes that materialized ``[H, K]`` tile and reduces heads with
``[H] @ [H, K]``.  The intermediate HBM traffic is therefore part of these
microbenchmarks; use XProf's custom-call duration to study each Pallas kernel,
not their sum as a prediction of the fused production scorer.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _first_dot_kernel(q_ref, keys_ref, similarities_out_ref):
    similarities_out_ref[0] = lax.dot_general(
        q_ref[0],
        keys_ref[0],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=("interpret",))
def first_dot_pallas(q, keys, *, interpret=False):
    """Compute one BF16/FP32-accumulating score tile per sequence."""
    num_seqs, num_heads, head_dim = q.shape
    block_k = keys.shape[1]
    q_spec = pl.BlockSpec(
        (1, num_heads, head_dim),
        lambda seq_id: (seq_id, 0, 0),
    )
    keys_spec = pl.BlockSpec(
        (1, block_k, head_dim),
        lambda seq_id: (seq_id, 0, 0),
    )
    output_spec = pl.BlockSpec(
        (1, num_heads, block_k),
        lambda seq_id: (seq_id, 0, 0),
    )
    return pl.pallas_call(
        _first_dot_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (num_seqs, num_heads, block_k),
            jnp.float32,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_seqs,),
            in_specs=[q_spec, keys_spec],
            out_specs=output_spec,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="dsa_score_first_dot",
    )(q, keys)


def _head_reduction_kernel(weights_ref, similarities_ref, scores_out_ref):
    scores_out_ref[0, 0] = lax.dot_general(
        weights_ref[0, 0].astype(jnp.float32),
        jnp.maximum(similarities_ref[0], jnp.float32(0.0)),
        dimension_numbers=(((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=("interpret",))
def head_reduction_pallas(weights, similarities, *, interpret=False):
    """Apply ReLU and reduce one materialized ``[H, K]`` tile per sequence."""
    num_seqs, num_heads, block_k = similarities.shape
    weights_3d = weights[:, None, :]
    weights_spec = pl.BlockSpec(
        (1, 1, num_heads),
        lambda seq_id: (seq_id, 0, 0),
    )
    similarities_spec = pl.BlockSpec(
        (1, num_heads, block_k),
        lambda seq_id: (seq_id, 0, 0),
    )
    output_spec = pl.BlockSpec(
        (1, 1, block_k),
        lambda seq_id: (seq_id, 0, 0),
    )
    scores = pl.pallas_call(
        _head_reduction_kernel,
        out_shape=jax.ShapeDtypeStruct((num_seqs, 1, block_k), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_seqs,),
            in_specs=[weights_spec, similarities_spec],
            out_specs=output_spec,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="dsa_score_head_reduction",
    )(weights_3d, similarities)
    return scores[:, 0, :]


def _latency_stats(samples_ms):
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "latency_ms": float(samples.mean()),
        "p50_latency_ms": float(np.percentile(samples, 50)),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "latencies_ms": samples.tolist(),
    }


def _time_once(run):
    start = time.perf_counter()
    jax.block_until_ready(run())
    return (time.perf_counter() - start) * 1e3


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=2048)
    parser.add_argument(
        "--first-dot-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--stage",
        choices=("first_dot", "head_reduction", "both"),
        default="both",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--trace-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.num_seqs < 1 or args.num_heads < 1:
        raise SystemExit("num_seqs and num_heads must be positive")
    if args.head_dim % 128 or args.block_k % 128:
        raise SystemExit("head_dim and block_k must be multiples of 128")
    if args.warmup < 0 or args.iters < 1 or args.profile_iters < 1:
        raise SystemExit("warmup must be non-negative and iteration counts positive")

    devices = jax.devices()
    if args.device_index < 0 or args.device_index >= len(devices):
        raise SystemExit(f"device-index must be in [0, {len(devices)})")
    device = devices[args.device_index]
    interpret = device.platform != "tpu"
    q_dtype = jnp.float32 if args.first_dot_dtype == "float32" else jnp.bfloat16
    key_q, key_k, key_w = jax.random.split(jax.random.key(args.seed), 3)
    with jax.default_device(device):
        q = jax.random.normal(
            key_q,
            (args.num_seqs, args.num_heads, args.head_dim),
            dtype=jnp.float32,
        ).astype(q_dtype)
        keys = jax.random.normal(
            key_k,
            (args.num_seqs, args.block_k, args.head_dim),
            dtype=jnp.bfloat16,
        )
        weights = jax.random.normal(
            key_w,
            (args.num_seqs, args.num_heads),
            dtype=jnp.float32,
        )

    def first_dot_run():
        return first_dot_pallas(q, keys, interpret=interpret)

    similarities = first_dot_run()
    jax.block_until_ready(similarities)

    def head_reduction_run():
        return head_reduction_pallas(
            weights,
            similarities,
            interpret=interpret,
        )

    runs = {
        "first_dot": first_dot_run,
        "head_reduction": head_reduction_run,
    }
    selected = tuple(runs) if args.stage == "both" else (args.stage,)

    first_dot_expected = jnp.einsum(
        "shd,skd->shk",
        q,
        keys,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(similarities),
        np.asarray(first_dot_expected),
    )
    reduction_output = head_reduction_run()
    reduction_expected = jnp.einsum(
        "sh,shk->sk",
        weights,
        jax.nn.relu(first_dot_expected),
    )
    np.testing.assert_array_equal(
        np.asarray(reduction_output),
        np.asarray(reduction_expected),
    )

    records = []
    for stage in selected:
        run = runs[stage]
        for _ in range(args.warmup):
            jax.block_until_ready(run())
        samples = [_time_once(run) for _ in range(args.iters)]
        stats = _latency_stats(samples)
        record = {
            "variant": stage,
            "jax_version": jax.__version__,
            "device": str(device),
            "num_seqs": args.num_seqs,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "block_k": args.block_k,
            "first_dot_dtype": args.first_dot_dtype,
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            **stats,
        }
        records.append(record)
        print(
            f"{stage}: mean={stats['latency_ms']:.6f} ms "
            f"p50={stats['p50_latency_ms']:.6f} ms "
            f"p95={stats['p95_latency_ms']:.6f} ms"
        )

        if args.trace_dir is not None:
            stage_trace_dir = args.trace_dir / stage
            stage_trace_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(stage_trace_dir)):
                for _ in range(args.profile_iters):
                    jax.block_until_ready(run())

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
