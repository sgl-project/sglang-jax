"""Benchmark serial vs ping-pong DSA indexer decode scoring.

Each active sequence contributes one complete ``[1, score_size]`` score tile.
The serial variant reproduces the pre-pipeline decode loop; the pipeline
variant calls the production implementation and overlaps scoring sequence
``n + 1`` with selecting sequence ``n``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.kernels.radix_topk.bench_dsa_indexer_topk import (
    BenchmarkShape,
    _bytes_by_input,
    _latency_stats,
    _make_inputs,
    _validate_output,
)
from sgl_jax.srt.kernels.dsa.ref import (
    _NEG_INF,
    _mask_and_compact_topk_indices,
    score_and_select_index_tokens,
)
from sgl_jax.srt.kernels.dsa.topk import select_indexer_topk


def _make_serial_run(shape: BenchmarkShape, topk_impl: str):
    """Reproduce the original score-then-select decode loop."""

    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        q_idx = inputs["q_idx"]
        idx_weights = inputs["idx_weights"].astype(jnp.float32)
        index_key_cache = inputs["index_key_cache"]
        seq_lens = inputs["seq_lens"]
        page_indices = inputs["page_indices"]
        cu_kv_lens = inputs["cu_kv_lens"]
        active_num_seqs = jnp.clip(inputs["distribution"][2], 0, shape.num_seqs)

        page_size = index_key_cache.shape[1]
        max_kv = shape.score_size
        kv_pos = jnp.arange(max_kv, dtype=jnp.int32)
        out = jnp.full((shape.num_seqs, shape.topk), -1, dtype=jnp.int32)

        def body(seq_id, serial_out):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices,
                cu_kv_lens[seq_id] // page_size,
                shape.pages_per_seq,
            )
            seq_keys = index_key_cache[seq_pages].reshape(max_kv, shape.head_dim)
            q_i = jax.lax.dynamic_slice_in_dim(q_idx, seq_id, 1, axis=0)
            weights_i = jax.lax.dynamic_slice_in_dim(idx_weights, seq_id, 1, axis=0)

            with jax.named_scope("dsa_indexer_decode_serial_score"):
                dots = jnp.einsum(
                    "thd,kd->thk",
                    q_i,
                    seq_keys,
                    preferred_element_type=jnp.float32,
                )
                scores = jnp.einsum(
                    "th,thk->tk",
                    weights_i,
                    jax.nn.relu(dots),
                )
                scores = jnp.where(kv_pos[None, :] < kv_len, scores, _NEG_INF)

            with jax.named_scope("dsa_indexer_decode_serial_topk"):
                values, indices = select_indexer_topk(
                    scores,
                    k=shape.topk,
                    implementation=topk_impl,
                )
                indices = _mask_and_compact_topk_indices(values, indices)

            return jax.lax.dynamic_update_slice_in_dim(
                serial_out,
                indices,
                seq_id,
                axis=0,
            )

        return jax.lax.fori_loop(0, active_num_seqs, body, out)

    return jax.jit(run)


def _make_pipeline_run(shape: BenchmarkShape, topk_impl: str):
    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        return score_and_select_index_tokens(
            inputs["q_idx"],
            inputs["idx_weights"],
            inputs["index_key_cache"],
            inputs["seq_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["cu_kv_lens"],
            inputs["distribution"],
            k=shape.topk,
            pages_per_seq=shape.pages_per_seq,
            one_token_per_seq=True,
            topk_impl=topk_impl,
        )

    return jax.jit(run)


def _compile_and_warmup(name, run, inputs, warmup_iters):
    start = time.perf_counter()
    output = run(inputs)
    jax.block_until_ready(output)
    compile_first_run_ms = (time.perf_counter() - start) * 1000.0
    print(f"{name}: compile + first run = {compile_first_run_ms:.2f} ms")

    for _ in range(warmup_iters):
        output = run(inputs)
        jax.block_until_ready(output)
    return output, compile_first_run_ms


def _time_once(run, inputs) -> float:
    start = time.perf_counter()
    output = run(inputs)
    jax.block_until_ready(output)
    return (time.perf_counter() - start) * 1000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=131072)
    parser.add_argument("--score-size", type=int, default=135168)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--topk-impl", choices=("exact_lax", "radix"), default="radix")
    parser.add_argument("--q-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trace-dir", type=pathlib.Path)
    parser.add_argument(
        "--profile-variant",
        choices=("serial", "pipeline", "both"),
        default="pipeline",
    )
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--summary-output", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_seqs < 1 or args.kv_len < 1 or args.score_size < args.kv_len:
        raise SystemExit(
            "num_seqs/kv_len must be positive and score_size must be >= kv_len"
        )
    if args.score_size % args.page_size:
        raise SystemExit("score_size must be divisible by page_size")
    if args.warmup < 0 or args.iters < 1 or args.profile_iters < 1:
        raise SystemExit(
            "warmup must be non-negative and iters/profile-iters must be positive"
        )

    shape = BenchmarkShape(
        num_seqs=args.num_seqs,
        prefix_len=args.kv_len - 1,
        extend_len=1,
        score_size=args.score_size,
        page_size=args.page_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        topk=args.topk,
    )
    shape.validate()

    devices = jax.local_devices()
    if not 0 <= args.device_index < len(devices):
        raise SystemExit(f"device-index must be in [0, {len(devices)})")
    device = devices[args.device_index]
    if args.topk_impl == "radix" and device.platform != "tpu":
        raise SystemExit("radix top-k requires a TPU")
    q_dtype = jnp.float32 if args.q_dtype == "float32" else jnp.bfloat16

    print(f"JAX {jax.__version__} | device={device}")
    print(
        f"decode shape: sequences={shape.num_seqs} tile=[1,{shape.score_size}] "
        f"logical_kv={shape.kv_len} H={shape.num_heads} D={shape.head_dim} "
        f"topk={shape.topk} impl={args.topk_impl}"
    )
    inputs = _make_inputs(shape, device, args.seed, q_dtype)
    input_bytes = _bytes_by_input(inputs)
    print(
        f"inputs={sum(input_bytes.values()) / (1 << 20):.1f} MiB "
        f"cache={input_bytes['index_key_cache'] / (1 << 20):.1f} MiB"
    )

    serial_run = _make_serial_run(shape, args.topk_impl)
    pipeline_run = _make_pipeline_run(shape, args.topk_impl)
    serial_output, serial_compile_ms = _compile_and_warmup(
        "serial", serial_run, inputs, args.warmup
    )
    pipeline_output, pipeline_compile_ms = _compile_and_warmup(
        "pipeline", pipeline_run, inputs, args.warmup
    )

    serial_host = np.asarray(serial_output)
    pipeline_host = np.asarray(pipeline_output)
    np.testing.assert_array_equal(
        np.sort(serial_host, axis=-1),
        np.sort(pipeline_host, axis=-1),
    )
    validation = _validate_output(pipeline_output, shape)
    print("validation: serial and pipeline select the same exact top-k set")

    latencies = {"serial": [], "pipeline": []}
    for iteration in range(args.iters):
        order = (
            (("serial", serial_run), ("pipeline", pipeline_run))
            if iteration % 2 == 0
            else (("pipeline", pipeline_run), ("serial", serial_run))
        )
        for name, run in order:
            latencies[name].append(_time_once(run, inputs))

    serial_stats = _latency_stats(latencies["serial"])
    pipeline_stats = _latency_stats(latencies["pipeline"])
    speedup = serial_stats["mean_ms"] / pipeline_stats["mean_ms"]
    reduction = 1.0 - pipeline_stats["mean_ms"] / serial_stats["mean_ms"]
    print(
        f"serial: mean={serial_stats['mean_ms']:.3f} ms "
        f"p50={serial_stats['p50_ms']:.3f} ms p95={serial_stats['p95_ms']:.3f} ms"
    )
    print(
        f"pipeline: mean={pipeline_stats['mean_ms']:.3f} ms "
        f"p50={pipeline_stats['p50_ms']:.3f} ms p95={pipeline_stats['p95_ms']:.3f} ms"
    )
    print(f"speedup={speedup:.3f}x latency_reduction={reduction:.2%}")

    trace_info = None
    if args.trace_dir is not None:
        profile_runs = {
            "serial": serial_run,
            "pipeline": pipeline_run,
        }
        selected_variants = (
            tuple(profile_runs)
            if args.profile_variant == "both"
            else (args.profile_variant,)
        )
        trace_info = {}
        for variant in selected_variants:
            variant_trace_dir = (
                args.trace_dir / variant
                if args.profile_variant == "both"
                else args.trace_dir
            )
            variant_trace_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"capturing {args.profile_iters} warmed {variant} iterations "
                f"to {variant_trace_dir}"
            )
            with jax.profiler.trace(str(variant_trace_dir)):
                for step in range(args.profile_iters):
                    with jax.profiler.StepTraceAnnotation(
                        f"dsa_indexer_decode_{variant}",
                        step_num=step,
                    ):
                        profile_output = profile_runs[variant](inputs)
                        jax.block_until_ready(profile_output)

            xplanes = sorted(variant_trace_dir.glob("plugins/profile/**/*.xplane.pb"))
            trace_jsons = sorted(
                variant_trace_dir.glob("plugins/profile/**/*.trace.json.gz")
            )
            if not xplanes:
                raise RuntimeError(f"no .xplane.pb found under {variant_trace_dir}")
            if not trace_jsons:
                raise RuntimeError(f"no .trace.json.gz found under {variant_trace_dir}")
            trace_info[variant] = {
                "trace_dir": str(variant_trace_dir.resolve()),
                "profile_iters": args.profile_iters,
                "xplane_count": len(xplanes),
                "trace_json_count": len(trace_jsons),
                "xplane_bytes": sum(path.stat().st_size for path in xplanes),
                "trace_json_bytes": sum(path.stat().st_size for path in trace_jsons),
            }
            print(
                f"{variant} trace: {len(xplanes)} xplane, "
                f"{len(trace_jsons)} trace.json.gz"
            )

    common = {
        "device": str(device),
        "jax_version": jax.__version__,
        "num_seqs": shape.num_seqs,
        "kv_len": shape.kv_len,
        "score_size": shape.score_size,
        "num_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "topk": shape.topk,
        "topk_impl": args.topk_impl,
        "q_dtype": args.q_dtype,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
    }
    records = [
        {
            **common,
            "variant": "serial",
            "compile_first_run_ms": serial_compile_ms,
            "latency_ms": serial_stats["mean_ms"],
            "p50_latency_ms": serial_stats["p50_ms"],
            "p95_latency_ms": serial_stats["p95_ms"],
            "latencies_ms": latencies["serial"],
        },
        {
            **common,
            "variant": "pipeline",
            "compile_first_run_ms": pipeline_compile_ms,
            "latency_ms": pipeline_stats["mean_ms"],
            "p50_latency_ms": pipeline_stats["p50_ms"],
            "p95_latency_ms": pipeline_stats["p95_ms"],
            "latencies_ms": latencies["pipeline"],
            "speedup_vs_serial": speedup,
            "latency_reduction_vs_serial": reduction,
        },
    ]
    summary = {
        "shape": common,
        "input_bytes": input_bytes,
        "validation": validation,
        "serial": serial_stats,
        "pipeline": pipeline_stats,
        "speedup_vs_serial": speedup,
        "latency_reduction_vs_serial": reduction,
        "traces": trace_info,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"metrics: {args.output.resolve()}")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"summary: {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
