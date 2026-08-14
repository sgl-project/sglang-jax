"""Benchmark gathered-JAX vs paged-Pallas DSA indexer decode scoring + top-k.

Each active sequence contributes one complete ``[1, score_size]`` score tile.
The serial variant gathers, scores, and selects each sequence independently.
On TPU, the batched variant calls the production paged-cache Pallas scorer and
then selects the complete ``[batch, score_size]`` matrix in one top-k invocation.
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
from sgl_jax.srt.kernels.dsa.indexer import _NEG_INF, _select_topk_indices
from sgl_jax.srt.kernels.dsa.paged_score import paged_decode_scores_pallas


def _make_gathered_score_run(shape: BenchmarkShape):
    """Materialize each sequence's keys, then compute its complete score row."""

    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        q_idx = inputs["q_idx"]
        idx_weights = inputs["idx_weights"].astype(jnp.float32)
        index_key_cache = inputs["index_key_cache"]
        seq_lens = inputs["seq_lens"]
        page_indices = inputs["page_indices"]
        cu_kv_lens = inputs["cu_kv_lens"]
        active_num_seqs = jnp.clip(inputs["distribution"][2], 0, shape.num_seqs)
        page_size = index_key_cache.shape[1]
        positions = jnp.arange(shape.score_size, dtype=jnp.int32)
        scores = jnp.full(
            (shape.num_seqs, shape.score_size),
            _NEG_INF,
            dtype=jnp.float32,
        )

        def body(seq_id, output):
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices,
                cu_kv_lens[seq_id] // page_size,
                shape.pages_per_seq,
            )
            seq_keys = index_key_cache[seq_pages].reshape(
                shape.score_size,
                shape.head_dim,
            )
            similarities = jnp.einsum(
                "hd,kd->hk",
                q_idx[seq_id],
                seq_keys,
                preferred_element_type=jnp.float32,
            )
            row = jnp.einsum(
                "h,hk->k",
                idx_weights[seq_id],
                jax.nn.relu(similarities),
            )
            row = jnp.where(positions < seq_lens[seq_id], row, _NEG_INF)
            return output.at[seq_id].set(row)

        return jax.lax.fori_loop(0, active_num_seqs, body, scores)

    return jax.jit(run)


def _make_paged_score_run(
    shape: BenchmarkShape,
    *,
    block_k: int,
    first_dot_bf16: bool,
    persistent_two_seq: bool,
    coalesce_page_dma: bool,
    interpret: bool,
):
    """Compute the score matrix with the direct paged-cache Pallas kernel."""

    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        return paged_decode_scores_pallas(
            inputs["q_idx"],
            inputs["idx_weights"],
            inputs["index_key_cache"],
            inputs["seq_lens"],
            inputs["page_indices"],
            inputs["cu_kv_lens"],
            inputs["distribution"],
            pages_per_seq=shape.pages_per_seq,
            block_k=block_k,
            first_dot_bf16=first_dot_bf16,
            persistent_two_seq=persistent_two_seq,
            coalesce_page_dma=coalesce_page_dma,
            interpret=interpret,
        )

    return jax.jit(run)


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
                indices = _select_topk_indices(
                    scores,
                    jnp.reshape(kv_len, (1,)),
                    k=shape.topk,
                    topk_impl=topk_impl,
                )

            return jax.lax.dynamic_update_slice_in_dim(
                serial_out,
                indices,
                seq_id,
                axis=0,
            )

        return jax.lax.fori_loop(0, active_num_seqs, body, out)

    return jax.jit(run)


def _make_batched_run(
    shape: BenchmarkShape,
    topk_impl: str,
    *,
    block_k: int,
    first_dot_bf16: bool,
    persistent_two_seq: bool,
    coalesce_page_dma: bool,
    interpret: bool,
):
    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        scores = paged_decode_scores_pallas(
            inputs["q_idx"],
            inputs["idx_weights"],
            inputs["index_key_cache"],
            inputs["seq_lens"],
            inputs["page_indices"],
            inputs["cu_kv_lens"],
            inputs["distribution"],
            pages_per_seq=shape.pages_per_seq,
            block_k=block_k,
            first_dot_bf16=first_dot_bf16,
            persistent_two_seq=persistent_two_seq,
            coalesce_page_dma=coalesce_page_dma,
            interpret=interpret,
        )
        return _select_topk_indices(
            scores,
            inputs["seq_lens"],
            k=shape.topk,
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
    parser.add_argument("--block-k", type=int, default=22528)
    parser.add_argument(
        "--first-dot-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--score-scheduler",
        choices=("independent", "persistent_two_seq"),
        default="persistent_two_seq",
    )
    parser.add_argument(
        "--page-dma",
        choices=("per_page", "coalesce_contiguous"),
        default="coalesce_contiguous",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trace-dir", type=pathlib.Path)
    parser.add_argument(
        "--profile-variant",
        choices=("gathered_score", "paged_score", "serial", "batched", "all"),
        default="batched",
    )
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--summary-output", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_seqs < 1 or args.kv_len < 1 or args.score_size < args.kv_len:
        raise SystemExit("num_seqs/kv_len must be positive and score_size must be >= kv_len")
    if args.score_size % args.page_size:
        raise SystemExit("score_size must be divisible by page_size")
    if args.block_k < 128 or args.block_k % 128:
        raise SystemExit("block_k must be a positive multiple of 128")
    if args.warmup < 0 or args.iters < 1 or args.profile_iters < 1:
        raise SystemExit("warmup must be non-negative and iters/profile-iters must be positive")

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
    first_dot_bf16 = args.first_dot_dtype == "bfloat16"
    persistent_two_seq = args.score_scheduler == "persistent_two_seq"
    coalesce_page_dma = args.page_dma == "coalesce_contiguous"

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

    gathered_score_run = _make_gathered_score_run(shape)
    paged_score_run = _make_paged_score_run(
        shape,
        block_k=args.block_k,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        coalesce_page_dma=coalesce_page_dma,
        interpret=device.platform != "tpu",
    )
    serial_run = _make_serial_run(shape, args.topk_impl)
    batched_run = _make_batched_run(
        shape,
        args.topk_impl,
        block_k=args.block_k,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        coalesce_page_dma=coalesce_page_dma,
        interpret=device.platform != "tpu",
    )
    gathered_score_output, gathered_score_compile_ms = _compile_and_warmup(
        "gathered_score",
        gathered_score_run,
        inputs,
        args.warmup,
    )
    paged_score_output, paged_score_compile_ms = _compile_and_warmup(
        "paged_score",
        paged_score_run,
        inputs,
        args.warmup,
    )
    serial_output, serial_compile_ms = _compile_and_warmup(
        "serial", serial_run, inputs, args.warmup
    )
    batched_output, batched_compile_ms = _compile_and_warmup(
        "batched", batched_run, inputs, args.warmup
    )

    gathered_score_host = np.asarray(gathered_score_output)
    paged_score_host = np.asarray(paged_score_output)
    finite = np.isfinite(gathered_score_host)
    gathered_finite = gathered_score_host[finite]
    paged_finite = paged_score_host[finite]
    score_max_abs_error = float(np.max(np.abs(gathered_finite - paged_finite)))
    score_allclose = bool(
        np.allclose(
            paged_finite,
            gathered_finite,
            rtol=1e-4,
            atol=1e-4,
        )
    )
    print(
        "score diagnostics: "
        f"paged_nan={np.count_nonzero(np.isnan(paged_score_host))} "
        f"paged_finite={np.count_nonzero(np.isfinite(paged_score_host))} "
        f"max_abs_error={score_max_abs_error:.6g} "
        f"allclose={score_allclose}"
    )

    serial_host = np.asarray(serial_output)
    batched_host = np.asarray(batched_output)
    serial_sorted = np.sort(serial_host, axis=-1)
    batched_sorted = np.sort(batched_host, axis=-1)
    topk_exact_rows = int(np.count_nonzero(np.all(serial_sorted == batched_sorted, axis=-1)))
    topk_intersections = [
        int(np.intersect1d(serial_host[row], batched_host[row]).size)
        for row in range(shape.num_seqs)
    ]
    topk_recall = float(np.mean(topk_intersections) / shape.topk)
    print(
        "top-k diagnostics: "
        f"exact_rows={topk_exact_rows}/{shape.num_seqs} "
        f"intersection={topk_intersections} recall={topk_recall:.8f}"
    )
    try:
        validation = _validate_output(batched_output, shape)
        validation["passed"] = True
        print(
            "validation: batched indices are valid; "
            f"score max_abs_error={score_max_abs_error:.6g}"
        )
    except AssertionError as error:
        validation = {"passed": False, "error": str(error)}
        print(f"validation failed: {error}")

    timed_runs = {
        "gathered_score": gathered_score_run,
        "paged_score": paged_score_run,
        "serial": serial_run,
        "batched": batched_run,
    }
    latencies = {name: [] for name in timed_runs}
    for iteration in range(args.iters):
        names = tuple(timed_runs)
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            latencies[name].append(_time_once(timed_runs[name], inputs))

    gathered_score_stats = _latency_stats(latencies["gathered_score"])
    paged_score_stats = _latency_stats(latencies["paged_score"])
    serial_stats = _latency_stats(latencies["serial"])
    batched_stats = _latency_stats(latencies["batched"])
    score_speedup = gathered_score_stats["mean_ms"] / paged_score_stats["mean_ms"]
    speedup = serial_stats["mean_ms"] / batched_stats["mean_ms"]
    reduction = 1.0 - batched_stats["mean_ms"] / serial_stats["mean_ms"]
    print(
        f"gathered_score: mean={gathered_score_stats['mean_ms']:.3f} ms "
        f"p50={gathered_score_stats['p50_ms']:.3f} ms "
        f"p95={gathered_score_stats['p95_ms']:.3f} ms"
    )
    print(
        f"paged_score: mean={paged_score_stats['mean_ms']:.3f} ms "
        f"p50={paged_score_stats['p50_ms']:.3f} ms "
        f"p95={paged_score_stats['p95_ms']:.3f} ms "
        f"speedup={score_speedup:.3f}x"
    )
    print(
        f"serial: mean={serial_stats['mean_ms']:.3f} ms "
        f"p50={serial_stats['p50_ms']:.3f} ms p95={serial_stats['p95_ms']:.3f} ms"
    )
    print(
        f"batched: mean={batched_stats['mean_ms']:.3f} ms "
        f"p50={batched_stats['p50_ms']:.3f} ms p95={batched_stats['p95_ms']:.3f} ms"
    )
    print(f"speedup={speedup:.3f}x latency_reduction={reduction:.2%}")

    trace_info = None
    if args.trace_dir is not None:
        profile_runs = {
            "gathered_score": gathered_score_run,
            "paged_score": paged_score_run,
            "serial": serial_run,
            "batched": batched_run,
        }
        selected_variants = (
            tuple(profile_runs) if args.profile_variant == "all" else (args.profile_variant,)
        )
        trace_info = {}
        for variant in selected_variants:
            variant_trace_dir = (
                args.trace_dir / variant if args.profile_variant == "all" else args.trace_dir
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
            trace_jsons = sorted(variant_trace_dir.glob("plugins/profile/**/*.trace.json.gz"))
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
            print(f"{variant} trace: {len(xplanes)} xplane, " f"{len(trace_jsons)} trace.json.gz")

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
        "block_k": args.block_k,
        "first_dot_dtype": args.first_dot_dtype,
        "score_scheduler": args.score_scheduler,
        "page_dma": args.page_dma,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
    }
    records = [
        {
            **common,
            "variant": "gathered_score",
            "compile_first_run_ms": gathered_score_compile_ms,
            "latency_ms": gathered_score_stats["mean_ms"],
            "p50_latency_ms": gathered_score_stats["p50_ms"],
            "p95_latency_ms": gathered_score_stats["p95_ms"],
            "latencies_ms": latencies["gathered_score"],
        },
        {
            **common,
            "variant": "paged_score",
            "compile_first_run_ms": paged_score_compile_ms,
            "latency_ms": paged_score_stats["mean_ms"],
            "p50_latency_ms": paged_score_stats["p50_ms"],
            "p95_latency_ms": paged_score_stats["p95_ms"],
            "latencies_ms": latencies["paged_score"],
            "speedup_vs_gathered_score": score_speedup,
            "score_max_abs_error": score_max_abs_error,
            "score_allclose": score_allclose,
        },
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
            "variant": "batched",
            "compile_first_run_ms": batched_compile_ms,
            "latency_ms": batched_stats["mean_ms"],
            "p50_latency_ms": batched_stats["p50_ms"],
            "p95_latency_ms": batched_stats["p95_ms"],
            "latencies_ms": latencies["batched"],
            "speedup_vs_serial": speedup,
            "latency_reduction_vs_serial": reduction,
            "topk_exact_rows": topk_exact_rows,
            "topk_intersections": topk_intersections,
            "topk_recall": topk_recall,
        },
    ]
    summary = {
        "shape": common,
        "input_bytes": input_bytes,
        "validation": {
            **validation,
            "score_max_abs_error": score_max_abs_error,
            "score_allclose": score_allclose,
            "topk_exact_rows": topk_exact_rows,
            "topk_intersections": topk_intersections,
            "topk_recall": topk_recall,
        },
        "gathered_score": gathered_score_stats,
        "paged_score": paged_score_stats,
        "score_speedup": score_speedup,
        "serial": serial_stats,
        "batched": batched_stats,
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
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"summary: {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
