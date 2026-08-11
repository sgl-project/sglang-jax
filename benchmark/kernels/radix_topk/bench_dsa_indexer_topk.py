"""Standalone GLM-5.2 DSA indexer score + top-k microbenchmark.

The default shape models two independent sequences with a 128K cached prefix
and a 1K extend each. It invokes only
``compute_scores_and_select_topk_indices``: no
model weights, server, HTTP traffic, prefix warmup, or MLA attention.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.indexer import (
    _INDEXER_QUERY_BLOCK_SIZE,
    compute_scores_and_select_topk_indices,
)


@dataclass(frozen=True)
class BenchmarkShape:
    num_seqs: int
    prefix_len: int
    extend_len: int
    score_size: int
    page_size: int
    num_heads: int
    head_dim: int
    topk: int

    @property
    def query_tokens(self) -> int:
        return self.num_seqs * self.extend_len

    @property
    def kv_len(self) -> int:
        return self.prefix_len + self.extend_len

    @property
    def pages_per_seq(self) -> int:
        return self.score_size // self.page_size

    def validate(self) -> None:
        positive_fields = {
            "num_seqs": self.num_seqs,
            "prefix_len": self.prefix_len,
            "extend_len": self.extend_len,
            "score_size": self.score_size,
            "page_size": self.page_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "topk": self.topk,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.score_size % self.page_size:
            raise ValueError(
                f"score_size={self.score_size} must be divisible by page_size={self.page_size}"
            )
        if self.score_size < self.kv_len:
            raise ValueError(
                f"score_size={self.score_size} is smaller than logical kv_len={self.kv_len}"
            )
        if self.topk > self.score_size:
            raise ValueError(f"topk={self.topk} exceeds score_size={self.score_size}")


def _make_inputs(
    shape: BenchmarkShape,
    device: jax.Device,
    seed: int,
    q_dtype: jnp.dtype,
) -> dict[str, jax.Array]:
    """Create production-like inputs on one selected device.

    Each sequence owns a distinct physical page range. Index-key BF16 matches
    the production cache; query dtype is selectable to measure the current FP32
    Hadamard output against an explicitly quantized BF16 scorer input.
    """

    key_q, key_w, key_cache = jax.random.split(jax.random.key(seed), 3)
    total_pages = shape.num_seqs * shape.pages_per_seq

    with jax.default_device(device):
        q_idx = jax.random.normal(
            key_q,
            (shape.query_tokens, shape.num_heads, shape.head_dim),
            dtype=jnp.float32,
        ).astype(q_dtype)
        idx_weights = jax.random.normal(
            key_w,
            (shape.query_tokens, shape.num_heads),
            dtype=jnp.float32,
        )
        index_key_cache = jax.random.normal(
            key_cache,
            (total_pages, shape.page_size, shape.head_dim),
            dtype=jnp.bfloat16,
        )
        seq_lens = jnp.full((shape.num_seqs,), shape.kv_len, dtype=jnp.int32)
        page_indices = jnp.arange(total_pages, dtype=jnp.int32)
        cu_q_lens = jnp.arange(shape.num_seqs + 1, dtype=jnp.int32) * shape.extend_len
        # The DSA ABI uses page-aligned KV capacity here as the packed
        # page_indices stride, rather than the logical sequence length.
        cu_kv_lens = jnp.arange(shape.num_seqs + 1, dtype=jnp.int32) * shape.score_size
        distribution = jnp.asarray([0, shape.num_seqs, shape.num_seqs], dtype=jnp.int32)

    inputs = {
        "q_idx": q_idx,
        "idx_weights": idx_weights,
        "index_key_cache": index_key_cache,
        "seq_lens": seq_lens,
        "page_indices": page_indices,
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "distribution": distribution,
    }
    jax.block_until_ready(inputs)
    return inputs


def _make_run(
    shape: BenchmarkShape,
    topk_impl: str,
    score_query_block_size: int,
):
    def run(inputs: dict[str, jax.Array]) -> jax.Array:
        with jax.named_scope("dsa_indexer_topk_microbench"):
            return compute_scores_and_select_topk_indices(
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
                one_token_per_seq=False,
                topk_impl=topk_impl,
                score_query_block_size=score_query_block_size,
            )

    return jax.jit(run)


def _validate_output(output: jax.Array, shape: BenchmarkShape) -> dict[str, int | bool]:
    local_q_pos = jnp.tile(jnp.arange(shape.extend_len, dtype=jnp.int32), shape.num_seqs)
    causal_max = shape.prefix_len + local_q_pos
    entry_valid = (output == -1) | ((output >= 0) & (output <= causal_max[:, None]))
    expected_valid_per_row = jnp.minimum(shape.topk, causal_max + 1)
    valid_per_row = (output >= 0).sum(axis=1)
    counts_match = jnp.all(valid_per_row == expected_valid_per_row)
    all_entries_valid = jnp.all(entry_valid)
    minimum = output.min()
    maximum = output.max()
    all_entries_valid, counts_match, minimum, maximum = jax.device_get(
        (all_entries_valid, counts_match, minimum, maximum)
    )
    if not bool(all_entries_valid):
        raise AssertionError("top-k output contains an out-of-range or non-causal index")
    if not bool(counts_match):
        raise AssertionError("top-k output has an unexpected number of valid indices")
    return {
        "all_entries_valid": bool(all_entries_valid),
        "valid_counts_match": bool(counts_match),
        "min_index": int(minimum),
        "max_index": int(maximum),
    }


def _bytes_by_input(inputs: dict[str, jax.Array]) -> dict[str, int]:
    return {name: int(array.size * array.dtype.itemsize) for name, array in inputs.items()}


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=131072)
    parser.add_argument("--extend-len", type=int, default=1024)
    parser.add_argument(
        "--score-size",
        type=int,
        default=135168,
        help="padded KV width used by scoring; production 128K bucket is 135168",
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--score-query-block-size",
        type=int,
        default=_INDEXER_QUERY_BLOCK_SIZE,
    )
    parser.add_argument(
        "--q-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="indexer query dtype at the scorer boundary",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--topk-impl",
        choices=("approx", "exact_lax", "radix"),
        default="radix",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3, help="post-compilation warmup iterations")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--trace-dir",
        type=pathlib.Path,
        help="optional XProf output root; warmup and timing remain outside the trace",
    )
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, help="optional JSON result path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.profile_iters <= 0:
        raise SystemExit("--warmup must be >= 0 and --iters/--profile-iters must be positive")

    shape = BenchmarkShape(
        num_seqs=args.num_seqs,
        prefix_len=args.prefix_len,
        extend_len=args.extend_len,
        score_size=args.score_size,
        page_size=args.page_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        topk=args.topk,
    )
    try:
        shape.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    devices = jax.local_devices()
    if not 0 <= args.device_index < len(devices):
        raise SystemExit(
            f"--device-index={args.device_index} is out of range for {len(devices)} local devices"
        )
    device = devices[args.device_index]
    if args.topk_impl == "radix" and device.platform != "tpu":
        raise SystemExit("--topk-impl=radix requires a TPU; use exact_lax for a CPU smoke test")
    if args.score_query_block_size <= 0:
        raise SystemExit("--score-query-block-size must be positive")
    q_dtype = jnp.float32 if args.q_dtype == "float32" else jnp.bfloat16

    print(f"JAX {jax.__version__} | selected device {args.device_index}: {device}")
    print(
        "scenario: "
        f"{shape.num_seqs} sequences x ({shape.prefix_len} prefix + "
        f"{shape.extend_len} extend), T={shape.query_tokens}, "
        f"logical_kv={shape.kv_len}, padded_score={shape.score_size}"
    )
    print(
        f"indexer: q=[{shape.query_tokens},{shape.num_heads},{shape.head_dim}] "
        f"cache=[{shape.num_seqs * shape.pages_per_seq},{shape.page_size},{shape.head_dim}] "
        f"topk={shape.topk} impl={args.topk_impl} "
        f"query_block={args.score_query_block_size} q_dtype={args.q_dtype}"
    )

    inputs = _make_inputs(shape, device, args.seed, q_dtype)
    input_bytes = _bytes_by_input(inputs)
    print(
        f"inputs ready: {sum(input_bytes.values()) / (1 << 20):.1f} MiB "
        f"(cache={input_bytes['index_key_cache'] / (1 << 20):.1f} MiB)"
    )
    run = _make_run(
        shape,
        args.topk_impl,
        args.score_query_block_size,
    )

    compile_start = time.perf_counter()
    output = run(inputs)
    jax.block_until_ready(output)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    print(f"compile + first run: {compile_ms:.2f} ms")

    for _ in range(args.warmup):
        output = run(inputs)
        jax.block_until_ready(output)

    latencies_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        output = run(inputs)
        jax.block_until_ready(output)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    validation = _validate_output(output, shape)
    stats = _latency_stats(latencies_ms)
    print(
        "latency: "
        f"mean={stats['mean_ms']:.3f} ms p50={stats['p50_ms']:.3f} ms "
        f"p95={stats['p95_ms']:.3f} ms min={stats['min_ms']:.3f} ms "
        f"max={stats['max_ms']:.3f} ms"
    )
    print(
        "validation: causal/range=ok valid-count=ok "
        f"index-range=[{validation['min_index']}, {validation['max_index']}]"
    )

    trace_dir = None
    if args.trace_dir is not None:
        trace_dir = args.trace_dir.expanduser().resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"capturing {args.profile_iters} compiled iterations to {trace_dir}")
        with jax.profiler.trace(str(trace_dir)):
            for step in range(args.profile_iters):
                with jax.profiler.StepTraceAnnotation("dsa_indexer_topk", step_num=step):
                    output = run(inputs)
                    jax.block_until_ready(output)
        xplanes = sorted(trace_dir.glob("plugins/profile/**/*.xplane.pb"))
        if not xplanes:
            raise RuntimeError(f"profile completed but no .xplane.pb was found under {trace_dir}")
        print(f"profile: {len(xplanes)} xplane file(s)")

    result = {
        "device": str(device),
        "jax_version": jax.__version__,
        "shape": asdict(shape),
        "query_tokens": shape.query_tokens,
        "logical_kv_len": shape.kv_len,
        "pages_per_seq": shape.pages_per_seq,
        "topk_impl": args.topk_impl,
        "score_query_block_size": args.score_query_block_size,
        "q_dtype": args.q_dtype,
        "input_bytes": input_bytes,
        "compile_first_run_ms": compile_ms,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "latency": stats,
        "latencies_ms": latencies_ms,
        "validation": validation,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
    }
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"result: {output_path}")


if __name__ == "__main__":
    main()
