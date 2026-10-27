"""Compare token-wise KV updates with the former page-sized scratch policy.

Run on a single TPU device, without model weights:
    python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
        --heads 1 8 32 --tokens 16 128 1024 --page-sizes 128 256 \
        --output /tmp/kv-slice-results.jsonl

Optional --slices values sweep alternative token-wise block sizes. Each result
includes an exact full-cache check, cold compilation time, and device timings
from the XLA module track. Cache donation matches model execution and avoids
timing an unrelated copy of the entire cache. This is not a serving benchmark.
"""

import argparse
import gzip
import importlib.metadata
import itertools
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    VMEM_HEADROOM_BYTES,
    VMEM_SIZE,
    get_num_slices_per_block,
    get_slot_mapping,
    kv_cache_update_impl,
)


def make_update(slices_per_block, max_slice_size):
    def update(new_kv, loc, cache):
        mapping = get_slot_mapping(
            slices_per_block,
            jnp.maximum(loc, 0),
            jnp.arange(loc.shape[0], dtype=jnp.int32),
            (loc != -1).astype(jnp.int32),
        )
        return kv_cache_update_impl(
            new_kv,
            mapping,
            cache,
            jnp.array([loc.shape[0]], dtype=jnp.int32),
            page_size=max_slice_size,
            num_slices_per_block=slices_per_block,
        )

    return jax.jit(update, donate_argnums=(2,))


def device_times(trace_dir, iterations):
    trace_file = next(Path(trace_dir).glob("plugins/profile/*/*.trace.json.gz"))
    with gzip.open(trace_file, "rt") as f:
        events = json.load(f)["traceEvents"]
    module_tracks = {
        (e["pid"], e["tid"])
        for e in events
        if e.get("name") == "thread_name" and e["args"]["name"] == "XLA Modules"
    }
    durations = [
        float(e["args"]["device_duration_ps"]) / 1e6
        for e in events
        if (e.get("pid"), e.get("tid")) in module_tracks
        and "jit_update" in e.get("name", "")
        and "device_duration_ps" in e.get("args", {})
    ]
    if len(durations) != iterations:
        raise RuntimeError(
            f"Expected {iterations} TPU module timings, found {len(durations)}; "
            "host timings cannot substitute for device measurements"
        )
    return durations


def benchmark_case(args, heads, tokens, page_size, head_dim):
    dtype = jnp.dtype(args.dtype)
    packing = 4 // dtype.itemsize
    capacity = max(args.cache_tokens, tokens * 2)
    capacity = (capacity + page_size - 1) // page_size * page_size
    rng = np.random.default_rng(args.seed)
    host_new = rng.standard_normal((tokens, 1, heads * 2 // packing, packing, head_dim))
    host_new = host_new.astype(dtype)
    host_cache = rng.standard_normal(
        (capacity // page_size, page_size, heads * 2 // packing, packing, head_dim)
    ).astype(dtype)
    if args.pattern == "random":
        loc = rng.choice(capacity, tokens, replace=False).astype(np.int32)
    else:
        loc = np.arange(tokens, dtype=np.int32) + 1
    loc[::7] = -1
    valid = loc != -1
    expected = host_cache.reshape(capacity, heads * 2, head_dim).copy()
    expected[loc[valid]] = host_new.reshape(tokens, heads * 2, head_dim)[valid]
    new_kv = jnp.asarray(host_new)
    device_loc = jnp.asarray(loc)
    cache_shape = jax.ShapeDtypeStruct(host_cache.shape, dtype)
    selected = get_num_slices_per_block(new_kv, cache_shape, page_size=1)
    bytes_per_page_slice = dtype.itemsize * page_size * heads * 2 * head_dim
    legacy = min(tokens, (VMEM_SIZE - VMEM_HEADROOM_BYTES) // bytes_per_page_slice)
    if legacy < 1:
        raise ValueError("The legacy policy cannot fit one page slice in VMEM")
    policies = [("legacy", legacy, page_size), ("selected", selected, 1)]
    policies.extend((f"slices_{s}", min(tokens, s), 1) for s in args.slices)
    for policy, slices, scratch_size in policies:
        cache = jnp.asarray(host_cache.copy())
        jax.block_until_ready((new_kv, device_loc, cache))
        fn = make_update(slices, scratch_size)
        start = time.perf_counter()
        compiled = fn.lower(new_kv, device_loc, cache).compile()
        compile_s = time.perf_counter() - start
        for _ in range(3):
            cache = compiled(new_kv, device_loc, cache)
        jax.block_until_ready(cache)
        np.testing.assert_array_equal(np.asarray(cache).reshape(expected.shape), expected)
        with tempfile.TemporaryDirectory(prefix="kv-slice-trace-") as trace_dir:
            with jax.profiler.trace(trace_dir):
                for i in range(args.iterations):
                    with jax.profiler.StepTraceAnnotation("kv_update", step_num=i):
                        cache = compiled(new_kv, device_loc, cache)
                        jax.block_until_ready(cache)
            timings = device_times(trace_dir, args.iterations)
        np.testing.assert_array_equal(np.asarray(cache).reshape(expected.shape), expected)
        yield {
            "policy": policy,
            "heads": heads,
            "tokens": tokens,
            "page_size": page_size,
            "head_dim": head_dim,
            "dtype": args.dtype,
            "pattern": args.pattern,
            "cache_tokens": capacity,
            "slices_per_block": slices,
            "scratch_slice_size": scratch_size,
            "scratch_bytes": slices * scratch_size * heads * 2 * head_dim * dtype.itemsize,
            "compile_s": compile_s,
            "device_us": timings,
            "median_us": float(np.median(timings)),
            "correct": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", default=[8])
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 128, 1024])
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--slices", type=int, nargs="*", default=[])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--pattern", choices=["random", "contiguous"], default="random")
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            *args.heads,
            *args.tokens,
            *args.page_sizes,
            *args.head_dims,
            *args.slices,
            args.cache_tokens,
            args.iterations,
        )
        <= 0
    ):
        parser.error("All dimensions, slice counts, and iteration counts must be positive")
    if any(dim % 128 for dim in args.head_dims):
        parser.error("Head dimensions must be multiples of 128")
    devices = jax.devices("tpu")
    if len(devices) != 1:
        parser.error("Run this benchmark on a single TPU device")
    metadata = {
        "device": devices[0].device_kind,
        "versions": {
            name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
        },
        "seed": args.seed,
        "iterations": args.iterations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(json.dumps({"environment": metadata}) + "\n")
        for config in itertools.product(args.heads, args.tokens, args.page_sizes, args.head_dims):
            for result in benchmark_case(args, *config):
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(json.dumps({k: v for k, v in result.items() if k != "device_us"}), flush=True)


if __name__ == "__main__":
    main()
