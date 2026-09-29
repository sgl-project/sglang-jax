"""Benchmark hca_step with native V4 caches, including all state/cache updates.

Run with PYTHONPATH=python:. python -m benchmark.kernels.hca.bench_hca.
S is the final context length; ragged requests extend from S*(4+r%4)//8 to S.
"""

import argparse
import gc
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.hca.hca import HCAMetadata, hca_step
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule

HIDDEN, HEADS, DIM, RATIO, WINDOW = 4096, 64, 512, 128, 128


def build_case(mode, batch, length, page_size, hidden=HIDDEN):
    prefixes = np.zeros(batch, np.int32)
    if mode == "decode":
        prefixes[:] = length - 1
    elif mode == "ragged":
        prefixes = length * (4 + np.arange(batch, dtype=np.int32) % 4) // 8
    queries = length - prefixes
    positions = np.concatenate([np.arange(p, length, dtype=np.int32) for p in prefixes])
    ids = np.repeat(np.arange(batch, dtype=np.int32), queries)
    slots = np.arange(batch, dtype=np.int32)[::-1].copy()
    page = page_size // RATIO
    # Each request owns an SWA ring and a separate compressed-history region.
    history_pages = (length + page_size - 1) // page_size
    history_stride = max(RATIO, history_pages)
    cache_rows = (batch + 1) * page_size
    schedule = get_hca_kernel_schedule(
        jax.devices()[0].device_kind,
        page_size=page_size,
        max_compressed_entries=max(1, length // RATIO),
        local_heads=HEADS,
        head_dim=DIM,
    )
    blocks = [
        (r, start)
        for r, n in enumerate(queries)
        if n > 1
        for start in range(0, int(n), schedule.query_block_size)
    ]
    metadata = HCAMetadata(
        **jax.tree.map(
            jax.device_put,
            dict(
                state_slots=slots[ids],
                query_seq_ids=ids,
                cu_q_lens=np.cumsum(np.r_[0, queries]).astype(np.int32),
                valid_token_mask=np.ones(len(positions), bool),
                boundary_token_indices=np.flatnonzero((positions + 1) % RATIO == 0).astype(
                    np.int32
                ),
                window_page_indices=np.repeat(
                    np.arange(1, batch + 1, dtype=np.int32), history_pages
                ),
                window_cu_kv_lens=np.arange(batch + 1, dtype=np.int32) * history_pages * page_size,
                seq_lens=np.full(batch, length, np.int32),
                compressed_kv_lens=np.full(batch, length // RATIO, np.int32),
                compressed_page_indices=np.concatenate(
                    [
                        (r + 1) * history_stride + np.arange(history_pages, dtype=np.int32)
                        for r in range(batch)
                    ]
                ),
                compressed_cu_kv_lens=np.arange(batch + 1, dtype=np.int32) * history_pages * page,
                query_block_request_ids=np.asarray([r for r, _ in blocks], np.int32),
                query_block_offsets=np.asarray([start for _, start in blocks], np.int32),
                decode_request_ids=np.flatnonzero(queries == 1).astype(np.int32),
            ),
        ),
        max_queries_per_request=int(queries.max()),
    )

    def host_inputs(seed):
        rng = np.random.default_rng(seed)

        def normal(shape, scale=1.0, dtype=ml_dtypes.bfloat16):
            return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(dtype)

        fused = normal((hidden, 2 * DIM), 0.02)
        angles = normal((length, 32), 0.02, np.float32)
        data = dict(
            hidden=normal((len(positions), hidden), 0.05),
            q=normal((len(positions), HEADS, DIM)),
            kv=normal((len(positions), DIM)),
            fused=fused,
            wkv=fused[:, :DIM].T.copy(),
            wgate=fused[:, DIM:].T.copy(),
            ape=normal((RATIO, DIM), 0.02, np.float32),
            norm=np.ones(DIM, ml_dtypes.bfloat16),
            cos=np.cos(angles),
            sin=np.sin(angles),
            positions=positions,
            sink=np.zeros(HEADS, np.float32),
        )
        state = np.zeros((batch + 2, RATIO, 2, DIM), np.float32)
        state[:, :, 1] = -np.inf
        swa = np.zeros((cache_rows, DIM), ml_dtypes.bfloat16)
        compressed = np.zeros(((batch + 1) * history_stride, 1, page, DIM), ml_dtypes.bfloat16)
        if np.any(prefixes):
            swa[page_size:] = normal(swa[page_size:].shape)
            compressed[history_stride:] = normal(compressed[history_stride:].shape)
            for slot, prefix in zip(slots, prefixes, strict=True):
                for p in range(max(0, int(prefix) - RATIO), int(prefix)):
                    state[slot, p % RATIO] = normal((2, DIM), 0.1, np.float32)
        return data, (state, swa, compressed)

    def execute(data, pools, md):
        return hca_step(
            data["hidden"],
            data["q"],
            data["kv"],
            *pools,
            data["wkv"],
            data["wgate"],
            data["ape"],
            data["norm"],
            data["cos"],
            data["sin"],
            data["positions"],
            data["sink"],
            md,
            mode="uniform" if mode == "prefill" else mode,
            schedule=schedule,
            page_size=page_size,
            softmax_scale=DIM**-0.5,
            fused_weight=data["fused"],
        )

    return jax.jit(execute, donate_argnums=(1,)), metadata, host_inputs, len(positions)


def device_time(path, call, count):
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with jax.profiler.trace(str(path), profiler_options=options):
        for i in range(count):
            jax.block_until_ready(call(i))
    traces = list(path.glob("plugins/profile/**/*.xplane.pb"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one device trace, found {len(traces)}")
    profile = jax.profiler.ProfileData.from_file(str(traces[0]))
    plane = profile.find_plane_with_name("/device:TPU:0")
    durations = [
        event.end_ns - event.start_ns
        for line in plane.lines
        if line.name == "XLA Modules"
        for event in line.events
    ]
    if len(durations) != count:
        raise RuntimeError(f"expected {count} kernel calls, found {len(durations)}")
    return sum(durations) / 1e6


def benchmark(mode, batch, length, page_size, profile_dir=None, hidden=HIDDEN):
    tokens = batch if mode == "decode" else batch * length
    if mode == "ragged":
        tokens = int(np.sum(length - length * (4 + np.arange(batch) % 4) // 8))
    limit = jax.devices()[0].memory_stats()["bytes_limit"]
    if 2 * tokens * HEADS * DIM * np.dtype(ml_dtypes.bfloat16).itemsize > limit:
        return {"status": "OOM"}
    function, metadata, host_inputs, tokens = build_case(mode, batch, length, page_size, hidden)
    host = host_inputs(200)
    data, pools = jax.tree.map(jax.device_put, host)
    executable = function.lower(data, pools, metadata).compile()
    output, *updated = executable(data, pools, metadata)
    # State's unused scores are -inf; attention and both KV arrays must be finite.
    for array in (output, updated[1], updated[2]):
        if not bool(jnp.all(jnp.isfinite(array))):
            raise AssertionError("non-finite output")
    del output, updated, data, pools, array
    gc.collect()
    memory = executable.memory_analysis()
    input_bytes = sum(a.nbytes for a in jax.tree.leaves(host))
    required = (
        4 * input_bytes
        + memory.output_size_in_bytes
        - memory.alias_size_in_bytes
        + memory.temp_size_in_bytes
    )
    resident = required <= limit - jax.devices()[0].memory_stats()["bytes_in_use"]
    buffers = (
        [jax.tree.map(jax.device_put, host if i == 0 else host_inputs(200 + i)) for i in range(4)]
        if resident
        else []
    )
    host_buffers = [] if resident else [host] + [host_inputs(200 + i) for i in range(1, 4)]
    del host

    def prepare(i):
        values = buffers[i % 4] if resident else jax.tree.map(jax.device_put, host_buffers[i % 4])
        jax.block_until_ready(values)
        return values

    def invoke(i, values):
        output, *updated = executable(*values, metadata)
        if resident:
            buffers[i % 4] = (values[0], tuple(updated))
        return output, updated

    for i in range(4):
        jax.block_until_ready(invoke(i, prepare(i)))
    for i in range(3):
        jax.block_until_ready(invoke(i, prepare(i)))
    trace_dir = (
        tempfile.TemporaryDirectory(prefix="hca-native-")
        if profile_dir is None
        else nullcontext(tempfile.mkdtemp(prefix=f"{mode}-b{batch}-s{length}-", dir=profile_dir))
    )
    with trace_dir as tmp:
        if resident:
            elapsed = device_time(Path(tmp), lambda i: invoke(i, buffers[i % 4]), 10)
        else:
            elapsed = 0.0
            for i in range(10):
                values = prepare(i)  # H2D finishes before profiling starts.
                elapsed += device_time(
                    Path(tmp) / str(i), lambda _, i=i, values=values: invoke(i, values), 1
                )
                del values
    return {
        "status": "ok",
        "mean_ms": elapsed / 10,
        "tokens": tokens,
        "input_rotation": "resident-4" if resident else "host-staged-4",
        **({"profile_dir": tmp} if profile_dir is not None else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("decode", "prefill", "ragged"),
        default=["decode", "prefill", "ragged"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--page-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--hidden-size", type=int, choices=(4096, 7168), default=HIDDEN)
    parser.add_argument("--profile-dir", type=Path, help="Keep XProf traces for hardware counters")
    args = parser.parse_args()
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        parser.error("requires a single TPU device")
    if min(*args.batch_sizes, *args.seq_lens) < 1:
        parser.error("batch sizes and context lengths must be positive")
    print(
        json.dumps(
            dict(
                device=jax.devices()[0].device_kind,
                jax=jax.__version__,
                page_size=args.page_size,
                hidden_size=args.hidden_size,
                warmup=3,
                timed=10,
                environment={
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("DSV4_HCA_", "HCA_"))
                    or k in ("DSV4_PAGED_ROW_RUN", "PALLAS_INTERPRET")
                },
            )
        ),
        flush=True,
    )
    for mode in args.modes:
        for batch in args.batch_sizes:
            for length in args.seq_lens:
                try:
                    result = benchmark(
                        mode, batch, length, args.page_size, args.profile_dir, args.hidden_size
                    )
                except jax.errors.JaxRuntimeError as error:
                    if "RESOURCE_EXHAUSTED" not in str(error) or "HBM" not in str(error):
                        raise
                    result = {"status": "OOM"}
                print(
                    json.dumps(dict(mode=mode, batch=batch, sequence=length, **result)), flush=True
                )
                jax.clear_caches()
                gc.collect()


if __name__ == "__main__":
    main()
