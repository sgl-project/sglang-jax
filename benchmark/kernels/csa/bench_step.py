"""Whole CSA step on native buffers: four rotations, three warmups, ten TPU timings."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from benchmark.kernels.csa_attention.bench_attention import inputs as attention_inputs
from sgl_jax.srt.kernels.csa import CSAMetadata, csa_step
from sgl_jax.srt.kernels.csa.tune import get_indexer_schedule
from sgl_jax.srt.kernels.csa_attention.tune import get_csa_attention_schedule
from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorBlock,
    CompressorMetadata,
)
from sgl_jax.srt.kernels.csa_compressor.tune import get_compressor_schedule


def make_inputs(batch, sequence, mode, seed):
    rng = np.random.default_rng(seed + 31)
    q, new, window, compressed, _, sink, md = attention_inputs(
        batch, sequence, mode, 64, seed, device=False, select_records=False
    )
    cu = md.cu_q_lens
    lengths = np.diff(cu)
    prefix = sequence - lengths
    pages = md.compressed_page_indices.reshape(batch, -1)
    positions = np.concatenate([np.arange(p, sequence) for p in prefix]).astype(np.int32)
    blocks = np.full((batch, int(max(lengths))), -1, np.int32)
    locations = np.empty(len(q), np.int32)
    for r, n in enumerate(lengths):
        ids = np.arange(cu[r], cu[r + 1])
        blocks[r, :n] = ids
        pos = positions[ids]
        locations[ids] = pages[r, pos // 128] * 32 + pos % 128 // 4
    compressor_md = CompressorMetadata(
        positions,
        cu,
        np.arange(batch, dtype=np.int32),
        locations,
        (CompressorBlock(blocks, np.arange(batch, dtype=np.int32)),),
    )
    distribution = [batch, batch, batch] if mode == "decode" else [0, 0, batch]
    metadata = CSAMetadata(compressor_md, md, pages, np.asarray(distribution, np.int32))

    def bf16(shape, scale=0.2):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)

    states = []
    for dim in (512, 128):
        state = np.zeros((batch + 1, 8, 4 * dim), np.float32)
        state[..., 2 * dim :] = -np.inf
        for r, p in enumerate(prefix):
            if p:
                state[r] = rng.normal(0, 0.2, state[r].shape)
        states.append(state)
    angles = np.arange(sequence)[:, None] * np.linspace(0.001, 0.1, 32)[None]
    data = (
        bf16((len(q), 4096)),
        q,
        new,
        bf16((len(q), 64, 128)),
        rng.uniform(0.1, 1, (len(q), 64)).astype(np.float32),
        bf16((4096, 2560), 4096**-0.5),
        *(rng.normal(0, 0.2, (4, 2 * d)).astype(np.float32) for d in (512, 128)),
        np.ones(512, np.float32),
        np.ones(128, np.float32),
        np.cos(angles).astype(np.float32),
        np.sin(angles).astype(np.float32),
        sink,
    )
    pools = (*states, window, compressed, bf16((*compressed.shape[:2], 128)))
    return data, pools, metadata


def measure(batch, sequence, mode, directory, *, host_buffers=False):
    schedule = get_csa_attention_schedule(jax.devices()[0].device_kind, decode=mode == "decode")
    compressor_schedule = get_compressor_schedule(4096, device_kind=jax.devices()[0].device_kind)
    indexer_schedule = get_indexer_schedule(128, schedule)

    def full_csa_step(data, pools, metadata):
        return csa_step(
            *data[:5],
            *pools,
            *data[5:],
            metadata,
            compressor_schedule=compressor_schedule,
            attention_schedule=schedule,
            indexer_schedule=indexer_schedule,
            softmax_scale=512**-0.5,
        )

    function = jax.jit(full_csa_step, donate_argnums=(1,))
    buffers = [make_inputs(batch, sequence, mode, seed) for seed in range(4)]
    if not host_buffers:
        buffers = [(jax.device_put(data), pools, jax.device_put(md)) for data, pools, md in buffers]
    jax.block_until_ready(buffers)
    counter = 0

    def call():
        nonlocal counter
        slot = counter % 4
        data, pools, md = buffers[slot]
        # Reset from identical snapshots; never feed a completed chunk back at its old positions.
        device_args = jax.device_put((data, pools, md))
        jax.block_until_ready(device_args)
        output, updates = jax.block_until_ready(function(*device_args))
        counter += 1
        return output

    for _ in range(4):
        output = call()
        if not bool(jnp.all(jnp.isfinite(output))):
            raise RuntimeError("nonfinite attention output")
        del output
    for _ in range(3):
        call()
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    directory.mkdir(parents=True, exist_ok=False)
    with jax.profiler.trace(str(directory), profiler_options=options):
        for _ in range(10):
            call()
    paths = list(directory.glob("plugins/profile/**/*.xplane.pb"))
    if len(paths) != 1:
        raise RuntimeError("expected one TPU trace")
    profile = jax.profiler.ProfileData.from_file(str(paths[0]))
    plane = profile.find_plane_with_name("/device:TPU:0")
    events = [event for line in plane.lines if line.name == "XLA Modules" for event in line.events]
    if len(events) != 10 or any("jit_full_csa_step" not in e.name for e in events):
        raise RuntimeError(f"unexpected timing boundary: {[e.name for e in events]}")
    samples = [e.duration_ns / 1e6 for e in events]
    # Hardware counters must exclude host staging, unlike the ten-call latency trace.
    device_args = jax.device_put(buffers[0])
    jax.block_until_ready(device_args)
    hardware_dir = directory / "hardware"
    with jax.profiler.trace(str(hardware_dir), profiler_options=options):
        jax.block_until_ready(function(*device_args))
    return dict(
        batch=batch,
        sequence=sequence,
        mode=mode,
        host_buffers=host_buffers,
        mean_ms=float(np.mean(samples)),
        samples_ms=samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--sequence", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["decode", "prefill", "ragged"],
        default=["decode", "prefill", "ragged"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--host-buffers",
        action="store_true",
        help="Stage one of four host snapshots to HBM before each kernel timing",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = [
        *root.glob("python/sgl_jax/srt/kernels/csa/*.py"),
        root / "python/sgl_jax/srt/kernels/csa_attention/csa_attention.py",
        root / "python/sgl_jax/srt/kernels/csa_compressor/compressor.py",
        root / "python/sgl_jax/srt/kernels/dsa/streamindex_topk.py",
        root / "python/sgl_jax/srt/kernels/csa_attention/tune.py",
        root / "python/sgl_jax/srt/kernels/csa_compressor/tune.py",
        root / "benchmark/kernels/csa_attention/bench_attention.py",
        Path(__file__),
    ]
    (args.output / "sources.json").write_text(
        json.dumps(
            {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(
                device=jax.devices()[0].device_kind,
                jax=jax.__version__,
                hidden=4096,
                heads=64,
                index_heads=64,
                warmup=3,
                timed=10,
                buffers=4,
            )
        ),
        flush=True,
    )
    for mode in args.mode:
        for batch in args.batch:
            for sequence in args.sequence:
                lengths = [
                    (
                        1
                        if mode == "decode"
                        else sequence if mode == "prefill" else max(1, sequence // (1 + r % 4))
                    )
                    for r in range(batch)
                ]
                q_and_output_bytes = 2 * sum(lengths) * 64 * 512 * 2
                memory_limit = jax.devices()[0].memory_stats()["bytes_limit"]
                try:
                    if q_and_output_bytes > memory_limit:
                        raise MemoryError(
                            f"Q+output alone need {q_and_output_bytes} bytes; device limit {memory_limit}"
                        )
                    row = measure(
                        batch,
                        sequence,
                        mode,
                        args.output / f"{mode}-b{batch}-s{sequence}",
                        host_buffers=args.host_buffers,
                    )
                except (MemoryError, jax.errors.JaxRuntimeError) as exc:
                    if not isinstance(exc, MemoryError) and not any(
                        s in str(exc).lower() for s in ("resource_exhausted", "out of memory")
                    ):
                        raise
                    row = dict(
                        batch=batch,
                        sequence=sequence,
                        mode=mode,
                        status="OOM",
                        reason=str(exc),
                        host_buffers=args.host_buffers,
                    )
                with (args.output / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                jax.clear_caches()
                gc.collect()
