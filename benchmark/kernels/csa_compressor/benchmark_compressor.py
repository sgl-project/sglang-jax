"""Native BF16 compressor: four rotating buffers, three warmups, ten device timings."""

import argparse
import gc
import json
import tempfile
import warnings
from pathlib import Path

import jax
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorBlock,
    CompressorMetadata,
    csa_compressor,
)
from sgl_jax.srt.kernels.csa_compressor.tune import get_compressor_schedule


def useful_io_bytes(mode, lengths, hidden):
    """Logical payload: unique inputs and changed outputs, excluding metadata/scratch.

    Decode emission reads seven old content/score halves and writes one full
    state row. Fresh prefill reads no old state; DMA padding/RMW is not useful IO.
    """
    dim = 512 + 128
    bf16_bytes, fp32_bytes = 2, 4
    tokens, batch = int(np.sum(lengths)), len(lengths)
    read = (tokens * hidden + hidden * 4 * dim) * bf16_bytes
    if mode == "decode":
        read += batch * 7 * 2 * dim * fp32_bytes
        state_rows, records, ape_rows, rope_rows = batch, batch, 1, 1
    else:
        state_rows = int(np.minimum(lengths, 8).sum())
        records = int((lengths // 4).sum())
        ape_rows, rope_rows = min(4, int(max(lengths))), int(max(lengths)) // 4
    read += (ape_rows * 2 * dim + dim + rope_rows * 2 * 32) * fp32_bytes
    write = state_rows * 4 * dim * fp32_bytes + records * dim * bf16_bytes
    return read + write


def benchmark(mode, batch, length, hidden, page_size, profile_root):
    # S is the final context length; decode emits a record at position S-1.
    lengths = np.full(batch, length if mode != "decode" else 1, np.int32)
    if mode == "ragged":
        lengths -= length * (np.arange(batch) % 4) // 8
    cu = np.cumsum(np.r_[0, lengths], dtype=np.int32)
    positions = np.concatenate(
        [np.arange(length - 1, length) if mode == "decode" else np.arange(n) for n in lengths]
    ).astype(np.int32)
    ids = np.full((batch, max(lengths)), -1, np.int32)
    locations = np.empty(cu[-1], np.int32)
    pages_per_request = (length + page_size - 1) // page_size
    for r, n in enumerate(lengths):
        ids[r, :n] = np.arange(cu[r], cu[r + 1])
        pos = positions[cu[r] : cu[r + 1]]
        page = 1 + r * pages_per_request + pos // page_size
        locations[cu[r] : cu[r + 1]] = page * (page_size // 4) + pos % page_size // 4
    metadata = CompressorMetadata(
        positions,
        cu,
        np.arange(batch, dtype=np.int32),
        locations,
        (CompressorBlock(ids, np.arange(batch, dtype=np.int32)),),
    )
    schedule = get_compressor_schedule(hidden, device_kind=jax.devices()[0].device_kind)
    function = jax.jit(
        lambda data, pools, md: csa_compressor(*data, *pools, md, schedule=schedule),
        donate_argnums=(1,),
    )
    buffers = []
    for seed in range(4):
        rng = np.random.default_rng(seed)

        def bf16(shape, scale=1):
            return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)

        angle = np.arange(length, dtype=np.float32)[:, None] * np.linspace(0.001, 0.1, 32)
        data = (
            bf16((cu[-1], hidden)),
            bf16((hidden, 2560), hidden**-0.5),
            *(rng.normal(0, 0.2, (4, 2 * d)).astype(np.float32) for d in (512, 128)),
            *(np.ones(d, np.float32) for d in (512, 128)),
            np.cos(angle).astype(np.float32),
            np.sin(angle).astype(np.float32),
        )
        states = []
        for d in (512, 128):
            state = np.zeros((batch + 1, 8, 4 * d), np.float32)
            state[..., 2 * d :] = -np.inf
            if mode == "decode":
                # Synthetic carried projection/score state; no history setup is timed.
                state[:batch] = rng.normal(0, 0.2, (batch, 8, 4 * d))
            states.append(state)
        caches = [
            np.zeros((1 + batch * pages_per_request, page_size // 4, d), ml_dtypes.bfloat16)
            for d in (512, 128)
        ]
        md = jax.tree.map(lambda a: np.array(a, copy=True), metadata)
        buffers.append(jax.tree.map(jax.device_put, (data, (*states, *caches), md)))
    executable = function.lower(*buffers[0]).compile()
    memory = executable.memory_analysis()

    def invoke(i):
        data, pools, md = buffers[i % 4]
        pools = executable(data, pools, md)
        buffers[i % 4] = data, pools, md
        return pools

    # First touch each allocation, then the same three warmups for every case.
    for i in range(4):
        jax.block_until_ready(invoke(i))
    for i in range(3):
        jax.block_until_ready(invoke(i))
    path = Path(tempfile.mkdtemp(prefix=f"{mode}-b{batch}-s{length}-d{hidden}-", dir=profile_root))
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = options.host_tracer_level = 0
    with jax.profiler.trace(str(path), profiler_options=options):
        for i in range(10):
            jax.block_until_ready(invoke(i))
    traces = list(path.glob("plugins/profile/**/*.xplane.pb"))
    if len(traces) != 1:
        raise RuntimeError(f"Expected one trace, got {len(traces)}")
    profile = jax.profiler.ProfileData.from_file(str(traces[0]))
    plane = profile.find_plane_with_name("/device:TPU:0")
    durations = [
        event.end_ns - event.start_ns
        for line in plane.lines
        if line.name == "XLA Modules"
        for event in line.events
    ]
    if len(durations) != 10:
        raise RuntimeError(f"Expected ten device calls, got {len(durations)}")
    seconds = float(np.mean(durations) / 1e9)
    payload = useful_io_bytes(mode, lengths, hidden)
    effective_gbs = payload / seconds / 1e9
    useful_tflops = 2 * int(cu[-1]) * hidden * 2560 / seconds / 1e12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        device_stats = dict(plane.stats)
    for _, pools, _ in buffers:
        for cache in pools[2:]:
            if not np.isfinite(np.asarray(cache).astype(np.float32)).all():
                raise AssertionError("Non-finite cache output")
    return dict(
        mode=mode,
        batch=batch,
        context=length,
        hidden=hidden,
        tokens=int(cu[-1]),
        mean_ms=float(np.mean(durations) / 1e6),
        useful_io_bytes=payload,
        effective_hbm_GBs=effective_gbs,
        effective_hbm_percent=100
        * effective_gbs
        / device_stats["peak_hbm_bw_gigabytes_per_second"],
        useful_mxu_percent=100 * useful_tflops / device_stats["peak_teraflops_per_second"],
        temporary_bytes=memory.temp_size_in_bytes,
        alias_bytes=memory.alias_size_in_bytes,
        profile=str(path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=("decode", "prefill", "ragged"),
        default=["decode", "prefill", "ragged"],
    )
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--length", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--hidden", nargs="+", type=int, default=[4096, 7168])
    parser.add_argument("--page-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--profile-root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()
    if any(v <= 0 for v in (*args.batch, *args.length, *args.hidden)) or any(
        s % 4 for s in args.length
    ):
        parser.error(
            "Dimensions must be positive; decode output-step lengths must be divisible by four"
        )
    args.profile_root.mkdir(parents=True, exist_ok=True)
    for hidden in args.hidden:
        for batch in args.batch:
            for length in args.length:
                for mode in args.mode:
                    print(
                        json.dumps(
                            benchmark(
                                mode, batch, length, hidden, args.page_size, args.profile_root
                            )
                        ),
                        flush=True,
                    )
                    jax.clear_caches()
                    gc.collect()


if __name__ == "__main__":
    main()
