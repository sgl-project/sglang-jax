"""Single-device compressor latency, from HBM operands to state/cache writes.

Run test/srt/kernels/csa_compressor/test_compressor.py first. Four independent input buffers
rotate; host-to-device input resets are outside device-module timing. Reported
host time also includes those resets and is not the operator latency.
"""

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorBlock,
    CompressorMetadata,
    csa_compressor,
)
from sgl_jax.srt.kernels.csa_compressor.tune import (
    CSA_COMPRESSION_RATIO,
    CSA_DUAL_PROJECTION_DIM,
    CSA_INDEX_PROJECTED_DIM,
    CSA_MAIN_PROJECTED_DIM,
    CSA_STATE_SLOTS,
    get_compressor_schedule,
)


def measure(call):
    jax.block_until_ready(call())  # Compile separately from warmup and timing.
    for _ in range(3):
        jax.block_until_ready(call())
    host_ms = []
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as directory:
        with jax.profiler.trace(directory, profiler_options=options):
            for _ in range(10):
                start = time.perf_counter_ns()
                jax.block_until_ready(call())
                host_ms.append((time.perf_counter_ns() - start) / 1e6)
        paths = list(Path(directory).glob("plugins/profile/**/*.xplane.pb"))
        if len(paths) != 1:
            raise RuntimeError("expected one TPU trace")
        profile = jax.profiler.ProfileData.from_file(str(paths[0]))
    device = profile.find_plane_with_name("/device:TPU:0")
    if device is None:
        raise RuntimeError("TPU device timing unavailable; host time is not a substitute")
    samples = [
        event.duration_ns / 1e6
        for line in device.lines
        if line.name == "XLA Modules"
        for event in line.events
    ]
    if len(samples) != 10:
        raise RuntimeError(f"expected 10 complete device modules, found {len(samples)}")
    return {
        "mean_ms": float(np.mean(samples)),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "host_mean_ms": float(np.mean(host_ms)),
    }


def make_inputs(
    pattern, batch, sequence, hidden, seed, *, cache_pages=0, cache_layout="contiguous"
):
    rng = np.random.default_rng(seed)
    if pattern.startswith("decode"):
        # Compare an emit boundary and its next non-emitting step; report actual context.
        sequence = (
            (sequence + CSA_COMPRESSION_RATIO - 1) // CSA_COMPRESSION_RATIO * CSA_COMPRESSION_RATIO
        )
        sequence += pattern == "decode_update"
        lengths = (1,) * batch
    elif pattern == "prefill":
        lengths = (sequence,) * batch
    elif batch == 1:
        # One request still exercises a multi-token prefix extend.
        lengths = (max(1, sequence // 2),)
    else:
        cycle = (1, max(1, sequence // 4), max(1, sequence // 2), sequence)
        lengths = tuple(cycle[r % len(cycle)] for r in range(batch))
    prefixes = tuple(sequence - n for n in lengths)
    cu = np.asarray((0, *np.cumsum(lengths)), np.int32)
    positions = np.concatenate(
        [np.arange(p, p + n, dtype=np.int32) for p, n in zip(prefixes, lengths, strict=True)]
    )
    page_size = 128
    pages_per_request = max(1, (sequence // 4 + page_size - 1) // page_size)
    pages = 1 + batch * pages_per_request
    if cache_pages:
        if cache_pages < pages:
            raise ValueError(f"cache_pages must be at least {pages} for this shape")
        pages = cache_pages
    locations = np.concatenate(
        [
            (1 + r * pages_per_request) * page_size + positions[cu[r] : cu[r + 1]] // 4
            for r in range(batch)
        ]
    ).astype(np.int32)
    locations = np.where((positions + 1) % 4 == 0, locations, -1).astype(np.int32)
    valid = locations >= 0
    placement_rng = np.random.default_rng(0)  # Same placement across rotating input buffers.
    if cache_layout == "paged":
        mapping = placement_rng.permutation(pages)
        locations[valid] = (
            mapping[locations[valid] // page_size] * page_size + locations[valid] % page_size
        )
    elif cache_layout in ("mixed", "holes"):
        # Perturb every fourth occupied page, leaving other pages DMA-friendly.
        for page in np.unique(locations[valid] // page_size)[3::4]:
            selected = valid & (locations // page_size == page)
            if cache_layout == "mixed":
                permutation = placement_rng.permutation(page_size)
                locations[selected] = (
                    page * page_size + permutation[locations[selected] % page_size]
                )
            else:
                locations[np.flatnonzero(selected)[-1]] = -1
    elif cache_layout == "cross_page":
        locations[valid] = (locations[valid] + 1) % (pages * page_size)
    elif cache_layout != "contiguous":
        raise ValueError(f"unknown cache_layout: {cache_layout}")
    blocks = []
    for length, slot in sorted(set(zip(lengths, [p % 4 for p in prefixes], strict=True))):
        requests = np.asarray(
            [
                r
                for r, (n, p) in enumerate(zip(lengths, prefixes, strict=True))
                if (n, p % 4) == (length, slot)
            ],
            np.int32,
        )
        leading = min(length, (-slot) % 4)
        end = leading + (length - leading) // 4 * 4
        intervals = [(i, i + 1) for i in range(leading)]
        if end > leading:
            intervals.append((leading, end))
        intervals.extend((i, i + 1) for i in range(end, length))
        for start, stop in intervals:
            tokens = cu[requests, None] + np.arange(start, stop, dtype=np.int32)[None]
            blocks.append(CompressorBlock(jnp.asarray(tokens), jnp.asarray(requests)))
    metadata = CompressorMetadata(
        jnp.asarray(positions),
        jnp.asarray(cu),
        jnp.arange(batch, dtype=jnp.int32),
        jnp.asarray(locations),
        tuple(blocks),
    )
    states = []
    for width in (CSA_MAIN_PROJECTED_DIM, CSA_INDEX_PROJECTED_DIM):
        state = rng.normal(0, 0.05, (batch, CSA_STATE_SLOTS, 2, width)).astype(np.float32)
        for r, prefix in enumerate(prefixes):
            if prefix < CSA_COMPRESSION_RATIO:
                empty = np.r_[
                    0:CSA_COMPRESSION_RATIO, CSA_COMPRESSION_RATIO + prefix : CSA_STATE_SLOTS
                ]
                state[r, empty, 0] = 0
                state[r, empty, 1] = -np.inf
        states.append(state)
    operands = [
        jnp.asarray(rng.normal(0, 0.3, (sum(lengths), hidden)), jnp.bfloat16),
        jnp.asarray(rng.normal(0, 0.005, (hidden, CSA_DUAL_PROJECTION_DIM)), jnp.bfloat16),
        jnp.asarray(rng.normal(0, 0.02, (4, CSA_MAIN_PROJECTED_DIM)), jnp.float32),
        jnp.asarray(rng.normal(0, 0.02, (4, CSA_INDEX_PROJECTED_DIM)), jnp.float32),
        jnp.ones(512, jnp.float32),
        jnp.ones(128, jnp.float32),
        jnp.ones((sequence + 1, 32), jnp.float32),
        jnp.zeros((sequence + 1, 32), jnp.float32),
        *[jnp.asarray(s) for s in states],
        jnp.zeros((pages, page_size, 4, 128), jnp.uint8),
        jnp.zeros((pages, page_size // 4, 4, 128), jnp.uint8),
        jnp.zeros((pages, page_size // 4, 4, 256), jnp.uint8),
    ]
    return operands, metadata, states, lengths


def check_finite(outputs, metadata):
    """Untimed smoke check; the independent correctness suite is a prerequisite."""
    if len(outputs) != 5:
        raise AssertionError("expected two states and three caches")
    states = [np.asarray(v) for v in outputs[:2]]
    ends = np.asarray(metadata.cu_q_lens)[1:]
    context = np.asarray(metadata.positions)[ends - 1] + 1
    slots = np.arange(CSA_STATE_SLOTS)[None, :]
    empty = (context[:, None] < CSA_COMPRESSION_RATIO) & (
        (slots < CSA_COMPRESSION_RATIO) | (slots >= CSA_COMPRESSION_RATIO + context[:, None])
    )
    for state in states:
        assert np.isfinite(state[:, :, 0]).all()
        scores = state[:, :, 1]
        assert np.isneginf(scores[empty]).all()
        assert np.isfinite(scores[~empty]).all()
    nope, rope, index = [
        np.asarray(v).reshape(-1, w) for v, w in zip(outputs[2:], (512, 128, 256), strict=True)
    ]
    for records, width, count in ((nope, 448, 7), (index, 128, 1)):
        assert np.isfinite(
            records[:, :width].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        ).all()
        assert np.isfinite(
            records[:, width : width + count].view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
        ).all()
    bits = (rope[:, :64].astype(np.uint16) << 8) | rope[:, 64:].astype(np.uint16)
    assert np.isfinite(bits.view(ml_dtypes.bfloat16).astype(np.float32)).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--sequence", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096, 7168])
    parser.add_argument(
        "--pattern",
        nargs="+",
        choices=("decode", "decode_emit", "decode_update", "prefill", "ragged"),
        default=["decode", "prefill", "ragged"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--cache-pages", type=int, default=0, help="128-record pages; 0 sizes to input"
    )
    parser.add_argument(
        "--cache-layout",
        choices=("contiguous", "paged", "mixed", "holes", "cross_page"),
        default="contiguous",
    )
    args = parser.parse_args()
    if any(v <= 0 for v in (*args.batch, *args.sequence, *args.hidden)):
        parser.error("batch, sequence and hidden must be positive")
    if args.cache_pages < 0:
        parser.error("cache-pages must be nonnegative")
    patterns = list(
        dict.fromkeys(
            p
            for pattern in args.pattern
            for p in (("decode_emit", "decode_update") if pattern == "decode" else (pattern,))
        )
    )
    environment = {name: version(name) for name in ("jax", "jaxlib", "libtpu")}
    report = []
    for hidden in args.hidden:
        schedule = get_compressor_schedule(hidden, device_kind=jax.devices()[0].device_kind)
        for pattern in patterns:
            for batch in args.batch:
                for sequence in args.sequence:
                    buffers = [
                        make_inputs(
                            pattern,
                            batch,
                            sequence,
                            hidden,
                            31000 + i,
                            cache_pages=args.cache_pages,
                            cache_layout=args.cache_layout,
                        )
                        for i in range(4)
                    ]
                    jax.block_until_ready([b[:2] for b in buffers])
                    # Validate every independent buffer before collecting any timings.
                    for operands, metadata, _, _ in buffers:
                        outputs = jax.block_until_ready(
                            csa_compressor(*operands, metadata, schedule=schedule)
                        )
                        operands[8:] = outputs
                        check_finite(outputs, metadata)
                    counter = 0

                    def call(buffers=buffers, schedule=schedule):
                        nonlocal counter
                        operands, metadata, initial_states, _ = buffers[counter % len(buffers)]
                        counter += 1
                        # Refresh only the small state; cache writes are deterministic
                        # overwrites. Every call sees the same prior overlap state.
                        operands[8:10] = [jax.device_put(s.copy()) for s in initial_states]
                        jax.block_until_ready(operands)
                        outputs = csa_compressor(*operands, metadata, schedule=schedule)
                        operands[8:] = outputs
                        return outputs

                    measurement = measure(call)
                    if args.profile_dir is not None:
                        trace_path = args.profile_dir / (
                            f"{pattern}-b{batch}-s{sequence}-h{hidden}"
                            f"-{args.cache_layout}-p{buffers[0][0][10].shape[0]}"
                        )
                        with jax.profiler.trace(str(trace_path)):
                            jax.block_until_ready(call())
                    row = dict(
                        pattern=pattern,
                        batch=batch,
                        sequence=sequence,
                        actual_context=int(np.asarray(buffers[0][1].positions).max()) + 1,
                        hidden=hidden,
                        cache_layout=args.cache_layout,
                        cache_pages=buffers[0][0][10].shape[0],
                        cache_bytes=sum(v.size * v.dtype.itemsize for v in buffers[0][0][10:]),
                        query_tokens=sum(buffers[0][3]),
                        query_lengths=list(buffers[0][3]),
                        prefix_lengths=[
                            int(np.asarray(buffers[0][1].positions)[int(start)])
                            for start in np.asarray(buffers[0][1].cu_q_lens)[:-1]
                        ],
                        device=jax.devices()[0].device_kind,
                        environment=environment,
                        schedule=asdict(schedule),
                        projection_operands="bfloat16",
                        projection_accumulator="float32",
                        dot_precision=jax.config.jax_default_matmul_precision or "DEFAULT",
                        warmup=3,
                        timed=10,
                        input_buffers=len(buffers),
                        **measurement,
                    )
                    report.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    del buffers
                    jax.clear_caches()


if __name__ == "__main__":
    main()
