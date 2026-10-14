"""Single-device HBM-to-HBM latency; run the correctness suite first."""

import argparse
import contextlib
import dataclasses
import itertools
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata, csa_joint_attention
from sgl_jax.srt.kernels.csa_attention.tune import get_csa_attention_schedule


def inputs(batch, sequence, mode, heads, seed, *, device=True):
    rng = np.random.default_rng(seed)
    if mode == "decode":
        lengths = [1] * batch
    elif mode == "prefill":
        lengths = [sequence] * batch
    else:
        lengths = [max(1, sequence // (1 + r % 4)) for r in range(batch)]
    cu = np.asarray([0, *np.cumsum(lengths)], np.int32)
    tokens = int(cu[-1])
    pages_per_request = max(1, (sequence + 127) // 128)
    pages = 1 + batch * pages_per_request
    q = rng.normal(0, 0.5, (tokens, heads, 512)).astype(ml_dtypes.bfloat16)
    new = rng.normal(0, 0.5, (tokens, 512)).astype(ml_dtypes.bfloat16)
    window = rng.normal(0, 0.5, ((batch + 1) * 128, 512)).astype(ml_dtypes.bfloat16)
    compressed = rng.normal(0, 0.5, (pages, 32, 512)).astype(ml_dtypes.bfloat16)
    selected = np.full((tokens, 512), -1, np.int32)
    for r, length in enumerate(lengths):
        for local in range(length):
            visible = (sequence - length + local + 1) // 4
            selected[cu[r] + local, : min(visible, 512)] = rng.permutation(visible)[:512]
    window_pages = rng.permutation(np.arange(1, batch + 1, dtype=np.int32))
    locations = np.full(tokens, -1, np.int32)
    for r, length in enumerate(lengths):
        local = np.arange(length)
        locations[cu[r] + local] = window_pages[r] * 128 + (sequence - length + local) % 128
    metadata = CSAAttentionMetadata(
        np.repeat(np.arange(batch, dtype=np.int32), lengths),
        cu,
        np.full(batch, sequence, np.int32),
        window_pages,
        np.arange(batch + 1, dtype=np.int32) * 128,
        rng.permutation(np.arange(1, pages, dtype=np.int32)),
        np.arange(batch + 1, dtype=np.int32) * pages_per_request * 32,
        np.full(batch, sequence // 4, np.int32),
        locations,
    )
    arrays = (
        q,
        new,
        window,
        compressed,
        selected,
        np.zeros(heads, np.float32),
        metadata,
    )
    return jax.tree.map(jnp.asarray, arrays) if device else arrays


def measure(buffers, schedule, *, host_buffers=False, profile_dir=None):
    rotation = itertools.cycle(buffers)

    def call():
        args = next(rotation)
        if host_buffers:
            # Stage independent inputs before device-module timing, not inside the operator.
            args = jax.tree.map(jax.device_put, args)
            jax.block_until_ready(args)
        return jax.block_until_ready(
            csa_joint_attention(
                *args,
                scale=512**-0.5,
                schedule=schedule,
                window_size=128,
                compression_ratio=4,
            )
        )

    for _ in range(4):
        call()  # Compile and first-touch each buffer outside timing.
    for _ in range(3):
        call()
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=False)
    context = (
        tempfile.TemporaryDirectory()
        if profile_dir is None
        else contextlib.nullcontext(str(profile_dir))
    )
    with context as directory:
        with jax.profiler.trace(directory, profiler_options=options):
            for _ in range(10):
                call()
        paths = list(Path(directory).glob("plugins/profile/**/*.xplane.pb"))
        if len(paths) != 1:
            raise RuntimeError("expected one TPU profile")
        profile = jax.profiler.ProfileData.from_file(str(paths[0]))
    device = profile.find_plane_with_name("/device:TPU:0")
    if device is None:
        raise RuntimeError("TPU device timing unavailable")
    events = [event for line in device.lines if line.name == "XLA Modules" for event in line.events]
    if len(events) != 10 or any("jit_csa_joint_attention" not in e.name for e in events):
        raise RuntimeError(
            f"expected ten complete attention modules, found {[e.name for e in events]}"
        )
    samples = [event.duration_ns / 1e6 for event in events]
    return {"mean_ms": float(np.mean(samples)), "samples_ms": samples}


def useful_work(args):
    q, new, _, _, selected, _, metadata = args
    selected, cu, lengths = map(np.asarray, (selected, metadata.cu_q_lens, metadata.seq_lens))
    pairs = 0
    unique_rows = 0
    written_rows = 0
    for r, length in enumerate(lengths):
        count = int(cu[r + 1] - cu[r])
        prefix = int(length) - count
        picks = selected[cu[r] : cu[r + 1]]
        pairs += int(np.minimum(np.arange(prefix + 1, length + 1), 128).sum())
        pairs += int((picks >= 0).sum())
        unique_rows += min(prefix, 127) + np.unique(picks[picks >= 0]).size
        written_rows += min(count, 128)
    # Compulsory payload only: no metadata, repeated DMA, padding, or replacement copies.
    payload = 2 * q.size * q.dtype.itemsize + new.size * new.dtype.itemsize
    payload += (unique_rows + written_rows) * q.shape[-1] * q.dtype.itemsize
    return {
        "useful_flops": 4 * pairs * q.shape[1] * q.shape[2],
        "compulsory_payload_bytes": int(payload),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--mode", choices=("decode", "prefill", "ragged"), default="decode")
    parser.add_argument("--query-tile", type=int, default=None)
    parser.add_argument("--write-run", type=int, default=None)
    parser.add_argument("--selected-tile", type=int, default=None)
    parser.add_argument("--profile-dir", type=Path, help="Keep XProf traces for hardware analysis")
    parser.add_argument(
        "--host-buffers",
        action="store_true",
        help="Rotate four host-backed inputs; stage to HBM before timing. One call must still fit HBM.",
    )
    args = parser.parse_args()
    if min(args.batch, args.sequence, args.heads) <= 0:
        parser.error("batch, sequence and heads must be positive")
    buffers = [
        inputs(args.batch, args.sequence, args.mode, args.heads, seed, device=not args.host_buffers)
        for seed in range(4)
    ]
    jax.block_until_ready(buffers)
    schedule = get_csa_attention_schedule(
        jax.devices()[0].device_kind, decode=args.mode == "decode"
    )
    if args.query_tile is not None:
        schedule = dataclasses.replace(schedule, query_tile=args.query_tile)
    if args.write_run is not None:
        schedule = dataclasses.replace(schedule, write_run=args.write_run)
    if args.selected_tile is not None:
        schedule = dataclasses.replace(schedule, selected_tile=args.selected_tile)
    work = useful_work(buffers[0])
    result = measure(
        buffers, schedule, host_buffers=args.host_buffers, profile_dir=args.profile_dir
    )
    print(
        json.dumps(
            {
                **vars(args),
                "query_tile": schedule.query_tile,
                "gather_backend": "sparsecore" if schedule.decode else "tensorcore",
                "device": jax.devices()[0].device_kind,
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "query_tokens": buffers[0][0].shape[0],
                "selected_tile": schedule.selected_tile,
                "write_run": schedule.write_run,
                "warmup": 3,
                "iterations": 10,
                "buffers": 4,
                **work,
                **result,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
