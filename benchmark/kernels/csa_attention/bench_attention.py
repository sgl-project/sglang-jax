"""Single-device HBM-to-HBM latency; run the correctness suite first."""

import argparse
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
from sgl_jax.srt.kernels.csa_attention.tune import (
    CSAAttentionSchedule,
    get_csa_attention_schedule,
)


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
    pages_per_request = max(1, (sequence // 4 + 127) // 128)
    pages = 1 + batch * pages_per_request
    q = rng.normal(0, 0.5, (tokens, heads, 512)).astype(ml_dtypes.bfloat16)
    new = rng.normal(0, 0.5, (tokens, 512)).astype(ml_dtypes.bfloat16)
    window = rng.normal(0, 0.5, (batch + 1, 64, 2, 512)).astype(ml_dtypes.bfloat16)
    nope = np.zeros((pages, 128, 512), np.uint8)
    nope[..., :448] = (
        rng.normal(0, 0.5, (pages, 128, 448)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    )
    nope[..., 448:455] = np.uint8(127)  # E8M0 scale = 1.
    bits = rng.normal(0, 0.5, (pages, 128, 64)).astype(ml_dtypes.bfloat16).view(np.uint16)
    rope = np.concatenate(((bits >> 8).astype(np.uint8), (bits & 255).astype(np.uint8)), axis=-1)
    selected = np.full((tokens, 512), -1, np.int32)
    for r, length in enumerate(lengths):
        for local in range(length):
            visible = (sequence - length + local + 1) // 4
            selected[cu[r] + local, : min(visible, 512)] = rng.permutation(visible)[:512]
    window_pages = rng.permutation(np.arange(1, batch + 1, dtype=np.int32))
    locations = np.full(tokens, -1, np.int32)
    for r, length in enumerate(lengths):
        local = np.arange(max(0, length - 128), length)
        locations[cu[r] + local] = window_pages[r] * 128 + (sequence - length + local) % 128
    metadata = CSAAttentionMetadata(
        np.repeat(np.arange(batch, dtype=np.int32), lengths),
        cu,
        np.full(batch, sequence, np.int32),
        window_pages,
        np.arange(batch + 1, dtype=np.int32) * 128,
        rng.permutation(np.arange(1, pages, dtype=np.int32)),
        np.arange(batch + 1, dtype=np.int32) * pages_per_request * 128,
        np.full(batch, sequence // 4, np.int32),
        locations,
    )
    arrays = (
        q,
        new,
        window,
        nope.reshape(pages, 128, 4, 128),
        rope.reshape(pages, 32, 4, 128),
        selected,
        np.zeros(heads, np.float32),
        metadata,
    )
    return jax.tree.map(jnp.asarray, arrays) if device else arrays


def measure(buffers, schedule, *, host_buffers=False):
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
                fp8_scale_block=64,
                rows_per_group=4,
            )
        )

    call()  # Compile outside timing.
    for _ in range(3):
        call()
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as directory:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--mode", choices=("decode", "prefill", "ragged"), default="decode")
    parser.add_argument("--query-tile", type=int, default=None)
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
        schedule = CSAAttentionSchedule(query_tile=args.query_tile)
    result = measure(buffers, schedule, host_buffers=args.host_buffers)
    print(
        json.dumps(
            {
                **vars(args),
                "query_tile": schedule.query_tile,
                "device": jax.devices()[0].device_kind,
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "query_tokens": buffers[0][0].shape[0],
                "selected_tile": schedule.selected_tile,
                "warmup": 3,
                "iterations": 10,
                "buffers": 4,
                **result,
            }
        )
    )


if __name__ == "__main__":
    main()
