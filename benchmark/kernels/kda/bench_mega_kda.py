"""Compare the existing chunked KDA prefill with the Mega KDA kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda.kda import chunk_kda_fwd
from sgl_jax.srt.kernels.kda.mega_kda import (
    kda_forward_inference,
    kda_forward_packed,
)

CHUNK_SIZE = 64
KEY_DIM = VALUE_DIM = 128
LOWER_BOUND = -5.0
SCALE = KEY_DIM**-0.5


def _normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.sum(value * value, axis=-1, keepdims=True) + 1e-6)
    return value.astype(jnp.bfloat16)


def _inputs(tokens: int, heads: int, segments: int, seed: int):
    if tokens % segments:
        raise ValueError("tokens must be divisible by segments")
    rng = np.random.default_rng(seed)
    shape = (1, tokens, heads, KEY_DIM)

    def bf16_normal(scale: float = 1.0):
        return jnp.asarray(
            rng.standard_normal(shape, dtype=np.float32) * scale,
            dtype=jnp.bfloat16,
        )

    lengths = np.full(segments, tokens // segments, dtype=np.int32)
    return (
        bf16_normal(),
        bf16_normal(),
        bf16_normal(1.5),
        jnp.asarray(
            rng.uniform(-4.5, 4.5, shape).astype(np.float32),
            dtype=jnp.bfloat16,
        ),
        jnp.asarray(
            rng.uniform(0.05, 0.95, shape[:-1]).astype(np.float32),
            dtype=jnp.bfloat16,
        ),
        jnp.asarray(rng.uniform(0.2, 3.0, (heads,)).astype(np.float32)),
        jnp.asarray(rng.uniform(-8.0, -1.5, (heads, KEY_DIM)).astype(np.float32)),
        jnp.asarray(
            rng.standard_normal(
                (segments, heads, KEY_DIM, VALUE_DIM),
                dtype=np.float32,
            )
            * 0.1
        ),
        jnp.asarray([0, *np.cumsum(lengths)], dtype=jnp.int32),
    )


def _block(value) -> None:
    jax.block_until_ready(value)


def _measure(fn, args, warmups: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        _block(fn(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        _block(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def _make_padded_mega(
    *,
    tokens: int,
    segments: int,
    max_segments_per_tile: int,
    lower_bound: float | None,
):
    """Build a Mega runner that includes repacking and output gathering.

    ``max_segments_per_tile=2`` is the minimum-padding layout supported by
    Mega's boundary path. ``max_segments_per_tile=1`` pads every request to a
    separate 64-token tile. Both layouts keep padding inside the jitted region
    so the benchmark measures the routing strategy rather than only the Pallas
    call.
    """
    segment_length = tokens // segments
    if segment_length > CHUNK_SIZE:
        raise ValueError("the padding study only supports segment lengths <= 64")

    segments_per_tile = min(max_segments_per_tile, CHUNK_SIZE // segment_length)
    padded_tokens = ((segments + segments_per_tile - 1) // segments_per_tile) * CHUNK_SIZE
    @jax.jit
    def padded_mega(q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens):
        source_positions = jnp.arange(tokens, dtype=jnp.int32)
        lengths = jnp.diff(cu_seqlens).astype(jnp.int32)
        source_segments = jnp.repeat(
            jnp.arange(segments, dtype=jnp.int32),
            lengths,
            total_repeat_length=tokens,
        )
        tile_indices = source_segments // segments_per_tile
        tile_source_starts = cu_seqlens[tile_indices * segments_per_tile]
        destination_positions = (
            tile_indices * CHUNK_SIZE + source_positions - tile_source_starts
        )
        segment_ids = jnp.zeros(padded_tokens, dtype=jnp.int32).at[
            destination_positions
        ].set(source_segments + 1)[None, :]

        def _repack(value):
            shape = list(value.shape)
            shape[1] = padded_tokens
            return jnp.zeros(shape, dtype=value.dtype).at[:, destination_positions].set(value)

        output, final_state = kda_forward_inference(
            _repack(q),
            _repack(k),
            _repack(v),
            _repack(g),
            _repack(beta),
            segment_ids=segment_ids,
            A_log=a_log,
            dt_bias=dt_bias.reshape(-1),
            scale=SCALE,
            initial_state=initial_state[None, ...],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
            chunk_size=CHUNK_SIZE,
            N_max=segments,
        )
        return output[:, destination_positions], final_state[0]

    return padded_mega, padded_tokens


def _direct_mega_supported(tokens: int, segments: int) -> bool:
    """Static equivalent of the runtime guard for equal-length segments."""
    segment_length = tokens // segments
    padded_tokens = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    starts = np.arange(segments, dtype=np.int32) * segment_length
    ends = starts + segment_length
    for tile_start in range(0, padded_tokens, CHUNK_SIZE):
        overlaps = np.sum((starts < tile_start + CHUNK_SIZE) & (ends > tile_start))
        if overlaps > 2:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--segments", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1550)
    parser.add_argument(
        "--padding-study",
        action="store_true",
        help=(
            "also compare minimum tile packing and one-request-per-tile padding; "
            "intended for short speculative extend batches"
        ),
    )
    parser.add_argument(
        "--gate-mode",
        choices=("bounded", "unbounded"),
        default="bounded",
        help="bounded selects K3's sigmoid gate; unbounded selects Kimi-Linear's softplus gate",
    )
    args = parser.parse_args()
    lower_bound = LOWER_BOUND if args.gate_mode == "bounded" else None

    arrays = list(_inputs(args.tokens, args.heads, args.segments, args.seed))
    if args.gate_mode == "unbounded":
        # Kimi-Linear produces FP32 beta even though q/k/v/g use BF16.
        arrays[4] = arrays[4].astype(jnp.float32)
    arrays = tuple(arrays)

    @jax.jit
    def chunked(q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens):
        result = chunk_kda_fwd(
            _normalize(q),
            _normalize(k),
            v,
            g,
            beta,
            scale=SCALE,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_size=CHUNK_SIZE,
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )
        return result[0], result[1]

    @jax.jit
    def mega(q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens):
        return kda_forward_packed(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=initial_state,
            lower_bound=lower_bound,
            chunk_size=CHUNK_SIZE,
        )

    runners = {"chunked": chunked}
    effective_tokens = {
        "chunked": args.segments
        * ((args.tokens // args.segments + CHUNK_SIZE - 1) // CHUNK_SIZE)
        * CHUNK_SIZE
    }
    if _direct_mega_supported(args.tokens, args.segments):
        runners["mega"] = mega
        effective_tokens["mega"] = (args.tokens + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    elif not args.padding_study:
        raise ValueError(
            "the packed layout puts more than two requests in one Mega tile; "
            "rerun with --padding-study to compare supported padded layouts"
        )

    if args.padding_study:
        tile_padded, tile_tokens = _make_padded_mega(
            tokens=args.tokens,
            segments=args.segments,
            max_segments_per_tile=2,
            lower_bound=lower_bound,
        )
        segment_padded, segment_tokens = _make_padded_mega(
            tokens=args.tokens,
            segments=args.segments,
            max_segments_per_tile=1,
            lower_bound=lower_bound,
        )
        runners["mega_tile_padded"] = tile_padded
        runners["mega_segment_padded"] = segment_padded
        effective_tokens["mega_tile_padded"] = tile_tokens
        effective_tokens["mega_segment_padded"] = segment_tokens

    results = {variant: runner(*arrays) for variant, runner in runners.items()}
    _block(tuple(results.values()))
    chunked_result = results["chunked"]
    for variant, result in results.items():
        if variant == "chunked":
            continue
        print(
            json.dumps(
                {
                    "event": "correctness",
                    "variant": variant,
                    "tokens": args.tokens,
                    "segment_length": args.tokens // args.segments,
                    "segments": args.segments,
                    "effective_tokens": effective_tokens[variant],
                    "gate_mode": args.gate_mode,
                    "output_max_abs": float(
                        jnp.max(
                            jnp.abs(
                                result[0].astype(jnp.float32)
                                - chunked_result[0].astype(jnp.float32)
                            )
                        )
                    ),
                    "state_max_abs": float(
                        jnp.max(jnp.abs(result[1] - chunked_result[1]))
                    ),
                },
                sort_keys=True,
            )
        )

    variants = tuple(runners)
    for round_index in range(args.rounds):
        offset = round_index % len(variants)
        order = variants[offset:] + variants[:offset]
        for variant in order:
            samples = _measure(
                runners[variant],
                arrays,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            print(
                json.dumps(
                    {
                        "event": "latency",
                        "variant": variant,
                        "round": round_index,
                        "tokens": args.tokens,
                        "segment_length": args.tokens // args.segments,
                        "effective_tokens": effective_tokens[variant],
                        "padding_ratio": effective_tokens[variant] / args.tokens,
                        "heads": args.heads,
                        "segments": args.segments,
                        "gate_mode": args.gate_mode,
                        "iterations": args.iterations,
                        "latency_ms": statistics.median(samples),
                        "median_ms": statistics.median(samples),
                        "mean_ms": statistics.mean(samples),
                        "p90_ms": float(np.percentile(samples, 90)),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
