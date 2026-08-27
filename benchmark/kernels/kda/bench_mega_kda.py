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
from sgl_jax.srt.kernels.kda.mega_kda import kda_forward_packed

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--segments", type=int, choices=(1, 4), default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1550)
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

    chunked_result = chunked(*arrays)
    mega_result = mega(*arrays)
    _block((chunked_result, mega_result))
    print(
        json.dumps(
            {
                "event": "correctness",
                "gate_mode": args.gate_mode,
                "output_max_abs": float(
                    jnp.max(
                        jnp.abs(
                            mega_result[0].astype(jnp.float32)
                            - chunked_result[0].astype(jnp.float32)
                        )
                    )
                ),
                "state_max_abs": float(jnp.max(jnp.abs(mega_result[1] - chunked_result[1]))),
            },
            sort_keys=True,
        )
    )

    runners = {"chunked": chunked, "mega": mega}
    for round_index in range(args.rounds):
        order = ("chunked", "mega") if round_index % 2 == 0 else ("mega", "chunked")
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
                        "heads": args.heads,
                        "segments": args.segments,
                        "gate_mode": args.gate_mode,
                        "iterations": args.iterations,
                        "median_ms": statistics.median(samples),
                        "mean_ms": statistics.mean(samples),
                        "p90_ms": float(np.percentile(samples, 90)),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
