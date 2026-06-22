"""BM2: recurrent-cache reuse-collapse K-sweep (one-time characterization).

Characterizes how many distinct cached prefix snapshots the recurrent cache
holds before reuse collapses, and asserts the empirical collapse "knee" falls
in an analytically predicted range -- a sizing/eviction correctness check for
the S5a hybrid-recurrent cache (subplan docs/review_v2_bm2_ksweep.md).

Recurrent state capacity is slot-count based (unlike token-scaled FULL KV), so
a controlled one-snapshot-per-prefix workload shows a visible capacity knee at
K* = dp_size * (S_rank - factor * C_rank), where S_rank = max_recurrent_state_size
/ dp_size, factor = 3 (1 running + 2 ping-pong track slots), and C_rank is the
per-rank in-flight reservation derived from the ACTUAL run (--parallel spread
over DP ranks), not a fixed constant.

This script POINTS AT an already-running server (the operator launches it,
multi-host for KDA); it does NOT launch a server. It talks HTTP only:
``/get_server_info`` (sizing), ``/flush_cache`` (cold each K point), and
``/generate`` with ``max_new_tokens=1`` reading ``meta_info.cached_tokens`` for
reuse accounting -- the same primitives as the cache-hit probe.

Per distinct-prefix count K::

    flush_cache
    WARM : send each of the K distinct prefixes once   -> populates the cache
    PROBE: resend each prefix + a short divergent suffix -> should reuse it
    record reuse_frac = sum(cached_tokens) / sum(prompt_tokens), probe TTFT, tput

Prefixes are deterministic (per-index salt, no RNG/timestamps) so the sweep is
reproducible; each crosses exactly one track boundary (length in [I, 2I), I =
recurrent_track_interval) -> one reusable snapshot per prefix.

Usage (client-only against a launched server)::

    python benchmark/hicache/bench_recurrent_reuse_sweep.py \
        --server-url http://<rank0>:30000 \
        --k-list 8 16 32 64 96 128 160 192 256 \
        --parallel 8 --output-json /tmp/bm2_ksweep.json

The HTTP sweep path is exercised by the controller against a live server; the
pure ``predict_knee`` math is covered by a CPU unittest (not in CI run_suite).
"""

import argparse
import asyncio
import json
import statistics
import time

import requests

from sgl_jax.test.kits.cache_hit_kit import (
    async_request_sglang_generate,
    flush_cache,
    gen_payload,
)


def predict_knee(
    max_recurrent_state_size: int,
    dp_size: int,
    C_rank: float,
    factor: int = 3,
    snapshots_per_prefix: int = 1,
) -> float:
    """Analytic collapse knee K* = dp_size * (S_rank - factor*C_rank) / snapshots.

    S_rank = max_recurrent_state_size / dp_size (slots per rank); factor = 3
    reserves 1 running + 2 ping-pong track slots per concurrent request; C_rank
    is the per-rank in-flight reservation. Example: size=192, dp=4, C_rank=4 ->
    S_rank=48, K* = 4 * (48 - 3*4) = 144.
    """
    s_rank = max_recurrent_state_size / dp_size
    cap_rank = s_rank - factor * C_rank
    return dp_size * cap_rank / snapshots_per_prefix


def derive_actual_C_rank(parallel: int, dp_size: int) -> float:
    """Per-rank in-flight reservation from the actual run, not a constant.

    The probe issues at most ``parallel`` requests concurrently; spread over
    ``dp_size`` ranks (assume even routing) each rank carries about
    ``parallel / dp_size`` concurrent requests, each reserving ``factor`` slots.
    Non-uniform routing is absorbed by the widened knee range, not here.
    """
    return max(1.0, parallel / dp_size)


def parse_args():
    p = argparse.ArgumentParser(
        description="BM2 recurrent-cache reuse-collapse K-sweep (one-time characterization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--server-url",
        required=True,
        help="Already-running server (operator launches; multi-host for KDA). Client-only.",
    )
    p.add_argument(
        "--k-list",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 96, 128, 160, 192, 256],
        help="Distinct-prefix counts K to sweep.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="In-flight concurrency for WARM/PROBE. Keep small for a clean capacity read.",
    )
    p.add_argument(
        "--suffix-tokens",
        type=int,
        default=8,
        help="Short divergent suffix length (keeps the shared prefix the reusable snapshot).",
    )
    # Sizing fallbacks: used only if /get_server_info omits the field.
    p.add_argument("--max-recurrent-state-size", type=int, default=None)
    p.add_argument("--dp-size", type=int, default=None)
    p.add_argument("--recurrent-track-interval", type=int, default=None)
    p.add_argument(
        "--plateau-frac",
        type=float,
        default=0.9,
        help="Knee = largest K with reuse_frac >= plateau-frac * max(reuse_frac).",
    )
    p.add_argument(
        "--knee-range-frac",
        type=float,
        default=0.4,
        help="Assertion half-width: knee must lie in [K*(1-f), K*(1+f)] (widened for DP routing).",
    )
    p.add_argument("--output-json", default=None)
    p.add_argument(
        "--no-assert",
        action="store_true",
        help="Report the knee-vs-K* verdict but skip the hard range assertion (curve-only run).",
    )
    return p.parse_args()


def get_server_sizing(args):
    """Read recurrent sizing from /get_server_info, falling back to CLI args."""
    info = requests.get(f"{args.server_url}/get_server_info", timeout=30).json()

    def pick(key, cli):
        val = info.get(key)
        return cli if val is None else val

    size = pick("max_recurrent_state_size", args.max_recurrent_state_size)
    dp_size = pick("dp_size", args.dp_size)
    interval = pick("recurrent_track_interval", args.recurrent_track_interval)
    tokenizer = info.get("tokenizer_path") or info.get("model_path")
    assert size is not None and dp_size is not None and interval is not None, (
        "missing recurrent sizing from /get_server_info; pass --max-recurrent-state-size, "
        "--dp-size, --recurrent-track-interval explicitly "
        f"(got size={size}, dp_size={dp_size}, interval={interval})"
    )
    return int(size), int(dp_size), int(interval), tokenizer


def make_prefix_ids(tokenizer, index: int, target_tokens: int):
    """Deterministic distinct prefix of ~target_tokens tokens (no RNG).

    The per-index tag makes prefixes truly distinct so accidental sharing does
    not blur the knee; padding words are index-seeded, never random.
    """
    head = f"Document {index} unique tag bm2-recurrent-reuse-sweep section. "
    ids = tokenizer.encode(head)
    filler = f"token{index}word "
    while len(ids) < target_tokens:
        ids = tokenizer.encode(head + filler * (target_tokens - len(ids) + 4))
    return ids[:target_tokens]


def make_suffix_ids(tokenizer, index: int, salt: int, n: int):
    """Short divergent suffix; salt differs WARM (7) vs PROBE (13) per index."""
    ids = tokenizer.encode(f" suffix{salt}-{index} end")
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


async def _send_round(payloads, url, parallel):
    sem = asyncio.Semaphore(parallel)

    async def one(payload):
        async with sem:
            return await async_request_sglang_generate(payload, url)

    return await asyncio.gather(*[asyncio.create_task(one(p)) for p in payloads])


def run_k_point(args, tokenizer, generate_url, K, interval):
    """Run one K point: flush -> WARM -> PROBE; return per-point metrics."""
    target_tokens = interval + interval // 2  # in [I, 2I) -> one track boundary
    prefixes = [make_prefix_ids(tokenizer, i, target_tokens) for i in range(K)]

    flush_cache(args.server_url)
    warm = [
        gen_payload(prefixes[i] + make_suffix_ids(tokenizer, i, 7, args.suffix_tokens), 1)
        for i in range(K)
    ]
    asyncio.run(_send_round(warm, generate_url, args.parallel))

    probe = [
        gen_payload(prefixes[i] + make_suffix_ids(tokenizer, i, 13, args.suffix_tokens), 1)
        for i in range(K)
    ]
    t0 = time.perf_counter()
    results = asyncio.run(_send_round(probe, generate_url, args.parallel))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.success]
    assert ok, f"K={K}: all probe requests failed"
    total_cached = sum(r.cached_tokens for r in ok)
    total_prompt = sum(r.prompt_len for r in ok)
    ttfts = sorted(r.ttft for r in ok)
    return {
        "K": K,
        "reuse_frac": total_cached / total_prompt if total_prompt else 0.0,
        "cached_tokens": total_cached,
        "prompt_tokens": total_prompt,
        "ttft_p50_ms": statistics.median(ttfts) * 1000.0,
        "ttft_p90_ms": ttfts[min(len(ttfts) - 1, int(0.9 * len(ttfts)))] * 1000.0,
        "throughput_req_s": len(ok) / wall if wall else 0.0,
    }


def detect_knee(curve, plateau_frac):
    """Knee = largest K with reuse_frac >= plateau_frac * max(reuse_frac)."""
    fracs = {pt["K"]: pt["reuse_frac"] for pt in curve}
    peak = max(fracs.values())
    return max(K for K, f in fracs.items() if f >= plateau_frac * peak)


def main():
    args = parse_args()
    print(f"args={args}\n", flush=True)

    from sgl_jax.bench_serving import get_tokenizer

    size, dp_size, interval, tokenizer_path = get_server_sizing(args)
    tokenizer = get_tokenizer(tokenizer_path)
    generate_url = f"{args.server_url}/generate"

    curve = []
    for K in sorted(args.k_list):
        pt = run_k_point(args, tokenizer, generate_url, K, interval)
        curve.append(pt)
        print(
            f"K={K:>5}  reuse_frac={pt['reuse_frac']:.4f}  "
            f"cached={pt['cached_tokens']}/{pt['prompt_tokens']}  "
            f"ttft_p50={pt['ttft_p50_ms']:.1f}ms  ttft_p90={pt['ttft_p90_ms']:.1f}ms  "
            f"tput={pt['throughput_req_s']:.1f}req/s",
            flush=True,
        )

    knee = detect_knee(curve, args.plateau_frac)
    C_rank = derive_actual_C_rank(args.parallel, dp_size)
    Kstar = predict_knee(size, dp_size, C_rank=C_rank, factor=3, snapshots_per_prefix=1)
    Kstar_lo = Kstar * (1 - args.knee_range_frac)
    Kstar_hi = Kstar * (1 + args.knee_range_frac)
    in_range = Kstar_lo <= knee <= Kstar_hi

    print(
        f"\nsize={size} dp_size={dp_size} interval={interval} parallel={args.parallel} "
        f"C_rank={C_rank:.2f}\n"
        f"K*={Kstar:.1f}  range=[{Kstar_lo:.1f}, {Kstar_hi:.1f}]  "
        f"empirical knee={knee}  -> {'PASS' if in_range else 'FAIL'}",
        flush=True,
    )

    if args.output_json:
        payload = {
            "args": vars(args),
            "sizing": {
                "max_recurrent_state_size": size,
                "dp_size": dp_size,
                "recurrent_track_interval": interval,
                "C_rank": C_rank,
            },
            "curve": curve,
            "knee": knee,
            "Kstar": Kstar,
            "Kstar_range": [Kstar_lo, Kstar_hi],
            "in_range": in_range,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote results to {args.output_json}")

    assert in_range or args.no_assert, (
        f"empirical knee {knee} outside predicted range "
        f"[{Kstar_lo:.1f}, {Kstar_hi:.1f}] (K*={Kstar:.1f}): "
        "likely an eviction/sizing/routing bug (knee << K*) or over-provisioning "
        "/ non-caching prefixes (knee >> K*)."
    )


if __name__ == "__main__":
    main()
