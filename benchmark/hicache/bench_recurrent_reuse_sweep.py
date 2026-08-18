"""Recurrent-cache reuse-collapse K-sweep (one-time characterization).

Characterizes how many distinct cached prefix snapshots the recurrent cache
holds before reuse collapses, and asserts the empirical collapse "knee" falls
in an analytically predicted range -- a sizing/eviction correctness check for
the hybrid-recurrent cache.

Recurrent state capacity is slot-count based (unlike token-scaled FULL KV), so
a controlled one-snapshot-per-prefix workload shows a visible capacity knee at
K* = dp_size * (S_rank - owned * C_rank), where S_rank = max_recurrent_state_size
/ dp_size, owned = request_owned_slots a running req consumes (1 running + ping-pong
track slots: 3 with overlap, 2 with --disable-overlap-schedule), and C_rank is the
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
    PROBE: resend each prefix (in a shuffled order) + a short divergent suffix
    record reuse_frac = sum(cached_tokens) / sum(prompt_tokens), probe TTFT, tput

Prefixes are deterministic (per-index salt, no RNG/timestamps) so the sweep is
reproducible; each crosses exactly one track boundary (length in [I, 2I), I =
recurrent_track_interval) -> one reusable snapshot per prefix. The PROBE order is
a fixed-seed shuffle, NOT the WARM order: replaying in the same order makes every
probe miss once K exceeds capacity (each prefix is evicted just before its turn),
which falsely reads as reuse=0; shuffling reflects the true retained fraction.

Usage (client-only against a launched server)::

    python benchmark/hicache/bench_recurrent_reuse_sweep.py \
        --server-url http://<rank0>:30000 \
        --k-list 8 16 32 64 96 128 160 192 256 \
        --parallel 8 --output-json /tmp/recurrent_reuse_sweep.json

The HTTP sweep path is exercised by the controller against a live server; the
pure ``predict_knee`` math is covered by a CPU unittest (not in CI run_suite).
"""

import argparse
import asyncio
import json
import random
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
    request_owned_slots: int = 3,
    snapshots_per_prefix: int = 1,
) -> float:
    """Analytic collapse knee K* = dp_size * (S_rank - owned*C_rank) / snapshots.

    S_rank = max_recurrent_state_size / dp_size (slots per rank); request_owned_slots
    is the slots a running req actually consumes (1 running + ping-pong track slots:
    3 with overlap, 2 with --disable-overlap-schedule); C_rank is the per-rank
    in-flight reservation. Example: size=192, dp=4, C_rank=4, owned=2 -> S_rank=48,
    K* = 4 * (48 - 2*4) = 160.
    """
    s_rank = max_recurrent_state_size / dp_size
    cap_rank = s_rank - request_owned_slots * C_rank
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
        description="Recurrent-cache reuse-collapse K-sweep (one-time characterization).",
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
        help="Client-side in-flight request load for WARM/PROBE (the production "
        "analog is live serving concurrency; the server cap is --max-running-requests, "
        "and the cache capacity that bounds K is --max-recurrent-state-size). Use "
        ">= dp_size: predict_knee assumes every dp rank is populated, but WARM only "
        "reaches ranks that carry an in-flight request, so parallel < dp_size "
        "under-fills ranks and lowers the empirical knee below the prediction.",
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
        "--request-owned-slots",
        type=int,
        default=None,
        help="Per-req recurrent slots consumed (overrides server-info derivation: "
        "3 overlap-on / 2 overlap-off extra-buffer, 1 otherwise).",
    )
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
    # Actual per-req consumption: 1 running + ping-pong track slots (2 with overlap,
    # 1 with --disable-overlap-schedule); 1 without extra-buffer.
    if args.request_owned_slots is not None:
        owned = args.request_owned_slots
    elif info.get("enable_recurrent_extra_buffer"):
        owned = 2 if info.get("disable_overlap_schedule") else 3
    else:
        owned = 1
    return int(size), int(dp_size), int(interval), tokenizer, int(owned)


def make_prefix_ids(tokenizer, index: int, target_tokens: int):
    """Deterministic distinct prefix of ~target_tokens tokens (no RNG).

    The per-index tag makes prefixes truly distinct so accidental sharing does
    not blur the knee; padding words are index-seeded, never random.
    """
    head = f"Document {index} unique tag recurrent-reuse-sweep section. "
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

    # Probe each prefix once, in a fixed-seed shuffled order (NOT the warm order).
    # Same-order replay makes every probe miss once K exceeds capacity (each
    # prefix is evicted just before its turn) -> false reuse=0; shuffling reflects
    # the true retained fraction.
    probe_order = list(range(K))
    random.Random(1234).shuffle(probe_order)
    probe = [
        gen_payload(prefixes[i] + make_suffix_ids(tokenizer, i, 13, args.suffix_tokens), 1)
        for i in probe_order
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


def detect_knee(curve, plateau_frac, eps=1e-9):
    """Knee = largest K with reuse_frac >= plateau_frac * max(reuse_frac).

    Returns None when no prefix was ever reused (peak reuse_frac ~ 0). There is
    no knee to locate in that case, and returning the largest K would falsely
    read as a full-capacity plateau -- the no-cache / broken-cache / all-miss
    signature, not a sizing result.
    """
    fracs = {pt["K"]: pt["reuse_frac"] for pt in curve}
    peak = max(fracs.values(), default=0.0)
    if peak <= eps:
        return None
    return max(K for K, f in fracs.items() if f >= plateau_frac * peak)


def main():
    args = parse_args()
    print(f"args={args}\n", flush=True)

    from sgl_jax.bench_serving import get_tokenizer

    size, dp_size, interval, tokenizer_path, owned = get_server_sizing(args)

    if args.parallel < dp_size and not args.no_assert:
        raise SystemExit(
            f"--parallel ({args.parallel}) < dp_size ({dp_size}): WARM only fills "
            f"ranks that carry an in-flight request, so the empirical knee falls "
            f"below predict_knee and the gate falsely fails. Use --parallel >= "
            f"{dp_size}, or pass --no-assert for a curve-only run."
        )

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
    Kstar = predict_knee(
        size, dp_size, C_rank=C_rank, request_owned_slots=owned, snapshots_per_prefix=1
    )
    Kstar_lo = Kstar * (1 - args.knee_range_frac)
    Kstar_hi = Kstar * (1 + args.knee_range_frac)
    no_reuse = knee is None
    in_range = (not no_reuse) and (Kstar_lo <= knee <= Kstar_hi)

    if no_reuse:
        print(
            "\nno reusable prefixes observed (peak reuse_frac ~ 0) -> no knee to "
            "locate. Expected with caching disabled or unreachable; otherwise the "
            "recurrent snapshot is not being cached/reused. -> FAIL",
            flush=True,
        )
    else:
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
                "request_owned_slots": owned,
                "C_rank": C_rank,
            },
            "curve": curve,
            "knee": knee,
            "no_reuse": no_reuse,
            "Kstar": Kstar,
            "Kstar_range": [Kstar_lo, Kstar_hi],
            "in_range": in_range,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote results to {args.output_json}")

    if no_reuse:
        assert args.no_assert, (
            "no reusable prefixes observed (peak reuse_frac ~ 0): caching is "
            "disabled/unreachable or the recurrent snapshot is not cached/reused."
        )
    else:
        assert in_range or args.no_assert, (
            f"empirical knee {knee} outside predicted range "
            f"[{Kstar_lo:.1f}, {Kstar_hi:.1f}] (K*={Kstar:.1f}): "
            "knee << K* means parallel < dp_size (WARM under-fills ranks) or an "
            "eviction/sizing/routing bug; knee >> K* means over-provisioning / "
            "non-caching prefixes."
        )


if __name__ == "__main__":
    main()
