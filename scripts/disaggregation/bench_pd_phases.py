#!/usr/bin/env python3
"""Clean per-phase PD profiling: traffic driver + server-log parser.

Why this exists: the existing ``bench_throughput.py`` sends P then sleeps a
fixed 0.5s before sending D, and runs with radix cache on. Both contaminate
TTFT. This driver removes every confounder so the three configs (no-pd /
unstaged / staged) can be compared on the SAME clean methodology:

  * goes through the mini_lb router (or a colocated server for no-pd) -- the
    production handshake, with NO artificial client-side sleep;
  * concurrency = 1, requests sent STRICTLY SEQUENTIALLY (await each fully
    before the next) so per-request phases never overlap or queue;
  * a UNIQUE random-token prompt per request, and the servers run with
    ``--disable-radix-cache``, so no prefix-cache hit can deflate TTFT;
  * ``max_new_tokens=1`` so client TTFT == time-to-first-token, isolating the
    prefill + KV-transfer path from the decode loop;
  * W warmup requests per input length absorb per-shape JIT compilation and
    the one-time transfer-link connect, then M measured requests are kept.

The real per-phase breakdown comes from the servers' own ``PD-TIME-STATS``
log lines (enabled by ``--enable-request-time-stats-logging``), NOT from this
client. Because driving is conc=1 sequential, those lines appear in request
order, so ``parse`` recovers each input length's median by POSITION -- drop
the first W ``role=<role>`` lines of each input-length group, take the median
of the next M. No req_id correlation needed.

Subcommands:
  drive  -- send W+M sequential streaming requests per input length; report
            client TTFT.
  parse  -- read one server log, emit median per-phase for a role, sliced per
            input length using the same input-lens / warmup / measure values.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time

import requests


def _unique_prompt(tokenizer, target_len: int, rng: random.Random) -> str:
    """A prompt of ~target_len tokens with fresh random ids each call."""
    vocab = tokenizer.vocab_size
    ids = [rng.randint(10, vocab - 1) for _ in range(target_len)]
    return tokenizer.decode(ids)


def _stream_ttft(url: str, payload: dict, timeout: float) -> float:
    """POST a streaming /generate and return seconds to the first SSE chunk."""
    payload = {**payload, "stream": True}
    t0 = time.perf_counter()
    with requests.post(f"{url}/generate", json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw[5:].strip() if raw.startswith("data:") else raw.strip()
            if not line or line == "[DONE]":
                continue
            # First non-empty data chunk == first token emitted.
            return time.perf_counter() - t0
    raise RuntimeError("stream ended with no token chunk")


def cmd_drive(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    url = args.url.rstrip("/")
    input_lens = [int(x) for x in args.input_lens.split(",")]
    rng = random.Random(args.seed)
    total = args.warmup + args.measure

    base = {
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    }

    rows = []
    for input_len in input_lens:
        print(
            f"--- input={input_len} (warmup={args.warmup}, measure={args.measure}) ---",
            flush=True,
        )
        ttfts = []
        for i in range(total):
            prompt = _unique_prompt(tokenizer, input_len, rng)
            payload = {**base, "text": prompt}
            try:
                ttft = _stream_ttft(url, payload, args.timeout)
            except Exception as e:  # noqa: BLE001 - surface and abort the group
                print(f"  request {i} error: {e}", flush=True)
                return 1
            tag = "warmup" if i < args.warmup else "measure"
            if tag == "measure":
                ttfts.append(ttft)
            print(f"  [{tag}] req {i}: ttft={ttft * 1000:.1f}ms", flush=True)
        rows.append(
            {
                "input_len": input_len,
                "measure": len(ttfts),
                "ttft_med_ms": round(statistics.median(ttfts) * 1000, 1),
                "ttft_min_ms": round(min(ttfts) * 1000, 1),
                "ttft_max_ms": round(max(ttfts) * 1000, 1),
            }
        )

    print("=" * 60, flush=True)
    print(f"{'Input':>6} {'N':>3} {'TTFT_med':>10} {'TTFT_min':>10} {'TTFT_max':>10}", flush=True)
    print("-" * 60, flush=True)
    for r in rows:
        print(
            f"{r['input_len']:>6} {r['measure']:>3} "
            f"{r['ttft_med_ms']:>9}m {r['ttft_min_ms']:>9}m {r['ttft_max_ms']:>9}m",
            flush=True,
        )
    print("=" * 60, flush=True)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"client TTFT saved to {args.output_file}", flush=True)
    return 0


_PHASE_RE = re.compile(r"(\w+)=([0-9.]+)ms")


def _parse_line(line: str) -> dict[str, float]:
    return {k: float(v) for k, v in _PHASE_RE.findall(line)}


def cmd_parse(args: argparse.Namespace) -> int:
    role = args.role
    input_lens = [int(x) for x in args.input_lens.split(",")]
    per_group = args.warmup + args.measure

    needle = f"PD-TIME-STATS role={role} "
    lines = []
    with open(args.log) as f:
        for line in f:
            idx = line.find(needle)
            if idx != -1:
                lines.append(_parse_line(line[idx:]))

    expected = len(input_lens) * per_group
    print(
        f"role={role}: found {len(lines)} PD-TIME-STATS lines "
        f"(expected {expected} = {len(input_lens)} lens x {per_group})",
        flush=True,
    )
    if len(lines) < expected:
        print(
            "  WARNING: fewer lines than expected -- some requests may have "
            "failed; per-input slicing may be misaligned.",
            flush=True,
        )

    # Phase labels in canonical order, per role. --fine selects the
    # SGL_JAX_PD_FINE_MARKS breakdown (pack/stage_misc/d2h_plus_send,
    # pull/scatter/decode_forward) emitted by the server in profiling mode.
    if getattr(args, "fine", False):
        order = {
            "prefill": [
                "queue",
                "forward",
                "kv_writeback",
                "sampling_finalize",
                "index_d2h",
                "gather",
                "pack",
                "stage_misc",
                "d2h_plus_send",
                "total",
            ],
            "decode": [
                "bootstrap",
                "prealloc_wait",
                "pull",
                "scatter",
                "decode_forward",
                "total",
            ],
        }[role]
    else:
        order = {
            "prefill": ["queue", "forward", "stage", "transfer", "total"],
            "decode": ["bootstrap", "prealloc_wait", "kv_wait", "decode", "total"],
        }[role]

    print(f"{'Input':>6} " + " ".join(f"{p:>13}" for p in order), flush=True)
    print("-" * (7 + 14 * len(order)), flush=True)
    out = []
    for j, input_len in enumerate(input_lens):
        lo = j * per_group + args.warmup
        hi = lo + args.measure
        group = lines[lo:hi]
        if not group:
            print(f"{input_len:>6} (no data)", flush=True)
            continue
        med = {}
        cells = []
        for p in order:
            vals = [g[p] for g in group if p in g]
            if vals:
                m = statistics.median(vals)
                med[p] = round(m, 1)
                cells.append(f"{m:>11.1f}m")
            else:
                cells.append(f"{'-':>13}")
        print(f"{input_len:>6} " + " ".join(cells), flush=True)
        out.append({"input_len": input_len, "role": role, "phases_med_ms": med})

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"per-phase medians saved to {args.output_file}", flush=True)
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    """Concurrent router load test to exercise host-pool backpressure.

    Fires ``num_requests`` non-streaming /generate calls at the router with a
    bounded concurrency. Each request has a unique random prompt and produces
    ``output_len`` tokens so the decode loop (and thus the full P->transfer->D
    pipeline) actually runs. When ``concurrency`` exceeds the prefill host
    pool size, the D1 admission gate must defer reqs back to the waiting queue
    (host backpressure). The proof that backpressure is *usable*: every request
    still completes with zero errors and no server crash, and throughput stays
    stable instead of collapsing.
    """
    import asyncio

    import aiohttp
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    url = args.url.rstrip("/")
    rng = random.Random(args.seed)

    prompts = [_unique_prompt(tokenizer, args.input_len, rng) for _ in range(args.num_requests)]

    async def _run() -> dict:
        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict] = []
        errors: list[str] = []

        async def _one(session: aiohttp.ClientSession, prompt: str) -> None:
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": args.output_len,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            }
            async with sem:
                t0 = time.perf_counter()
                try:
                    timeout = aiohttp.ClientTimeout(total=args.timeout)
                    async with session.post(
                        f"{url}/generate", json=payload, timeout=timeout
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            errors.append(f"HTTP {resp.status}: {body[:120]}")
                            return
                        data = await resp.json()
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))
                    return
                lat = time.perf_counter() - t0
                meta = data.get("meta_info", {})
                results.append(
                    {
                        "latency": lat,
                        "completion_tokens": meta.get("completion_tokens", 0),
                        "prompt_tokens": meta.get("prompt_tokens", 0),
                    }
                )

        async with aiohttp.ClientSession() as session:
            t_start = time.perf_counter()
            await asyncio.gather(*(_one(session, p) for p in prompts))
            wall = time.perf_counter() - t_start
        return {"results": results, "errors": errors, "wall": wall}

    print(
        f"--- stress: input={args.input_len} output={args.output_len} "
        f"conc={args.concurrency} num_requests={args.num_requests} ---",
        flush=True,
    )
    out = asyncio.run(_run())
    results, errors, wall = out["results"], out["errors"], out["wall"]

    completed = len(results)
    n_err = len(errors)
    lats = sorted(r["latency"] for r in results)
    out_tok = sum(r["completion_tokens"] for r in results)

    def _pct(p: float) -> float:
        if not lats:
            return 0.0
        k = min(len(lats) - 1, int(round(p / 100.0 * (len(lats) - 1))))
        return lats[k]

    print("=" * 60, flush=True)
    print(f"completed         : {completed}/{args.num_requests}", flush=True)
    print(f"errors            : {n_err}", flush=True)
    print(f"wall_time_s       : {wall:.2f}", flush=True)
    print(f"req_throughput_s  : {completed / wall:.2f}" if wall else "n/a", flush=True)
    print(f"out_tok_per_s     : {out_tok / wall:.1f}" if wall else "n/a", flush=True)
    if lats:
        print(f"latency_s p50/p99 : {_pct(50):.2f} / {_pct(99):.2f}", flush=True)
        print(f"latency_s min/max : {lats[0]:.2f} / {lats[-1]:.2f}", flush=True)
    if errors:
        print("--- first errors ---", flush=True)
        for e in errors[:5]:
            print(f"  {e}", flush=True)
    print("=" * 60, flush=True)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(
                {
                    "input_len": args.input_len,
                    "output_len": args.output_len,
                    "concurrency": args.concurrency,
                    "num_requests": args.num_requests,
                    "completed": completed,
                    "errors": n_err,
                    "wall_time_s": round(wall, 2),
                    "req_throughput_s": round(completed / wall, 2) if wall else None,
                    "out_tok_per_s": round(out_tok / wall, 1) if wall else None,
                    "latency_p50_s": round(_pct(50), 3),
                    "latency_p99_s": round(_pct(99), 3),
                },
                f,
                indent=2,
            )
        print(f"stress summary saved to {args.output_file}", flush=True)

    # Backpressure is "usable" only if nothing was dropped.
    return 0 if n_err == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean per-phase PD profiling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("drive", help="send sequential streaming traffic")
    d.add_argument("--url", required=True, help="router URL (PD) or server URL (no-pd)")
    d.add_argument("--model-path", required=True)
    d.add_argument("--input-lens", default="512,1024,2048,4096")
    d.add_argument("--warmup", type=int, default=3)
    d.add_argument("--measure", type=int, default=10)
    d.add_argument("--seed", type=int, default=1234)
    d.add_argument("--timeout", type=float, default=600.0)
    d.add_argument("--output-file", default=None)
    d.set_defaults(func=cmd_drive)

    p = sub.add_parser("parse", help="aggregate per-phase medians from a server log")
    p.add_argument("--log", required=True)
    p.add_argument("--role", required=True, choices=["prefill", "decode"])
    p.add_argument("--input-lens", default="512,1024,2048,4096")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=10)
    p.add_argument(
        "--fine",
        action="store_true",
        help="parse SGL_JAX_PD_FINE_MARKS fine-grained phase labels",
    )
    p.add_argument("--output-file", default=None)
    p.set_defaults(func=cmd_parse)

    s = sub.add_parser("stress", help="concurrent router load test (host backpressure)")
    s.add_argument("--url", required=True, help="router URL")
    s.add_argument("--model-path", required=True)
    s.add_argument("--input-len", type=int, default=4096)
    s.add_argument("--output-len", type=int, default=32)
    s.add_argument("--concurrency", type=int, default=48)
    s.add_argument("--num-requests", type=int, default=192)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--timeout", type=float, default=600.0)
    s.add_argument("--output-file", default=None)
    s.set_defaults(func=cmd_stress)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
