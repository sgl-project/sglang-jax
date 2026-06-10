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

    # Phase labels in canonical order, per role.
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
    p.add_argument("--output-file", default=None)
    p.set_defaults(func=cmd_parse)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
