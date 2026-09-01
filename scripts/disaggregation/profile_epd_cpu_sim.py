#!/usr/bin/env python3
"""Arm profiling across the EPD CPU-sim tiers, drive image requests, and stop.

Pairs with scripts/disaggregation/run_epd_cpu_sim.sh. Captures a jax.profiler
trace on every encoder process and on the language server (prefill+decode),
all under one profiler dir, so you can align them into a single EPD flame graph.

Example:
    python scripts/disaggregation/profile_epd_cpu_sim.py \
        --lang-url http://127.0.0.1:30000 \
        --encoder-url http://127.0.0.1:31001 \
        --image https://.../demo.jpeg --n-requests 4 --max-tokens 32

Only the standard library is required.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

DEFAULT_PROFILER_DIR = "/tmp/epd-sim-profile"


def _post(url: str, payload: dict | None, timeout: float = 1200.0) -> dict:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _image_content(image: str) -> dict:
    """Build an OpenAI image_url content block; inline local files as data URIs."""
    if image.startswith(("http://", "https://", "data:")):
        url = image
    else:
        mime = mimetypes.guess_type(image)[0] or "image/jpeg"
        with open(image, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        url = f"data:{mime};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}


def _chat_request(args, image_block: dict) -> dict:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    image_block,
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        # Force a fixed number of decode steps regardless of the (meaningless
        # under --simulate-compute) sampled tokens, so decode is actually
        # exercised and profiled.
        "ignore_eos": True,
    }


def _arm(url: str, output_dir: str, args) -> None:
    body = {"output_dir": output_dir}
    if args.host_tracer_level is not None:
        body["host_tracer_level"] = args.host_tracer_level
    if args.python_tracer_level is not None:
        body["python_tracer_level"] = args.python_tracer_level
    print(f"  start_profile -> {url}  ({output_dir})")
    _post(f"{url}/start_profile", body)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--encoder-url",
        action="append",
        default=[],
        help="Encoder base URL (repeat for multiple encoders).",
    )
    p.add_argument("--image", required=True, help="Image URL or local file path.")
    p.add_argument("--prompt", default="Describe this image in detail.")
    p.add_argument("--model", default="model")
    p.add_argument("--n-requests", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--profiler-dir",
        default=DEFAULT_PROFILER_DIR,
        help="Must match PROFILER_DIR used by run_epd_cpu_sim.sh.",
    )
    p.add_argument(
        "--host-tracer-level",
        type=int,
        default=None,
        help="XProf host tracer level (0-3). Default keeps the server default.",
    )
    p.add_argument(
        "--python-tracer-level",
        type=int,
        default=1,
        help="XProf python tracer level. 1 (default) = full per-call Python "
        "frames in Perfetto (zoom in to resolve the many tiny slices; keep the "
        "workload small, e.g. --n-requests 1 --max-tokens 8, to avoid the 1M-event "
        "truncation). 0 = stage annotations only (clean flame-graph view).",
    )
    args = p.parse_args()

    encoder_urls = args.encoder_url or ["http://127.0.0.1:31001"]
    image_block = _image_content(args.image)

    # Warmup outside the trace window to keep the flame graph focused.
    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}")
        _post(f"{args.lang_url}/v1/chat/completions", _chat_request(args, image_block))

    print("arming profilers:")
    for idx, url in enumerate(encoder_urls):
        _arm(url, os.path.join(args.profiler_dir, f"encoder_{idx}"), args)
    _arm(args.lang_url, os.path.join(args.profiler_dir, "language"), args)

    t0 = time.monotonic()
    for i in range(args.n_requests):
        r = _post(f"{args.lang_url}/v1/chat/completions", _chat_request(args, image_block))
        usage = r.get("usage", {}) if isinstance(r, dict) else {}
        print(f"request {i + 1}/{args.n_requests}  usage={usage}")
    elapsed = time.monotonic() - t0

    print("stopping profilers:")
    for url in encoder_urls:
        _post(f"{url}/stop_profile", None)
        print(f"  stop_profile -> {url}")
    _post(f"{args.lang_url}/stop_profile", None)
    print(f"  stop_profile -> {args.lang_url}")

    print(
        f"\n{args.n_requests} requests in {elapsed:.2f}s "
        f"({elapsed / max(1, args.n_requests) * 1000:.1f} ms/req)"
    )
    print(f"\nTraces under {args.profiler_dir}:")
    print("  encoder_*/plugins/profile/.../*.trace.json.gz")
    print("  language/plugins/profile/.../*.trace.json.gz")
    print("\nView the full EPD chain:")
    print(
        "  - Drag all trace.json.gz files into https://ui.perfetto.dev/ (multi-trace, same clock)"
    )
    print(f"  - or: xprof --logdir={args.profiler_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
