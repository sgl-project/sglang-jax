#!/usr/bin/env python3
"""Audit prefix-cache hits in the server-log window for an EvalScope run."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


CACHED_TOKEN_RE = re.compile(r"#cached-token:\s*(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--start-line", type=int, required=True)
    parser.add_argument("--expected-min-hits", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    server_log = pathlib.Path(args.server_log)
    lines = server_log.read_text(errors="replace").splitlines()
    window = lines[args.start_line :]
    prefill_lines = [line for line in window if "Prefill batch." in line]
    cached_values = []
    hit_lines = []
    for line in prefill_lines:
        match = CACHED_TOKEN_RE.search(line)
        cached = int(match.group(1)) if match else 0
        cached_values.append(cached)
        if cached > 0:
            hit_lines.append(line)

    report = {
        "passed": len(hit_lines) >= args.expected_min_hits,
        "server_log": str(server_log),
        "start_line": args.start_line,
        "window_line_count": len(window),
        "prefill_line_count": len(prefill_lines),
        "cache_hit_line_count": len(hit_lines),
        "expected_min_hits": args.expected_min_hits,
        "cached_tokens_sum": sum(cached_values),
        "cached_tokens_max": max(cached_values, default=0),
        "sample_hit_lines": hit_lines[:32],
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GLM52_EVALSCOPE_SERVER_PREFIX_CACHE_AUDIT", json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("EvalScope server-log window did not contain enough prefix-cache hits")


if __name__ == "__main__":
    main()
