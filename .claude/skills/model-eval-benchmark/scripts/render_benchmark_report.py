#!/usr/bin/env python3
"""Render a benchmark markdown report from accuracy and speed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pick_score(metrics: dict) -> str:
    if not metrics:
        return ""
    for key in ("score", "final_score", "acc", "accuracy"):
        value = metrics.get(key)
        if value is not None:
            return str(value)
    if len(metrics) == 1:
        only_value = next(iter(metrics.values()))
        if only_value is not None:
            return str(only_value)
    return ""


def _read_csv_rows(path: str | None) -> tuple[list[str], list[list[str]]]:
    if not path:
        return [], []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _format_paths(paths: list[str]) -> str:
    if not paths:
        return "- None"
    return "\n".join(f"- `{path}`" for path in paths)


def _parse_kv(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        pairs.append((key, value))
    return pairs


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output markdown path.")
    parser.add_argument("--dataset", required=True, help="Accuracy dataset name.")
    parser.add_argument("--accuracy-command", required=True, help="Executed accuracy command.")
    parser.add_argument("--speed-command", required=True, help="Executed speed command(s).")
    parser.add_argument("--target-resource", required=True, help="Target machine or cluster.")
    parser.add_argument("--endpoint", required=True, help="Server endpoint.")
    parser.add_argument("--model-path", required=True, help="Served model path.")
    parser.add_argument("--sample-count", type=int, required=True, help="Accuracy sample count.")
    parser.add_argument("--server-id", default="", help="Server PID or Falcon exp_id.")
    parser.add_argument("--accuracy-json", default="", help="Accuracy metrics json path.")
    parser.add_argument("--accuracy-artifact", action="append", default=[], help="Accuracy artifact path. Repeatable.")
    parser.add_argument("--accuracy-note", action="append", default=[], help="Extra accuracy result note. Repeatable.")
    parser.add_argument("--speed-output-dir", default="", help="Speed benchmark output directory.")
    parser.add_argument("--speed-summary-csv", default="", help="Speed summary csv path.")
    parser.add_argument("--speed-artifact", action="append", default=[], help="Speed artifact path. Repeatable.")
    parser.add_argument("--speed-note", action="append", default=[], help="Extra speed result note. Repeatable.")
    parser.add_argument("--speed-kv", action="append", default=[], help="Extra speed metadata as KEY=VALUE. Repeatable.")
    args = parser.parse_args()

    accuracy_metrics = _read_json(args.accuracy_json)
    speed_header, speed_rows = _read_csv_rows(args.speed_summary_csv)
    speed_pairs = _parse_kv(args.speed_kv)

    accuracy_score = _pick_score(accuracy_metrics)

    accuracy_lines = [
        f"- Target resource: `{args.target_resource}`",
        f"- Endpoint: `{args.endpoint}`",
        f"- Model path: `{args.model_path}`",
        f"- Sample count: `{args.sample_count}`",
        f"- Score: `{accuracy_score}`",
    ]
    if args.server_id:
        accuracy_lines.append(f"- Server ID: `{args.server_id}`")
    if args.accuracy_json:
        accuracy_lines.append(f"- Accuracy metrics file: `{args.accuracy_json}`")
    if args.accuracy_artifact:
        accuracy_lines.append("- Result artifacts:")
        accuracy_lines.append(_format_paths(args.accuracy_artifact))
    accuracy_lines.extend(f"- {note}" for note in args.accuracy_note)

    speed_lines = [
        f"- Target resource: `{args.target_resource}`",
        f"- Endpoint: `{args.endpoint}`",
    ]
    if args.speed_output_dir:
        speed_lines.append(f"- Output directory: `{args.speed_output_dir}`")
    if args.speed_summary_csv:
        speed_lines.append(f"- Summary file: `{args.speed_summary_csv}`")
    for key, value in speed_pairs:
        speed_lines.append(f"- {key}: `{value}`")
    if speed_header:
        speed_lines.append("")
        speed_lines.append(_markdown_table(speed_header, speed_rows))
    if args.speed_artifact:
        speed_lines.append("")
        speed_lines.append("- Raw result files:")
        speed_lines.append(_format_paths(args.speed_artifact))
    speed_lines.extend(f"- {note}" for note in args.speed_note)

    report = f"""# Benchmark

## Accuracy Benchmark

### {args.dataset}

```bash
{args.accuracy_command}
```

#### Results

{chr(10).join(accuracy_lines)}

## Speed Benchmark

```bash
{args.speed_command}
```

#### Results

{chr(10).join(speed_lines)}
"""

    Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
