#!/usr/bin/env python3
"""Audit EvalScope review caches for complete, replayable agent traces."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
from typing import Any


THINK_TAGS = ("<think>", "</think>")
NONZERO_EXIT_RE = re.compile(r"\[exit\s+([1-9][0-9]*)\]")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-max-steps", type=int, required=True)
    parser.add_argument("--require-tools", action="store_true")
    parser.add_argument("--require-reasoning-separation", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    work_dir = pathlib.Path(args.work_dir)
    output = pathlib.Path(args.output)
    review_files = sorted((work_dir / "reviews").glob("**/*.jsonl"))
    rows: list[tuple[pathlib.Path, dict[str, Any]]] = []
    issues: list[str] = []
    quality_issues: list[str] = []
    for review_file in review_files:
        try:
            rows.extend((review_file, row) for row in _load_jsonl(review_file))
        except ValueError as exc:
            issues.append(str(exc))

    if len(rows) != args.expected_samples:
        issues.append(f"expected {args.expected_samples} review rows, found {len(rows)}")

    overall_events: collections.Counter[str] = collections.Counter()
    overall_reasoning_parts = 0
    overall_reasoning_chars = 0
    overall_nonzero_tool_exits: collections.Counter[int] = collections.Counter()
    sample_summaries = []
    for review_file, row in rows:
        sample_id = row.get("index")
        trace = row.get("agent_trace")
        sample_issues: list[str] = []
        sample_quality_issues: list[str] = []
        if not isinstance(trace, dict):
            sample_issues.append("missing agent_trace")
            events: list[dict[str, Any]] = []
        else:
            events = trace.get("events") or []
            if trace.get("strategy") != "function_calling":
                sample_issues.append(f"unexpected strategy={trace.get('strategy')!r}")
            if trace.get("max_steps") != args.expected_max_steps:
                sample_issues.append(
                    f"expected max_steps={args.expected_max_steps}, got {trace.get('max_steps')!r}"
                )

        event_counts: collections.Counter[str] = collections.Counter()
        tool_names: list[str] = []
        call_ids: set[str] = set()
        result_ids: set[str] = set()
        tool_errors: list[dict[str, Any]] = []
        trace_errors: list[dict[str, Any]] = []
        reasoning_part_count = 0
        reasoning_char_count = 0
        nonzero_tool_exits: collections.Counter[int] = collections.Counter()
        leaked_think_fields: list[str] = []
        timestamps_valid = True
        for event in events:
            event_type = str(event.get("type"))
            event_counts[event_type] += 1
            overall_events[event_type] += 1
            timestamp = event.get("timestamp")
            timestamps_valid = timestamps_valid and isinstance(timestamp, (int, float)) and timestamp > 0
            payload = event.get("payload") or {}
            if event_type == "tool_call":
                if payload.get("name"):
                    tool_names.append(str(payload["name"]))
                if payload.get("id"):
                    call_ids.add(str(payload["id"]))
            elif event_type == "tool_result":
                if payload.get("id"):
                    result_ids.add(str(payload["id"]))
                if payload.get("error"):
                    tool_errors.append(payload)
            elif event_type == "error":
                trace_errors.append(payload)

        messages = row.get("messages")
        if not isinstance(messages, list):
            sample_issues.append("missing or invalid messages")
            messages = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                sample_issues.append(f"message {message_index} is not an object")
                continue
            role = message.get("role")
            content = message.get("content")
            content_parts = content if isinstance(content, list) else [content]
            for part_index, part in enumerate(content_parts):
                field_name = f"messages[{message_index}].content[{part_index}]"
                if isinstance(part, str):
                    text = part
                elif isinstance(part, dict) and part.get("type") == "reasoning":
                    text = part.get("reasoning")
                    if isinstance(text, str) and text:
                        reasoning_part_count += 1
                        reasoning_char_count += len(text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                else:
                    text = None
                if isinstance(text, str) and any(tag in text for tag in THINK_TAGS):
                    leaked_think_fields.append(field_name)
                if role == "tool" and isinstance(text, str):
                    for match in NONZERO_EXIT_RE.finditer(text):
                        nonzero_tool_exits[int(match.group(1))] += 1
            if role == "assistant":
                for call_index, call in enumerate(message.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    arguments = (call.get("function") or {}).get("arguments")
                    if isinstance(arguments, str):
                        serialized_arguments = arguments
                    else:
                        serialized_arguments = json.dumps(arguments, sort_keys=True, default=str)
                    if any(tag in serialized_arguments for tag in THINK_TAGS):
                        leaked_think_fields.append(
                            f"messages[{message_index}].tool_calls[{call_index}].function.arguments"
                        )
            if role == "tool" and message.get("error"):
                sample_quality_issues.append(f"tool message {message_index} records an error")

        if leaked_think_fields:
            sample_issues.append(f"thinking tags leaked into {sorted(set(leaked_think_fields))}")
        if args.require_reasoning_separation and reasoning_part_count < 1:
            sample_quality_issues.append("no structured reasoning content recorded for this sample")
        if nonzero_tool_exits:
            sample_quality_issues.append(
                "tool messages contain non-zero command exits: "
                + ", ".join(f"exit {code} x{count}" for code, count in sorted(nonzero_tool_exits.items()))
            )
        overall_reasoning_parts += reasoning_part_count
        overall_reasoning_chars += reasoning_char_count
        overall_nonzero_tool_exits.update(nonzero_tool_exits)

        if not timestamps_valid:
            sample_issues.append("one or more trace events have invalid timestamps")
        if event_counts["model_generate"] < 1:
            sample_issues.append("missing model_generate event")
        if event_counts["submit"] < 1 and not trace_errors:
            sample_issues.append("missing terminal submit or error event")
        if trace_errors:
            messages = [str(payload.get("message")) for payload in trace_errors]
            sample_quality_issues.append(
                f"trace terminated with {len(trace_errors)} error event(s): {messages}"
            )
        if tool_errors:
            sample_quality_issues.append(
                f"trace contains {len(tool_errors)} failed tool result(s)"
            )
        if args.require_tools:
            if event_counts["tool_call"] < 1:
                sample_issues.append("missing tool_call event")
            if event_counts["tool_result"] < 1:
                sample_issues.append("missing tool_result event")
            if call_ids != result_ids:
                sample_issues.append(
                    f"tool call/result id mismatch calls={sorted(call_ids)} results={sorted(result_ids)}"
                )

        for issue in sample_issues:
            issues.append(f"sample {sample_id}: {issue}")
        for issue in sample_quality_issues:
            quality_issues.append(f"sample {sample_id}: {issue}")
        score = ((row.get("sample_score") or {}).get("score") or {})
        sample_summaries.append(
            {
                "sample_id": sample_id,
                "review_file": str(review_file.relative_to(work_dir)),
                "framework": trace.get("framework") if isinstance(trace, dict) else None,
                "strategy": trace.get("strategy") if isinstance(trace, dict) else None,
                "environment": trace.get("environment") if isinstance(trace, dict) else None,
                "max_steps": trace.get("max_steps") if isinstance(trace, dict) else None,
                "event_count": len(events),
                "event_counts": dict(sorted(event_counts.items())),
                "tool_names": tool_names,
                "reasoning_part_count": reasoning_part_count,
                "reasoning_char_count": reasoning_char_count,
                "nonzero_tool_exit_codes": dict(sorted(nonzero_tool_exits.items())),
                "total_usage": trace.get("total_usage") if isinstance(trace, dict) else None,
                "score": score,
                "issues": sample_issues,
                "quality_issues": sample_quality_issues,
            }
        )

    if args.require_reasoning_separation and overall_reasoning_parts < 1:
        issues.append("thinking is enabled but no structured reasoning content was recorded")

    report = {
        "passed": not issues,
        "work_dir": str(work_dir),
        "expected_samples": args.expected_samples,
        "review_files": [str(path.relative_to(work_dir)) for path in review_files],
        "review_rows": len(rows),
        "trace_rows": sum(1 for _, row in rows if isinstance(row.get("agent_trace"), dict)),
        "event_counts": dict(sorted(overall_events.items())),
        "reasoning_part_count": overall_reasoning_parts,
        "reasoning_char_count": overall_reasoning_chars,
        "nonzero_tool_exit_codes": dict(sorted(overall_nonzero_tool_exits.items())),
        "samples": sample_summaries,
        "issues": issues,
        "quality_issues": quality_issues,
        "accuracy_is_gate": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GLM52_EVALSCOPE_TRACE_AUDIT", json.dumps(report, sort_keys=True))
    if issues:
        raise SystemExit("EvalScope agent trace audit failed")


if __name__ == "__main__":
    main()
