#!/usr/bin/env python3
"""Validate one structured tool call against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import urllib.error
import urllib.request
from typing import Any


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"tool arguments must be a JSON string or object, got {type(value).__name__}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"decoded tool arguments must be an object, got {type(parsed).__name__}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--expected-value", default="falcon-officeqa-smoke")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call echo_smoke exactly once with value "
                    f"{args.expected_value!r}. Do not answer in plain text."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo_smoke",
                    "description": "Return a smoke-test marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": [args.expected_value],
                            }
                        },
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "echo_smoke"}},
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "seed": 3,
        "stream": False,
        # The server must override this to false for named-tool grammar mode.
        # This directly exercises the GLM tool-call fix merged into epic.
        "chat_template_kwargs": {"enable_thinking": True},
        "rid": "falcon-officeqa-smoke-tool-preflight",
    }
    (output_dir / "request.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        report = {"ok": False, "endpoint": endpoint, "error": repr(exc)}
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise

    (output_dir / "response.json").write_text(raw_body + ("" if raw_body.endswith("\n") else "\n"))
    report: dict[str, Any] = {"ok": False, "endpoint": endpoint, "http_status": status}
    try:
        body = json.loads(raw_body)
        choice = body["choices"][0]
        message = choice["message"]
        calls = message["tool_calls"]
        if not calls:
            raise ValueError("response contains no tool_calls")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content not in (None, ""):
            raise ValueError("named-tool grammar response unexpectedly contains reasoning_content")
        visible_content = message.get("content") or ""
        if "<think>" in visible_content or "</think>" in visible_content:
            raise ValueError("thinking tags leaked into named-tool response content")
        function = calls[0]["function"]
        tool_args = _parse_arguments(function["arguments"])
        if function.get("name") != "echo_smoke":
            raise ValueError(f"unexpected tool name: {function.get('name')!r}")
        if tool_args.get("value") != args.expected_value:
            raise ValueError(f"unexpected tool value: {tool_args.get('value')!r}")
        report.update(
            ok=True,
            finish_reason=choice.get("finish_reason"),
            tool_call_count=len(calls),
            tool_name=function.get("name"),
            tool_arguments=tool_args,
            reasoning_content_empty=True,
            usage=body.get("usage"),
        )
    except Exception as exc:
        report["error"] = repr(exc)

    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GLM52_OPENAI_TOOL_CALL_VALIDATION", json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("OpenAI structured tool-call preflight failed")


if __name__ == "__main__":
    main()
