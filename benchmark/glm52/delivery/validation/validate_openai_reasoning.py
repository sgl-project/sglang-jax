#!/usr/bin/env python3
"""Validate GLM thinking-field separation on an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.error
import urllib.request
from typing import Any


THINK_TAGS = ("<think>", "</think>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Compute 17 * 24. Think through it briefly, then state the final answer.",
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "seed": 3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "rid": "falcon-officeqa-smoke-reasoning-preflight",
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
        reasoning_content = message.get("reasoning_content")
        content = message.get("content")
        if not isinstance(reasoning_content, str) or not reasoning_content.strip():
            raise ValueError("response has no non-empty reasoning_content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response has no non-empty final content")
        leaked_tags = [tag for tag in THINK_TAGS if tag in reasoning_content or tag in content]
        if leaked_tags:
            raise ValueError(f"thinking tags leaked into response fields: {leaked_tags}")
        report.update(
            ok=True,
            finish_reason=choice.get("finish_reason"),
            reasoning_chars=len(reasoning_content),
            content_chars=len(content),
            usage=body.get("usage"),
        )
    except Exception as exc:
        report["error"] = repr(exc)

    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GLM52_OPENAI_REASONING_VALIDATION", json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("OpenAI reasoning-content preflight failed")


if __name__ == "__main__":
    main()
