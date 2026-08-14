#!/usr/bin/env python3
"""Fail fast unless a repeated OpenAI request reports cached prompt tokens."""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.error
import urllib.request
from typing import Any


def _post(endpoint: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _cached_tokens(body: dict[str, Any]) -> int:
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--minimum-cached-tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    unique_prefix = " ".join(
        f"falcon-officeqa-cache-probe-{index:03d}" for index in range(192)
    )
    base_payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": f"Cache this exact validation prefix. {unique_prefix}\nReply only READY.",
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "seed": 3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    responses: list[dict[str, Any]] = []
    for attempt in (1, 2):
        payload = dict(base_payload)
        payload["rid"] = f"falcon-officeqa-prefix-cache-preflight-{attempt}"
        (output_dir / f"request-{attempt}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        status, raw_body = _post(endpoint, payload, args.timeout)
        (output_dir / f"response-{attempt}.json").write_text(
            raw_body + ("" if raw_body.endswith("\n") else "\n")
        )
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {"raw_body": raw_body}
        responses.append(
            {
                "attempt": attempt,
                "http_status": status,
                "cached_tokens": _cached_tokens(body),
                "usage": body.get("usage") if isinstance(body, dict) else None,
            }
        )

    second_cached = responses[1]["cached_tokens"]
    report = {
        "ok": all(response["http_status"] == 200 for response in responses)
        and second_cached >= args.minimum_cached_tokens,
        "endpoint": endpoint,
        "minimum_cached_tokens": args.minimum_cached_tokens,
        "responses": responses,
        "second_request_cached_tokens": second_cached,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("GLM52_PREFIX_CACHE_VALIDATION", json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("OpenAI prefix-cache preflight failed")


if __name__ == "__main__":
    main()
