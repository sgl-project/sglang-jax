#!/usr/bin/env python3
"""Validate that the OpenAI endpoint accepts a prompt above the legacy 135K cap."""

from __future__ import annotations

import argparse
import hashlib
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


def _prompt_tokens(tokenizer: Any, content: str) -> int:
    messages = [{"role": "user", "content": content}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(token_ids)


def _build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    unit = "Falcon long context capacity probe. "
    suffix = "\nReply with exactly OK."
    low, high = 1, target_tokens
    while _prompt_tokens(tokenizer, unit * high + suffix) < target_tokens:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if _prompt_tokens(tokenizer, unit * mid + suffix) < target_tokens:
            low = mid + 1
        else:
            high = mid
    content = unit * low + suffix
    return content, _prompt_tokens(tokenizer, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=150_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--flush-url")
    args = parser.parse_args()

    if args.target_prompt_tokens <= 135_168:
        raise SystemExit("target-prompt-tokens must exceed the legacy 135168-token cap")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from sgl_jax.srt.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    content, local_prompt_tokens = _build_prompt(tokenizer, args.target_prompt_tokens)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "seed": 3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "rid": "falcon-officeqa-long-context-preflight",
    }
    request_summary = {
        "endpoint": endpoint,
        "local_prompt_tokens": local_prompt_tokens,
        "max_tokens": args.max_tokens,
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    (output_dir / "request-summary.json").write_text(
        json.dumps(request_summary, indent=2, sort_keys=True) + "\n"
    )

    status, raw_body = _post(endpoint, payload, args.timeout)
    (output_dir / "response.json").write_text(
        raw_body + ("" if raw_body.endswith("\n") else "\n")
    )
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        body = {"raw_body": raw_body}
    usage = body.get("usage") if isinstance(body, dict) else None
    server_prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    report = {
        "ok": status == 200 and server_prompt_tokens >= args.target_prompt_tokens,
        "http_status": status,
        "target_prompt_tokens": args.target_prompt_tokens,
        "local_prompt_tokens": local_prompt_tokens,
        "server_prompt_tokens": server_prompt_tokens,
        "max_tokens": args.max_tokens,
        "finish_reason": (
            ((body.get("choices") or [{}])[0]).get("finish_reason")
            if isinstance(body, dict)
            else None
        ),
    }

    if args.flush_url:
        flush_status, flush_body = _post(args.flush_url, {}, args.timeout)
        report["flush_cache"] = {"http_status": flush_status, "body": flush_body}
        report["ok"] = report["ok"] and flush_status == 200

    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("GLM52_OPENAI_LONG_CONTEXT_VALIDATION", json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("OpenAI long-context preflight failed")


if __name__ == "__main__":
    main()
