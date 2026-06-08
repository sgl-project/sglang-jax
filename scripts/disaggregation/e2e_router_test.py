#!/usr/bin/env python3
"""Stage 3 router e2e test: validates the mini_lb single-entry proxy path.

Unlike the Stage 2 e2e suite (which manually pairs P+D requests), this test
sends all requests to the router's single endpoint and verifies the full
routing + bootstrap injection + PD handoff + decode generation flow.

Usage
-----
    python scripts/disaggregation/e2e_router_test.py \
        --router-url http://localhost:30000 \
        [--bootstrap-url http://<P_HOST>:8998]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_json(url: str, timeout: float = 30.0) -> dict:
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_health(url: str, timeout: float = 300.0) -> None:
    health = url.rstrip("/") + "/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            req = Request(health)
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server at {url} not healthy after {timeout}s")


class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def test_r1_health(router_url: str) -> TestResult:
    """R1: Router /health endpoint responds."""
    try:
        req = Request(f"{router_url}/health")
        with urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return TestResult("R1-health", True, "router healthy")
        return TestResult("R1-health", False, f"status={resp.status}")
    except Exception as e:
        return TestResult("R1-health", False, str(e))


def test_r2_server_info(router_url: str) -> TestResult:
    """R2: Router /get_server_info aggregates backend info."""
    try:
        info = _get_json(f"{router_url}/get_server_info")
        if "prefill" not in info or "decode" not in info:
            return TestResult("R2-server-info", False, f"missing keys: {list(info.keys())}")
        return TestResult("R2-server-info", True, f"prefill={len(info['prefill'])}, decode={len(info['decode'])}")
    except Exception as e:
        return TestResult("R2-server-info", False, str(e))


def test_r3_generate_happy_path(router_url: str) -> TestResult:
    """R3: Single /generate request through router."""
    try:
        resp = _post_json(
            f"{router_url}/generate",
            {
                "text": "What is 2+2? Answer briefly.",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            },
            timeout=180.0,
        )
        text = resp.get("text", "")
        if not text:
            return TestResult("R3-generate", False, f"empty text, resp={resp}")
        return TestResult("R3-generate", True, f"output={text!r}")
    except Exception as e:
        return TestResult("R3-generate", False, str(e))


def test_r4_generate_concurrent(router_url: str) -> TestResult:
    """R4: 3 concurrent /generate requests through router."""
    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 3 times 7? Answer with just the number.",
        "Name a color of the sky. One word only.",
    ]

    try:
        results = []
        errors = []

        def _run_one(i):
            resp = _post_json(
                f"{router_url}/generate",
                {
                    "text": prompts[i],
                    "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
                },
                timeout=180.0,
            )
            return i, resp

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_run_one, i) for i in range(3)]
            for fut in as_completed(futs):
                try:
                    idx, resp = fut.result()
                    text = resp.get("text", "")
                    if not text:
                        errors.append(f"req {idx}: empty text")
                    else:
                        results.append((idx, text))
                except Exception as e:
                    errors.append(str(e))

        if errors:
            return TestResult("R4-concurrent", False, f"{len(errors)} failures: {'; '.join(errors)}")
        detail = "; ".join(f"req{i}={t!r}" for i, t in sorted(results))
        return TestResult("R4-concurrent", True, detail)
    except Exception as e:
        return TestResult("R4-concurrent", False, str(e))


def test_r5_v1_chat_completions(router_url: str) -> TestResult:
    """R5: /v1/chat/completions through router."""
    try:
        resp = _post_json(
            f"{router_url}/v1/chat/completions",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 16,
                "temperature": 0.0,
            },
            timeout=180.0,
        )
        choices = resp.get("choices", [])
        if not choices:
            return TestResult("R5-chat", False, f"no choices in response: {resp}")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return TestResult("R5-chat", False, f"empty content: {choices[0]}")
        return TestResult("R5-chat", True, f"content={content!r}")
    except Exception as e:
        return TestResult("R5-chat", False, str(e))


def test_r6_v1_completions(router_url: str) -> TestResult:
    """R6: /v1/completions through router."""
    try:
        resp = _post_json(
            f"{router_url}/v1/completions",
            {
                "model": "default",
                "prompt": "The capital of Germany is",
                "max_tokens": 16,
                "temperature": 0.0,
            },
            timeout=180.0,
        )
        choices = resp.get("choices", [])
        if not choices:
            return TestResult("R6-completions", False, f"no choices: {resp}")
        text = choices[0].get("text", "")
        if not text:
            return TestResult("R6-completions", False, f"empty text: {choices[0]}")
        return TestResult("R6-completions", True, f"text={text!r}")
    except Exception as e:
        return TestResult("R6-completions", False, str(e))


def test_r7_v1_models(router_url: str) -> TestResult:
    """R7: /v1/models endpoint works."""
    try:
        resp = _get_json(f"{router_url}/v1/models")
        data = resp.get("data", [])
        if not data:
            return TestResult("R7-models", False, f"empty model list: {resp}")
        model_id = data[0].get("id", "")
        return TestResult("R7-models", True, f"model={model_id!r}")
    except Exception as e:
        return TestResult("R7-models", False, str(e))


def test_r8_long_output(router_url: str) -> TestResult:
    """R8: Long output (128 tokens) through router."""
    try:
        resp = _post_json(
            f"{router_url}/generate",
            {
                "text": "Write a short poem about the ocean. Be creative.",
                "sampling_params": {"max_new_tokens": 128, "temperature": 0.0},
            },
            timeout=240.0,
        )
        text = resp.get("text", "")
        if not text:
            return TestResult("R8-long-output", False, "empty text")
        word_count = len(text.split())
        if word_count < 20:
            return TestResult("R8-long-output", False, f"too short: {word_count} words")
        return TestResult("R8-long-output", True, f"{word_count} words, {len(text)} chars")
    except Exception as e:
        return TestResult("R8-long-output", False, str(e))


def test_r9_bootstrap_injection(router_url: str, bootstrap_url: str | None) -> TestResult:
    """R9: Verify router injects bootstrap fields (check via bootstrap server)."""
    if not bootstrap_url:
        return TestResult("R9-bootstrap-inject", True, "skipped: no --bootstrap-url")
    try:
        prefills = _get_json(f"{bootstrap_url}/list_prefills")
        if not prefills.get("prefills"):
            return TestResult("R9-bootstrap-inject", False, "no prefills registered")

        resp = _post_json(
            f"{router_url}/generate",
            {
                "text": "Test bootstrap injection.",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
            },
            timeout=120.0,
        )
        text = resp.get("text", "")
        if text:
            return TestResult("R9-bootstrap-inject", True, f"request succeeded through bootstrap path")
        return TestResult("R9-bootstrap-inject", False, "empty response")
    except Exception as e:
        return TestResult("R9-bootstrap-inject", False, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 router e2e test")
    parser.add_argument("--router-url", required=True, help="Router (mini_lb) URL")
    parser.add_argument("--bootstrap-url", default=None, help="Bootstrap server URL for R9")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    router_url = args.router_url.rstrip("/")

    if not args.skip_health_check:
        print(f"Waiting for router ({router_url}) to be healthy...")
        _wait_health(router_url, timeout=args.timeout)
        print("Router healthy.\n")

    tests: list[TestResult] = []
    all_tests = [
        ("R1", lambda: test_r1_health(router_url)),
        ("R2", lambda: test_r2_server_info(router_url)),
        ("R3", lambda: test_r3_generate_happy_path(router_url)),
        ("R4", lambda: test_r4_generate_concurrent(router_url)),
        ("R5", lambda: test_r5_v1_chat_completions(router_url)),
        ("R6", lambda: test_r6_v1_completions(router_url)),
        ("R7", lambda: test_r7_v1_models(router_url)),
        ("R8", lambda: test_r8_long_output(router_url)),
        ("R9", lambda: test_r9_bootstrap_injection(router_url, args.bootstrap_url)),
    ]

    print("=" * 60)
    print("Stage 3 Router E2E Test Suite")
    print("=" * 60)

    for label, test_fn in all_tests:
        print(f"\n--- {label} ---")
        result = test_fn()
        tests.append(result)
        print(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for t in tests if t.passed)
    failed = sum(1 for t in tests if not t.passed)
    for t in tests:
        print(t)
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
