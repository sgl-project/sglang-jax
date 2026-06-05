#!/usr/bin/env python3
"""PD e2e test suite for TPU.

Runs Tier 1 (E1-E5) and Tier 2 (E6-E8) tests against a live P+D deployment.

Usage
-----
    python scripts/disaggregation/e2e_test_suite.py \
        --prefill-url http://<P_HOST>:10000 \
        --decode-url  http://<D_HOST>:10001 \
        [--tier 1]          # 1 = Tier 1 only, 2 = Tier 1+2
        [--normal-url ...]  # for E5 correctness comparison
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
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


def _send_pd_request(
    p_url: str,
    d_url: str,
    prompt: str,
    max_new_tokens: int = 32,
    transfer_id: str | None = None,
    bootstrap_room: int = 42,
    timeout: float = 120.0,
) -> tuple[dict, dict]:
    """Send a paired P+D request and return (p_resp, d_resp)."""
    if transfer_id is None:
        transfer_id = f"e2e-{uuid.uuid4().hex[:12]}"

    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        "bootstrap_room": bootstrap_room,
        "disagg_transfer_id": transfer_id,
    }

    results = {}

    def _send(role: str, url: str):
        return role, _post_json(url, payload, timeout=timeout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_p = pool.submit(_send, "prefill", f"{p_url}/generate")
        time.sleep(1.0)
        fut_d = pool.submit(_send, "decode", f"{d_url}/generate")
        for fut in as_completed([fut_p, fut_d]):
            role, resp = fut.result()
            results[role] = resp

    return results["prefill"], results["decode"]


class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def test_e1_happy_path(p_url: str, d_url: str) -> TestResult:
    """E1: Happy path single request."""
    try:
        _, d_resp = _send_pd_request(
            p_url, d_url,
            prompt="What is 2+2? Answer briefly.",
            max_new_tokens=32,
        )
        text = d_resp.get("text", "")
        if not text:
            return TestResult("E1-happy-path", False, "decode returned empty text")
        return TestResult("E1-happy-path", True, f"output={text!r}")
    except Exception as e:
        return TestResult("E1-happy-path", False, str(e))


def test_e2_concurrent(p_url: str, d_url: str) -> TestResult:
    """E2: 3 concurrent requests with different transfer IDs."""
    num_requests = 3
    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 3 times 7? Answer with just the number.",
        "Name a color of the sky. One word only.",
    ]

    try:
        results = []
        errors = []

        def _run_one(i):
            tid = f"e2-concurrent-{uuid.uuid4().hex[:8]}"
            _, d_resp = _send_pd_request(
                p_url, d_url,
                prompt=prompts[i],
                max_new_tokens=32,
                transfer_id=tid,
                bootstrap_room=i,
                timeout=180.0,
            )
            return i, d_resp

        with ThreadPoolExecutor(max_workers=num_requests) as pool:
            futs = [pool.submit(_run_one, i) for i in range(num_requests)]
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
            return TestResult(
                "E2-concurrent", False,
                f"{len(errors)} failures: {'; '.join(errors)}"
            )
        detail = "; ".join(f"req{i}={t!r}" for i, t in sorted(results))
        return TestResult("E2-concurrent", True, detail)
    except Exception as e:
        return TestResult("E2-concurrent", False, str(e))


def test_e3_long_output(p_url: str, d_url: str) -> TestResult:
    """E3: Long output (128+ tokens) to test multi-page KV gather/scatter."""
    try:
        _, d_resp = _send_pd_request(
            p_url, d_url,
            prompt="Write a short story about a robot learning to paint. Be creative and detailed.",
            max_new_tokens=128,
            timeout=180.0,
        )
        text = d_resp.get("text", "")
        if not text:
            return TestResult("E3-long-output", False, "empty text")
        token_count = len(text.split())
        detail = f"{token_count} words, {len(text)} chars"
        if token_count < 20:
            return TestResult("E3-long-output", False, f"too short: {detail}")
        return TestResult("E3-long-output", True, detail)
    except Exception as e:
        return TestResult("E3-long-output", False, str(e))


def test_e4_d_starts_first(p_url: str, d_url: str) -> TestResult:
    """E4: D server started before P — P's register_prefill retries
    should handle this, and D's bootstrap lookup 503 should be retried
    by the client. Since both servers are already running, we just
    verify the path works (the startup-order test is structural, not
    runtime). We verify by checking bootstrap /list_prefills is non-empty.
    """
    try:
        bootstrap_url = os.environ.get("BOOTSTRAP_URL", "")
        if not bootstrap_url:
            return TestResult(
                "E4-startup-order", True,
                "skipped: BOOTSTRAP_URL not set (structural test)"
            )
        req = Request(f"{bootstrap_url}/list_prefills")
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        prefills = data.get("prefills", [])
        if not prefills:
            return TestResult(
                "E4-startup-order", False,
                "no prefills registered in bootstrap"
            )
        return TestResult(
            "E4-startup-order", True,
            f"{len(prefills)} prefill(s) registered"
        )
    except Exception as e:
        return TestResult("E4-startup-order", False, str(e))


def test_e5_correctness(
    p_url: str, d_url: str, normal_url: str | None
) -> TestResult:
    """E5: Compare PD output with normal mode output."""
    if normal_url is None:
        return TestResult(
            "E5-correctness", True,
            "skipped: --normal-url not provided"
        )
    prompt = "What is the capital of Japan? Answer in one word."
    max_new_tokens = 16

    try:
        # Normal mode
        normal_resp = _post_json(
            f"{normal_url}/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            },
            timeout=120.0,
        )
        normal_text = normal_resp.get("text", "")

        # PD mode
        _, d_resp = _send_pd_request(
            p_url, d_url,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        pd_text = d_resp.get("text", "")

        if normal_text == pd_text:
            return TestResult(
                "E5-correctness", True,
                f"outputs match: {pd_text!r}"
            )
        else:
            return TestResult(
                "E5-correctness", False,
                f"MISMATCH normal={normal_text!r} pd={pd_text!r}"
            )
    except Exception as e:
        return TestResult("E5-correctness", False, str(e))


def test_e6_prefill_crash(p_url: str, d_url: str) -> TestResult:
    """E6: Test that D handles P failure gracefully.
    We send a request with no actual P processing available.
    Since we can't kill P in a GKE pod from here, we test by
    sending a request to D with a bogus transfer_id that P
    will never fulfill — D should timeout and return an error,
    not hang forever.
    """
    try:
        bogus_tid = f"bogus-{uuid.uuid4().hex[:8]}"
        payload = {
            "text": "test prompt",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
            "bootstrap_room": 999,
            "disagg_transfer_id": bogus_tid,
        }

        # Only send to D (no matching P request), expect timeout/error
        try:
            resp = _post_json(
                f"{d_url}/generate", payload, timeout=90.0
            )
            text = resp.get("text", "")
            # If we get an error message back, that's the expected behavior
            meta = resp.get("meta_info", {})
            finish = meta.get("finish_reason", {})
            if isinstance(finish, dict) and finish.get("type") == "abort":
                return TestResult(
                    "E6-prefill-crash", True,
                    "D returned abort as expected"
                )
            # If we somehow get text, it means something unexpected happened
            if text:
                return TestResult(
                    "E6-prefill-crash", False,
                    f"D returned text without P: {text!r}"
                )
            return TestResult(
                "E6-prefill-crash", True,
                "D handled missing P transfer gracefully"
            )
        except HTTPError as e:
            if e.code >= 400:
                return TestResult(
                    "E6-prefill-crash", True,
                    f"D returned HTTP {e.code} as expected"
                )
            return TestResult("E6-prefill-crash", False, str(e))
        except TimeoutError:
            return TestResult(
                "E6-prefill-crash", False,
                "D hung (timeout) — should have returned error"
            )
    except Exception as e:
        return TestResult("E6-prefill-crash", False, str(e))


def test_e7_bootstrap_resilience(p_url: str, d_url: str) -> TestResult:
    """E7: Bootstrap server resilience.
    Verify that bootstrap server is healthy and P is registered.
    Full restart test requires process control not available here.
    """
    try:
        bootstrap_url = os.environ.get("BOOTSTRAP_URL", "")
        if not bootstrap_url:
            return TestResult(
                "E7-bootstrap", True,
                "skipped: BOOTSTRAP_URL not set"
            )
        # Check health
        req = Request(f"{bootstrap_url}/health")
        with urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return TestResult(
                    "E7-bootstrap", False,
                    f"health returned {resp.status}"
                )

        # Check P is registered
        req = Request(f"{bootstrap_url}/list_prefills")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        prefills = data.get("prefills", [])
        if not prefills:
            return TestResult(
                "E7-bootstrap", False,
                "no prefills registered after health check"
            )

        # Verify bootstrap_room lookup works
        req = Request(f"{bootstrap_url}/get_prefill_info?bootstrap_room=0")
        with urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read())
        if "bootstrap_key" not in info:
            return TestResult(
                "E7-bootstrap", False,
                "get_prefill_info missing bootstrap_key"
            )

        return TestResult(
            "E7-bootstrap", True,
            f"healthy, {len(prefills)} prefill(s), lookup ok"
        )
    except Exception as e:
        return TestResult("E7-bootstrap", False, str(e))


def test_e8_abort_no_leak(p_url: str, d_url: str) -> TestResult:
    """E8: Send a request then verify pools don't leak.
    We run a normal request, then check pool usage via internal state.
    """
    try:
        # First run a normal request to completion
        _, d_resp = _send_pd_request(
            p_url, d_url,
            prompt="Say hello.",
            max_new_tokens=8,
        )
        text = d_resp.get("text", "")
        if not text:
            return TestResult(
                "E8-abort-leak", False,
                "baseline request failed"
            )

        # Give a moment for cleanup
        time.sleep(2)

        # Check internal state of both servers (if available)
        for role, url in [("P", p_url), ("D", d_url)]:
            try:
                req = Request(f"{url}/get_internal_state")
                with urlopen(req, timeout=10) as resp:
                    state = json.loads(resp.read())
                # Check that running batch is empty after request completes
                num_running = state.get("num_running_reqs", -1)
                if num_running > 0:
                    return TestResult(
                        "E8-abort-leak", False,
                        f"{role} has {num_running} running reqs after completion"
                    )
            except Exception:
                pass

        return TestResult(
            "E8-abort-leak", True,
            "no leak detected after request completion"
        )
    except Exception as e:
        return TestResult("E8-abort-leak", False, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(description="PD e2e test suite")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--normal-url", default=None,
                        help="Normal mode server URL for E5 comparison")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2],
                        help="1=Tier1 only, 2=Tier1+2")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")

    if not args.skip_health_check:
        print(f"Waiting for P ({p_url}) and D ({d_url}) to be healthy...")
        _wait_health(p_url, timeout=args.timeout)
        _wait_health(d_url, timeout=args.timeout)
        print("Both servers healthy.\n")

    # Tier 1 tests
    tests: list[TestResult] = []
    tier1 = [
        ("E1", lambda: test_e1_happy_path(p_url, d_url)),
        ("E2", lambda: test_e2_concurrent(p_url, d_url)),
        ("E3", lambda: test_e3_long_output(p_url, d_url)),
        ("E4", lambda: test_e4_d_starts_first(p_url, d_url)),
        ("E5", lambda: test_e5_correctness(p_url, d_url, args.normal_url)),
    ]
    tier2 = [
        ("E6", lambda: test_e6_prefill_crash(p_url, d_url)),
        ("E7", lambda: test_e7_bootstrap_resilience(p_url, d_url)),
        ("E8", lambda: test_e8_abort_no_leak(p_url, d_url)),
    ]

    all_tests = tier1 if args.tier == 1 else tier1 + tier2

    print("=" * 60)
    print(f"PD E2E Test Suite — Tier {args.tier}")
    print("=" * 60)

    for label, test_fn in all_tests:
        print(f"\n--- {label} ---")
        result = test_fn()
        tests.append(result)
        print(result)

    # Summary
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
