#!/usr/bin/env python3
"""Real TPU dual-process PD e2e test.

Two independent machines: Machine A (prefill) and Machine B (decode).
Bootstrap server runs on Machine A.

Usage
-----
Step 1: Start bootstrap server on Machine A:
    python -m sgl_jax.srt.disaggregation.run_bootstrap \
        --host 0.0.0.0 --port 8998

Step 2: Start P server on Machine A:
    python -m sgl_jax.launch_server \
        --model-path Qwen/Qwen3-8B \
        --port 10000 \
        --page-size 128 \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-url http://<MACHINE_A_IP>:8998

Step 3: Start D server on Machine B:
    python -m sgl_jax.launch_server \
        --model-path Qwen/Qwen3-8B \
        --port 10001 \
        --page-size 128 \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-url http://<MACHINE_A_IP>:8998

Step 4: Run this test from any machine:
    python scripts/disaggregation/e2e_two_host.py \
        --prefill-url http://<MACHINE_A_IP>:10000 \
        --decode-url  http://<MACHINE_B_IP>:10001 \
        [--prompt "What is 2+2?"] \
        [--max-new-tokens 32]

The script sends the same prompt to both P and D with a shared
transfer ID. P prefills and serves KV to D; D generates tokens.
P's response completes only after D pulls the KV (SUCCESS ack).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


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


def main() -> int:
    parser = argparse.ArgumentParser(description="PD e2e two-host test")
    parser.add_argument("--prefill-url", required=True,
                        help="P server URL, e.g. http://10.0.0.1:10000")
    parser.add_argument("--decode-url", required=True,
                        help="D server URL, e.g. http://10.0.0.2:10001")
    parser.add_argument("--prompt", default="What is 2+2? Answer briefly.",
                        help="Prompt to send")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds")
    args = parser.parse_args()

    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")

    if not args.skip_health_check:
        print(f"Waiting for P ({p_url}) and D ({d_url}) to be healthy...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(_wait_health, p_url),
                pool.submit(_wait_health, d_url),
            ]
            for f in as_completed(futs):
                f.result()
        print("Both servers healthy.")

    transfer_id = f"e2e-{uuid.uuid4().hex[:12]}"
    bootstrap_room = 42

    payload_base = {
        "text": args.prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
        },
        "bootstrap_room": bootstrap_room,
        "disagg_transfer_id": transfer_id,
    }

    print(f"\ntransfer_id={transfer_id}  bootstrap_room={bootstrap_room}")
    print(f"prompt: {args.prompt!r}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    # P's /generate will block until D pulls KV (SUCCESS ack).
    # D's /generate will pull KV from P, decode, then return tokens.
    # Send both concurrently.
    p_generate = f"{p_url}/generate"
    d_generate = f"{d_url}/generate"

    print(f"\nSending to P ({p_generate}) and D ({d_generate}) concurrently...")
    t0 = time.monotonic()

    results = {}
    errors = {}

    def _send(role: str, url: str):
        try:
            resp = _post_json(url, payload_base, timeout=args.timeout)
            return role, resp
        except Exception as exc:
            return role, exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Send P first, then D after a brief delay so P has time to
        # start prefilling before D tries to pull.
        fut_p = pool.submit(_send, "prefill", p_generate)
        time.sleep(1.0)
        fut_d = pool.submit(_send, "decode", d_generate)

        for fut in as_completed([fut_p, fut_d]):
            role, result = fut.result()
            if isinstance(result, Exception):
                errors[role] = result
                print(f"  [{role}] ERROR: {result}")
            else:
                results[role] = result
                elapsed = time.monotonic() - t0
                print(f"  [{role}] done in {elapsed:.2f}s")

    print("\n" + "=" * 60)

    if errors:
        print("ERRORS:")
        for role, exc in errors.items():
            print(f"  {role}: {exc}")
        return 1

    print("PREFILL response:")
    print(json.dumps(results.get("prefill", {}), indent=2, ensure_ascii=False))
    print()
    print("DECODE response:")
    d_resp = results.get("decode", {})
    print(json.dumps(d_resp, indent=2, ensure_ascii=False))

    # Validate
    ok = True

    # D should have generated text
    d_text = d_resp.get("text", "")
    if not d_text:
        print("\nFAIL: decode response has no text")
        ok = False
    else:
        print(f"\nGenerated text: {d_text!r}")

    elapsed_total = time.monotonic() - t0
    print(f"\nTotal e2e time: {elapsed_total:.2f}s")

    if ok:
        print("\nPASS")
        return 0
    else:
        print("\nFAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
