#!/usr/bin/env python3
import json
import os
import subprocess
import time

import requests

RUN_ID = os.environ.get("RUN_ID", "").strip()
if not RUN_ID:
    with open("/tmp/current_specdecode_route_run_id", "r") as f:
        RUN_ID = f.read().strip()

BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30271")
LOG = os.environ.get("SGLANG_RANK0_LOG", f"/tmp/sglang_{RUN_ID}_rank0.log")
PROFILE_DIR = os.environ.get("PROFILE_DIR", f"/tmp/profile_{RUN_ID}/decode5_after_decode32")
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "32"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TARGET_DECODE32_STEPS = int(os.environ.get("TARGET_DECODE32_STEPS", "6"))
START_AFTER_DECODE32_STEPS = int(os.environ.get("START_AFTER_DECODE32_STEPS", "1"))
WAIT_TIMEOUT_S = float(os.environ.get("WAIT_TIMEOUT_S", "420"))


def count_decode32():
    try:
        with open(LOG, "r", errors="ignore") as f:
            return sum(1 for line in f if "Decode batch. #running-req: 32" in line)
    except FileNotFoundError:
        return 0


def post_json(path, payload):
    return requests.post(BASE + path, json=payload, timeout=180)


def start_requests():
    procs = []
    for i in range(NUM_REQUESTS):
        payload = {
            "model": "/data/pc",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Deterministically continue a numbered arithmetic list. "
                        f"Request {i}. Keep producing concise steps."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        }
        procs.append(
            subprocess.Popen(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    str(max(int(WAIT_TIMEOUT_S) + 60, 300)),
                    "-H",
                    "Content-Type: application/json",
                    "-o",
                    f"/tmp/curl_{RUN_ID}_{i}.out",
                    "-X",
                    "POST",
                    BASE + "/v1/chat/completions",
                    "-d",
                    json.dumps(payload),
                ]
            )
        )
    return procs


def stop_requests(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    time.sleep(0.5)
    for p in procs:
        if p.poll() is None:
            p.kill()


def main():
    base_count = count_decode32()
    print(f"RUN_ID={RUN_ID}", flush=True)
    print(f"LOG={LOG}", flush=True)
    print(f"PROFILE_DIR={PROFILE_DIR}", flush=True)
    print(f"BASELINE_DECODE32={base_count}", flush=True)
    print(f"START_AFTER_DECODE32_STEPS={START_AFTER_DECODE32_STEPS}", flush=True)

    procs = start_requests()
    print(f"STARTED_CURLS={len(procs)} MAX_TOKENS={MAX_TOKENS}", flush=True)

    profile_started = False
    target_count = None
    start_time = time.time()
    try:
        while time.time() - start_time < WAIT_TIMEOUT_S:
            c = count_decode32()
            elapsed = time.time() - start_time
            if not profile_started and c >= base_count + START_AFTER_DECODE32_STEPS:
                print(
                    f"TRIGGER_DECODE32_COUNT={c} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                os.makedirs(PROFILE_DIR, exist_ok=True)
                resp = post_json(
                    "/start_profile",
                    {
                        "output_dir": PROFILE_DIR,
                        "activities": ["CPU", "TPU"],
                        "profile_by_stage": False,
                        "profile_stages": None,
                    },
                )
                print(
                    f"PROFILE_START={{status: {resp.status_code}, text: {resp.text[:200]!r}}}",
                    flush=True,
                )
                profile_started = True
                target_count = c + TARGET_DECODE32_STEPS

            if profile_started and c >= target_count:
                print(f"STOP_DECODE32_COUNT={c} elapsed={elapsed:.1f}s", flush=True)
                break
            if profile_started and not any(p.poll() is None for p in procs):
                print(f"CLIENTS_DONE_DECODE32_COUNT={c} elapsed={elapsed:.1f}s", flush=True)
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for decode32/profile window")
    finally:
        if profile_started:
            try:
                resp = requests.post(BASE + "/stop_profile", timeout=300)
                print(
                    f"PROFILE_STOP={{status: {resp.status_code}, text: {resp.text[:200]!r}}}",
                    flush=True,
                )
            except Exception as exc:
                print(f"PROFILE_STOP_ERROR={exc!r}", flush=True)
        stop_requests(procs)

    print(f"FINAL_DECODE32={count_decode32()}", flush=True)
    print("CLIENT_EXIT_CODES=" + ",".join(str(p.poll()) for p in procs), flush=True)


if __name__ == "__main__":
    main()
