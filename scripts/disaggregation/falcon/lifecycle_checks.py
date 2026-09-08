"""Serving lifecycle checks using supported P/D control endpoints.

Call run(driver, router_port, expected_tokens) while an otherwise idle P/D pair
is running. This tests real pause/retract and allocator recovery, independently
of performance measurements. Delayed native-send injection is separate.
"""

import concurrent.futures
import json
import threading
import time

import requests

PORTS = (30000, 30010)


def control(port, endpoint, payload=None):
    response = requests.post(f"http://127.0.0.1:{port}/{endpoint}", json=payload or {}, timeout=120)
    response.raise_for_status()


def states():
    result = {}
    for port in PORTS:
        response = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=30)
        response.raise_for_status()
        result[str(port)] = response.json()["internal_states"]
    return result


def idle(baseline=None):
    deadline = time.monotonic() + 120
    last = None
    while time.monotonic() < deadline:
        last = states()
        drained = True
        for port, ranks in last.items():
            for i, rank in enumerate(ranks):
                for key in [
                    "waiting_queue_size",
                    "running_batch_size",
                    "pending_dp_reqs_size",
                    "disagg_prefill_queue_size",
                    "disagg_prealloc_queue_size",
                    "disagg_transfer_queue_size",
                    "req_to_token_pool_used",
                ]:
                    if rank.get(key, 0):
                        drained = False
                if (
                    baseline
                    and rank["available_kv_tokens"] != baseline[port][i]["available_kv_tokens"]
                ):
                    drained = False
        if drained:
            return last
        time.sleep(0.2)
    raise AssertionError(f"allocator/queue did not drain: {last}")


def stream(driver, port, started, finished):
    tokens = []
    body = driver.payload(4096, 1024, stream=True)
    body.pop("return_logprob")
    body.pop("logprob_start_len")
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json=body,
            stream=True,
            timeout=600,
        )
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024):
            if line.startswith(b"data:") and b"[DONE]" not in line:
                event = json.loads(line[5:])
                tokens = event.get("output_ids")
                if tokens is None:
                    tokens = [
                        v[1] for v in event.get("meta_info", {}).get("output_token_logprobs", [])
                    ]
                if tokens:
                    started.set()
        return tokens
    finally:
        finished.set()


def run(driver, router_port, expected_tokens):
    baseline = idle()
    results = []
    for mode in ["in_place", "retract"]:
        print("LIFECYCLE_START " + mode, flush=True)
        started, finished = threading.Event(), threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stream, driver, router_port, started, finished)
            assert started.wait(180), "request never emitted a token"
            assert not finished.is_set(), "request completed before lifecycle injection"
            try:
                # D pause is injected during confirmed active decode.
                control(30010, "pause_generation", {"mode": mode})
                snapshot = states()
                assert all(r["engine_paused"] for r in snapshot["30010"])
                assert not finished.is_set(), "request completed before pause took effect"
                control(30010, "continue_generation")
                assert future.result(timeout=600) == expected_tokens, mode
            finally:
                control(30010, "continue_generation")
        after = idle(baseline)
        print("LIFECYCLE_PASSED " + mode, flush=True)
        results.append({"case": mode, "status": "passed", "paused_state": snapshot, "after": after})
    # HTTP pause blocks tokenizer admission, so it cannot create a scheduler
    # queue for abort_all. Use a real concurrent PD burst and observe D queues.
    print("LIFECYCLE_START queued-cancel", flush=True)
    debug_dir = driver.OUT / "python-stacks"
    if debug_dir.exists():
        (debug_dir / "trigger").touch()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(
                requests.post,
                f"http://127.0.0.1:{router_port}/generate",
                json=driver.payload(16384, 1024),
                timeout=90,
            )
            for _ in range(16)
        ]
        deadline = time.monotonic() + 30
        while True:
            queued = states()
            if any(
                r.get("disagg_prealloc_queue_size", 0) or r.get("disagg_transfer_queue_size", 0)
                for r in queued["30010"]
            ):
                break
            assert time.monotonic() < deadline, "no D scheduler queue observed"
            time.sleep(0.02)
        # Repeated cancellation also catches requests still entering through
        # HTTP at the first cancellation; every observed PD queue is drained.
        deadline = time.monotonic() + 60
        while not all(f.done() for f in futures):
            for port in PORTS:
                control(port, "abort_request", {"abort_all": True})
            if time.monotonic() >= deadline:
                diagnostics = {
                    "done": sum(f.done() for f in futures),
                    "total": len(futures),
                    "states": states(),
                }
                (driver.OUT / "cancel-diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2)
                )
                raise AssertionError("cancelled requests did not terminate")
            time.sleep(0.2)
        responses = [f.result() for f in futures]
    aborted = 0
    for response in responses:
        body = response.json()
        finish = body.get("meta_info", {}).get("finish_reason") or {}
        if response.status_code >= 400 or finish.get("type") == "abort":
            aborted += 1
    assert aborted, "no cancellation response observed"
    after = idle(baseline)
    fresh = driver.generate(router_port, 4096, 1024)
    assert fresh["token_ids"] == expected_tokens
    results.append(
        {
            "case": "queued-cancel-and-reuse",
            "status": "passed",
            "queued": queued,
            "aborted": aborted,
            "after": after,
        }
    )
    print("LIFECYCLE_PASSED queued-cancel", flush=True)
    return results
