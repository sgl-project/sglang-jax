"""Bounded completion-delay, late cancellation and decode peer-failure checks."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import requests
from transformers import AutoTokenizer

import lifecycle_checks as lifecycle
import single_pod_pd as driver

REPORT = {
    "status": "running",
    "checks": [],
    "scope": "delayed completion visibility, not artificially delayed DMA; same-pod peer process failure",
}
DELAY = 2.0
original_launch = driver.launch
EVENTS = driver.OUT / "native-events"


def save():
    (driver.OUT / "fault-summary.json").write_text(json.dumps(REPORT, indent=2))


def launch(name, command, chips=None):
    values = {}
    if "sgl_jax.launch_server" in command:
        command = list(command)
        command[command.index("--disaggregation-max-inflight-transfers") + 1] = "1"
        command += [
            "--disaggregation-pull-timeout-seconds",
            "30",
            "--disaggregation-ack-timeout-seconds",
            "30",
            "--disaggregation-orphan-reaper-interval-seconds",
            "1",
        ]
        if DELAY:
            values = {
                "PYTHONPATH": "/tmp/pd-fault-hooks:" + os.environ.get("PYTHONPATH", ""),
                "PD_TEST_HOLD_COMPLETION_S": str(DELAY),
                "PD_TEST_EVENT_DIR": str(EVENTS),
            }
    if "sgl_jax.srt.disaggregation.launch_router" in command:
        command = [*command, "--request-timeout-secs", "60"]
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        return original_launch(name, command, chips)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def wait_completion(start):
    end = time.monotonic() + 90
    while time.monotonic() < end:
        for path in EVENTS.glob("*.jsonl"):
            for line in path.read_text().splitlines()[-8:]:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event["at"] >= start:
                    return event
        time.sleep(0.02)
    raise TimeoutError("no delayed native completion observed")


def raw_request(port):
    return requests.post(
        f"http://127.0.0.1:{port}/generate", json=driver.payload(16384, 1024), timeout=90
    )


def main():
    global DELAY
    driver.OUT.mkdir(parents=True, exist_ok=True)
    EVENTS.mkdir()
    REPORT["completion_hook_sha256"] = hashlib.sha256(
        Path("/tmp/pd-fault-hooks/sitecustomize.py").read_bytes()
    ).hexdigest()
    save()
    prerequisite = Path(os.environ["PD_REQUIRE_SUMMARY"])
    assert json.loads(prerequisite.read_text())["status"] == "passed_implemented_correctness_checks"
    reference = None
    for line in (prerequisite.parent / "B0-requests.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["input"] == 4096 and item["output"] == 256:
            reference = item["token_ids"]
            break
    assert reference
    driver.TOKENIZER = AutoTokenizer.from_pretrained(driver.MODEL)
    driver.launch = launch
    port = driver.boot("delayed", True, True, True)
    baseline = lifecycle.idle()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: driver.generate(port, 4096, 256), range(4)))
    assert all(item["token_ids"] == reference for item in outputs)
    lifecycle.idle(baseline)
    REPORT["checks"].append({"name": "backpressure-window1-delay2s", "status": "passed"})
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        start = time.time()
        future = pool.submit(raw_request, port)
        marker = wait_completion(start)
        assert not future.done()
        for server in lifecycle.PORTS:
            lifecycle.control(server, "abort_request", {"abort_all": True})
        response = future.result(timeout=90)
        if response.status_code < 400:
            body = response.json()
            assert body.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort", body
    after = lifecycle.idle(baseline)
    assert driver.generate(port, 4096, 256)["token_ids"] == reference
    lifecycle.idle(baseline)
    REPORT["checks"].append(
        {
            "name": "cancel-during-held-completion-and-reuse",
            "status": "passed",
            "marker": marker,
            "after": after,
        }
    )
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        start = time.time()
        future = pool.submit(raw_request, port)
        marker = wait_completion(start)
        target = next(proc for proc, _, name in driver.CHILDREN if name.endswith("-decode"))
        assert not future.done()
        os.killpg(target.pid, signal.SIGKILL)
        try:
            response = future.result(timeout=90)
            assert response.status_code >= 400, response.text[:500]
            failure = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            failure = repr(exc)
    producer = next(proc for proc, _, name in driver.CHILDREN if name.endswith("-prefill"))
    assert producer.poll() is None, "P exited with its peer"
    deadline = time.monotonic() + 60
    while True:
        response = requests.get("http://127.0.0.1:30000/get_server_info", timeout=15)
        response.raise_for_status()
        ranks = response.json()["internal_states"]
        if all(
            rank["available_kv_tokens"] == baseline["30000"][i]["available_kv_tokens"]
            and rank["req_to_token_pool_used"] == 0
            for i, rank in enumerate(ranks)
        ):
            break
        if time.monotonic() > deadline:
            raise AssertionError(f"P did not recover after peer loss: {ranks}")
        time.sleep(0.2)
    REPORT["checks"].append(
        {
            "name": "decode-peer-kill-bounded-error-P-recovery",
            "status": "passed",
            "failure": failure,
            "marker": marker,
        }
    )
    save()
    driver.stop()
    DELAY = 0
    port = driver.boot("restart", True, True, True)
    assert driver.generate(port, 4096, 256)["token_ids"] == reference
    lifecycle.idle()
    REPORT["checks"].append({"name": "restart-after-peer-kill", "status": "passed"})
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise TimeoutError(f"fault suite interrupted: {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        driver.stop()
        print("FAULT_SUMMARY=" + json.dumps(REPORT), flush=True)
