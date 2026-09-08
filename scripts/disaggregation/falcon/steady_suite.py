"""Continuous B1/PD steady-state comparisons, token-checked soak, and short profiles."""

import asyncio
import json
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lifecycle_checks as lifecycle
import requests
import single_pod_pd as driver
from steady_client import run
from transformers import AutoTokenizer

REPORT = {"status": "running", "measurements": [], "profiles": [], "soak": None}
MIXED = [(4096, 256), (16384, 256), (4096, 1024), (16384, 1024)]


def save():
    target = driver.OUT / "steady-summary.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(REPORT, indent=2))
    temp.replace(target)


def bodies(shapes):
    result = []
    for length, output in shapes:
        for variant in range(4):
            body = driver.payload(length, output, stream=True, variant=variant)
            body.pop("return_logprob")
            body.pop("logprob_start_len")
            result.append(body)
    return result


def measure(tag, port, shapes, concurrency, duration=180, expected=None, warmup=60):
    baseline = lifecycle.idle()
    samples = []
    stop = threading.Event()
    origin = time.perf_counter()

    def sample():
        while not stop.is_set():
            try:
                samples.append({"at_s": time.perf_counter() - origin, "roles": lifecycle.states()})
            except Exception as exc:
                samples.append({"at_s": time.perf_counter() - origin, "error": repr(exc)})
            stop.wait(1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with (driver.OUT / f"{tag}-requests.jsonl").open("w") as handle:

            def record(item):
                handle.write(json.dumps(item) + "\n")
                handle.flush()

            result = asyncio.run(
                run(
                    f"http://127.0.0.1:{port}/generate",
                    bodies(shapes),
                    concurrency,
                    warmup,
                    duration,
                    on_complete=record,
                    expected=expected,
                )
            )
    finally:
        stop.set()
        sampler.join(timeout=65)
        (driver.OUT / f"{tag}-pool-samples.json").write_text(json.dumps(samples))
    result.update(tag=tag, shapes=shapes, baseline=baseline, after=lifecycle.idle(baseline))
    measured = [s for s in samples if warmup <= s["at_s"] < warmup + duration and "roles" in s]
    active = 0
    for sample in measured:
        p = sample["roles"]["30000"][0]
        d = sample["roles"]["30010"][0]
        if d["running_batch_size"] and (
            not p["cur_batch_is_none"]
            or not p["chunked_req_is_none"]
            or p["disagg_prefill_queue_size"]
            or d["disagg_prealloc_queue_size"]
            or d["disagg_transfer_queue_size"]
        ):
            active += 1
    result.update(
        pool_samples=len(measured),
        simultaneous_decode_and_pd_activity_samples=active,
        simultaneous_decode_and_pd_activity_fraction=active / len(measured) if measured else 0,
        sample_errors=sum("error" in s for s in samples),
    )
    assert result["cohort_requests"] > 0 and result["window_output_tokens"] > 0, result
    assert measured and active, "no sampled evidence of decode overlapping incoming PD work"
    assert result["sample_errors"] == 0, "pool observation failed"
    REPORT["measurements"].append(result)
    save()
    print(
        "STEADY_MEASUREMENT="
        + json.dumps(
            {k: v for k, v in result.items() if k not in {"baseline", "after", "tokens_per_second"}}
        ),
        flush=True,
    )
    return result


def profile(tag, port):
    # Separate from benchmark: profiling/export may block either scheduler.
    # Stop both roles concurrently after a short capture to bound artifact size.
    def load():
        return asyncio.run(run(f"http://127.0.0.1:{port}/generate", bodies(MIXED), 16, 60, 120))

    with ThreadPoolExecutor(max_workers=3) as pool:
        traffic = pool.submit(load)
        time.sleep(60)
        started = []
        try:
            for role, server_port in [("prefill", 30000), ("decode", 30010)]:
                path = driver.OUT / "profiles" / tag / role
                response = requests.post(
                    f"http://127.0.0.1:{server_port}/start_profile",
                    json={
                        "output_dir": str(path),
                        "host_tracer_level": 1,
                        "python_tracer_level": 0,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                started.append(server_port)
            time.sleep(5)
        finally:
            futures = [
                pool.submit(requests.post, f"http://127.0.0.1:{p}/stop_profile", timeout=300)
                for p in started
            ]
            for future in futures:
                future.result().raise_for_status()
        traffic.result(timeout=720)
    files = list((driver.OUT / "profiles" / tag).rglob("*.xplane.pb"))
    assert len(files) >= 2 and all(p.stat().st_size for p in files)
    REPORT["profiles"].append(
        {
            "tag": tag,
            "capture_s": 5,
            "traffic": "continuous mixed C16",
            "files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
        }
    )
    lifecycle.idle()
    save()


def main():
    driver.OUT.mkdir(parents=True, exist_ok=True)
    required = Path(os.environ["PD_REQUIRE_SUMMARY"])
    assert json.loads(required.read_text())["status"] == "passed_implemented_correctness_checks"
    driver.TOKENIZER = AutoTokenizer.from_pretrained(driver.MODEL)
    save()
    for repeat in range(3):
        order = [("B1", False), ("PD", True)]
        if repeat % 2:
            order.reverse()
        for name, overlap in order:
            port = driver.boot(f"steady-{repeat}-{name}", overlap, overlap, True)
            for length in (4096, 16384):
                for concurrency in (16, 32):
                    measure(
                        f"{repeat}-{name}-{length}-c{concurrency}",
                        port,
                        [(length, 256)],
                        concurrency,
                    )
            measure(f"{repeat}-{name}-mixed-c16", port, MIXED, 16)
            driver.stop()
    for name, po, do in [("P-only", True, False), ("D-only", False, True)]:
        port = driver.boot(name, po, do, True)
        measure(name + "-mixed-c16", port, MIXED, 16)
        driver.stop()
    # Independent non-PD references for every payload; verify every soak request.
    port = driver.boot("soak-reference", standalone=True)
    expected = []
    for body in bodies(MIXED):
        response = requests.post(
            f"http://127.0.0.1:{port}/generate", json={**body, "stream": False}, timeout=600
        )
        response.raise_for_status()
        expected.append(response.json()["output_ids"])
    (driver.OUT / "soak-reference.json").write_text(json.dumps(expected))
    port = driver.boot("soak-PD", True, True, True)
    REPORT["soak"] = measure("soak-PD-mixed-c32", port, MIXED, 32, duration=1200, expected=expected)
    save()
    driver.stop()
    for name, overlap in [("B1", False), ("PD", True)]:
        port = driver.boot("profile-" + name, overlap, overlap, True)
        profile(name, port)
        driver.stop()
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("steady suite interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        driver.stop()
        print(
            "STEADY_SUMMARY="
            + json.dumps(
                {
                    "status": REPORT["status"],
                    "measurements": len(REPORT["measurements"]),
                    "error": REPORT.get("error"),
                }
            ),
            flush=True,
        )
