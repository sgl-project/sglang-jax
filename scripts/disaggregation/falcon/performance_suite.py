"""Same-pod PD performance ablation without logprob serialization overhead."""

import asyncio
import json
import os
from pathlib import Path
import signal
import time
import threading
import traceback

import aiohttp
import requests
from transformers import AutoTokenizer

import single_pod_pd as driver

REPORT = {
    "status": "running",
    "return_logprob": False,
    "measurements": [],
    "profiles": [],
    "scope": "single-pod TP4/DP1/EP4 Qwen3-30B-A3B; cross-host networking not evaluated",
}


def save():
    path = driver.OUT / "performance-summary.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(REPORT, indent=2))
    tmp.replace(path)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


async def request(session, port, length, output, variant):
    body = driver.payload(length, output, stream=True, variant=variant)
    body.pop("return_logprob")
    body.pop("logprob_start_len")
    begin = time.perf_counter()
    events = []
    done = False
    async with session.post(f"http://127.0.0.1:{port}/generate", json=body) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data:"):
                continue
            if line[5:].strip() == b"[DONE]":
                done = True
                break
            data = json.loads(line[5:])
            if "error" in data:
                raise RuntimeError(data)
            meta = data.get("meta_info", {})
            count = int(meta.get("completion_tokens", 0))
            events.append((time.perf_counter() - begin, count))
    assert done and events and events[-1][1] == output, (done, events[-3:], output)
    timed_tokens = [(t, n) for t, n in events if n > 0]
    assert timed_tokens
    itls = []
    deltas = []
    for (previous_time, previous_count), (now, count) in zip(timed_tokens, timed_tokens[1:]):
        delta = count - previous_count
        assert delta >= 0, "completion token count decreased"
        if delta:
            itls.append((now - previous_time) / delta)
            deltas.append(delta)
    return {
        "input": length,
        "output": output,
        "variant": variant,
        "ttft_s": timed_tokens[0][0],
        "e2e_s": time.perf_counter() - begin,
        "itl_s": itls,
        "max_event_token_delta": max(deltas, default=0),
    }


async def batch(port, length, output, concurrency, n):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600), connector=aiohttp.TCPConnector(limit=concurrency)
    ) as session:

        async def run(i):
            async with semaphore:
                return await request(session, port, length, output, i % 4)

        begin = time.perf_counter()
        results = await asyncio.gather(*(run(i) for i in range(n)))
        return results, time.perf_counter() - begin


def measure(tag, port, length, output, concurrency):
    # Warm the actual concurrency/shape, independently of measured traffic.
    asyncio.run(batch(port, length, output, concurrency, concurrency))
    results = []
    seconds = 0
    snapshots = []
    stop_sampling = threading.Event()

    def sample():
        while not stop_sampling.is_set():
            snapshot = {"at": time.time(), "roles": {}}
            for role, server_port in [("prefill", 30000), ("decode", 30010)]:
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{server_port}/get_server_info", timeout=3
                    )
                    response.raise_for_status()
                    snapshot["roles"][role] = response.json()["internal_states"]
                except requests.RequestException as exc:
                    snapshot[role + "_error"] = repr(exc)
            snapshots.append(snapshot)
            stop_sampling.wait(1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        while seconds < 20 or len(results) < max(16, concurrency * 2):
            values, elapsed = asyncio.run(
                batch(port, length, output, concurrency, max(8, concurrency))
            )
            results.extend(values)
            seconds += elapsed
    finally:
        stop_sampling.set()
        sampler.join(timeout=8)
        with (driver.OUT / f"{tag}-pool-samples.jsonl").open("a") as handle:
            for snapshot in snapshots:
                handle.write(
                    json.dumps(
                        {"input": length, "output": output, "concurrency": concurrency, **snapshot}
                    )
                    + "\n"
                )
    for item in results:
        driver.record(tag, item)
    ttft = [r["ttft_s"] for r in results]
    e2e = [r["e2e_s"] for r in results]
    itl = [x for r in results for x in r["itl_s"]]
    report = {
        "tag": tag,
        "input": length,
        "output": output,
        "concurrency": concurrency,
        "completed": len(results),
        "failed": 0,
        "pool_sample_period_s": 1,
        "sampled_peak_kv_tokens": {
            role: max(
                (
                    rank["memory_usage"]["token_capacity"] - rank["available_kv_tokens"]
                    for snapshot in snapshots
                    for rank in snapshot["roles"].get(role, [])
                ),
                default=None,
            )
            for role in ("prefill", "decode")
        },
        "measurement_s": seconds,
        "request_per_s": len(results) / seconds,
        "output_token_per_s": len(results) * output / seconds,
        "ttft_p50_s": percentile(ttft, 0.5),
        "ttft_p95_s": percentile(ttft, 0.95),
        "itl_p50_s": percentile(itl, 0.5),
        "itl_p95_s": percentile(itl, 0.95),
        "e2e_p50_s": percentile(e2e, 0.5),
        "e2e_p95_s": percentile(e2e, 0.95),
        "max_event_token_delta": max(r["max_event_token_delta"] for r in results),
    }
    REPORT["measurements"].append(report)
    save()
    print("PERFORMANCE_MEASUREMENT=" + json.dumps(report), flush=True)


def profile(tag, port):
    record = {"tag": tag, "host_start_ns": time.time_ns(), "roles": {}}
    started = []
    try:
        for role, server_port in [("prefill", 30000), ("decode", 30010)]:
            path = driver.OUT / "profiles" / tag / role
            requests.post(
                f"http://127.0.0.1:{server_port}/start_profile",
                json={"output_dir": str(path), "host_tracer_level": 2, "python_tracer_level": 0},
                timeout=120,
            ).raise_for_status()
            started.append(server_port)
            status = requests.get(f"http://127.0.0.1:{server_port}/profile_status", timeout=30)
            status.raise_for_status()
            assert status.json()["status"] == "in_progress", status.text
            record["roles"][role] = str(path)
        asyncio.run(batch(port, 16384, 256, 16, 32))
    finally:
        for server_port in started:
            requests.post(
                f"http://127.0.0.1:{server_port}/stop_profile", timeout=180
            ).raise_for_status()
    record["files"] = {}
    for role, directory in record["roles"].items():
        files = list(Path(directory).rglob("*.xplane.pb"))
        assert files and all(path.stat().st_size > 0 for path in files), directory
        record["files"][role] = [
            {"path": str(path), "bytes": path.stat().st_size} for path in files
        ]
    record["host_end_ns"] = time.time_ns()
    REPORT["profiles"].append(record)
    save()


def main():
    driver.OUT.mkdir(parents=True, exist_ok=True)
    required = os.environ.get("PD_REQUIRE_SUMMARY")
    if not required:
        raise RuntimeError("PD_REQUIRE_SUMMARY must name the passed correctness report")
    prerequisite = json.loads(open(required).read())
    if prerequisite.get("status") != "passed_implemented_correctness_checks":
        raise RuntimeError(f"correctness prerequisite did not pass: {prerequisite.get('status')}")
    fault_required = os.environ.get("PD_REQUIRE_FAULT_SUMMARY")
    if fault_required:
        fault = json.loads(Path(fault_required).read_text())
        if fault.get("status") != "passed":
            raise RuntimeError(f"fault prerequisite did not pass: {fault.get('status')}")
        REPORT["fault_prerequisite"] = fault_required
    stream_required = os.environ.get("PD_REQUIRE_STREAM_SUMMARY")
    if stream_required:
        stream = json.loads(Path(stream_required).read_text())
        if stream.get("status") != "passed":
            raise RuntimeError(f"stream prerequisite did not pass: {stream.get('status')}")
        REPORT["stream_prerequisite"] = stream_required
    eos_required = os.environ.get("PD_REQUIRE_EOS_SUMMARY")
    if eos_required:
        eos = json.loads(Path(eos_required).read_text())
        if eos.get("status") != "passed":
            raise RuntimeError(f"EOS prerequisite did not pass: {eos.get('status')}")
        REPORT["eos_prerequisite"] = eos_required
    REPORT["correctness_prerequisite"] = required
    driver.TOKENIZER = AutoTokenizer.from_pretrained(driver.MODEL)
    save()
    cases = [(4096, 256, c) for c in (1, 4, 16, 32)]
    cases += [(16384, 256, c) for c in (1, 16, 32)]
    cases += [(16384, 1024, 16)]
    for repeat in range(3):
        order = [("B1", False, False), ("PD", True, True)]
        if repeat % 2:
            order.reverse()
        for name, po, do in order:
            tag = f"{repeat}-{name}"
            port = driver.boot(tag, po, do, True)
            for length, output, concurrency in cases:
                measure(tag, port, length, output, concurrency)
            if repeat == 2:
                profile(tag, port)
            driver.stop()
    for name, po, do in [("P-only", True, False), ("D-only", False, True)]:
        port = driver.boot(name, po, do, True)
        for length in (4096, 16384):
            measure(name, port, length, 256, 16)
        driver.stop()
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise TimeoutError(f"performance interrupted: {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        driver.stop()
        print("PERFORMANCE_SUMMARY=" + json.dumps(REPORT), flush=True)
