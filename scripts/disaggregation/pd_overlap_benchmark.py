"""Closed-loop SSE load with explicit warmup, measurement, and drain boundaries.

One persistent connection pool and independent workers replenish requests without
batch barriers. Throughput counts observed token deltas in [start, end), including
requests crossing either boundary. Latency uses the start-in-window cohort,
including its drain, so slow requests are not silently censored.
"""

# Manual entry point: no accelerator allocation or CI registration.
import argparse
import asyncio
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from ipaddress import IPv4Address
from pathlib import Path

import aiohttp
import requests
from pd_overlap_check import idle
from transformers import AutoTokenizer


@dataclass
class Window:
    start: float
    end: float
    tokens: int = 0
    bins: dict = field(default_factory=dict)

    def observe(self, at, delta):
        if delta < 0:
            raise AssertionError("completion token count decreased")
        if self.start <= at < self.end:
            self.tokens += delta
            second = int(at - self.start)
            self.bins[second] = self.bins.get(second, 0) + delta


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


async def run(
    url,
    payloads,
    concurrency,
    warmup_s,
    duration_s,
    *,
    request_timeout_s=600,
    stagger_s=2,
):
    if not payloads or concurrency < 1 or duration_s <= 0 or warmup_s < stagger_s:
        raise ValueError("invalid workload or measurement window")
    origin = time.perf_counter()
    window = Window(origin + warmup_s, origin + warmup_s + duration_s)
    results = []
    active = 0
    peak_active = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=request_timeout_s),
        connector=aiohttp.TCPConnector(limit=concurrency),
        read_bufsize=2**20,
    ) as session:

        async def worker(slot):
            nonlocal active, peak_active
            await asyncio.sleep(stagger_s * slot / concurrency)
            sequence = 0
            while time.perf_counter() < window.end:
                index = (slot + sequence * concurrency) % len(payloads)
                sequence += 1
                body = payloads[index]
                begin = time.perf_counter()
                active += 1
                peak_active = max(peak_active, active)
                count = 0
                first = previous = None
                itls = []
                done = False
                max_delta = 0
                try:
                    async with session.post(url, json=body) as response:
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
                            now = time.perf_counter()
                            new_count = int(data.get("meta_info", {}).get("completion_tokens", 0))
                            delta = new_count - count
                            window.observe(now, delta)
                            if delta:
                                max_delta = max(max_delta, delta)
                                if first is None:
                                    first = now
                                elif previous is not None:
                                    itls.append((now - previous) / delta)
                                previous = now
                            count = new_count
                    wanted = body["sampling_params"]["max_new_tokens"]
                    if not done or count != wanted or first is None:
                        raise AssertionError(("incomplete stream", done, count, wanted))
                    end = time.perf_counter()
                    result = {
                        "payload_index": index,
                        "slot": slot,
                        "sequence": sequence,
                        "start_s": begin - origin,
                        "end_s": end - origin,
                        "cohort": window.start <= begin < window.end,
                        "output": count,
                        "ttft_s": first - begin,
                        "e2e_s": end - begin,
                        "itl_s": itls,
                        "max_event_token_delta": max_delta,
                    }
                    results.append(result)
                finally:
                    active -= 1

        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    cohort = [r for r in results if r["cohort"]]
    itls = [t for r in cohort for t in r["itl_s"]]
    return {
        "mode": "continuous_closed_loop",
        "concurrency": concurrency,
        "warmup_s": warmup_s,
        "measurement_s": duration_s,
        "drain_s": max(0, time.perf_counter() - window.end),
        "window_output_tokens": window.tokens,
        "output_token_per_s": window.tokens / duration_s,
        "tokens_per_second": [window.bins.get(i, 0) for i in range(int(duration_s))],
        "completed_total": len(results),
        "cohort_requests": len(cohort),
        "failed": 0,
        "peak_client_active": peak_active,
        "requests_crossing_start": sum(r["start_s"] < warmup_s < r["end_s"] for r in results),
        "requests_crossing_end": sum(
            r["start_s"] < warmup_s + duration_s < r["end_s"] for r in results
        ),
        "ttft_p50_s": percentile([r["ttft_s"] for r in cohort], 0.5),
        "ttft_p95_s": percentile([r["ttft_s"] for r in cohort], 0.95),
        "e2e_p50_s": percentile([r["e2e_s"] for r in cohort], 0.5),
        "e2e_p95_s": percentile([r["e2e_s"] for r in cohort], 0.95),
        "itl_p50_s": percentile(itls, 0.5),
        "itl_p95_s": percentile(itls, 0.95),
        "max_event_token_delta": max((r["max_event_token_delta"] for r in results), default=0),
    }


MIXED = [(4096, 256), (4096, 1024), (16384, 256), (16384, 1024)]
WORKLOADS = [
    ("4k-c16", [(4096, 256)], 16),
    ("4k-c32", [(4096, 256)], 32),
    ("16k-c16", [(16384, 256)], 16),
    ("16k-c32", [(16384, 256)], 32),
    ("mixed-c16", MIXED, 16),
]


class Servers:
    def __init__(self, args):
        self.args = args
        self.children = []
        self.env = os.environ.copy()
        self.env.update(
            ALLOW_MULTIPLE_LIBTPU_LOAD="true",
            TPU_CHIPS_PER_PROCESS_BOUNDS="1,2,1",
            TPU_PROCESS_BOUNDS="1,1,1",
            NO_PROXY="*",
            no_proxy="*",
        )
        for key in (
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_HOST_BOUNDS",
            "TPU_VISIBLE_CHIPS",
            "TPU_MESH_CONTROLLER_ADDRESS",
            "TPU_MESH_CONTROLLER_PORT",
            "ENABLE_MULTI_NUMA",
        ):
            self.env.pop(key, None)

    def launch(self, label, command, chips=None):
        env = self.env.copy()
        if chips:
            env.update(
                TPU_VISIBLE_CHIPS=chips,
                JAX_COMPILATION_CACHE_DIR=str(self.args.output / "cache" / chips),
            )
        log = (self.args.output / f"{label}.log").open("w")
        proc = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.children.append((proc, log))
        with (self.args.output / "commands.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "argv": command, "chips": chips}) + "\n")

    def ready(self, port):
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if any(proc.poll() is not None for proc, _ in self.children):
                raise RuntimeError("server exited; inspect logs")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=3).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"service on port {port} did not become healthy")

    def stop(self):
        for proc, _ in self.children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, log in self.children:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            log.close()
        self.children.clear()

    def boot(self, label, overlap):
        self.stop()
        a = self.args
        self.launch(
            label + "-bootstrap",
            [
                sys.executable,
                "-m",
                "sgl_jax.srt.disaggregation.run_bootstrap",
                "--host",
                "0.0.0.0",
                "--port",
                "8998",
            ],
        )
        self.ready(8998)
        common = [
            "--model-path",
            a.model,
            "--disaggregation-host-ip",
            a.host_ip,
            "--disaggregation-bootstrap-url",
            f"http://{a.host_ip}:8998",
            *"""--host 0.0.0.0 --nnodes 1 --tp-size 4 --dp-size 1 --ep-size 4
            --moe-backend epmoe --dtype bfloat16 --kv-cache-dtype bf16
            --attention-backend fa --page-size 128 --chunked-prefill-size 2048
            --context-length 32768 --max-running-requests 32 --max-total-tokens 262144
            --mem-fraction-static 0.8 --disable-radix-cache --skip-server-warmup
            --precompile-bs-paddings 1 4 16 32 --precompile-token-paddings 128 512 2048
            --disaggregation-use-raiden --disaggregation-max-inflight-transfers 4
            --disaggregation-enable-chunk-prefill-transfer""".split(),
        ]
        if overlap:
            common += ["--disaggregation-enable-overlap-schedule"]
        for role, port, side, chips in [
            ("prefill", 30000, 9600, "0,1"),
            ("decode", 30010, 9700, "2,3"),
        ]:
            self.launch(
                label + "-" + role,
                [
                    sys.executable,
                    "-m",
                    "sgl_jax.launch_server",
                    *common,
                    "--disaggregation-mode",
                    role,
                    "--port",
                    str(port),
                    "--disaggregation-transfer-port",
                    str(port + 1),
                    "--disaggregation-side-channel-port",
                    str(side),
                ],
                chips,
            )
        self.ready(30000)
        self.ready(30010)
        self.launch(
            label + "-router",
            [
                sys.executable,
                "-m",
                "sgl_jax.srt.disaggregation.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://{a.host_ip}:30000",
                "8998",
                "--decode",
                "http://127.0.0.1:30010",
                "--prefill-bootstrap-host",
                a.host_ip,
                "--max-concurrent-requests",
                "32",
                "--host",
                "0.0.0.0",
                "--port",
                "30020",
            ],
        )
        self.ready(30020)


def payloads(tokenizer, shapes):
    bodies = []
    for length, output in shapes:
        for variant in range(4):
            seed = tokenizer.encode(
                "Explain the following sequence carefully. " + str(variant) + " ",
                add_special_tokens=False,
            )
            if not seed:
                raise ValueError("tokenizer returned an empty seed")
            bodies.append(
                {
                    "input_ids": (seed * (length // len(seed) + 1))[:length],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": output,
                        "ignore_eos": True,
                    },
                    "stream": True,
                }
            )
    return bodies


def main():
    parser = argparse.ArgumentParser(
        description="Manual single-pod v7x-8 Qwen3-30B-A3B steady PD overlap A/B. Requires installed Raiden and exclusive access to chips 0-3; does not allocate resources."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-ip", required=True, help="Routable IP of this host/pod")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--soak-seconds", type=int, default=1200, help="Mixed C32 PD load after the matrix; 0 skips"
    )
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.warmup < 2 or args.duration <= 0 or args.repetitions < 1 or args.soak_seconds < 0:
        parser.error("invalid workload duration/repetition count")
    if args.output.exists():
        parser.error("output directory must be new to avoid mixing runs")
    try:
        host_ip = IPv4Address(args.host_ip)
    except ValueError:
        parser.error("host-ip must be a local IPv4 address")
    if host_ip.is_unspecified or host_ip.is_multicast:
        parser.error("host-ip must identify a specific local interface")
    # Check both endpoints so health probes cannot accept another serving job.
    for host in dict.fromkeys(("127.0.0.1", str(host_ip))):
        for port in (8998, 30000, 30001, 30010, 30011, 30020, 9600, 9700):
            with socket.socket() as sock:
                try:
                    sock.bind((host, port))
                except OSError as exc:
                    parser.error(f"cannot reserve {host}:{port}: {exc}; use an idle host/pod")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    servers = Servers(args)
    report = {
        "status": "running",
        "model": args.model,
        "windows": [],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "limitations": [
            "Single process per role; not multi-host SPMD",
            "No DMA attribution",
            "Soak checks stream length and recovery, not every token against a non-PD reference",
        ],
    }
    (args.output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    urls = ["http://127.0.0.1:30000", "http://127.0.0.1:30010"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2))

    def measure(label, name, shapes, concurrency, duration):
        _, before = idle(urls)
        result = asyncio.run(
            run(
                "http://127.0.0.1:30020/generate",
                payloads(tokenizer, shapes),
                concurrency,
                args.warmup,
                duration,
            )
        )
        _, after = idle(urls, before)
        result.update(label=label, workload=name, capacity_before=before, capacity_after=after)
        report["windows"].append(result)
        save()

    try:
        for repeat in range(args.repetitions):
            for name in ["B1", "PD"] if repeat % 2 == 0 else ["PD", "B1"]:
                label = f"{repeat}-{name}"
                print(f"Starting {label}", flush=True)
                servers.boot(label, name == "PD")
                if repeat == 0:
                    check = [
                        sys.executable,
                        str(Path(__file__).with_name("pd_overlap_check.py")),
                        "--model",
                        args.model,
                        "--reference",
                        str(args.output / "reference.json"),
                    ]
                    if name == "B1":
                        check += ["--record-reference"]
                    subprocess.run(check, check=True)
                for workload, shapes, concurrency in WORKLOADS:
                    measure(label, workload, shapes, concurrency, args.duration)
        if args.soak_seconds:
            servers.boot("soak-PD", True)
            measure("soak-PD", "mixed-c32", MIXED, 32, args.soak_seconds)
        rows = [
            "| Workload | B1 tok/s | PD tok/s | Paired gain | TTFT p95 B1 / PD (s) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, _, _ in WORKLOADS:
            groups = [
                [
                    w
                    for w in report["windows"]
                    if w["workload"] == name and w["label"].endswith("-" + role)
                ]
                for role in ("B1", "PD")
            ]
            b, p = groups
            gain = statistics.median(
                y["output_token_per_s"] / x["output_token_per_s"] - 1 for x, y in zip(b, p)
            )
            med = lambda xs, key: statistics.median(x[key] for x in xs)
            rows.append(
                f"| {name} | {med(b, 'output_token_per_s'):.1f} | {med(p, 'output_token_per_s'):.1f} | {gain:+.1%} | {med(b, 'ttft_p95_s'):.3f} / {med(p, 'ttft_p95_s'):.3f} |"
            )
        (args.output / "summary.md").write_text("\n".join(rows) + "\n")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        servers.stop()


if __name__ == "__main__":
    main()
