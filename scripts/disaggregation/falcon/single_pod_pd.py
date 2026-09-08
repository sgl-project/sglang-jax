"""Bounded single-pod P/D correctness and performance experiment driver."""

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import requests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ["PD_OUT"])
MODEL = os.environ["MODEL_PATH"]
HOST = (
    os.environ.get("PD_HOST_IP")
    or subprocess.check_output(["hostname", "-I"], text=True).split()[0]
)
CHILDREN = []
SUMMARY = {
    "status": "running",
    "stages": [],
    "limitations": [
        "Single-pod results do not measure cross-host networking.",
        "Delayed native send, exact transfer-phase cancellation, and peer failure injection remain follow-up coverage.",
        "Device traces require analysis before claiming physical compute/transfer overlap.",
    ],
}


def save():
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=2))


def launch(name, command, chips=None):
    env = os.environ.copy()
    if chips:
        env["TPU_VISIBLE_CHIPS"] = chips
        env["JAX_COMPILATION_CACHE_DIR"] = "/tmp/tpu_logs/jax-cache/" + chips.replace(",", "-")
    path = OUT / (name + ".log")
    handle = path.open("w")
    proc = subprocess.Popen(
        command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    CHILDREN.append((proc, handle, name))
    with (OUT / "commands.jsonl").open("a") as f:
        f.write(json.dumps({"name": name, "command": command, "chips": chips}) + "\n")
    print("START", name, proc.pid, flush=True)
    return proc


def check():
    for proc, _, name in CHILDREN:
        if proc.poll() is not None:
            raise RuntimeError(
                f'{name} exited {proc.returncode}: {(OUT / (name + ".log")).read_text()[-5000:]}'
            )


def stop():
    for proc, _, _ in CHILDREN:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for proc, handle, _ in CHILDREN:
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        handle.close()
    CHILDREN.clear()


def health(port):
    end = time.monotonic() + 1800
    while time.monotonic() < end:
        check()
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"health port {port}")


COMMON = [
    "--model-path",
    MODEL,
    "--nnodes",
    "1",
    "--tp-size",
    "4",
    "--dp-size",
    "1",
    "--ep-size",
    "4",
    "--moe-backend",
    "epmoe",
    "--dtype",
    "bfloat16",
    "--kv-cache-dtype",
    "bf16",
    "--attention-backend",
    "fa",
    "--page-size",
    "128",
    "--chunked-prefill-size",
    "2048",
    "--context-length",
    "32768",
    "--max-running-requests",
    "32",
    "--max-total-tokens",
    "262144",
    "--mem-fraction-static",
    "0.8",
    "--disable-radix-cache",
    "--skip-server-warmup",
    "--precompile-bs-paddings",
    "1",
    "4",
    "16",
    "32",
    "--precompile-token-paddings",
    "128",
    "512",
    "2048",
    "--enable-request-time-stats-logging",
    "--host",
    "0.0.0.0",
]


def boot(group, p_overlap=False, d_overlap=False, chunk=False, standalone=False):
    stop()
    if standalone:
        launch(
            group + "-standalone",
            [sys.executable, "-m", "sgl_jax.launch_server", *COMMON, "--port", "30010"],
            "2,3",
        )
        health(30010)
        return 30010
    launch(
        group + "-bootstrap",
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
    health(8998)
    for role, port, side, chips, overlap in [
        ("prefill", 30000, 9600, "0,1", p_overlap),
        ("decode", 30010, 9700, "2,3", d_overlap),
    ]:
        args = [
            sys.executable,
            "-m",
            "sgl_jax.launch_server",
            *COMMON,
            "--port",
            str(port),
            "--disaggregation-mode",
            role,
            "--disaggregation-use-raiden",
            "--disaggregation-host-ip",
            HOST,
            "--disaggregation-bootstrap-url",
            f"http://{HOST}:8998",
            "--disaggregation-transfer-port",
            str(port + 1),
            "--disaggregation-side-channel-port",
            str(side),
            "--disaggregation-max-inflight-transfers",
            "4",
        ]
        if overlap:
            args += ["--disaggregation-enable-overlap-schedule"]
        if chunk:
            args += ["--disaggregation-enable-chunk-prefill-transfer"]
        launch(group + "-" + role, args, chips)
    health(30000)
    health(30010)
    launch(
        group + "-router",
        [
            sys.executable,
            "-m",
            "sgl_jax.srt.disaggregation.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            f"http://{HOST}:30000",
            "8998",
            "--decode",
            "http://127.0.0.1:30010",
            "--prefill-bootstrap-host",
            HOST,
            "--max-concurrent-requests",
            "32",
            "--host",
            "0.0.0.0",
            "--port",
            "30020",
        ],
    )
    health(30020)
    return 30020


TOKENIZER = None


def payload(length, output, stream=False, variant=0):
    tokens = TOKENIZER.encode(
        "Explain the following sequence carefully. " + str(variant) + " ", add_special_tokens=False
    )
    ids = (tokens * (length // len(tokens) + 1))[:length]
    return {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": output, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": -1,
        "stream": stream,
    }


def generate(port, length, output, stream=False, variant=0):
    start = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json=payload(length, output, stream, variant),
        timeout=600,
        stream=stream,
    )
    response.raise_for_status()
    if stream:
        events = []
        for line in response.iter_lines(chunk_size=1024):
            if line.startswith(b"data:") and b"[DONE]" not in line:
                events.append((time.perf_counter() - start, json.loads(line[5:])))
        assert events, "empty stream"
        result = events[-1][1]
        times = [t for t, e in events if e.get("meta_info", {}).get("completion_tokens", 0) > 0]
        ttft = times[0] if times else None
    else:
        result = response.json()
        ttft = None
        events = []
    assert "error" not in result, result
    meta = result.get("meta_info", {})
    assert (
        len(meta.get("output_token_logprobs", [])) == output
    ), "duplicated/missing output logprobs"
    token_ids = result.get("output_ids")
    if token_ids is None:
        token_ids = [v[1] for v in meta.get("output_token_logprobs", [])]
    assert len(token_ids) == output, {
        "expected": output,
        "actual": len(token_ids),
        "result": result,
    }
    if stream:
        previous = []
        for _, event in events:
            now = event.get("output_ids")
            if now is None:
                now = [v[1] for v in event.get("meta_info", {}).get("output_token_logprobs", [])]
            assert now[: len(previous)] == previous, "non-monotonic stream token sequence"
            previous = now
    return {
        "input": length,
        "output": output,
        "variant": variant,
        "token_ids": token_ids,
        "elapsed_s": time.perf_counter() - start,
        "ttft_s": ttft,
        "meta_info": meta,
        "stream_event_times": [t for t, _ in events],
    }


def record(group, result):
    with (OUT / f"{group}-requests.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    global TOKENIZER
    OUT.mkdir(parents=True, exist_ok=True)
    save()
    # Do not run serving unless both independent process probes finish successfully.
    probes = [
        launch(
            "probe-" + role,
            [
                sys.executable,
                str(Path(__file__).with_name("split_chip_probe.py")),
                "--role",
                role,
                "--out",
                str(OUT / "probe"),
            ],
            chips,
        )
        for role, chips in [("producer", "0,1"), ("consumer", "2,3")]
    ]
    for proc in probes:
        rc = proc.wait(timeout=600)
        if rc:
            raise RuntimeError(f"probe exited {rc}")
    stop()
    SUMMARY["stages"].append({"stage": "S0-page-integrity", "status": "passed"})
    save()
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    cases = [
        (127, 32),
        (128, 32),
        (129, 32),
        (2047, 32),
        (2048, 32),
        (2049, 32),
        (4096, 256),
        (16384, 256),
        (4096, 1024),
    ]
    reference = {}
    port = boot("reference", standalone=True)
    for length, output in cases:
        result = generate(port, length, output)
        record("reference", result)
        reference[length, output] = result["token_ids"]
    stop()
    SUMMARY["stages"].append({"stage": "S0-model-and-reference", "status": "passed"})
    save()
    baseline = {}
    groups = [
        ("B0", False, False, False),
        ("B1", False, False, True),
        ("P", True, False, True),
        ("D", False, True, True),
        ("PD", True, True, True),
        ("PD-no-chunk", True, True, False),
    ]
    for group, po, do, chunk in groups:
        port = boot(group, po, do, chunk)
        import lifecycle_checks

        initial_capacity = lifecycle_checks.idle()
        for length, output in cases:
            result = generate(port, length, output)
            record(group, result)
            lifecycle_checks.idle(initial_capacity)
            target = reference if group == "B0" else baseline
            assert (
                result["token_ids"] == target[length, output]
            ), f"{group} token mismatch {length}/{output}"
            if group == "B0":
                baseline[length, output] = result["token_ids"]
        SUMMARY["stages"].append(
            {"stage": "S1-" + group + "-nonstream", "status": "passed", "cases": len(cases)}
        )
        save()
        print("NONSTREAM_PASSED", group, flush=True)
        streamed = generate(port, 2049, 32, stream=True)
        record(group, streamed)
        assert streamed["token_ids"] == baseline[2049, 32]
        for concurrency in [4, 16]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = list(
                    executor.map(lambda _: generate(port, 4096, 256), range(concurrency))
                )
            for result in results:
                record(group, result)
                assert (
                    result["token_ids"] == baseline[4096, 256]
                ), f"{group} concurrent token mismatch"
        recovered = lifecycle_checks.idle(initial_capacity)
        SUMMARY["stages"].append(
            {
                "stage": "S1-" + group,
                "status": "passed",
                "initial_capacity": initial_capacity,
                "recovered_capacity": recovered,
            }
        )
        save()
        if group == "PD":
            import lifecycle_checks

            results = lifecycle_checks.run(sys.modules[__name__], port, baseline[4096, 1024])
            SUMMARY["stages"].append(
                {"stage": "S2-controls-and-recovery", "status": "passed", "results": results}
            )
            save()
        stop()
    if os.environ.get("PD_RUN_MODE") == "correctness":
        SUMMARY["status"] = "passed_implemented_correctness_checks"
        save()
        return
    # Warm each configuration, alternate order across three measured passes.
    for repetition in range(3):
        order = [("B1", False), ("PD", True)]
        if repetition % 2:
            order.reverse()
        for group, overlap in order:
            tag = f"perf-{repetition}-{group}"
            port = boot(tag, overlap, overlap, True)
            for length, output in [(4096, 256), (16384, 1024)]:
                generate(port, length, output)
                for concurrency in [1, 4, 16, 32]:
                    n = max(32, concurrency * 4)
                    started = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                        results = list(
                            executor.map(
                                lambda i: generate(
                                    port, length, output, stream=True, variant=i % 4
                                ),
                                range(n),
                            )
                        )
                    elapsed = time.perf_counter() - started
                    for result in results:
                        record(tag, result)
                    SUMMARY["stages"].append(
                        {
                            "stage": "S3-" + tag,
                            "status": "passed",
                            "input": length,
                            "output": output,
                            "concurrency": concurrency,
                            "requests": n,
                            "elapsed_s": elapsed,
                            "tokens_per_s": n * output / elapsed,
                            "ttft_s": sorted(
                                r["ttft_s"] for r in results if r["ttft_s"] is not None
                            ),
                        }
                    )
                    save()
            if repetition == 2:
                for role, server_port in [("prefill", 30000), ("decode", 30010)]:
                    requests.post(
                        f"http://127.0.0.1:{server_port}/start_profile",
                        json={
                            "output_dir": str(OUT / "profiles" / group / role),
                            "host_tracer_level": 2,
                            "python_tracer_level": 0,
                        },
                        timeout=120,
                    ).raise_for_status()
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    list(executor.map(lambda _: generate(port, 16384, 256), range(32)))
                for server_port in [30000, 30010]:
                    requests.post(
                        f"http://127.0.0.1:{server_port}/stop_profile", timeout=120
                    ).raise_for_status()
            stop()
    SUMMARY["status"] = "passed_implemented_checks"
    SUMMARY["stages"].append(
        {
            "stage": "S2-native-delay-peer-failure",
            "status": "not_run",
            "reason": "Requires targeted native injection; not inferred from successful traffic.",
        }
    )
    save()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise TimeoutError(f"runner interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        SUMMARY["status"] = "failed"
        SUMMARY["error"] = repr(exc)
        SUMMARY["traceback"] = traceback.format_exc()
        save()
        traceback.print_exc()
        raise
    finally:
        stop()
        print("EXPERIMENT_SUMMARY=" + json.dumps(SUMMARY), flush=True)
