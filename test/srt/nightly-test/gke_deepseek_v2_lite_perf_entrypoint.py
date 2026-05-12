import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import get_benchmark_args

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
TP_SIZE = 16
DEFAULT_NNODES = 4
SERVER_PORT = 30000
DIST_INIT_PORT = 10011
CONTROL_PORT = 18080
NUM_PROMPTS = 24
MAX_CONCURRENCY = 8
INPUT_LEN = 1024
OUTPUT_LEN = 128

_control_state = {"done": False, "exit_code": 0}


def log(message):
    print(f"[gke-deepseek-v2-lite-perf] {message}", flush=True)


def get_env(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_model_path():
    candidates = [
        "/models/model_scope/deepseek-ai/DeepSeek-V2-Lite",
        "/models/model_scope/DEEPSEEK_V2_LITE",
        "/models/DeepSeek-V2-Lite",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            log(f"Using cached model path: {candidate}")
            return candidate
    log(f"Model cache miss; falling back to Hugging Face model id: {MODEL_ID}")
    return MODEL_ID


class ControlHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/status":
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(_control_state).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_control_server():
    server = ThreadingHTTPServer(("0.0.0.0", CONTROL_PORT), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f"Control server started on port {CONTROL_PORT}")
    return server


def wait_for_done(control_url, server_process, timeout_seconds=3600):
    log(f"Waiting for rank 0 done signal from {control_url}")
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        try:
            with urllib.request.urlopen(control_url, timeout=10) as response:
                state = json.loads(response.read().decode("utf-8"))
            if state.get("done"):
                exit_code = int(state.get("exit_code", 0))
                log(f"Received done signal with exit_code={exit_code}")
                return exit_code
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            log(f"Done signal not available yet: {exc}")

        return_code = server_process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited before done signal: {return_code}")

        time.sleep(10)

    raise TimeoutError(f"Done signal did not arrive within {timeout_seconds} seconds")


def wait_for_server(base_url, process, timeout_seconds=1800):
    log(f"Waiting for server health at {base_url}")
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited while waiting for health: {return_code}")

        try:
            response = requests.get(f"{base_url}/health_generate?multimodal=False", timeout=30)
            if response.status_code == 200:
                log("Server is ready")
                return
            log(f"Health check returned status={response.status_code}")
        except requests.RequestException as exc:
            log(f"Health check failed: {exc}")

        time.sleep(10)

    raise TimeoutError(f"Server did not become ready within {timeout_seconds} seconds")


def launch_server(model_path, rank, nnodes, workload_name, headless_service_name):
    dist_init_addr = f"{workload_name}-0.{headless_service_name}:{DIST_INIT_PORT}"
    cmd = [
        sys.executable,
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        model_path,
        "--trust-remote-code",
        "--skip-server-warmup",
        "--random-seed",
        "3",
        "--mem-fraction-static",
        "0.8",
        "--max-prefill-tokens",
        "8192",
        "--download-dir",
        "/dev/shm/",
        "--dtype",
        "bfloat16",
        "--tp-size",
        str(TP_SIZE),
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(rank),
        "--dist-init-addr",
        dist_init_addr,
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--page-size",
        "128",
        "--context-length",
        "8192",
        "--device",
        "tpu",
    ]
    log("Launching server:")
    log(" ".join(cmd))
    return subprocess.Popen(cmd, stdout=None, stderr=None, env=os.environ.copy())


def run_perf(model_path, nnodes):
    base_url = f"http://127.0.0.1:{SERVER_PORT}"
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name="random",
        tokenizer=model_path,
        num_prompts=NUM_PROMPTS,
        random_input_len=INPUT_LEN,
        random_output_len=OUTPUT_LEN,
        max_concurrency=MAX_CONCURRENCY,
        random_range_ratio=1.0,
        request_rate=float("inf"),
        seed=42,
        warmup_requests=0,
        backend="sgl-jax",
    )
    args.output_file = "/dev/null"

    log(
        "Running benchmark: "
        f"num_prompts={NUM_PROMPTS}, concurrency={MAX_CONCURRENCY}, "
        f"input_len={INPUT_LEN}, output_len={OUTPUT_LEN}"
    )
    metrics = run_benchmark(args)

    summary = {
        "model": MODEL_ID,
        "nnodes": nnodes,
        "tp_size": TP_SIZE,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "concurrency": MAX_CONCURRENCY,
        "num_prompts": NUM_PROMPTS,
        "completed": metrics.get("completed"),
        "median_ttft_ms": metrics.get("median_ttft_ms"),
        "median_itl_ms": metrics.get("median_itl_ms"),
        "input_throughput": metrics.get("input_throughput"),
        "output_throughput": metrics.get("output_throughput"),
        "request_throughput": metrics.get("request_throughput"),
    }
    log("Benchmark summary:")
    log(json.dumps(summary, indent=2, sort_keys=True))

    if metrics.get("completed") != NUM_PROMPTS:
        raise RuntimeError(f"Expected completed={NUM_PROMPTS}, got {metrics.get('completed')}")
    if metrics.get("output_throughput", 0) <= 0:
        raise RuntimeError(
            f"Expected positive output_throughput, got {metrics.get('output_throughput')}"
        )
    if metrics.get("median_ttft_ms", 0) <= 0:
        raise RuntimeError(f"Expected positive median_ttft_ms, got {metrics.get('median_ttft_ms')}")


def main():
    rank = int(get_env("JOB_COMPLETION_INDEX"))
    nnodes = int(get_env("NNODES", str(DEFAULT_NNODES)))
    workload_name = get_env("WORKLOAD_NAME")
    headless_service_name = get_env("HEADLESS_SERVICE_NAME")

    log(f"rank={rank}, nnodes={nnodes}, workload_name={workload_name}")
    log(f"TPU_PROCESS_ADDRESSES={os.environ.get('TPU_PROCESS_ADDRESSES')}")
    log(f"TPU_WORKER_HOSTNAMES={os.environ.get('TPU_WORKER_HOSTNAMES')}")

    model_path = resolve_model_path()
    control_server = start_control_server() if rank == 0 else None
    server_process = launch_server(model_path, rank, nnodes, workload_name, headless_service_name)
    exit_code = 0

    try:
        if rank == 0:
            wait_for_server(f"http://127.0.0.1:{SERVER_PORT}", server_process)
            run_perf(model_path, nnodes)
            _control_state["done"] = True
            _control_state["exit_code"] = 0
        else:
            control_url = f"http://{workload_name}-0.{headless_service_name}:{CONTROL_PORT}/status"
            exit_code = wait_for_done(control_url, server_process)
    except Exception as exc:
        exit_code = 1
        log(f"ERROR: {exc}")
        if rank == 0:
            _control_state["done"] = True
            _control_state["exit_code"] = exit_code
        raise
    finally:
        if rank == 0:
            log("Keeping control server alive for worker ranks")
            time.sleep(30)
        log("Stopping server process")
        kill_process_tree(server_process.pid)
        if control_server is not None:
            control_server.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
