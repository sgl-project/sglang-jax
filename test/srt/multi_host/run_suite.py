import argparse
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make sibling modules (multi_host_suite, test_*.py suites) importable regardless
# of cwd or invocation style. Python normally only adds the script's directory
# to sys.path[0] when run as `python file.py`; this ensures other entry points
# (e.g. `python -c`, runners that change cwd) still work.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from multi_host_suite import (
    AccuracyCase,
    ModelRun,
    MultiHostSuite,
    PerfCase,
    RuntimeConfig,
    build_other_server_args,
    dry_run_suite,
)

DIST_INIT_PORT = 10011
SERVER_PORT = 30000
CONTROL_PORT = 18080

_control_state = {"done": False, "exit_code": 0}


def _publish_state(exit_code: int) -> None:
    """Set terminal state visible to worker ranks via /status.

    Writes exit_code BEFORE done so any worker that observes ``done == True``
    is guaranteed to read the matching exit_code in the same poll. Without
    this ordering a worker can race the two assignments and report success
    while rank 0 actually failed.
    """
    _control_state["exit_code"] = exit_code
    _control_state["done"] = True


def _reset_state() -> None:
    _control_state["done"] = False
    _control_state["exit_code"] = 0


def _log(message: str) -> None:
    print(f"[multi-host-suite] {message}", flush=True)


def _get_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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


def start_control_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", CONTROL_PORT), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _log(f"Control server started on port {CONTROL_PORT}")
    return server


def wait_for_done(control_url: str, server_process, timeout_seconds: int = 3600) -> int:
    _log(f"Waiting for rank 0 done signal from {control_url}")
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        try:
            with urllib.request.urlopen(control_url, timeout=10) as response:
                state = json.loads(response.read().decode("utf-8"))
            if state.get("done"):
                exit_code = int(state.get("exit_code", 0))
                _log(f"Received done signal with exit_code={exit_code}")
                return exit_code
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"Done signal not available yet: {exc}")

        return_code = server_process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited before done signal: {return_code}")

        time.sleep(10)

    raise TimeoutError(f"Done signal did not arrive within {timeout_seconds} seconds")


def build_runtime_config(node_rank: int | None = None) -> RuntimeConfig:
    workload_name = _get_env("WORKLOAD_NAME")
    headless_service_name = _get_env("HEADLESS_SERVICE_NAME")
    resolved_node_rank = int(_get_env("JOB_COMPLETION_INDEX") if node_rank is None else node_rank)
    nnodes = int(_get_env("NNODES", "4"))
    return RuntimeConfig(
        nnodes=nnodes,
        node_rank=resolved_node_rank,
        dist_init_addr=f"{workload_name}-0.{headless_service_name}:{DIST_INIT_PORT}",
        host="0.0.0.0",
        port=SERVER_PORT,
    )


def run_case(case: PerfCase | AccuracyCase, model_path: str, port: int) -> None:
    if isinstance(case, PerfCase):
        run_perf_case(case, model_path, port)
        return
    if isinstance(case, AccuracyCase):
        run_accuracy_case(case, port)
        return

    raise NotImplementedError(f"Unsupported case type: {type(case).__name__}")


def run_perf_case(case: PerfCase, model_path: str, port: int) -> None:
    from sgl_jax.bench_serving import run_benchmark
    from sgl_jax.test.test_utils import get_benchmark_args

    base_url = f"http://127.0.0.1:{port}"
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name="random",
        tokenizer=model_path,
        num_prompts=case.num_prompts,
        random_input_len=case.input_len,
        random_output_len=case.output_len,
        max_concurrency=case.max_concurrency,
        random_range_ratio=1.0,
        request_rate=case.request_rate,
        seed=case.seed,
        warmup_requests=0,
        backend="sgl-jax",
    )
    args.output_file = "/dev/null"
    args.flush_cache = case.flush_cache

    _log(
        "Running perf case "
        f"name={case.name}, num_prompts={case.num_prompts}, "
        f"concurrency={case.max_concurrency}, input_len={case.input_len}, "
        f"output_len={case.output_len}, request_rate={case.request_rate}, "
        f"seed={case.seed}, flush_cache={case.flush_cache}"
    )
    metrics = run_benchmark(args)

    summary = {
        "case": case.name,
        "completed": metrics.get("completed"),
        "median_ttft_ms": metrics.get("median_ttft_ms"),
        "median_itl_ms": metrics.get("median_itl_ms"),
        "input_throughput": metrics.get("input_throughput"),
        "output_throughput": metrics.get("output_throughput"),
        "request_throughput": metrics.get("request_throughput"),
    }
    _log("Perf summary:")
    _log(json.dumps(summary, indent=2, sort_keys=True))

    if metrics.get("completed") != case.num_prompts:
        raise RuntimeError(f"Expected completed={case.num_prompts}, got {metrics.get('completed')}")
    if metrics.get("output_throughput", 0) <= 0:
        raise RuntimeError(
            f"Expected positive output_throughput, got {metrics.get('output_throughput')}"
        )
    if metrics.get("median_ttft_ms", 0) <= 0:
        raise RuntimeError(f"Expected positive median_ttft_ms, got {metrics.get('median_ttft_ms')}")


def run_accuracy_case(case: AccuracyCase, port: int) -> None:
    api_url = f"http://127.0.0.1:{port}/v1"
    cmd = [
        "evalscope",
        "eval",
        "--model",
        case.model_id,
        "--api-url",
        api_url,
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        case.dataset,
        "--eval-batch-size",
        str(case.eval_batch_size),
    ]
    if case.generation_config:
        cmd.extend(["--generation-config", json.dumps(case.generation_config)])
    if case.limit is not None:
        cmd.extend(["--limit", str(case.limit)])
    if case.timeout is not None:
        cmd.extend(["--timeout", str(case.timeout)])

    _log(
        "Running accuracy case "
        f"name={case.name}, dataset={case.dataset}, "
        f"eval_batch_size={case.eval_batch_size}, "
        f"generation_config={case.generation_config}, limit={case.limit}, "
        f"timeout={case.timeout}"
    )
    _log(f"Command: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"evalscope exited with code {completed.returncode} for case={case.name}"
        )
    _log(f"Accuracy case {case.name} completed (warn-only mode, accuracy not gated)")


def run_model_run(model_run: ModelRun, runtime_cfg: RuntimeConfig) -> int:
    from sgl_jax.srt.utils import kill_process_tree
    from sgl_jax.test.test_utils import popen_launch_server

    runtime_cfg = dataclasses.replace(runtime_cfg, port=model_run.model.port)
    _log(
        f"Launching model run={model_run.name}, rank={runtime_cfg.node_rank}, port={runtime_cfg.port}"
    )
    is_rank0 = runtime_cfg.node_rank == 0
    _reset_state()
    control_server = start_control_server() if is_rank0 else None
    server_process = None
    exit_code = 0

    try:
        base_url = f"http://{runtime_cfg.host}:{runtime_cfg.port}"
        server_process = popen_launch_server(
            model=model_run.model.model_path,
            base_url=base_url,
            timeout=1800,
            other_args=build_other_server_args(model_run.model, runtime_cfg),
        )

        if is_rank0:
            for case in model_run.cases:
                run_case(case, model_run.model.model_path, runtime_cfg.port)
        else:
            workload_name = _get_env("WORKLOAD_NAME")
            headless_service_name = _get_env("HEADLESS_SERVICE_NAME")
            control_url = f"http://{workload_name}-0.{headless_service_name}:{CONTROL_PORT}/status"
            exit_code = wait_for_done(control_url, server_process)
    except Exception:
        exit_code = 1
        raise
    finally:
        if is_rank0:
            # Always publish — covers success, case failure, and popen_launch_server
            # failure (where server_process never got assigned). Without this,
            # workers spin on /status until wait_for_done's 60-min timeout when
            # rank 0 dies during launch.
            _publish_state(exit_code)
            _log("Keeping control server alive for worker ranks")
            time.sleep(30)
        if server_process is not None:
            _log("Stopping server process")
            kill_process_tree(server_process.pid)
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
        if control_server is not None:
            control_server.shutdown()

    return exit_code


def run_suite(suite: MultiHostSuite, runtime_cfg: RuntimeConfig) -> int:
    _log(f"Running suite={suite.name}, target={suite.target}, rank={runtime_cfg.node_rank}")
    exit_code = 0
    for model_run in suite.runs:
        exit_code = run_model_run(model_run, runtime_cfg)
        if exit_code != 0:
            return exit_code
    return exit_code


def _discover_suite_modules() -> list[str]:
    """List sibling test_*.py files (without .py), sorted for determinism."""
    return sorted(
        f[:-3] for f in os.listdir(_SELF_DIR) if f.startswith("test_") and f.endswith(".py")
    )


def get_suites() -> dict[str, MultiHostSuite]:
    suites = {}
    for module_name in _discover_suite_modules():
        module = importlib.import_module(module_name)
        if not hasattr(module, "get_suites"):
            continue
        for suite in module.get_suites():
            if suite.name in suites:
                raise ValueError(f"Duplicate multi-host suite name: {suite.name}")
            suites[suite.name] = suite
    return suites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-host SGL-JAX suites")
    parser.add_argument("--suite", choices=sorted(get_suites()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_suites(suite_name: str | None) -> list[MultiHostSuite]:
    suites = get_suites()
    if suite_name is not None:
        return [suites[suite_name]]
    return list(suites.values())


def main() -> int:
    args = parse_args()
    suites = select_suites(args.suite)

    if args.dry_run:
        print(json.dumps([dry_run_suite(suite) for suite in suites], indent=2, sort_keys=True))
        return 0

    runtime_cfg = build_runtime_config()
    for suite in suites:
        exit_code = run_suite(suite, runtime_cfg)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
