import argparse
import dataclasses
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

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from accuracy_case_runner import run_accuracy_case
from multi_host_suite import (
    AccuracyCase,
    ModelRun,
    MultiHostSuite,
    PerfCase,
    RuntimeConfig,
    dry_run_suite,
)
from perf_case_runner import run_perf_case
from profile_loader import LaunchProfile, build_other_server_args, load_profile

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
    control_port = int(_get_env("CONTROL_PORT"))
    server = ThreadingHTTPServer(("0.0.0.0", control_port), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _log(f"Control server started on port {control_port}")
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
    dist_init_port = int(_get_env("DIST_INIT_PORT"))
    server_port = int(_get_env("SERVER_PORT"))
    return RuntimeConfig(
        nnodes=nnodes,
        node_rank=resolved_node_rank,
        dist_init_addr=f"{workload_name}-0.{headless_service_name}:{dist_init_port}",
        host="0.0.0.0",
        port=server_port,
    )


def run_case(case: PerfCase | AccuracyCase, profile: LaunchProfile) -> None:
    if isinstance(case, PerfCase):
        run_perf_case(case, profile)
        return
    if isinstance(case, AccuracyCase):
        run_accuracy_case(case, profile)
        return

    raise NotImplementedError(f"Unsupported case type: {type(case).__name__}")


def run_model_run(model_run: ModelRun, runtime_cfg: RuntimeConfig) -> int:
    from sgl_jax.srt.utils import kill_process_tree
    from sgl_jax.test.test_utils import popen_launch_server

    profile = load_profile(model_run.launch_profile)
    runtime_cfg = dataclasses.replace(runtime_cfg, port=profile.port)
    _log(
        f"Launching model run={profile.name}, target={profile.target}, "
        f"rank={runtime_cfg.node_rank}, port={runtime_cfg.port}"
    )
    is_rank0 = runtime_cfg.node_rank == 0
    _reset_state()
    control_server = start_control_server() if is_rank0 else None
    server_process = None
    exit_code = 0

    try:
        base_url = f"http://{runtime_cfg.host}:{runtime_cfg.port}"
        server_process = popen_launch_server(
            model=profile.model_path,
            base_url=base_url,
            timeout=1800,
            other_args=build_other_server_args(profile, runtime_cfg),
        )

        if is_rank0:
            failed_cases: list[tuple[str, BaseException]] = []
            for case in model_run.cases:
                try:
                    run_case(case, profile)
                except Exception as exc:
                    failed_cases.append((case.name, exc))
                    _log(f"Case {case.name} failed: {exc!r}")
            if failed_cases:
                exit_code = 1
                _log(f"Run {profile.name} failed cases: " f"{[name for name, _ in failed_cases]}")
        else:
            workload_name = _get_env("WORKLOAD_NAME")
            headless_service_name = _get_env("HEADLESS_SERVICE_NAME")
            control_url = (
                f"http://{workload_name}-0.{headless_service_name}:"
                f"{int(_get_env('CONTROL_PORT'))}/status"
            )
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
    _log(f"Running suite={suite.name}, rank={runtime_cfg.node_rank}")
    exit_code = 0
    for model_run in suite.runs:
        exit_code = run_model_run(model_run, runtime_cfg)
        if exit_code != 0:
            return exit_code
    return exit_code


def _resolve_launch_profile(run: ModelRun, base_dir: Path) -> ModelRun:
    if Path(run.launch_profile).is_absolute():
        return run
    return dataclasses.replace(run, launch_profile=str(base_dir / run.launch_profile))


# Explicit suite registry. To add a new suite: insert a new entry below.
SUITES: dict[str, MultiHostSuite] = {
    "mimo-flash-pref-test": MultiHostSuite(
        name="mimo-flash-pref-test",
        runs=[
            ModelRun(
                launch_profile="launch_profiles/mimo-flash-v6e-4x4.yaml",
                cases=[
                    AccuracyCase(
                        name="mimo-flash-gsm8k",
                        dataset="gsm8k",
                        model_id="XiaomiMiMo/MiMo-V2-Flash",
                        eval_batch_size=32,
                        generation_config={"temperature": 0.8, "top_p": 0.95},
                    ),
                    AccuracyCase(
                        name="mimo-flash-aime25",
                        dataset="aime25",
                        model_id="XiaomiMiMo/MiMo-V2-Flash",
                        eval_batch_size=32,
                        generation_config={"temperature": 0.8, "top_p": 0.95},
                    ),
                ],
            ),
        ],
    ),
}


def get_suites() -> dict[str, MultiHostSuite]:
    base_dir = Path(_SELF_DIR)
    return {
        name: dataclasses.replace(
            suite, runs=[_resolve_launch_profile(r, base_dir) for r in suite.runs]
        )
        for name, suite in SUITES.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-host SGL-JAX suites")
    parser.add_argument("--suite", choices=sorted(get_suites()))
    parser.add_argument(
        "--target",
        default=os.environ.get("TARGET"),
        help="Filter suites by launch profile target (e.g. v6e-4x4). Defaults to $TARGET.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_suites(suite_name: str | None, target: str | None) -> list[MultiHostSuite]:
    suites = get_suites()
    if suite_name is not None:
        return [suites[suite_name]]
    if target:
        filtered = [
            s
            for s in suites.values()
            if all(load_profile(r.launch_profile).target == target for r in s.runs)
        ]
        if not filtered:
            raise ValueError(f"No suite matched target={target}")
        return filtered
    return list(suites.values())


def main() -> int:
    args = parse_args()
    suites = select_suites(args.suite, args.target)

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
