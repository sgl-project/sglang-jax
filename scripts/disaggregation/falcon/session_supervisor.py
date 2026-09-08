"""Bounded Falcon development session with serial, auditable test commands."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = Path("/tmp/tpu_logs/pd-session")
RESULTS = Path(os.environ["PD_OUT"])
ARTIFACT = Path(os.environ["ARTIFACT_LOCAL_DIR"]) / "pd-results"
DEADLINE = time.monotonic() + 21000
child = None


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def archive():
    shutil.copytree(RESULTS, ARTIFACT, dirs_exist_ok=True)


def stop_child():
    global child
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
    child = None


def main():
    global child
    STATE.mkdir(parents=True, exist_ok=True)
    commands = STATE / "commands"
    commands.mkdir(exist_ok=True)
    first = commands / "001-correctness.json"
    write(
        first,
        {
            "id": "correctness-01",
            "command": [
                sys.executable,
                str(ROOT / "scripts/disaggregation/falcon/single_pod_pd.py"),
            ],
            "env": {"PD_RUN_MODE": "correctness"},
            "timeout_s": 7200,
        },
    )
    manifest = json.loads((RESULTS / "source-manifest.json").read_text())
    diagnostic = manifest.get("diagnostics_only", False)
    automatic = manifest.get("pr_validation", False) or diagnostic
    if automatic and not diagnostic:
        write(
            commands / "002-pr-validation.json",
            {
                "id": "pr-validation",
                "command": [
                    sys.executable,
                    str(ROOT / "scripts/disaggregation/falcon/pr_validation_suite.py"),
                ],
                "timeout_s": 18000,
            },
        )
        write(
            commands / "003-steady-ablation.json",
            {
                "id": "steady-ablation",
                "command": [
                    sys.executable,
                    str(ROOT / "scripts/disaggregation/falcon/steady_ablation.py"),
                ],
                "env": {
                    "PD_REQUIRE_SUMMARY": str(RESULTS / "correctness-01/summary.json"),
                },
                "timeout_s": 3600,
            },
        )
    if diagnostic:
        for number, name, runner, limit in [
            (2, "steady-ablation", "steady_ablation.py", 3600),
            (3, "profiles", "profile_suite.py", 2400),
        ]:
            write(
                commands / f"{number:03}-{name}.json",
                {
                    "id": name,
                    "command": [
                        sys.executable,
                        str(ROOT / "scripts/disaggregation/falcon" / runner),
                    ],
                    "env": {"PD_REQUIRE_SUMMARY": str(RESULTS / "correctness-01/summary.json")},
                    "timeout_s": limit,
                },
            )
    completed = set()
    last_exit = 1
    while time.monotonic() < DEADLINE:
        finish = STATE / "finish.json"
        if finish.exists():
            code = int(json.loads(finish.read_text()).get("exit_code", last_exit))
            archive()
            return code
        pending = [p for p in sorted(commands.glob("*.json")) if p.name not in completed]
        if not pending:
            if automatic:
                return last_exit
            write(
                STATE / "status.json",
                {
                    "state": "awaiting_command",
                    "completed": sorted(completed),
                    "last_exit_code": last_exit,
                    "deadline_remaining_s": int(DEADLINE - time.monotonic()),
                },
            )
            time.sleep(5)
            continue
        path = pending[0]
        spec = json.loads(path.read_text())
        name = spec["id"]
        if Path(name).name != name:
            raise ValueError("command id must be a basename")
        out = RESULTS / name
        out.mkdir(exist_ok=False)
        write(out / "command.json", spec)
        (out / "source.patch").write_bytes(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        )
        write(
            out / "runner-sha256.json",
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "scripts/disaggregation/falcon").glob("*.py")
            },
        )
        runner_source = out / "runner-source"
        runner_source.mkdir()
        for source in (ROOT / "scripts/disaggregation/falcon").iterdir():
            if source.suffix in {".py", ".sh"}:
                shutil.copy2(source, runner_source / source.name)
        env = dict(os.environ, **spec.get("env", {}), PD_OUT=str(out))
        status = {"state": "running", "id": name, "started_at": time.time(), "out": str(out)}
        write(STATE / "status.json", status)
        print("SESSION_COMMAND_START=" + json.dumps(status), flush=True)
        child = subprocess.Popen(spec["command"], cwd=ROOT, env=env)
        try:
            last_exit = child.wait(
                timeout=min(spec.get("timeout_s", 7200), DEADLINE - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            stop_child()
            last_exit = 124
        child = None
        completed.add(path.name)
        status.update(state="finished", exit_code=last_exit, finished_at=time.time())
        write(out / "session-result.json", status)
        archive()
        print("SESSION_COMMAND_FINISHED=" + json.dumps(status), flush=True)
        if automatic and last_exit != 0:
            return last_exit
    return 124


if __name__ == "__main__":

    def interrupted(*_):
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    finally:
        stop_child()
        archive()
