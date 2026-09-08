"""Run ready-for-PR gates serially; fail fast and preserve each runner's report."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["PD_OUT"])
REPORT = {"status": "running", "stages": []}


def save():
    (OUT / "pr-validation-summary.json").write_text(json.dumps(REPORT, indent=2))


def main():
    correctness = OUT.parent / "correctness-01/summary.json"
    assert json.loads(correctness.read_text())["status"] == "passed_implemented_correctness_checks"
    for name, runner, report, limit in [
        ("stream", "stream_regression.py", "stream-summary.json", 1800),
        ("fault", "fault_suite.py", "fault-summary.json", 1800),
        ("eos", "eos_checks.py", "eos-summary.json", 1800),
        ("mixed", "mixed_requests.py", "mixed-summary.json", 1800),
        ("steady", "steady_suite.py", "steady-summary.json", 14000),
    ]:
        directory = OUT / name
        directory.mkdir(exist_ok=False)
        stage = {"name": name, "status": "running", "started_at": time.time()}
        REPORT["stages"].append(stage)
        save()
        env = dict(os.environ, PD_OUT=str(directory), PD_REQUIRE_SUMMARY=str(correctness))
        subprocess.run([sys.executable, str(HERE / runner)], env=env, check=True, timeout=limit)
        result = json.loads((directory / report).read_text())
        assert result["status"] == "passed", result
        stage.update(status="passed", finished_at=time.time())
        save()
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc))
        save()
        raise
