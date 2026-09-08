"""Capture PD profiles separately, allowing for blocking JAX trace export."""

import json
import os
import signal
import traceback
from pathlib import Path

import steady_suite as suite
from transformers import AutoTokenizer


def main():
    suite.driver.OUT.mkdir(parents=True, exist_ok=True)
    assert (
        json.loads(Path(os.environ["PD_REQUIRE_SUMMARY"]).read_text())["status"]
        == "passed_implemented_correctness_checks"
    )
    suite.driver.TOKENIZER = AutoTokenizer.from_pretrained(suite.driver.MODEL)
    # stop_profile blocks the scheduler while exporting. An observed 80–93 s
    # export exceeds the normal 30 s pull / 60 s ack budget, even in B1.
    # These overrides apply only to profiling, never throughput or fault tests.
    original = suite.driver.COMMON[:]
    suite.driver.COMMON += [
        "--disaggregation-pull-timeout-seconds",
        "300",
        "--disaggregation-ack-timeout-seconds",
        "300",
    ]
    suite.REPORT["profiling_only_timeout_s"] = {"pull": 300, "ack": 300}
    suite.REPORT["timing_scope"] = "Profile traffic/export latencies are not benchmark results"
    suite.save()
    try:
        for name, overlap in [("B1", False), ("PD", True)]:
            port = suite.driver.boot("profile-" + name, overlap, overlap, True)
            suite.profile(name, port)
            suite.driver.stop()
    finally:
        suite.driver.COMMON[:] = original
    suite.REPORT["status"] = "passed"
    suite.save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("profile suite interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        suite.REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        suite.save()
        raise
    finally:
        suite.driver.stop()
        print("PROFILE_SUMMARY=" + json.dumps(suite.REPORT), flush=True)
