"""Isolate the 4K/C32 steady TTFT tradeoff with contemporaneous role ablations."""

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
    assert (
        json.loads(Path(os.environ["PD_REQUIRE_STEADY_SUMMARY"]).read_text())["status"] == "passed"
    )
    suite.driver.TOKENIZER = AutoTokenizer.from_pretrained(suite.driver.MODEL)
    suite.REPORT["purpose"] = "4K/256 C32 TTFT: same-window B1, D-only, P-only and PD controls"
    suite.save()
    for name, po, do in [
        ("B1", False, False),
        ("D-only", False, True),
        ("P-only", True, False),
        ("PD", True, True),
    ]:
        port = suite.driver.boot("ablation-" + name, po, do, True)
        suite.measure("ablation-" + name + "-4096-c32", port, [(4096, 256)], 32)
        suite.driver.stop()
    suite.REPORT["status"] = "passed"
    suite.save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("steady ablation interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        suite.REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        suite.save()
        raise
    finally:
        suite.driver.stop()
        print(
            "STEADY_ABLATION_SUMMARY="
            + json.dumps(
                {
                    "status": suite.REPORT["status"],
                    "measurements": len(suite.REPORT["measurements"]),
                    "error": suite.REPORT.get("error"),
                }
            ),
            flush=True,
        )
