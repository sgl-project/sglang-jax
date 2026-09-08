"""Isolate the 4K/C32 steady TTFT tradeoff with contemporaneous role ablations."""

import asyncio
import json
import os
import signal
import traceback
from pathlib import Path

import lifecycle_checks
import performance_suite
import steady_rate_client
import steady_suite as suite
from transformers import AutoTokenizer


def matched_rate(tag, port, length, rate):
    baseline = lifecycle_checks.idle()
    with (suite.driver.OUT / (tag + "-requests.jsonl")).open("w") as handle:

        async def send(session, sequence):
            return await performance_suite.request(session, port, length, 256, sequence % 4)

        def record(result):
            handle.write(json.dumps(result) + "\n")
            handle.flush()

        result = asyncio.run(steady_rate_client.run(send, rate, 60, 180, on_complete=record))
    result.update(
        tag=tag, input=length, output=256, baseline=baseline, after=lifecycle_checks.idle(baseline)
    )
    assert abs(result["actual_cohort_rps"] - rate) <= 1 / 180, result
    assert result["arrival_lateness_max_s"] < 0.1, result
    suite.REPORT.setdefault("matched_rate", []).append(result)
    suite.save()
    print(
        "MATCHED_RATE="
        + json.dumps({k: v for k, v in result.items() if k not in {"baseline", "after"}}),
        flush=True,
    )


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
    for name, overlap in [("B1", False), ("PD", True)]:
        port = suite.driver.boot("matched-rate-" + name, overlap, overlap, True)
        for length, rate in [(4096, 5.0), (16384, 1.2)]:
            matched_rate(f"rate-{name}-{length}", port, length, rate)
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
