"""Bounded long logprob SSE regression with efficient client reads."""

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import single_pod_pd as driver
from transformers import AutoTokenizer

REPORT = {"status": "running", "checks": []}


def save():
    (driver.OUT / "stream-summary.json").write_text(json.dumps(REPORT, indent=2))


def main():
    driver.OUT.mkdir(parents=True, exist_ok=True)
    if "--client" in sys.argv:
        driver.TOKENIZER = AutoTokenizer.from_pretrained(driver.MODEL)
        result = driver.generate(30020, 4096, 1024, stream=True)
        (driver.OUT / "long-stream-result.json").write_text(json.dumps(result))
        return
    prior = Path(os.environ["PD_REQUIRE_SUMMARY"])
    assert json.loads(prior.read_text())["status"] == "passed_implemented_correctness_checks"
    expected = next(
        json.loads(line)["token_ids"]
        for line in (prior.parent / "B0-requests.jsonl").read_text().splitlines()
        if json.loads(line)["input"] == 4096 and json.loads(line)["output"] == 1024
    )
    save()
    driver.boot("long-stream", True, True, True)
    started = time.monotonic()
    subprocess.run([sys.executable, __file__, "--client"], check=True, timeout=90)
    result = json.loads((driver.OUT / "long-stream-result.json").read_text())
    assert result["token_ids"] == expected
    REPORT["checks"].append(
        {
            "case": "4K-1024-stream-logprobs",
            "status": "passed",
            "elapsed_s": time.monotonic() - started,
        }
    )
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("stream regression interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    if "--client" in sys.argv:
        main()
    else:
        try:
            main()
        except BaseException as exc:
            REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            save()
            raise
        finally:
            driver.stop()
            print("STREAM_SUMMARY=" + json.dumps(REPORT), flush=True)
