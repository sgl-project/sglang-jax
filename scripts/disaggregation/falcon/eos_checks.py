"""Natural EOS must match non-PD output and reclaim the original KV capacity."""

import json
import os
import signal
import traceback
from pathlib import Path

import lifecycle_checks as lifecycle
import requests
import single_pod_pd as driver
from transformers import AutoTokenizer

REPORT = {"status": "running", "checks": []}


def save():
    (driver.OUT / "eos-summary.json").write_text(json.dumps(REPORT, indent=2))


def generate(port, ids):
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 128, "ignore_eos": False},
            "return_logprob": True,
            "logprob_start_len": -1,
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert result["meta_info"]["finish_reason"]["type"] == "stop", result
    assert result["meta_info"]["completion_tokens"] < 128
    return result


def main():
    driver.OUT.mkdir(parents=True, exist_ok=True)
    prerequisite = Path(os.environ["PD_REQUIRE_SUMMARY"])
    assert json.loads(prerequisite.read_text())["status"] == "passed_implemented_correctness_checks"
    tokenizer = AutoTokenizer.from_pretrained(driver.MODEL)
    prompts = [
        "Reply with exactly the word hello.",
        "Read this context: " + "blue " * 4096 + "\nReply with exactly the word hello.",
    ]
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    reference_port = driver.boot("eos-reference", standalone=True)
    reference = [generate(reference_port, ids) for ids in inputs]
    driver.stop()
    port = driver.boot("eos-PD", True, True, True)
    baseline = lifecycle.idle()
    for ids, expected in zip(inputs, reference):
        actual = generate(port, ids)
        assert actual["output_ids"] == expected["output_ids"]
        assert actual["meta_info"]["finish_reason"] == expected["meta_info"]["finish_reason"]
        after = lifecycle.idle(baseline)
        REPORT["checks"].append(
            {
                "input_ids": ids,
                "reference": expected,
                "actual": actual,
                "after": after,
                "status": "passed",
            }
        )
        save()
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("EOS checks interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        driver.stop()
        print(
            "EOS_SUMMARY="
            + json.dumps(
                {
                    "status": REPORT["status"],
                    "checks": len(REPORT["checks"]),
                    "error": REPORT.get("error"),
                }
            ),
            flush=True,
        )
