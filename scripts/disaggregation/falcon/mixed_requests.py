"""Mixed prompts detect request/KV cross-contamination hidden by identical inputs."""

import concurrent.futures
import json
import os
import signal
import traceback
from pathlib import Path

import lifecycle_checks as lifecycle
import requests
import single_pod_pd as driver
from transformers import AutoTokenizer

REPORT = {"status": "running", "groups": []}


def save():
    (driver.OUT / "mixed-summary.json").write_text(json.dumps(REPORT, indent=2))


def generate(port, ids):
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 64,
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert len(result["output_ids"]) == 64, result
    return result["output_ids"]


def main():
    driver.OUT.mkdir(parents=True, exist_ok=True)
    prerequisite = Path(os.environ["PD_REQUIRE_SUMMARY"])
    assert json.loads(prerequisite.read_text())["status"] == "passed_implemented_correctness_checks"
    tokenizer = AutoTokenizer.from_pretrained(driver.MODEL)
    inputs = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (f"context{i} " * (2048 if i < 4 else 8192))
                    + f"\nIgnore the context. Reply with exactly the unique identifier TEST{i}XYZ.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for i in range(8)
    ]
    assert all(len(ids) < 32700 for ids in inputs)
    port = driver.boot("mixed-reference", standalone=True)
    expected = [generate(port, ids) for ids in inputs]
    assert len({tuple(tokens) for tokens in expected}) == len(
        inputs
    ), "reference outputs are not distinct"
    REPORT["input_lengths"] = [len(ids) for ids in inputs]
    REPORT["reference_tokens"] = expected
    save()
    for group, overlap in [("B1", False), ("PD", True)]:
        port = driver.boot("mixed-" + group, overlap, overlap, True)
        baseline = lifecycle.idle()
        for repeat in range(2):
            order = [(i * 3 + repeat) % 8 for i in range(16)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(generate, port, inputs[i]) for i in order]
                outputs = [future.result(timeout=180) for future in futures]
            assert outputs == [expected[i] for i in order], (group, repeat)
            after = lifecycle.idle(baseline)
            REPORT["groups"].append(
                {
                    "group": group,
                    "repeat": repeat,
                    "order": order,
                    "status": "passed",
                    "after": after,
                }
            )
            save()
    REPORT["status"] = "passed"
    save()


if __name__ == "__main__":

    def interrupted(*_):
        raise TimeoutError("mixed checks interrupted")

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
            "MIXED_SUMMARY="
            + json.dumps(
                {
                    "status": REPORT["status"],
                    "groups": len(REPORT["groups"]),
                    "error": REPORT.get("error"),
                }
            ),
            flush=True,
        )
