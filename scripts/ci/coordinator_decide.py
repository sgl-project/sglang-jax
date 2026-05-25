"""Coordinator decision logic for the CI pipeline.

Reads GitHub context from environment variables (set by the YAML env: block),
computes which jobs should run, and writes results to $GITHUB_OUTPUT.
"""

import json
import os
import sys


def detect_draft(event_name, pr_draft):
    if event_name == "pull_request":
        return pr_draft if pr_draft in ("true", "false") else "false"
    return "false"


def detect_labels(event_name, pr_labels_json):
    label_flags = {
        "run_full": False,
        "requires_4tpu": False,
        "run_perf": False,
        "run_perf_trace": False,
        "run_accuracy_extra": False,
    }

    if event_name != "pull_request":
        return label_flags

    try:
        labels = json.loads(pr_labels_json) if pr_labels_json else []
    except (json.JSONDecodeError, TypeError):
        labels = []
    if not isinstance(labels, list):
        labels = []

    label_to_flag = {
        "test:full": "run_full",
        "test:multi-chip": "requires_4tpu",
        "test:perf": "run_perf",
        "test:perf-trace": "run_perf_trace",
        "test:accuracy-extra": "run_accuracy_extra",
    }
    for label_name, flag_name in label_to_flag.items():
        if label_name in labels:
            label_flags[flag_name] = True

    return label_flags


def summarize(main_package, pallas_kernel):
    run_main_test = main_package == "true"
    run_pallas_bench = pallas_kernel == "true"
    return {
        "run_main_test": run_main_test,
        "run_pallas_bench": run_pallas_bench,
    }


def _bool_str(value):
    return "true" if value else "false"


def write_outputs(outputs):
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            for name, value in outputs.items():
                f.write(f"{name}={value}\n")
    else:
        print("(GITHUB_OUTPUT not set — printing only)", file=sys.stderr)


def main():
    event_name = os.environ.get("EVENT_NAME", "")
    pr_draft = os.environ.get("PR_DRAFT", "")
    pr_labels = os.environ.get("PR_LABELS", "[]")
    main_package = os.environ.get("MAIN_PACKAGE", "false")
    pallas_kernel = os.environ.get("PALLAS_KERNEL", "false")

    is_draft = detect_draft(event_name, pr_draft)
    label_flags = detect_labels(event_name, pr_labels)
    decisions = summarize(main_package, pallas_kernel)

    outputs = {}
    for name, value in label_flags.items():
        outputs[name] = _bool_str(value)
    for name, value in decisions.items():
        outputs[name] = _bool_str(value)

    write_outputs(outputs)

    print("=== Coordinator Decisions ===")
    print(f"main_package: {main_package}")
    print(f"pallas_kernel: {pallas_kernel}")
    print(f"is_draft: {is_draft}")
    for name, value in outputs.items():
        print(f"{name}: {value}")

    if is_draft == "true":
        print("::error::PR is a draft — mark as ready for review to run CI")
        sys.exit(1)


if __name__ == "__main__":
    main()
