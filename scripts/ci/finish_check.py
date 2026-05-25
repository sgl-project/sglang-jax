"""Finish-gate logic for the CI pipeline.

Reads job results and coordinator flags from environment variables,
validates that all required jobs succeeded, and exits non-zero on failure.
"""

import os
import sys

MANDATORY_JOBS = {
    "run_main_test": [
        ("unit-test-1-tpu", "R_UNIT_1"),
        ("unit-test-4-tpu", "R_UNIT_4"),
        ("unit-test-cpu", "R_CPU"),
        ("e2e-test-1-tpu", "R_E2E_1"),
        ("e2e-test-4-tpu", "R_E2E_4"),
        ("accuracy-test-1-tpu", "R_ACC_1"),
        ("accuracy-test-4-tpu", "R_ACC_4"),
        ("performance-test-1-tpu", "R_PERF_1"),
        ("performance-test-4-tpu", "R_PERF_4"),
    ],
}

OPTIONAL_JOBS = [
    ("multi-chip-extra-test-4-tpu", "R_MULTI_CHIP", ["requires_4tpu", "run_full"]),
    ("accuracy-extra-test-1-tpu", "R_ACC_EXTRA", ["run_accuracy_extra", "run_full"]),
    ("perf-extra-test-1-tpu", "R_PERF_EXTRA", ["run_perf", "run_full"]),
    ("perf-trace-test-1-tpu", "R_PERF_TRACE", ["run_perf_trace", "run_full"]),
]

PALLAS_JOBS = [
    ("pallas-kernel-benchmark", "R_PALLAS"),
]


def check_jobs(env=None):
    """Returns list of (job_name, result) for each failed job."""
    if env is None:
        env = os.environ

    def get(name):
        return env.get(name, "")

    failures = []

    coordinator = get("R_COORDINATOR")
    if coordinator != "success":
        failures.append(("coordinator", coordinator))
        return failures

    flags = {
        "run_main_test": get("RUN_MAIN_TEST"),
        "run_pallas_bench": get("RUN_PALLAS_BENCH"),
        "requires_4tpu": get("REQUIRES_4TPU"),
        "run_full": get("RUN_FULL"),
        "run_accuracy_extra": get("RUN_ACCURACY_EXTRA"),
        "run_perf": get("RUN_PERF"),
        "run_perf_trace": get("RUN_PERF_TRACE"),
    }

    for flag_name, jobs in MANDATORY_JOBS.items():
        if flags.get(flag_name) == "true":
            for job_name, env_key in jobs:
                result = get(env_key)
                if result != "success":
                    failures.append((job_name, result))

    if flags.get("run_main_test") == "true":
        for job_name, env_key, trigger_flags in OPTIONAL_JOBS:
            if any(flags.get(f) == "true" for f in trigger_flags):
                result = get(env_key)
                if result != "success":
                    failures.append((job_name, result))

    if flags.get("run_pallas_bench") == "true":
        for job_name, env_key in PALLAS_JOBS:
            result = get(env_key)
            if result != "success":
                failures.append((job_name, result))

    return failures


def print_summary():
    def get(name):
        return os.environ.get(name, "")

    print("=== Job Results ===")
    print(f"coordinator: {get('R_COORDINATOR')}")
    print("--- Stage 1 ---")
    print(f"unit-test-1-tpu: {get('R_UNIT_1')}")
    print(f"unit-test-cpu: {get('R_CPU')}")
    print("--- Stage 2 ---")
    print(f"unit-test-4-tpu: {get('R_UNIT_4')}")
    print(f"e2e-test-1-tpu: {get('R_E2E_1')}")
    print(f"e2e-test-4-tpu: {get('R_E2E_4')}")
    print(f"accuracy-test-1-tpu: {get('R_ACC_1')}")
    print(f"accuracy-test-4-tpu: {get('R_ACC_4')}")
    print(f"multi-chip-extra-test-4-tpu: {get('R_MULTI_CHIP')}")
    print(f"accuracy-extra-test-1-tpu: {get('R_ACC_EXTRA')}")
    print("--- Stage 3 ---")
    print(f"performance-test-1-tpu: {get('R_PERF_1')}")
    print(f"performance-test-4-tpu: {get('R_PERF_4')}")
    print(f"perf-extra-test-1-tpu: {get('R_PERF_EXTRA')}")
    print(f"perf-trace-test-1-tpu: {get('R_PERF_TRACE')}")
    print("--- Independent ---")
    print(f"pallas-kernel-benchmark: {get('R_PALLAS')}")
    print()


def main():
    print_summary()

    failures = check_jobs()
    for job_name, result in failures:
        print(f"::error::{job_name} did not succeed: {result}")

    if failures:
        sys.exit(1)

    print("All required test jobs passed")


if __name__ == "__main__":
    main()
