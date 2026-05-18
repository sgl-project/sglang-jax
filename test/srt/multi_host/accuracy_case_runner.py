"""Accuracy case runner: drives evalscope against a given server URL."""

import json
import subprocess

from multi_host_suite import AccuracyCase


def run_accuracy_case(case: AccuracyCase, port: int) -> None:
    api_url = f"http://127.0.0.1:{port}/v1"
    cmd = [
        "evalscope",
        "eval",
        "--model",
        case.model_id,
        "--api-url",
        api_url,
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        case.dataset,
        "--eval-batch-size",
        str(case.eval_batch_size),
    ]
    if case.generation_config:
        cmd.extend(["--generation-config", json.dumps(case.generation_config)])
    if case.limit is not None:
        cmd.extend(["--limit", str(case.limit)])
    if case.timeout is not None:
        cmd.extend(["--timeout", str(case.timeout)])

    print(
        "[multi-host-suite] Running accuracy case "
        f"name={case.name}, dataset={case.dataset}, "
        f"eval_batch_size={case.eval_batch_size}, "
        f"generation_config={case.generation_config}, limit={case.limit}, "
        f"timeout={case.timeout}",
        flush=True,
    )
    print(f"[multi-host-suite] Command: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"evalscope exited with code {completed.returncode} for case={case.name}"
        )
    print(
        f"[multi-host-suite] Accuracy case {case.name} completed "
        "(warn-only mode, accuracy not gated)",
        flush=True,
    )
