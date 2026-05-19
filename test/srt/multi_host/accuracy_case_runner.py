"""Accuracy case runner: drives run_eval against a given server URL."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_TEST_SRT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _TEST_SRT not in sys.path:
    sys.path.insert(0, _TEST_SRT)

from multi_host_suite import AccuracyCase
from profile_loader import LaunchProfile


def run_accuracy_case(case: AccuracyCase, profile: LaunchProfile) -> None:
    from run_eval import run_eval

    gen = case.generation_config or {}
    args = SimpleNamespace(
        base_url=f"http://127.0.0.1:{profile.port}",
        host=None,
        port=None,
        model=case.model_id,
        eval_name=case.dataset,
        num_examples=case.limit,
        num_threads=case.eval_batch_size,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 2048),
    )

    print(
        f"[multi-host-suite] Running accuracy case "
        f"name={case.name}, dataset={case.dataset}, "
        f"num_threads={args.num_threads}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}, "
        f"limit={case.limit}",
        flush=True,
    )
    metrics = run_eval(args)

    results_dir = os.environ.get("RESULTS_DIR")
    if results_dir:
        summary = {
            "type": "accuracy",
            "case": case.name,
            "profile": profile.name,
            "target": profile.target,
            "dataset": case.dataset,
            "model_id": case.model_id,
            **(metrics if isinstance(metrics, dict) else {}),
        }
        out_path = Path(results_dir) / f"{case.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))

    print(
        f"[multi-host-suite] Accuracy case {case.name} finished "
        "(warn-only mode, accuracy not gated)",
        flush=True,
    )
