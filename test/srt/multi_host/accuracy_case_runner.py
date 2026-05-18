"""Accuracy case runner: drives run_eval against a given server URL."""

import os
import sys
from types import SimpleNamespace

_TEST_SRT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _TEST_SRT not in sys.path:
    sys.path.insert(0, _TEST_SRT)

from multi_host_suite import AccuracyCase


def run_accuracy_case(case: AccuracyCase, port: int) -> None:
    from run_eval import run_eval

    gen = case.generation_config or {}
    args = SimpleNamespace(
        base_url=f"http://127.0.0.1:{port}",
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
    run_eval(args)
    print(
        f"[multi-host-suite] Accuracy case {case.name} finished "
        "(warn-only mode, accuracy not gated)",
        flush=True,
    )
