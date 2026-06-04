"""Perf case runner: drives bench_serving against a given server URL."""

import json
import os
from pathlib import Path

from multi_host_suite import PerfCase
from profile_loader import LaunchProfile


def run_perf_case(case: PerfCase, profile: LaunchProfile) -> None:
    from sgl_jax.bench_serving import run_benchmark
    from sgl_jax.test.test_utils import get_benchmark_args

    base_url = f"http://127.0.0.1:{profile.port}"
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name="random",
        tokenizer=profile.model_path,
        num_prompts=case.num_prompts,
        random_input_len=case.input_len,
        random_output_len=case.output_len,
        max_concurrency=case.max_concurrency,
        random_range_ratio=1.0,
        request_rate=case.request_rate,
        seed=case.seed,
        warmup_requests=0,
        backend="sgl-jax",
    )
    args.output_file = "/dev/null"
    args.flush_cache = case.flush_cache

    print(
        "[multi-host-suite] Running perf case "
        f"name={case.name}, num_prompts={case.num_prompts}, "
        f"concurrency={case.max_concurrency}, input_len={case.input_len}, "
        f"output_len={case.output_len}, request_rate={case.request_rate}, "
        f"seed={case.seed}, flush_cache={case.flush_cache}",
        flush=True,
    )
    metrics = run_benchmark(args)

    summary = {
        "type": "perf",
        "case": case.name,
        "profile": profile.name,
        "target": profile.target,
        **metrics,
    }
    # default=float handles numpy scalars returned by bench_serving.
    summary_json = json.dumps(summary, indent=2, sort_keys=True, default=float)
    print("[multi-host-suite] Perf summary:", flush=True)
    print(f"[multi-host-suite] {summary_json}", flush=True)

    results_dir = os.environ.get("RESULTS_DIR")
    if results_dir:
        out_path = Path(results_dir) / f"{case.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary_json)

    if metrics.get("completed") != case.num_prompts:
        raise RuntimeError(f"Expected completed={case.num_prompts}, got {metrics.get('completed')}")
