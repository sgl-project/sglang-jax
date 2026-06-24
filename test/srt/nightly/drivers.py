"""Case drivers: build args + invoke run_eval / run_benchmark for one case.

``run_eval_for_case`` (accuracy) is shared by the single- and multi-host runners.
``run_benchmark_for_case`` (perf) is single-host only — the multi-host perf runner
calls ``run_benchmark`` inline. ``run_bench_for_case`` (BenchCase) shells out to a
standalone ``benchmark/hicache`` bench and is shared by both host runners. Each host
runner does its own logging / gating and feeds the returned metrics to ``results.py``
(``build_*_result`` + ``write_*``).
"""

import os
import subprocess
import sys

_NIGHTLY_DIR = os.path.dirname(os.path.abspath(__file__))
if _NIGHTLY_DIR not in sys.path:
    sys.path.insert(0, _NIGHTLY_DIR)

# Repo root = .../test/srt/nightly -> .../test/srt -> .../test -> repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_NIGHTLY_DIR)))

from cases import AccuracyCase, BenchCase, PerfCase  # noqa: E402


def run_eval_for_case(case: AccuracyCase, base_url: str):
    """Drive ``run_eval`` for one case against a live server at ``base_url``.

    Returns ``(metrics, started_at, finished_at)``.
    """
    import time
    from types import SimpleNamespace

    from run_eval import run_eval

    gen = case.generation_config or {}
    # Forward the full sampler config. run_eval routes the SGLang-only params
    # (SGLANG_EXTRA_SAMPLING_PARAMS) into extra_body; cherry-picking a subset
    # here would let a case set e.g. top_k in generation_config, record it in
    # the summary, yet silently drop it before it reaches the sampler.
    args = SimpleNamespace(
        base_url=base_url,
        host=None,
        port=None,
        model=case.model_id,
        eval_name=case.dataset,
        num_examples=case.limit,
        num_threads=case.eval_batch_size,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 2048),
        top_p=gen.get("top_p"),
        top_k=gen.get("top_k"),
        min_p=gen.get("min_p"),
        presence_penalty=gen.get("presence_penalty"),
        repetition_penalty=gen.get("repetition_penalty"),
        frequency_penalty=gen.get("frequency_penalty"),
        seed=gen.get("seed"),
        chat_template_kwargs=gen.get("chat_template_kwargs"),
    )
    started_at = time.time()
    metrics = run_eval(args)
    finished_at = time.time()
    return metrics, started_at, finished_at


def run_benchmark_for_case(
    case: PerfCase,
    base_url: str,
    tokenizer: str,
    *,
    profile: bool = False,
    profile_num_steps: int | None = None,
):
    """Drive ``run_benchmark`` for one perf sweep point against a live server.

    ``tokenizer`` is the served model path (PerfCase carries no model_id — the
    model comes from the launch profile). When ``profile`` is set, bench_serving
    drives /start_profile + /stop_profile (the server writes the trace to
    $SGLANG_JAX_PROFILER_DIR); ``profile=False`` keeps args.profile=None,
    bench_serving's "off" sentinel. Returns the ``metrics`` dict.
    """
    from sgl_jax.bench_serving import run_benchmark
    from sgl_jax.test.test_utils import get_benchmark_args

    args = get_benchmark_args(
        base_url=base_url,
        dataset_name="random",
        device="tpu",
        tokenizer=tokenizer,
        num_prompts=case.num_prompts,
        random_input_len=case.input_len,
        random_output_len=case.output_len,
        max_concurrency=case.max_concurrency,
        random_range_ratio=1.0,
        request_rate=case.request_rate,
        seed=case.seed,
        warmup_requests=0,
    )
    args.output_file = "/dev/null"
    args.flush_cache = case.flush_cache
    args.profile = True if profile else None
    if profile and profile_num_steps is not None:
        args.profile_num_steps = profile_num_steps

    return run_benchmark(args)


def run_bench_for_case(
    case: BenchCase, base_url: str | None = None
) -> tuple[dict, tuple[str, str] | None]:
    """Run one BenchCase as a subprocess; gate on its exit code.

    Builds ``python <repo>/<script> [--server-url ...] [--compare ...] <argv>
    [--output-json ...]`` and runs it from the repo root with inherited stdio (so
    a long sweep streams live to the CI log). The bench owns its own gate
    (``--strict`` / a knee assert); this maps ``returncode != 0`` to a tagged
    ``threshold`` failure. A dashboard record is written to ``$RESULTS_DIR``.

    Returns ``(result, fail)`` where ``fail`` is ``None`` on pass or
    ``("threshold", msg)`` on a nonzero exit, matching the other case drivers.
    """
    from results import write_result

    results_dir = os.environ.get("RESULTS_DIR")
    # The bench's own --output-json write (json.dump) does not create parents, so
    # ensure RESULTS_DIR exists before the subprocess runs (write_result mkdirs
    # too, but that is after the bench has already tried to write).
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(_REPO_ROOT, case.script)]
    if case.server == "runner":
        if not base_url:
            raise ValueError(f"{case.name}: server='runner' needs a base_url")
        cmd += ["--server-url", base_url]
    if case.compare_inputs:
        if not results_dir:
            raise RuntimeError(f"{case.name}: compare_inputs needs RESULTS_DIR to resolve inputs")
        cmd += ["--compare", *[os.path.join(results_dir, n) for n in case.compare_inputs]]
    cmd += list(case.argv)
    if case.output_json and results_dir:
        cmd += ["--output-json", os.path.join(results_dir, case.output_json)]

    print(f"[bench-runner] {case.name}: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode

    result = {
        "type": "bench",
        "case": case.name,
        "script": case.script,
        "server": case.server,
        "argv": list(case.argv),
        "returncode": rc,
        "passed": rc == 0,
    }
    write_result(result, case.name)

    fail = None if rc == 0 else ("threshold", f"{case.name}: bench exited {rc} (see logs)")
    print(f"[bench-runner] {case.name}: {'PASS' if rc == 0 else 'FAIL'} (exit {rc})", flush=True)
    return result, fail
