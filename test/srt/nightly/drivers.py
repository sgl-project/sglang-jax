"""Case drivers: build args + invoke run_eval / run_benchmark for one case.

``run_eval_for_case`` (accuracy) is shared by the single- and multi-host runners.
``run_benchmark_for_case`` (perf) is single-host only — the multi-host perf runner
calls ``run_benchmark`` inline. Each host runner does its own logging / gating and
feeds the returned metrics to ``results.py`` (``build_*_result`` + ``write_*``).
"""

import os
import sys

_NIGHTLY_DIR = os.path.dirname(os.path.abspath(__file__))
if _NIGHTLY_DIR not in sys.path:
    sys.path.insert(0, _NIGHTLY_DIR)

from cases import AccuracyCase, PerfCase  # noqa: E402


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
        dataset_name=case.workload,
        device="tpu",
        tokenizer=tokenizer,
        num_prompts=case.num_prompts,
        random_input_len=case.input_len,
        random_output_len=case.output_len,
        max_concurrency=case.max_concurrency,
        random_range_ratio=1.0,
        request_rate=case.request_rate,
        seed=case.seed,
        warmup_requests=case.warmup_requests,
        gsp_num_groups=case.gsp_num_groups,
        gsp_prompts_per_group=case.gsp_prompts_per_group,
        gsp_system_prompt_len=case.gsp_system_prompt_len,
        gsp_question_len=case.gsp_question_len,
        gsp_output_len=case.output_len,
        gsp_range_ratio=case.gsp_range_ratio,
    )
    args.output_file = "/dev/null"
    args.flush_cache = case.flush_cache
    args.profile = True if profile else None
    if profile and profile_num_steps is not None:
        args.profile_num_steps = profile_num_steps

    return run_benchmark(args)
