"""Standalone A/B serving benchmark for the unified radix cache.

Compares cache configurations on a single TPU pod:
  - ``no-cache``  (``--disable-radix-cache``): no prefix reuse, the floor.
  - ``radix``     (default RadixCache): the current production baseline.
  - ``unified``   (``--enable-unified-radix-tree``): the UnifiedRadixCache.
  - ``unified-recurrent`` (``--enable-unified-radix-tree
    --enable-recurrent-extra-buffer``): the hybrid-recurrent cache.

For each (config x workload) it launches one server, runs ``bench_serving``
``--repeats`` times (cold each rep via ``/flush_cache``), drops the first
``--drop-first`` warmup reps, then reports pooled TTFT percentiles, mean+/-std
throughput, and mean cache-hit-rate, plus a few soft acceptance checks.

This is throughput/TTFT evidence for the unified radix cache. It is run
MANUALLY on a v6e-4 host (e.g. sky-yh-v6e4); it is NOT registered in CI.
Servers are launched sequentially -- one TPU pod, never two servers at once.

Usage (on the TPU host, from the repo root)::

    python benchmark/hicache/bench_unified_radix_ab.py \
        --model Qwen/Qwen3-8B --tp-size 4 --page-size 128 --port 20000 \
        --configs no-cache radix unified --workloads random gsp \
        --output-json /tmp/radix_ab.json

MoE usage (Qwen3-MoE under tp/ep/dp)::

    python benchmark/hicache/bench_unified_radix_ab.py \
        --model /models/Qwen3-30B-A3B --tp-size 4 --ep-size 4 --moe-backend epmoe \
        --page-size 256 --configs radix unified --output-json /tmp/moe_ab.json

External-server / multi-host usage:
    ``popen_launch_server`` is single-host and cannot start an ``nnodes>1``
    server. For multi-host KDA the operator launches the 4-pod server manually,
    then runs this harness client-only against rank0 with ``--server-url`` and a
    single ``--configs`` entry (a result label). A/B = run twice (e.g. no-cache
    and unified-recurrent) and merge the two ``--output-json`` files offline::

        python benchmark/hicache/bench_unified_radix_ab.py \
            --server-url http://<rank0>:30000 --configs unified-recurrent \
            --disable-overlap-schedule --workloads gsp --output-json /tmp/rec.json

Multi-chip note:
    --tp-size is the TOTAL chip count; --dp-size sub-partitions it (attention runs
    at tp_size // dp_size). At --tp-size > 1, --precompile-bs-paddings values must be
    divisible by --tp-size (validated at startup): a decode batch not divisible by the
    device count hits a pre-existing tp>1 sampler lax.cond sharding bug. Widen the set
    to cover higher --parallel.

GSP disk-cache warning:
    The generated-shared-prefix dataset is pickled under ``~/.cache/sglang``
    keyed by the GSP shape, ``--seed``, and tokenizer *class* (not the model), so
    switching models with the same tokenizer class reuses stale prompts -- use a
    fresh ``--seed`` or cache dir.
"""

import argparse
import json
import time

import numpy as np

CONFIG_ARGS = {
    "no-cache": ["--disable-radix-cache"],
    "radix": [],
    "unified": ["--enable-unified-radix-tree"],
    "unified-recurrent": [
        "--enable-unified-radix-tree",
        "--enable-recurrent-extra-buffer",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="A/B serving benchmark: no-cache / RadixCache / UnifiedRadixCache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--ep-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--moe-backend", default=None, help="e.g. epmoe; emitted only when set")
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--port", type=int, default=20000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--drop-first", type=int, default=2)
    p.add_argument(
        "--configs",
        nargs="+",
        choices=["no-cache", "radix", "unified", "unified-recurrent"],
        default=["no-cache", "radix", "unified"],
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        choices=["random", "gsp", "mooncake"],
        default=["random", "gsp"],
    )
    p.add_argument(
        "--parallel",
        nargs="+",
        type=int,
        default=[64],
        help="Client-side in-flight request load(s) — the same knob as the reuse "
        "sweep's --parallel (an asyncio.Semaphore), NOT a server capacity knob (the "
        "server cap is --max-running-requests). Pass several values to sweep "
        "concurrency on one launched server. With cache_aware DP routing, keep each "
        ">= dp_size so all ranks receive load.",
    )
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--random-input-len", type=int, default=1024)
    p.add_argument("--random-output-len", type=int, default=128)
    p.add_argument("--gsp-num-groups", type=int, default=16)
    p.add_argument("--gsp-prompts-per-group", type=int, default=8)
    p.add_argument("--gsp-system-prompt-len", type=int, default=2048)
    p.add_argument("--gsp-question-len", type=int, default=128)
    p.add_argument("--gsp-output-len", type=int, default=128)
    p.add_argument(
        "--mooncake-workload",
        default="conversation",
        help="Mooncake trace to replay for the 'mooncake' workload (conversation "
        "has the highest prefix reuse).",
    )
    p.add_argument(
        "--mooncake-trace-path",
        default=None,
        help="Local path to the mooncake trace .jsonl. When unset, bench_serving "
        "downloads it to /tmp; pre-stage it (e.g. on a model bucket) for offline / "
        "deterministic runs.",
    )
    p.add_argument(
        "--mooncake-slowdown-factor",
        type=float,
        default=1.0,
        help="Scales trace inter-arrival times (<1 compresses the replay to a "
        "shorter wall-clock; 1.0 = real time).",
    )
    p.add_argument(
        "--mooncake-num-rounds",
        type=int,
        default=1,
        help="Conversation rounds replayed per trace session. >1 re-sends the "
        "growing history each round, so round N shares a prefix with round N-1 — "
        "this is what exercises recurrent prefix reuse (rounds=1 has no reuse).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--attention-backend", default="fa")
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=256,
        help="Server running-batch cap. Lower it for large models on few chips "
        "(their KV budget can't hold 256 in flight).",
    )
    p.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Server --context-length; emitted only when set. Bound it for large "
        "models — their full context can force max_running_requests to 0 on few chips.",
    )
    p.add_argument(
        "--max-recurrent-state-size",
        type=int,
        default=None,
        help="Server --max-recurrent-state-size, emitted only for the "
        "unified-recurrent config. Cap it for large models: the default ratio-based "
        "sizing reserves ~0.9 of HBM and leaves a negative KV budget.",
    )
    p.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=512,
        help=(
            "Prefill chunk size. Emitted to the server command ONLY for the "
            "unified-recurrent config (it also sets the recurrent track interval); "
            "the dense configs use the server default to stay byte-identical."
        ),
    )
    p.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help=(
            "Pass --disable-overlap-schedule to the server (correctness-critical "
            "for KDA/Kimi-Linear recurrent decode)."
        ),
    )
    p.add_argument(
        "--server-url",
        default=None,
        help=(
            "Run client-only against an already-launched server at this URL "
            "(e.g. a multi-host nnodes>1 server). When set, the harness does NOT "
            "launch or kill a server; exactly one --configs entry is required "
            "(it is only a result label)."
        ),
    )
    p.add_argument(
        "--precompile-bs-paddings",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64],
        help=(
            "Decode batch buckets the running batch pads up to. At --tp-size > 1 "
            "every value must be divisible by --tp-size (a non-divisible decode "
            "batch hits a pre-existing tp>1 sampler sharding bug); cover the "
            "concurrency range."
        ),
    )
    p.add_argument(
        "--precompile-token-paddings",
        nargs="+",
        type=int,
        default=[2048],
        help="Prefill token buckets to precompile (off-bucket shapes JIT on demand).",
    )
    p.add_argument("--output-json", default=None)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any soft-target check fails.",
    )
    p.add_argument(
        "--dataset-random-ids",
        action="store_true",
        help=(
            "Use dataset_name 'random-ids' instead of 'random' (synthetic token "
            "ids) when the host has no ShareGPT access."
        ),
    )
    p.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="RESULT_JSON",
        help=(
            "Offline A/B gate: merge the 'aggregates' from two or more "
            "--output-json files (each a single-config run) and run the soft-target "
            "gate across them, no serving. This lets runs that produce one config "
            "at a time form an A/B gate. Combine with --strict to fail on a missed "
            "target."
        ),
    )
    args = p.parse_args()
    if args.compare:
        # Offline merge of prior results: no server is launched, so the
        # server-shape validations below do not apply.
        return args
    assert args.page_size >= 128, f"--page-size must be >= 128, got {args.page_size}"
    if args.tp_size > 1:
        bad = [b for b in args.precompile_bs_paddings if b % args.tp_size != 0]
        assert not bad, (
            f"--precompile-bs-paddings {bad} not divisible by --tp-size "
            f"{args.tp_size} (hits the tp>1 sampler sharding bug)"
        )
    if args.server_url is not None:
        assert len(args.configs) == 1, (
            "--server-url runs client-only against one already-launched server, so "
            f"exactly one --configs entry is required (got {args.configs}). The config "
            "name is only a result label; run twice and merge the two JSON outputs for A/B."
        )
    return args


def _per_run_args(args, base_url, workload, parallel):
    """Build a per-run bench_serving SimpleNamespace via get_benchmark_args."""
    from sgl_jax.test.test_utils import get_benchmark_args

    if workload == "random":
        run_args = get_benchmark_args(
            base_url=base_url,
            dataset_name="random-ids" if args.dataset_random_ids else "random",
            device="tpu",
            tokenizer=args.model,
            num_prompts=args.num_prompts,
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            random_range_ratio=1.0,
            max_concurrency=parallel,
            seed=args.seed,
            warmup_requests=1,
        )
    elif workload == "gsp":
        num_prompts = args.gsp_num_groups * args.gsp_prompts_per_group
        run_args = get_benchmark_args(
            base_url=base_url,
            dataset_name="generated-shared-prefix",
            device="tpu",
            tokenizer=args.model,
            num_prompts=num_prompts,
            max_concurrency=parallel,
            seed=args.seed,
            warmup_requests=1,
            gsp_num_groups=args.gsp_num_groups,
            gsp_prompts_per_group=args.gsp_prompts_per_group,
            gsp_system_prompt_len=args.gsp_system_prompt_len,
            gsp_question_len=args.gsp_question_len,
            gsp_output_len=args.gsp_output_len,
            gsp_range_ratio=1.0,
        )
    elif workload == "mooncake":
        # Trace replay: bench_serving owns the rows via its timed generator, so
        # backend="sglang" selects that path (the same /generate request func as
        # sgl-jax). num_rounds>1 re-sends the growing history per session, which is
        # what exercises recurrent prefix reuse; the per-rep completeness check
        # below scales the expected count by num_rounds.
        run_args = get_benchmark_args(
            base_url=base_url,
            dataset_name="mooncake",
            device="tpu",
            tokenizer=args.model,
            num_prompts=args.num_prompts,
            max_concurrency=parallel,
            seed=args.seed,
            warmup_requests=1,
            backend="sglang",
            dataset_path=args.mooncake_trace_path or "",
            mooncake_workload=args.mooncake_workload,
            use_trace_timestamps=True,
            mooncake_slowdown_factor=args.mooncake_slowdown_factor,
            mooncake_num_rounds=args.mooncake_num_rounds,
        )
    else:
        raise ValueError(f"unknown workload {workload!r}")

    # run_benchmark always appends a per-run JSONL; metrics come from its return
    # value, so /dev/null avoids stray files without losing data.
    run_args.output_file = "/dev/null"
    run_args.flush_cache = True  # flush after warmup -> each rep starts cold
    return run_args


def run_config(args, config):
    """Run all workloads x repeats for ``config``.

    Launches one server for ``config`` unless ``--server-url`` is set, in which
    case it runs client-only against that already-launched server (no launch, no
    kill) -- the path for multi-host nnodes>1 servers ``popen_launch_server``
    cannot start.
    """
    from sgl_jax.bench_serving import run_benchmark
    from sgl_jax.srt.utils import kill_process_tree
    from sgl_jax.test.test_utils import popen_launch_server

    external = args.server_url is not None
    base_url = args.server_url if external else f"http://127.0.0.1:{args.port}"

    # config -> {workload -> {parallel -> [res dict per rep]}}
    results = {w: {p: [] for p in args.parallel} for w in args.workloads}

    def run_reps():
        for workload in args.workloads:
            for parallel in args.parallel:
                for rep in range(args.repeats):
                    print(
                        f"\n=== config={config} workload={workload} "
                        f"parallel={parallel} rep={rep + 1}/{args.repeats} ===",
                        flush=True,
                    )
                    run_args = _per_run_args(args, base_url, workload, parallel)
                    try:
                        res = run_benchmark(run_args)
                    except Exception as e:  # noqa: BLE001
                        # Don't let one failed rep (OOM, transient hiccup, or
                        # run_benchmark's NameError on zero completions) abort the sweep.
                        print(
                            f"!!! rep failed (config={config} workload={workload} "
                            f"parallel={parallel} rep={rep + 1}): {e!r}; skipping this rep",
                            flush=True,
                        )
                        continue
                    # A partially-failed rep would skew stats (failed requests carry
                    # ttft=0.0 and missing tokens); count it as failed instead.
                    # Mooncake replays num_rounds requests per session, so the
                    # expected completion is num_prompts * num_rounds (1 for the
                    # other workloads).
                    completed = res.get("completed", 0)
                    expected = run_args.num_prompts * run_args.mooncake_num_rounds
                    if completed != expected:
                        print(
                            f"!!! rep incomplete (config={config} workload={workload} "
                            f"parallel={parallel} rep={rep + 1}): "
                            f"completed={completed}/{expected}; skipping this rep",
                            flush=True,
                        )
                        continue
                    results[workload][parallel].append(res)

    if external:
        # Operator already launched the matching server (e.g. multi-host KDA);
        # just drive the client loop against it.
        run_reps()
        return results

    common = [
        "--trust-remote-code",
        "--skip-server-warmup",
        "--dtype",
        "bfloat16",
        "--random-seed",
        "3",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--page-size",
        str(args.page_size),
        "--tp-size",
        str(args.tp_size),
        "--attention-backend",
        args.attention_backend,
        "--download-dir",
        "/dev/shm/",
    ]
    if args.context_length is not None:
        common += ["--context-length", str(args.context_length)]
    common += ["--precompile-bs-paddings", *[str(b) for b in args.precompile_bs_paddings]]
    common += ["--precompile-token-paddings", *[str(t) for t in args.precompile_token_paddings]]
    # Emit parallelism/MoE flags only when set, so the default dense command stays
    # byte-identical and main-compatible. --tp-size is the TOTAL chip count;
    # --dp-size sub-partitions it (attention runs at tp_size // dp_size).
    if args.dp_size > 1:
        common += ["--dp-size", str(args.dp_size)]
    if args.ep_size > 1:
        common += ["--ep-size", str(args.ep_size)]
    if args.moe_backend:
        common += ["--moe-backend", args.moe_backend]
    # Recurrent flags. Emit --chunked-prefill-size only for the recurrent config
    # (it also sets the recurrent track interval); the dense configs keep the
    # server default (4096) so their launch command stays byte-identical.
    if config == "unified-recurrent":
        common += ["--chunked-prefill-size", str(args.chunked_prefill_size)]
        if args.max_recurrent_state_size is not None:
            common += ["--max-recurrent-state-size", str(args.max_recurrent_state_size)]
    if args.disable_overlap_schedule:
        common += ["--disable-overlap-schedule"]

    process = popen_launch_server(
        args.model,
        base_url=base_url,
        timeout=1800,
        other_args=common + CONFIG_ARGS[config],
        check_cache_miss=False,
        # A stray SGLANG_JAX_ENABLE_UNIFIED_RADIX_TREE=1 in the caller's env
        # would silently turn the baselines into unified (env can only force
        # the flag ON); pin it off -- "unified" gets the flag via CLI.
        env={"SGLANG_JAX_ENABLE_UNIFIED_RADIX_TREE": "0"},
    )
    try:
        run_reps()
    finally:
        kill_process_tree(process.pid)
        process.wait()
        time.sleep(10)  # let the TPU free before the next config

    return results


def _sample_std(x):
    """Sample std (ddof=1); 0.0 for a single kept rep (ddof=1 would be nan)."""
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def aggregate(args, raw):
    """Aggregate raw[config][workload][parallel] = [res, ...] into per-cell stats."""
    agg = {}
    for config, by_workload in raw.items():
        agg[config] = {}
        for workload, by_parallel in by_workload.items():
            agg[config][workload] = {}
            for parallel, reps in by_parallel.items():
                kept = reps[args.drop_first :]
                if not kept:
                    continue
                pooled_ttft = []
                for res in kept:
                    # bench_serving's per-request ttfts are unfiltered: failed
                    # requests carry the 0.0 default, which would fake-improve
                    # percentiles. Reps are already completeness-checked; this
                    # guards the per-request layer.
                    pooled_ttft.extend(t for t in (res.get("ttfts") or []) if t > 0)
                if pooled_ttft:
                    p50, p95, p99 = np.percentile(pooled_ttft, [50, 95, 99]) * 1000.0
                else:
                    p50 = p95 = p99 = float("nan")

                out_tps = np.array([res["output_throughput"] for res in kept], dtype=float)
                total_tps = np.array([res["total_throughput"] for res in kept], dtype=float)
                hit_rate = np.array([res.get("cache_hit_rate", 0.0) for res in kept], dtype=float)

                agg[config][workload][parallel] = {
                    "p50_ttft_ms": float(p50),
                    "p95_ttft_ms": float(p95),
                    "p99_ttft_ms": float(p99),
                    "out_tok_s_mean": float(out_tps.mean()),
                    "out_tok_s_std": _sample_std(out_tps),
                    "total_tok_s_mean": float(total_tps.mean()),
                    "total_tok_s_std": _sample_std(total_tps),
                    "hit_rate_mean": float(hit_rate.mean()),
                    "kept_reps": len(kept),
                }
    return agg


def print_table(args, agg):
    header = (
        f"{'workload':<10} {'parallel':>8} {'config':<18} {'p50_ttft':>10} "
        f"{'p95_ttft':>10} {'p99_ttft':>10} {'out_tok/s':>20} {'total_tok/s':>20} {'hit_rate':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for workload in args.workloads:
        for parallel in args.parallel:
            for config in args.configs:
                cell = agg.get(config, {}).get(workload, {}).get(parallel)
                if cell is None:
                    continue
                out_str = f"{cell['out_tok_s_mean']:.1f}+/-{cell['out_tok_s_std']:.1f}"
                total_str = f"{cell['total_tok_s_mean']:.1f}+/-{cell['total_tok_s_std']:.1f}"
                print(
                    f"{workload:<10} {parallel!s:>8} {config:<18} "
                    f"{cell['p50_ttft_ms']:>10.2f} {cell['p95_ttft_ms']:>10.2f} "
                    f"{cell['p99_ttft_ms']:>10.2f} {out_str:>20} "
                    f"{total_str:>20} {cell['hit_rate_mean']:>9.4f}"
                )
    print("=" * len(header))


def check_sweep_complete(args, raw):
    """Fail loudly if any requested (config, workload) cell lacks enough reps.

    Failed reps are skipped, and empty cells drop out of the table + soft-target
    checks -- so without this an all-failed cell would look like a clean pass.
    Every requested cell needs >= drop_first + 1 successful reps (>= 1 survives
    the warmup drop); fewer is a sweep failure.
    """
    print("\n--- sweep completeness ---")
    min_required = args.drop_first + 1
    complete = True
    for config in args.configs:
        for workload in args.workloads:
            for parallel in args.parallel:
                n = len(raw.get(config, {}).get(workload, {}).get(parallel, []))
                if n < min_required:
                    complete = False
                    tag = "FAIL"
                elif n < args.repeats:
                    tag = "WARN"
                else:
                    tag = "PASS"
                print(
                    f"[{tag}] {config}/{workload}/p{parallel}: {n}/{args.repeats} reps ok "
                    f"(need >= {min_required})"
                )
    return complete


def soft_targets(args, agg):
    """Print PASS/WARN per check. Returns ``(all_pass, n_gates_fired)``.

    ``n_gates_fired`` counts the A/B comparison checks that actually ran (both
    sides present) across every ``--parallel`` point. A run where no gate fired
    (e.g. only one config supplied) must not be mistaken for a pass; callers
    using ``--strict`` should fail when it is zero.
    """
    print("\n--- soft-target report ---")
    all_pass = True
    fired = 0

    def have(config, workload, parallel):
        return agg.get(config, {}).get(workload, {}).get(parallel) is not None

    rc = "unified-recurrent"
    for parallel in args.parallel:
        t = f"@p{parallel}"

        # 1. random: unified degradation < 5% vs radix total throughput.
        if (
            "random" in args.workloads
            and have("unified", "random", parallel)
            and have("radix", "random", parallel)
        ):
            u = agg["unified"]["random"][parallel]["total_tok_s_mean"]
            r = agg["radix"]["random"][parallel]["total_tok_s_mean"]
            ok = u >= 0.95 * r
            all_pass = all_pass and ok
            fired += 1
            print(
                f"[{'PASS' if ok else 'WARN'}] random{t}: unified.total_tok/s ({u:.1f}) "
                f">= 0.95 * radix.total_tok/s ({0.95 * r:.1f})"
            )

        # 2. gsp: unified hit-rate >= radix hit-rate.
        if (
            "gsp" in args.workloads
            and have("unified", "gsp", parallel)
            and have("radix", "gsp", parallel)
        ):
            u = agg["unified"]["gsp"][parallel]["hit_rate_mean"]
            r = agg["radix"]["gsp"][parallel]["hit_rate_mean"]
            ok = u >= r
            all_pass = all_pass and ok
            fired += 1
            print(
                f"[{'PASS' if ok else 'WARN'}] gsp{t}: unified.hit_rate ({u:.4f}) "
                f">= radix.hit_rate ({r:.4f})"
            )

        # 3. gsp: unified p50 TTFT < no-cache p50 TTFT (warn if ratio >= 0.9).
        if (
            "gsp" in args.workloads
            and have("unified", "gsp", parallel)
            and have("no-cache", "gsp", parallel)
        ):
            u = agg["unified"]["gsp"][parallel]["p50_ttft_ms"]
            nc = agg["no-cache"]["gsp"][parallel]["p50_ttft_ms"]
            ratio = u / nc if nc else float("nan")
            ok = ratio < 0.9
            all_pass = all_pass and ok
            fired += 1
            print(
                f"[{'PASS' if ok else 'WARN'}] gsp{t}: unified.p50_ttft ({u:.2f}) "
                f"vs no-cache.p50_ttft ({nc:.2f}), ratio={ratio:.3f} (want < 0.9)"
            )

        # Recurrent A/B {no-cache, unified-recurrent} (no radix baseline). The
        # hit-rate band is calibrated by the controller after the first run, so
        # gsp hit-rate is reported, not gated.
        if "gsp" in args.workloads and have(rc, "gsp", parallel):
            hr = agg[rc]["gsp"][parallel]["hit_rate_mean"]
            print(f"[REPORT] gsp{t}: {rc}.hit_rate = {hr:.4f}")
            if have("no-cache", "gsp", parallel):
                u = agg[rc]["gsp"][parallel]["p50_ttft_ms"]
                nc = agg["no-cache"]["gsp"][parallel]["p50_ttft_ms"]
                ratio = u / nc if nc else float("nan")
                ok = ratio < 0.9
                all_pass = all_pass and ok
                fired += 1
                print(
                    f"[{'PASS' if ok else 'WARN'}] gsp{t}: {rc}.p50_ttft ({u:.2f}) "
                    f"vs no-cache.p50_ttft ({nc:.2f}), ratio={ratio:.3f} (want < 0.9)"
                )

        # random (a.k.a. low-concurrency): recurrent throughput overhead <= 5% vs
        # no-cache.
        if (
            "random" in args.workloads
            and have(rc, "random", parallel)
            and have("no-cache", "random", parallel)
        ):
            u = agg[rc]["random"][parallel]["total_tok_s_mean"]
            nc = agg["no-cache"]["random"][parallel]["total_tok_s_mean"]
            ok = u >= 0.95 * nc
            all_pass = all_pass and ok
            fired += 1
            print(
                f"[{'PASS' if ok else 'WARN'}] random{t}: {rc}.total_tok/s ({u:.1f}) "
                f">= 0.95 * no-cache.total_tok/s ({0.95 * nc:.1f})"
            )

        # mooncake: REPORT-only. The trace's throughput is output-only (the rows
        # are owned by the timed generator), and a stable TTFT threshold isn't
        # calibrated yet, so report recurrent-vs-no-cache, don't gate.
        if "mooncake" in args.workloads and have(rc, "mooncake", parallel):
            hr = agg[rc]["mooncake"][parallel]["hit_rate_mean"]
            print(f"[REPORT] mooncake{t}: {rc}.hit_rate = {hr:.4f}")
            if have("no-cache", "mooncake", parallel):
                u = agg[rc]["mooncake"][parallel]["p50_ttft_ms"]
                nc = agg["no-cache"]["mooncake"][parallel]["p50_ttft_ms"]
                ratio = u / nc if nc else float("nan")
                print(
                    f"[REPORT] mooncake{t}: {rc}.p50_ttft ({u:.2f}) "
                    f"vs no-cache.p50_ttft ({nc:.2f}), ratio={ratio:.3f}"
                )

    return all_pass, fired


def compare_mode(args) -> tuple[bool, int]:
    """Merge the aggregates from multiple result JSONs and run the A/B gate.

    Each input is one single-config run; together they form the A/B that a
    single run constrained to one config cannot. Reuses ``print_table`` /
    ``soft_targets`` on the union of configs+workloads. Returns
    ``(all_pass, n_gates_fired)``; a single-file compare fires no gate.
    """
    merged: dict = {}
    workloads: list[str] = []
    parallels: list = []
    for path in args.compare:
        with open(path) as f:
            payload = json.load(f)
        for config, by_workload in (payload.get("aggregates") or {}).items():
            dst = merged.setdefault(config, {})
            for workload, by_parallel in by_workload.items():
                wdst = dst.setdefault(workload, {})
                if workload not in workloads:
                    workloads.append(workload)
                for parallel, cell in by_parallel.items():
                    wdst[parallel] = cell
                    if parallel not in parallels:
                        parallels.append(parallel)
    if not merged:
        raise SystemExit(f"--compare: no 'aggregates' found in {args.compare}")
    # print_table / soft_targets iterate these; drive them off the merged keys.
    args.configs = list(merged.keys())
    args.workloads = workloads
    args.parallel = parallels
    print(
        f"compare: merged {len(args.compare)} file(s) -> "
        f"configs={args.configs} workloads={workloads} parallel={parallels}\n"
    )
    print_table(args, merged)
    return soft_targets(args, merged)


def main():
    args = parse_args()
    if args.compare:
        ok, fired = compare_mode(args)
        if fired == 0:
            print(
                "\nNOTE: no A/B gate fired — compare needs both sides of a pair "
                "for the same workload (e.g. no-cache/gsp + unified-recurrent/gsp, "
                "or radix/* + unified/*). Supply both result JSONs."
            )
        if args.strict and (not ok or fired == 0):
            raise SystemExit(1)
        return

    # Resolve the model against the test model cache (SGLANG_JAX_MODEL_CACHE) like
    # the rest of the harness: a bare HF id picks up a local CI checkpoint, while
    # absolute paths / cache misses pass through unchanged. Done after the compare
    # branch so an offline --compare merge stays jax-free.
    from sgl_jax.test.test_utils import _local_or_hf

    args.model = _local_or_hf(args.model)

    print(f"args={args}\n", flush=True)

    raw = {}
    for config in args.configs:
        raw[config] = run_config(args, config)

    agg = aggregate(args, raw)
    print_table(args, agg)
    complete = check_sweep_complete(args, raw)
    targets_pass, _fired = soft_targets(args, agg)
    all_pass = complete and targets_pass

    if args.server_url is not None and args.strict:
        # Single-config external run: the A/B targets compare two configs, so none
        # of them fire here. Say so, so --strict is not mistaken for a passed gate.
        print(
            "\nNOTE: single-config external run — the A/B targets need two configs "
            "and did not run. Run each config separately, then gate with "
            "`--compare run_a.json run_b.json --strict`."
        )

    if args.output_json:
        payload = {
            "args": vars(args),
            "raw": raw,
            "aggregates": agg,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote results to {args.output_json}")

    if args.strict and not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
