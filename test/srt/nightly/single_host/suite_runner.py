"""Single-host nightly suites (v6e-1 / v6e-4) — accuracy and perf.

The single-host analogue of ``multi_host/suite_runner.py``: a suite is a list of
runs, each run one launch profile + the cases evaluated against the server it
launches (nnodes=1). Run it directly (the 4-TPU daily CI jobs do)::

    cd test/srt
    python3 nightly/single_host/suite_runner.py --suite accuracy-text-models-v6e-4
    python3 nightly/single_host/suite_runner.py --suite perf-text-models-v6e-4

``AccuracyCase`` / ``PerfCase`` are dispatched by type to their case runner.
Pass/fail is conveyed through the process exit code (same tagged codes as the
multi-host runner) so CI can classify the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cases import (  # noqa: E402
    GSM8K_GENERATION_CONFIG,
    AccuracyCase,
    BenchCase,
    PerfCase,
    PerfParams,
    SuiteError,
    perf_sweep_cases,
)

# Case runners and sgl_jax are imported lazily in run_one(), not here, so
# --caselist can enumerate cases without jax (it runs on a plain CPU runner).

# Exit codes consumed by CI (same scheme as multi_host/suite_runner.py).
EXIT_OK = 0
EXIT_INFRA = 10  # server / runtime infra failure → caller may retry
EXIT_THRESHOLD = 20  # case finished but score below threshold → do not retry
EXIT_CASE_CRASH = 30  # case raised unexpectedly → bug, do not retry


@dataclass(frozen=True)
class SingleHostRun:
    """One server launch + one or more cases on a single host.

    Mirrors the multi-host ``ModelRun``: ``launch_profile`` is a profile filename
    under ``launch_profiles/`` carrying the full server config (one profile per
    config — no per-run overrides), and ``cases`` are evaluated in order against
    the single launched server.

    ``launch_profile`` may be ``None`` for a run of self-launching / offline
    ``BenchCase``s (the single-host A/B bench launches its own servers): the runner
    then launches nothing and just drives the cases.
    """

    launch_profile: str | None
    cases: tuple[AccuracyCase | PerfCase | BenchCase, ...]

    def __post_init__(self):
        # Allow call sites to pass a list for readability; freeze to a tuple.
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))


@dataclass(frozen=True)
class SingleHostSuite:
    name: str
    runs: list[SingleHostRun]


def _gsm8k_case(name: str, model_id: str, threshold: float, eval_batch_size: int) -> AccuracyCase:
    return AccuracyCase(
        name=name,
        dataset="gsm8k",
        model_id=model_id,
        eval_batch_size=eval_batch_size,
        generation_config=GSM8K_GENERATION_CONFIG,
        score_threshold=threshold,
    )


def _ablation_case(label: str, dp: int, ep: int) -> BenchCase:
    """A self-launching recurrent A/B BenchCase for one dp/ep mesh point.

    tp4 is fixed (4 chips); ep shards experts so dp>1 fits the 35B. The bench
    sweeps the --parallel list on one launched server per config (no relaunch per
    concurrency point), so each mesh point is two cold 35B loads. Report-only (no
    --strict). Mooncake uses bench_serving's /tmp trace fallback — pre-stage
    /tmp/conversation_trace.jsonl on the pod (see the validation runbook).
    """
    return BenchCase(
        name=f"ablation-{label}",
        script="benchmark/hicache/bench_unified_radix_ab.py",
        server="self",
        output_json=f"ablation_{label}.json",
        argv=(
            "--model",
            "Qwen/Qwen3.5-35B-A3B",
            "--tp-size",
            "4",
            "--dp-size",
            str(dp),
            "--ep-size",
            str(ep),
            "--page-size",
            "128",
            "--context-length",
            "8192",
            "--max-running-requests",
            "16",
            "--chunked-prefill-size",
            "512",
            "--max-recurrent-state-size",
            "96",
            "--mem-fraction-static",
            "0.8",
            "--configs",
            "no-cache",
            "unified-recurrent",
            "--workloads",
            "gsp",
            "random",
            "mooncake",
            "--parallel",
            "8",
            "16",
            "--num-prompts",
            "64",
            "--repeats",
            "2",
            "--drop-first",
            "1",
            "--disable-overlap-schedule",
            "--precompile-bs-paddings",
            "8",
            "16",
            "--mooncake-workload",
            "conversation",
            "--mooncake-slowdown-factor",
            "0.1",
            "--mooncake-num-rounds",
            "2",
        ),
    )


SUITES: dict[str, SingleHostSuite] = {
    "accuracy-text-models-v6e-4": SingleHostSuite(
        name="accuracy-text-models-v6e-4",
        runs=[
            # 1.1.1 Qwen3-8B dense × fa, TP4 (native variant out of scope)
            SingleHostRun(
                launch_profile="qwen3-8b-fa-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-8b-fa", "Qwen/Qwen3-8B", 0.85, 128)],
            ),
            # Qwen3-32B dense × fa, TP4 — correctness gate for the perf 32B config.
            # Threshold 0.92 = measured gsm8k ~0.97 less ~0.05 margin to absorb
            # run-to-run noise (same convention as the MoE/dense cases above).
            SingleHostRun(
                launch_profile="qwen3-32b-fa-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-32b-fa", "Qwen/Qwen3-32B", 0.92, 128)],
            ),
            # 1.1.2 DeepSeek-Coder-V2-Lite-Instruct × {mla, fa_mha}, DP4×TP4
            SingleHostRun(
                launch_profile="ds-v2-lite-mla-v6e-4.yaml",
                cases=[
                    _gsm8k_case(
                        "ds-v2-lite-mla", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", 0.855, 128
                    )
                ],
            ),
            SingleHostRun(
                launch_profile="ds-v2-lite-fa-mha-v6e-4.yaml",
                cases=[
                    _gsm8k_case(
                        "ds-v2-lite-fa-gqa",
                        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                        0.85,
                        128,
                    )
                ],
            ),
            # 1.1.3 Qwen3-MoE-30B-A3B × {epmoe, fused}, TP4/EP4
            SingleHostRun(
                launch_profile="qwen3-moe-epmoe-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-moe-epmoe", "Qwen/Qwen3-30B-A3B", 0.92, 64)],
            ),
            SingleHostRun(
                launch_profile="qwen3-moe-fused-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-moe-fused", "Qwen/Qwen3-30B-A3B", 0.92, 64)],
            ),
            # 1.1.4 Kimi-Linear-48B-A3B-Instruct × hybrid_linear, TP4
            SingleHostRun(
                launch_profile="kimi-linear-v6e-4.yaml",
                cases=[
                    _gsm8k_case(
                        "kimi-linear-hybrid", "moonshotai/Kimi-Linear-48B-A3B-Instruct", 0.89, 64
                    )
                ],
            ),
            # 1.1.5 Qwen3-MoE-30B-A3B-FP8 (static FP8 checkpoint) × {epmoe, fused},
            # TP4/EP4, drift gate vs bf16. epmoe gated one notch below fused to
            # absorb run-to-run gsm8k noise observed in CI.
            SingleHostRun(
                launch_profile="qwen3-moe-fp8-epmoe-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-moe-fp8-epmoe", "Qwen/Qwen3-30B-A3B-FP8", 0.92, 64)],
            ),
            SingleHostRun(
                launch_profile="qwen3-moe-fp8-fused-v6e-4.yaml",
                cases=[_gsm8k_case("qwen3-moe-fp8-fused", "Qwen/Qwen3-30B-A3B-FP8", 0.93, 64)],
            ),
        ],
    ),
    # 4-TPU perf sweeps. One server per model/config; concurrency points filtered
    # to what each server can actually run concurrently (decode is KV-bound:
    # dense ceiling ~41, MoE ~118 at i4096/o1024). Prefill (c{8,32} × i{4k,8k} × o1)
    # fills for all; decode (× i4k × o1024) is trimmed per model. radix cache
    # disabled. One full profile per config, at the largest decode point that fills.
    "perf-text-models-v6e-4": SingleHostSuite(
        name="perf-text-models-v6e-4",
        runs=[
            # 2.1.1 Qwen3-32B dense, TP4 + fa. Decode KV ceiling ~41 -> {16,32};
            # repr/trace at c32 (c64 exceeds the ceiling).
            SingleHostRun(
                launch_profile="qwen3-32b-perf-v6e-4.yaml",
                cases=perf_sweep_cases(
                    "qwen3-32b",
                    PerfParams(
                        decode_concurrencies=(16, 32),
                        profile_point=(32, 4096, 1024),
                        floors={"out_tps": 865.6},
                        prefill_floor_point=(32, 4096, 1),
                        prefill_floors={"in_tps": 15629.7},
                    ),
                ),
            ),
            # 2.1.2a Qwen3-MoE-30B-A3B, full-EP (TP4/EP4), moe-backend=epmoe.
            # Decode KV ceiling ~118 -> {32,64,96}; repr/trace at c64.
            SingleHostRun(
                launch_profile="qwen3-moe-epmoe-perf-v6e-4.yaml",
                cases=perf_sweep_cases(
                    "qwen3-moe-epmoe",
                    PerfParams(
                        decode_concurrencies=(32, 64, 96),
                        floors={"out_tps": 1534.1},
                        prefill_floor_point=(32, 4096, 1),
                        prefill_floors={"in_tps": 24039.9},
                    ),
                ),
            ),
            # 2.1.2b Qwen3-MoE-30B-A3B, full-EP (TP4/EP4), moe-backend=fused.
            # Same KV ceiling as epmoe (same checkpoint) -> {32,64,96}; repr at c64.
            SingleHostRun(
                launch_profile="qwen3-moe-fused-perf-v6e-4.yaml",
                cases=perf_sweep_cases(
                    "qwen3-moe-fused",
                    PerfParams(
                        decode_concurrencies=(32, 64, 96),
                        floors={"out_tps": 1601.3},
                        prefill_floor_point=(32, 4096, 1),
                        prefill_floors={"in_tps": 31954.6},
                    ),
                ),
            ),
        ],
    ),
    # Recurrent A/B perf gate (Next Work item 4, Phase 2): no-cache vs
    # unified-recurrent on the GDN-hybrid Qwen3.5-35B-A3B, single-host dp=1 tp=4.
    # The A/B bench self-launches both servers sequentially (one pod), so the run
    # carries no launch_profile; its own --strict gate fires the soft targets
    # (gsp p50-TTFT < 0.9x no-cache; random throughput >= 0.95x no-cache). Server
    # knobs mirror the page-128 extra-buffer determinism e2e (context bounded so
    # the 35B leaves a positive KV budget; bs-paddings divisible by tp*t_packing).
    "recurrent-ab-perf-v6e-4": SingleHostSuite(
        name="recurrent-ab-perf-v6e-4",
        runs=[
            SingleHostRun(
                launch_profile=None,
                cases=[
                    BenchCase(
                        name="recurrent-ab",
                        script="benchmark/hicache/bench_unified_radix_ab.py",
                        server="self",
                        output_json="recurrent_ab.json",
                        argv=(
                            "--model",
                            "Qwen/Qwen3.5-35B-A3B",
                            "--tp-size",
                            "4",
                            "--page-size",
                            "128",
                            "--context-length",
                            "8192",
                            "--max-running-requests",
                            "16",
                            "--chunked-prefill-size",
                            "512",
                            "--max-recurrent-state-size",
                            "96",
                            "--mem-fraction-static",
                            "0.8",
                            "--configs",
                            "no-cache",
                            "unified-recurrent",
                            "--workloads",
                            "gsp",
                            "random",
                            "--parallel",
                            "16",
                            "--num-prompts",
                            "64",
                            "--repeats",
                            "3",
                            "--drop-first",
                            "1",
                            "--disable-overlap-schedule",
                            "--precompile-bs-paddings",
                            "8",
                            "16",
                            "--strict",
                        ),
                    ),
                ],
            ),
        ],
    ),
    # Recurrent dp>1 cache_aware reuse gate, SINGLE-host (Next Work item 4, Phase 3
    # re-homed to single-host — nightly CI has no multi-host v6e capacity). One
    # v6e-4 runs tp4/dp2/ep4: dp2 gives cache_aware something to route across, ep4
    # shards experts so the 35B fits at dp>1, tp4 keeps the fused-MoE t_packing
    # trivial. Two SingleHostRuns (gate + contrast) launch sequentially on one host
    # — single-host has no multi-host 2nd-launch sync hang, so they share a suite.
    # Validated on a v6e-4 node: cache_aware reuse plateau 0.66, knee 64 in
    # K*=72 range [43.2, 100.8] -> PASS; min_running is the lower-reuse baseline.
    "recurrent-cache-aware-v6e-4": SingleHostSuite(
        name="recurrent-cache-aware-v6e-4",
        runs=[
            # Gate: assert the empirical reuse knee lands within predict_knee's range.
            SingleHostRun(
                launch_profile="recurrent-qwen35-cache-aware-v6e-4.yaml",
                cases=[
                    BenchCase(
                        name="reuse-sweep-cache-aware",
                        script="benchmark/hicache/bench_recurrent_reuse_sweep.py",
                        server="runner",
                        output_json="reuse_cache_aware.json",
                        argv=(
                            "--parallel",
                            "8",
                            "--k-list",
                            "8",
                            "32",
                            "64",
                            "96",
                            "128",
                        ),
                    ),
                ],
            ),
            # Contrast (--no-assert): min_running reuse is ~1/dp-bounded, reported only.
            SingleHostRun(
                launch_profile="recurrent-qwen35-min-running-v6e-4.yaml",
                cases=[
                    BenchCase(
                        name="reuse-sweep-min-running",
                        script="benchmark/hicache/bench_recurrent_reuse_sweep.py",
                        server="runner",
                        output_json="reuse_min_running.json",
                        argv=(
                            "--parallel",
                            "8",
                            "--k-list",
                            "8",
                            "32",
                            "64",
                            "96",
                            "128",
                            "--no-assert",
                        ),
                    ),
                ],
            ),
        ],
    ),
    # Offline ablation harness (manual; NOT wired into the nightly workflow). Sweeps
    # the dp/ep mesh and a --parallel concurrency list for the recurrent A/B on one
    # v6e-4. tp4 is fixed; ep shards experts so dp>1 fits. Report-only (no --strict)
    # — heavy (several cold 35B loads), run on demand. See the validation runbook.
    "recurrent-ablation-v6e-4": SingleHostSuite(
        name="recurrent-ablation-v6e-4",
        runs=[
            SingleHostRun(
                launch_profile=None,
                cases=[
                    _ablation_case("dp1-ep1", dp=1, ep=1),
                    _ablation_case("dp1-ep4", dp=1, ep=4),
                    _ablation_case("dp2-ep4", dp=2, ep=4),
                ],
            ),
        ],
    ),
}


def _log(message: str) -> None:
    print(f"[single-host-suite] {message}", flush=True)


def _gate_accuracy(case: AccuracyCase, result: dict) -> tuple[str, str] | None:
    if case.score_threshold is None:
        return None
    if result["score"] is None:
        return ("case", f"{case.name}: eval produced no score")
    if not result["passed"]:
        return (
            "threshold",
            f"{case.name}: score={result['score']:.4f} below "
            f"threshold={case.score_threshold:.4f}",
        )
    return None


def run_one(run: SingleHostRun) -> None:
    """Launch one server for ``run`` and evaluate its cases.

    Collects per-case failures across all of ``run.cases`` — both gating misses
    and unexpected crashes from the case runner — and raises a single SuiteError
    tagged with the dominant kind (a crash maps to ``case`` so it surfaces as
    EXIT_CASE_CRASH, not retryable infra). Only genuine infra errors (profile
    load, server launch) propagate as-is. ``AccuracyCase`` / ``PerfCase`` are
    dispatched by type to their case runner; ``BenchCase`` shells out to a
    standalone bench (``run.launch_profile`` is ``None`` for a self-launching /
    offline BenchCase run, so no server is launched here).
    """
    from drivers import run_bench_for_case

    from sgl_jax.srt.utils import kill_process_tree

    process = None
    spec = None
    profile = None
    if run.launch_profile is not None:
        # Lazy: pull jax in only when a server-backed case actually runs (keeps
        # --caselist and BenchCase-only runs jax-free in the parent).
        from accuracy_case_runner import load_profile_file, profile_server_spec

        from sgl_jax.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            popen_launch_server,
        )

        profile = load_profile_file(run.launch_profile)
        spec = profile_server_spec(profile)
        _log(f"launching {profile.name} (model={spec['model']}, base_url={spec['base_url']})")
        process = popen_launch_server(
            spec["model"],
            spec["base_url"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=spec["other_args"],
        )
    else:
        _log(f"no launch profile — self-launching/offline cases ({len(run.cases)})")

    failures: list[tuple[str, str]] = []
    try:
        for case in run.cases:
            try:
                if isinstance(case, BenchCase):
                    base_url = spec["base_url"] if (spec and case.server == "runner") else None
                    _result, fail = run_bench_for_case(case, base_url)
                elif isinstance(case, PerfCase):
                    if spec is None:
                        raise ValueError(f"{case.name}: PerfCase requires a launch_profile")
                    from perf_case_runner import run_perf_case

                    _result, fail = run_perf_case(
                        case, spec["base_url"], spec["model"], profile.name, profile.target
                    )
                else:
                    if spec is None:
                        raise ValueError(f"{case.name}: AccuracyCase requires a launch_profile")
                    from accuracy_case_runner import run_accuracy_case

                    result = run_accuracy_case(case, spec["base_url"], profile.name, profile.target)
                    fail = _gate_accuracy(case, result)
            except Exception as exc:  # noqa: BLE001 — a case crash is a bug, not infra
                _log(f"{case.name}: CRASH — {exc!r}")
                failures.append(("case", f"{case.name}: {exc!r}"))
                continue
            if fail:
                _log(f"{case.name}: FAIL ({fail[0]}) — {fail[1]}")
                failures.append(fail)
            else:
                _log(f"{case.name}: PASS")
    finally:
        if process is not None:
            kill_process_tree(process.pid)

    if failures:
        kinds = {kind for kind, _ in failures}
        # A case-crash dominates a threshold miss: surface the bug.
        kind = "case" if "case" in kinds else "threshold"
        raise SuiteError(kind=kind, message="; ".join(msg for _, msg in failures))


def run_suite(suite: SingleHostSuite) -> int:
    """Run every SingleHostRun (independent servers); report all failures."""
    total_cases = sum(len(r.cases) for r in suite.runs)
    _log(f"running suite={suite.name} ({len(suite.runs)} runs, {total_cases} cases)")
    had_infra = had_crash = had_threshold = False
    for run in suite.runs:
        try:
            run_one(run)
        except SuiteError as exc:
            _log(f"{run.launch_profile}: FAIL ({exc.kind}) — {exc}")
            if exc.kind == "threshold":
                had_threshold = True
            else:
                had_crash = True
        except Exception as exc:  # noqa: BLE001 — surface as infra failure
            _log(f"{run.launch_profile}: infra error — {exc!r}")
            had_infra = True

    if had_infra:
        return EXIT_INFRA
    if had_crash:
        return EXIT_CASE_CRASH
    if had_threshold:
        return EXIT_THRESHOLD
    _log(f"suite {suite.name}: all cases passed")
    return EXIT_OK


def _dry_run(suite: SingleHostSuite) -> dict:
    return {
        "suite": suite.name,
        "runs": [
            {
                "launch_profile": run.launch_profile,
                "cases": [
                    {
                        "case": case.name,
                        # accuracy-only fields; absent on PerfCase
                        "model_id": getattr(case, "model_id", None),
                        "score_threshold": getattr(case, "score_threshold", None),
                    }
                    for case in run.cases
                ],
            }
            for run in suite.runs
        ],
    }


def _caselist() -> list[dict]:
    """Runnable /run-nightly cases across all suites, as [{"suite", "case"}].

    Perf exposes only its representative point (PerfCase.capture_trace) — the
    sweep's other points only make sense as a set; accuracy cases have no such
    attr so all are listed. Touches only the stdlib `cases` catalog (no jax), so
    the slash handler can call this on a plain CPU runner.
    """
    out: list[dict] = []
    for suite_name, suite in SUITES.items():
        for run in suite.runs:
            for case in run.cases:
                if getattr(case, "capture_trace", True) is False:
                    continue
                out.append({"suite": suite_name, "case": case.name})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-host SGL-JAX nightly suites")
    parser.add_argument("--suite", choices=sorted(SUITES))
    parser.add_argument(
        "--cases",
        help="Comma-separated case names; run only these (e.g. for /run-nightly targeting).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--caselist",
        action="store_true",
        help="Print runnable case_keys as JSON and exit (no jax needed).",
    )
    return parser.parse_args()


def _select_cases(suite: SingleHostSuite, cases: str | None) -> SingleHostSuite:
    if not cases:
        return suite
    want = {c.strip() for c in cases.split(",") if c.strip()}
    selected: list[SingleHostRun] = []
    for run in suite.runs:
        keep = tuple(c for c in run.cases if c.name in want)
        if keep:
            selected.append(SingleHostRun(launch_profile=run.launch_profile, cases=keep))
    found = {c.name for run in selected for c in run.cases}
    missing = want - found
    if missing:
        raise SystemExit(f"unknown case_key(s): {sorted(missing)}")
    return SingleHostSuite(name=suite.name, runs=selected)


def main() -> int:
    args = parse_args()
    if args.caselist:
        print(json.dumps(_caselist(), indent=2, sort_keys=True))
        return EXIT_OK
    if not args.suite:
        raise SystemExit("--suite is required (unless --caselist)")
    suite = _select_cases(SUITES[args.suite], args.cases)
    if args.dry_run:
        print(json.dumps(_dry_run(suite), indent=2, sort_keys=True))
        return EXIT_OK
    return run_suite(suite)


if __name__ == "__main__":
    sys.exit(main())
