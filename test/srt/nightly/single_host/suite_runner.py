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
    AbsolutePerfBaseline,
    AccuracyCase,
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
    """

    launch_profile: str
    cases: tuple[AccuracyCase | PerfCase, ...]

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
            SingleHostRun(
                launch_profile="recurrent-qwen35-perf-v6e-4.yaml",
                cases=[
                    PerfCase(
                        name="recurrent-qwen35-gsp",
                        workload="generated-shared-prefix",
                        input_len=2176,
                        output_len=128,
                        num_prompts=128,
                        max_concurrency=16,
                        warmup_requests=1,
                        flush_cache=True,
                        gsp_num_groups=16,
                        gsp_prompts_per_group=8,
                        gsp_system_prompt_len=2048,
                        gsp_question_len=128,
                        absolute_baselines={
                            "ttft_ms": AbsolutePerfBaseline(value=1512.2, tolerance=0.02),
                            "total_tps": AbsolutePerfBaseline(value=5795.3, tolerance=0.02),
                        },
                        use_trailing_baseline=False,
                        expose_in_caselist=True,
                    ),
                    PerfCase(
                        name="recurrent-qwen35-random",
                        input_len=1024,
                        output_len=128,
                        num_prompts=64,
                        max_concurrency=16,
                        warmup_requests=1,
                        flush_cache=True,
                        absolute_baselines={
                            "ttft_ms": AbsolutePerfBaseline(value=4400.3, tolerance=0.02),
                            "total_tps": AbsolutePerfBaseline(value=1729.3, tolerance=0.02),
                        },
                        use_trailing_baseline=False,
                        expose_in_caselist=True,
                    ),
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
    load, server launch) propagate as-is. ``AccuracyCase`` and ``PerfCase`` are
    dispatched by type to their case runner.
    """
    # Lazy: pull jax in only to actually run a case (keeps --caselist jax-free).
    from accuracy_case_runner import (
        load_profile_file,
        profile_server_spec,
        run_accuracy_case,
    )
    from perf_case_runner import run_perf_case

    from sgl_jax.srt.utils import kill_process_tree
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
    failures: list[tuple[str, str]] = []
    try:
        for case in run.cases:
            try:
                if isinstance(case, PerfCase):
                    result, fail = run_perf_case(
                        case, spec["base_url"], spec["model"], profile.name, profile.target
                    )
                else:
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

    Perf exposes its representative trace point or an explicitly selected case;
    accuracy cases are all listed. Touches only the stdlib `cases` catalog (no
    jax), so the slash handler can call this on a plain CPU runner.
    """
    out: list[dict] = []
    for suite_name, suite in SUITES.items():
        for run in suite.runs:
            for case in run.cases:
                if not (
                    getattr(case, "capture_trace", True)
                    or getattr(case, "expose_in_caselist", False)
                ):
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
