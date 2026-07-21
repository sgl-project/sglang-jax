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
    the single launched server. A ``BenchCase`` here runs client-only against that
    server (``server="runner"``); self-launching / offline benches go in a
    ``SelfLaunchRun`` instead, so this type always means exactly one server.
    """

    launch_profile: str
    cases: tuple[AccuracyCase | PerfCase | BenchCase, ...]

    def __post_init__(self):
        # Allow call sites to pass a list for readability; freeze to a tuple.
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))


@dataclass(frozen=True)
class SelfLaunchRun:
    """A run with no runner-launched server: its ``BenchCase``s either launch their
    own servers (``server="self"``, e.g. the A/B bench starts a server per config)
    or are offline (``server="none"``, e.g. an ``--compare`` gate). A separate type
    keeps SingleHostRun's "one run == one launched server" invariant intact.
    """

    cases: tuple[BenchCase, ...]

    def __post_init__(self):
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))
        bad = [c for c in self.cases if not isinstance(c, BenchCase) or c.server == "runner"]
        if bad:
            raise ValueError(
                "SelfLaunchRun holds only self-launching/offline BenchCases "
                f"(server in {{self, none}}); got {[getattr(c, 'name', c) for c in bad]}"
            )


@dataclass(frozen=True)
class SingleHostSuite:
    name: str
    runs: list[SingleHostRun | SelfLaunchRun]
    expose_in_caselist: bool = True


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
    /tmp/conversation_trace.jsonl on the pod before running.
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


def _recurrent_ab_measurement(config: str, output_json: str) -> BenchCase:
    """Run one side of the recurrent A/B against a profile-launched server."""
    return BenchCase(
        name=f"recurrent-ab-{config}",
        case_key="recurrent-ab",
        script="benchmark/hicache/bench_unified_radix_ab.py",
        server="runner",
        output_json=output_json,
        timeout=30 * 60,
        argv=(
            "--model",
            "Qwen/Qwen3.5-35B-A3B",
            "--configs",
            config,
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
    # Recurrent A/B perf gate: no-cache vs
    # unified-recurrent on the GDN-hybrid Qwen3.5-35B-A3B, single-host dp=1 tp=4.
    # Each side runs client-only against a profile-launched server; the final
    # offline case merges their JSON and applies the --strict soft targets
    # (gsp p50-TTFT < 0.9x no-cache; random throughput >= 0.95x no-cache). Server
    # knobs mirror the page-128 extra-buffer determinism e2e (context bounded so
    # the 35B leaves a positive KV budget; bs-paddings divisible by tp*t_packing).
    "recurrent-ab-perf-v6e-4": SingleHostSuite(
        name="recurrent-ab-perf-v6e-4",
        runs=[
            SingleHostRun(
                launch_profile="recurrent-qwen35-ab-no-cache-v6e-4.yaml",
                cases=[_recurrent_ab_measurement("no-cache", "recurrent_ab_no_cache.json")],
            ),
            SingleHostRun(
                launch_profile="recurrent-qwen35-ab-unified-v6e-4.yaml",
                cases=[_recurrent_ab_measurement("unified-recurrent", "recurrent_ab_unified.json")],
            ),
            SelfLaunchRun(
                cases=[
                    BenchCase(
                        name="recurrent-ab-compare",
                        case_key="recurrent-ab",
                        script="benchmark/hicache/bench_unified_radix_ab.py",
                        server="none",
                        compare_inputs=(
                            "recurrent_ab_no_cache.json",
                            "recurrent_ab_unified.json",
                        ),
                        timeout=60,
                        argv=("--strict",),
                    ),
                ],
            ),
        ],
    ),
    # Recurrent dp>1 cache_aware reuse gate, SINGLE-host (re-homed from multi-host —
    # nightly CI has no multi-host v6e capacity). One
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
                        timeout=60 * 60,
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
                        timeout=60 * 60,
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
    # — heavy (several cold 35B loads), run on demand.
    "recurrent-ablation-v6e-4": SingleHostSuite(
        name="recurrent-ablation-v6e-4",
        expose_in_caselist=False,
        runs=[
            SelfLaunchRun(
                cases=[
                    _ablation_case("dp1-ep1", dp=1, ep=1),
                    _ablation_case("dp1-ep4", dp=1, ep=4),
                    _ablation_case("dp2-ep4", dp=2, ep=4),
                ],
            ),
        ],
    ),
    # Recurrent-cache accuracy gate: Qwen3.5-35B-A3B tp4/dp2 with
    # the recurrent radix cache ON, gsm8k greedy non-thinking → score >= 0.95
    # (0.97 measured, 3-repeat spread 0.000 → min-0.02). Goes through the AccuracyCase framework (profile +
    # run_accuracy_case) like accuracy-text-models-v6e-4; the e2e determinism test
    # stays a unittest in run_suite.py (it's a 2-run byte-identical comparison).
    "recurrent-accuracy-v6e-4": SingleHostSuite(
        name="recurrent-accuracy-v6e-4",
        runs=[
            SingleHostRun(
                launch_profile="recurrent-qwen35-accuracy-v6e-4.yaml",
                cases=[
                    AccuracyCase(
                        name="recurrent-gsm8k",
                        dataset="gsm8k",
                        model_id="Qwen/Qwen3.5-35B-A3B",
                        eval_batch_size=16,
                        # Non-thinking + max_tokens 2048 (repo GSM8K convention) so
                        # answers fit the bounded (8192) context at dp2; greedy →
                        # deterministic gate.
                        generation_config={
                            "temperature": 0.0,
                            "max_tokens": 2048,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        limit=200,
                        score_threshold=0.95,
                    ),
                ],
            ),
            # MMLU (thinking) gate: cache-on serving accuracy on a harder benchmark
            # at dp2/ep4/context-32768 (ep4 fits the 32k thinking CoT that dp2/8192
            # can't). Official Qwen3.5 Thinking/general sampling (temp 1.0). Calibrated
            # over 6 seeds @ limit 200: 0.90-0.93, mean 0.914, sigma 0.011 →
            # threshold 0.88 = min(min-0.02, mean-3sigma). seed pinned to a calibrated
            # point for reproducibility (threshold has margin over all 6).
            SingleHostRun(
                launch_profile="recurrent-qwen35-mmlu-v6e-4.yaml",
                cases=[
                    AccuracyCase(
                        name="recurrent-mmlu-thinking",
                        dataset="mmlu",
                        model_id="Qwen/Qwen3.5-35B-A3B",
                        eval_batch_size=16,
                        generation_config={
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 1.5,
                            "repetition_penalty": 1.0,
                            "seed": 11,
                            # < context (32768): max_tokens is the completion cap and
                            # input+completion must fit the context, so it can't equal
                            # it. 16384 is ample for mmlu thinking CoT.
                            "max_tokens": 16384,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                        limit=200,
                        score_threshold=0.88,
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


def _eval_cases(cases, spec, profile) -> None:
    """Evaluate ``cases`` against an already-launched server (``spec``/``profile``)
    or, when both are ``None``, drive self-launching / offline ``BenchCase``s.

    Collects per-case failures — both gating misses and unexpected crashes from the
    case runner — and raises a single SuiteError tagged with the dominant kind (a
    crash maps to ``case`` so it surfaces as EXIT_CASE_CRASH, not retryable infra).
    """
    from drivers import run_bench_for_case

    failures: list[tuple[str, str]] = []
    for case in cases:
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

    if failures:
        kinds = {kind for kind, _ in failures}
        # Infra dominates because it is retryable; otherwise a case crash dominates
        # a threshold miss so the bug is not reported as a model regression.
        if "infra" in kinds:
            kind = "infra"
        elif "case" in kinds:
            kind = "case"
        else:
            kind = "threshold"
        raise SuiteError(kind=kind, message="; ".join(msg for _, msg in failures))


def run_one(run: SingleHostRun) -> None:
    """Launch the run's profile server, evaluate its cases, then tear it down.

    Only genuine infra errors (profile load, server launch) propagate as-is;
    per-case outcomes are funneled through ``_eval_cases`` into a SuiteError.
    """
    from accuracy_case_runner import load_profile_file, profile_server_spec

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
        check_cache_miss=spec["check_cache_miss"],
    )
    try:
        _eval_cases(run.cases, spec=spec, profile=profile)
    finally:
        kill_process_tree(process.pid)


def run_self_launch(run: SelfLaunchRun) -> None:
    """Drive self-launching / offline BenchCases — the runner launches no server."""
    _log(f"self-launching/offline cases ({len(run.cases)}) — no runner server")
    _eval_cases(run.cases, spec=None, profile=None)


def run_suite(suite: SingleHostSuite) -> int:
    """Run every run (independent servers / self-launching benches); report all failures."""
    total_cases = sum(len(r.cases) for r in suite.runs)
    _log(f"running suite={suite.name} ({len(suite.runs)} runs, {total_cases} cases)")
    had_infra = had_crash = had_threshold = False
    for run in suite.runs:
        label = run.launch_profile if isinstance(run, SingleHostRun) else "self-launch"
        try:
            if isinstance(run, SingleHostRun):
                run_one(run)
            else:
                run_self_launch(run)
        except SuiteError as exc:
            _log(f"{label}: FAIL ({exc.kind}) — {exc}")
            if exc.kind == "infra":
                had_infra = True
            elif exc.kind == "threshold":
                had_threshold = True
            else:
                had_crash = True
        except Exception as exc:  # noqa: BLE001 — surface as infra failure
            _log(f"{label}: infra error — {exc!r}")
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
                "launch_profile": getattr(run, "launch_profile", None),
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
        if not suite.expose_in_caselist:
            continue
        seen: set[str] = set()
        for run in suite.runs:
            for case in run.cases:
                if getattr(case, "capture_trace", True) is False:
                    continue
                case_key = getattr(case, "case_key", None) or case.name
                if case_key in seen:
                    continue
                seen.add(case_key)
                out.append({"suite": suite_name, "case": case_key})
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
    selected: list[SingleHostRun | SelfLaunchRun] = []
    for run in suite.runs:
        keep = tuple(c for c in run.cases if (getattr(c, "case_key", None) or c.name) in want)
        if keep:
            if isinstance(run, SingleHostRun):
                selected.append(SingleHostRun(launch_profile=run.launch_profile, cases=keep))
            else:
                selected.append(SelfLaunchRun(cases=keep))
    found = {getattr(c, "case_key", None) or c.name for run in selected for c in run.cases}
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
