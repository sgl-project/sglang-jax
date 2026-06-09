"""Single-host accuracy helpers, used by ``single_host/suite_runner.py``.

Mirrors the multi-host ``multi_host/accuracy_case_runner.py``: loads a launch
profile, turns it into popen_launch_server kwargs for a single host, and runs
one ``AccuracyCase`` against a live server — emitting the shared
``accuracy_result.v1.yaml`` schema via ``results``.
The suite runner (``single_host/suite_runner.py``) owns server launch and gating.
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cases import AccuracyCase  # noqa: E402
from drivers import run_eval_for_case  # noqa: E402
from profiles import (  # noqa: E402
    LaunchProfile,
    RuntimeConfig,
    build_other_server_args,
    load_profile,
)
from results import build_accuracy_result, write_result  # noqa: E402

from sgl_jax.test.test_utils import (  # noqa: E402
    _local_or_hf,
    is_in_ci,
    write_github_step_summary,
)

_LAUNCH_PROFILES_DIR = os.path.join(_NIGHTLY_DIR, "launch_profiles")

# dist-init coordination port = HTTP port + this offset, kept distinct from the
# server's HTTP port.
_DIST_INIT_PORT_OFFSET = 5000

# Default gsm8k sampling for the 4-TPU accuracy gates: greedy decode.
# Cases reference this so every AccuracyCase carries an explicit
# generation_config, matching the per-case convention.
GSM8K_GENERATION_CONFIG = {"temperature": 0.0, "max_tokens": 2048}


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_accuracy_case(
    case: AccuracyCase,
    base_url: str,
    profile_name: str,
    target: str,
) -> dict:
    """Run a single AccuracyCase against an already-running server.

    Writes structured JSON to ``$RESULTS_DIR/<case.name>.json`` when set, and a
    GitHub step summary line when running in CI. Returns the full result dict
    (including ``score`` and ``passed``); assertion is left to the caller.
    """
    print(
        f"[accuracy-runner] Running case name={case.name}, "
        f"dataset={case.dataset}, model={case.model_id}, "
        f"base_url={base_url}, threads={case.eval_batch_size}, limit={case.limit}",
        flush=True,
    )

    metrics, started_at, finished_at = run_eval_for_case(case, base_url)

    result = build_accuracy_result(case, profile_name, target, metrics, started_at, finished_at)

    out_path = write_result(result, case.name)
    if out_path is not None:
        print(f"[accuracy-runner] Wrote result to {out_path}", flush=True)

    score = result["score"]
    threshold = case.score_threshold
    if result["passed"] is True:
        status = "PASS"
    elif result["passed"] is False:
        status = "FAIL"
    else:
        status = "N/A"

    if is_in_ci():
        write_github_step_summary(
            f"### {case.name}\n" f"score={_fmt(score)}  threshold={_fmt(threshold)}  **{status}**\n"
        )

    print(
        f"[accuracy-runner] {case.name}: score={_fmt(score)}, "
        f"threshold={_fmt(threshold)}, {status}",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# Launch profiles (test/srt/launch_profiles/*.yaml)
# ---------------------------------------------------------------------------


def load_profile_file(filename: str) -> LaunchProfile:
    """Load a launch profile from ``launch_profiles/<filename>``.

    One profile per config (mirroring the multi-host ``ModelRun`` model): the
    full server config lives in the YAML, so there is no per-run override layer.
    """
    return load_profile(os.path.join(_LAUNCH_PROFILES_DIR, filename))


def profile_server_spec(profile: LaunchProfile) -> dict:
    """Build popen_launch_server kwargs for a single-host launch of ``profile``.

    Constructs a single-host RuntimeConfig (nnodes=1, localhost dist-init),
    resolves ``model_path`` through the local model cache, and derives the HTTP
    base_url from the profile port.
    """
    runtime = RuntimeConfig(
        nnodes=1,
        node_rank=0,
        dist_init_addr=f"127.0.0.1:{profile.port + _DIST_INIT_PORT_OFFSET}",
        port=profile.port,
    )
    return {
        "model": _local_or_hf(profile.model_path),
        "base_url": f"http://127.0.0.1:{profile.port}",
        "other_args": build_other_server_args(profile, runtime),
    }
