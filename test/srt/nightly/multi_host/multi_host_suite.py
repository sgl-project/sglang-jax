from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export the host-neutral case contracts from cases.py so `from multi_host_suite
# import ...` keeps working. Kept stdlib-only (no profiles/pydantic) so suite_runner.py's
# module import stays light (e.g. for --dry-run); RuntimeConfig comes from profiles where used.
from cases import AccuracyCase, BenchCase, PerfCase, SuiteError  # noqa: E402,F401


@dataclass(frozen=True)
class ModelRun:
    launch_profile: str  # filename under launch_profiles/; resolved to absolute path
    cases: list[PerfCase | AccuracyCase | BenchCase]


@dataclass(frozen=True)
class MultiHostSuite:
    name: str
    runs: list[ModelRun]


def dry_run_suite(suite: MultiHostSuite) -> dict:
    return {
        "suite": suite.name,
        "runs": [_dry_run_model_run(run) for run in suite.runs],
    }


def _dry_run_model_run(run: ModelRun) -> dict:
    from profile_loader import load_profile

    profile = load_profile(run.launch_profile)
    return {
        "name": profile.name,
        "target": profile.target,
        "launch_profile": run.launch_profile,
        "model_path": profile.model_path,
        "parallelism": {
            "tp_size": profile.tp_size,
            "dp_size": profile.dp_size,
            "ep_size": profile.ep_size,
        },
        "cases": [_dry_run_case(case) for case in run.cases],
    }


def _dry_run_case(case: PerfCase | AccuracyCase | BenchCase) -> dict:
    if isinstance(case, BenchCase):
        case_type = "bench"
    elif isinstance(case, PerfCase):
        case_type = "perf"
    else:
        case_type = "accuracy"
    return {
        "name": case.name,
        "type": case_type,
    }
