"""Accuracy case runner: drives run_eval against a given server URL.

The result JSON written to ``${RESULTS_DIR}/<case>.json`` is built by the
shared emitter in ``test/srt/nightly/results.py`` — the single source of the
``accuracy_result.v1.yaml`` shape, used by both the single-host and multi-host
accuracy nightlies.
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from drivers import run_eval_for_case
from multi_host_suite import AccuracyCase, SuiteError
from profile_loader import LaunchProfile
from results import build_accuracy_result, write_result


def run_accuracy_case(case: AccuracyCase, profile: LaunchProfile) -> None:
    base_url = f"http://127.0.0.1:{profile.port}"
    gen = case.generation_config or {}

    print(
        f"[multi-host-suite] Running accuracy case "
        f"name={case.name}, dataset={case.dataset}, "
        f"num_threads={case.eval_batch_size}, "
        f"temperature={gen.get('temperature', 0.0)}, max_tokens={gen.get('max_tokens', 2048)}, "
        f"top_p={gen.get('top_p')}, top_k={gen.get('top_k')}, min_p={gen.get('min_p')}, "
        f"presence_penalty={gen.get('presence_penalty')}, "
        f"repetition_penalty={gen.get('repetition_penalty')}, "
        f"frequency_penalty={gen.get('frequency_penalty')}, seed={gen.get('seed')}, "
        f"chat_template_kwargs={gen.get('chat_template_kwargs')}, "
        f"limit={case.limit}",
        flush=True,
    )

    metrics, started_at, finished_at = run_eval_for_case(case, base_url)

    summary = build_accuracy_result(
        case, profile.name, profile.target, metrics, started_at, finished_at
    )

    out_path = write_result(summary, case.name)
    if out_path is not None:
        print(f"[multi-host-suite] Wrote accuracy summary to {out_path}", flush=True)
    else:
        print(
            f"[multi-host-suite] RESULTS_DIR unset; skipping accuracy summary write",
            flush=True,
        )

    score = summary["score"]
    if case.score_threshold is not None:
        if score is None:
            raise SuiteError(
                kind="case",
                message=(
                    f"Accuracy case {case.name} produced no score; cannot evaluate "
                    f"against threshold={case.score_threshold}"
                ),
            )
        if score < case.score_threshold:
            raise SuiteError(
                kind="threshold",
                message=(
                    f"Accuracy case {case.name} score={score:.4f} below "
                    f"threshold={case.score_threshold:.4f}"
                ),
            )
        print(
            f"[multi-host-suite] Accuracy case {case.name} passed: "
            f"score={score:.4f} >= threshold={case.score_threshold:.4f}",
            flush=True,
        )
    else:
        print(
            f"[multi-host-suite] Accuracy case {case.name} finished "
            f"(no threshold set, score={score})",
            flush=True,
        )
