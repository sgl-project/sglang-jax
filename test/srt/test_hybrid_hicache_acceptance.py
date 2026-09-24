"""Offline contract tests for the manual FULL+SWA L2 serving acceptance."""

import copy
import hashlib
import json
import sys

import pytest
from hybrid_hicache_acceptance import check, main


def fixture():
    manifest = {
        "return_logprob": True,
        "sampling_params": {"no_stop_trim": True, "skip_special_tokens": False},
        "cases": [
            {
                "rid": "reuse-r0",
                "kind": "normal",
                "dp_rank": 0,
                "max_new_tokens": 2,
                "expected_finish_reason": "length",
                "l2_components": ["FULL", "SWA"],
            },
            {
                "rid": "retract-r0",
                "kind": "retract",
                "dp_rank": 0,
                "max_new_tokens": 2,
                "expected_finish_reason": "length",
            },
            {"rid": "abort-r0", "kind": "abort", "dp_rank": 0, "max_new_tokens": 2},
        ],
    }
    results = [
        {
            "output_ids": [11, 12],
            "meta_info": {
                "id": "reuse-r0",
                "dp_rank": 0,
                "completion_tokens": 2,
                "finish_reason": {"type": "length", "length": 2},
            },
            "events": [
                {"type": t, "component": c, "dp_rank": 0, "source": "trace:123"}
                for c in ("FULL", "SWA")
                for t in ("d2h", "device_evict", "slot_overwrite", "h2d")
            ],
        },
        {
            "output_ids": [21, 22],
            "meta_info": {
                "id": "retract-r0",
                "dp_rank": 0,
                "completion_tokens": 2,
                "finish_reason": {"type": "length", "length": 2},
            },
            "events": [
                {"type": t, "dp_rank": 0, "source": "trace:456"}
                for t in ("retract", "reschedule", "complete")
            ],
        },
        {
            "output_ids": [],
            "meta_info": {
                "id": "abort-r0",
                "dp_rank": 0,
                "completion_tokens": 0,
                "finish_reason": {"type": "abort"},
            },
            "events": [
                {"type": t, "dp_rank": 0, "source": "trace:789"} for t in ("cancelled", "cleanup")
            ],
        },
    ]
    for result in results:
        result["meta_info"]["output_token_logprobs"] = [
            [-0.25, token, None] for token in result["output_ids"]
        ]
    off = {
        "return_logprob": True,
        "sampling_params": copy.deepcopy(manifest["sampling_params"]),
        "results": copy.deepcopy(results),
    }
    on = copy.deepcopy(off)
    return manifest, {
        "off": off,
        "off_repeat": copy.deepcopy(off),
        "on": on,
        "on_repeat": copy.deepcopy(on),
    }


def test_valid_four_runs_pass():
    manifest, runs = fixture()
    assert check(manifest, runs) == []


def test_equal_truncated_outputs_fail():
    manifest, runs = fixture()
    for run in runs.values():
        run["results"][0]["output_ids"] = [11]
    assert any("completion_tokens" in error for error in check(manifest, runs))


def test_equal_truncated_stop_fails_expected_length_contract():
    manifest, runs = fixture()
    for run in runs.values():
        run["results"][0]["output_ids"] = [11]
        run["results"][0]["meta_info"]["completion_tokens"] = 1
        run["results"][0]["meta_info"]["finish_reason"] = {"type": "stop"}
    assert any("expected terminal" in error for error in check(manifest, runs))


def test_empty_normal_output_fails_even_when_all_runs_match():
    manifest, runs = fixture()
    for run in runs.values():
        run["results"][0]["output_ids"] = []
        run["results"][0]["meta_info"]["completion_tokens"] = 0
        run["results"][0]["meta_info"]["finish_reason"] = {"type": "stop"}
    assert any("empty normal output" in error for error in check(manifest, runs))


def test_missing_or_duplicate_rid_fails():
    manifest, runs = fixture()
    runs["on"]["results"].pop()
    assert any("missing" in error for error in check(manifest, runs))
    manifest, runs = fixture()
    runs["on"]["results"].append(copy.deepcopy(runs["on"]["results"][0]))
    assert any("duplicate" in error for error in check(manifest, runs))


def test_equal_errors_fail():
    manifest, runs = fixture()
    for run in runs.values():
        run["results"][0]["error"] = "OOM"
    assert any("error" in error for error in check(manifest, runs))


def test_wrong_rank_and_missing_transfer_evidence_fail():
    manifest, runs = fixture()
    runs["on"]["results"][0]["meta_info"]["dp_rank"] = 1
    runs["on"]["results"][0]["events"] = []
    errors = check(manifest, runs)
    assert any("dp_rank" in error for error in errors)
    assert any("h2d" in error for error in errors)


def test_retract_requires_resume_and_normal_completion():
    manifest, runs = fixture()
    runs["on"]["results"][1]["events"] = []
    assert any("reschedule" in error for error in check(manifest, runs))


def test_abort_is_separate_from_normal_output_comparison():
    manifest, runs = fixture()
    runs["on"]["results"][2]["output_ids"] = [99]
    runs["on"]["results"][2]["meta_info"]["output_token_logprobs"] = [[-0.5, 99, None]]
    assert check(manifest, runs) == []


def test_abort_requires_cleanup_evidence():
    manifest, runs = fixture()
    runs["on"]["results"][2]["events"].pop()
    assert any("cleanup" in error for error in check(manifest, runs))


@pytest.mark.parametrize("run_name", ["off", "off_repeat", "on", "on_repeat"])
def test_all_four_runs_must_freeze_logprob_collection(run_name):
    manifest, runs = fixture()
    runs[run_name]["return_logprob"] = False
    assert any(f"{run_name}: return_logprob" in error for error in check(manifest, runs))


def test_manifest_requires_explicit_logprob_collection():
    manifest, runs = fixture()
    manifest.pop("return_logprob")
    assert any("manifest" in error and "return_logprob" in error for error in check(manifest, runs))


@pytest.mark.parametrize("run_name", ["off", "off_repeat", "on", "on_repeat"])
@pytest.mark.parametrize("name, value", [("no_stop_trim", False), ("skip_special_tokens", True)])
def test_all_runs_must_preserve_full_output_tokens(run_name, name, value):
    manifest, runs = fixture()
    runs[run_name]["sampling_params"][name] = value
    assert any(f"{run_name}: sampling_params" in error for error in check(manifest, runs))


@pytest.mark.parametrize(
    "entries, expected_error",
    [
        (None, "missing output_token_logprobs"),
        ([], "length"),
        ([[float("nan"), 11, None], [-0.2, 12, None]], "finite"),
        ([[float("inf"), 11, None], [-0.2, 12, None]], "finite"),
        ([[None, 11, None], [-0.2, 12, None]], "finite"),
        ([[True, 11, None], [-0.2, 12, None]], "finite"),
        ([[-0.2, 12, None], [-0.2, 11, None]], "token_id"),
        ([[-0.2, 11.0, None], [-0.2, 12, None]], "token_id"),
        ([[-0.2], [-0.2, 12, None]], "entry"),
    ],
)
def test_logprob_payload_must_be_complete_finite_and_token_aligned(entries, expected_error):
    manifest, runs = fixture()
    runs["on"]["results"][0]["meta_info"]["output_token_logprobs"] = entries
    report = {}
    errors = check(manifest, runs, report=report)
    assert any("on/reuse-r0" in error and expected_error in error for error in errors)
    assert report["logprobs"]["reuse-r0"]["runs"]["on"]["status"] == "invalid"


def test_logprob_differences_are_reported_without_an_invented_tolerance():
    manifest, runs = fixture()
    for name in ("on", "on_repeat"):
        runs[name]["results"][0]["meta_info"]["output_token_logprobs"][1][0] = -1.0
    report = {}
    assert check(manifest, runs, report=report) == []
    comparisons = report["logprobs"]["reuse-r0"]["comparisons"]
    assert comparisons["off/on"] == {
        "status": "compared",
        "token_count": 2,
        "different_count": 1,
        "max_abs_diff": 0.75,
        "mean_abs_diff": 0.375,
        "first_different_index": 1,
    }
    assert comparisons["off/off_repeat"]["different_count"] == 0
    assert comparisons["on/on_repeat"]["different_count"] == 0
    assert report["logprobs"]["abort-r0"]["comparisons"]["off/on"]["status"] == "not_compared"


def test_output_ids_remain_a_hard_gate_even_with_valid_logprobs():
    manifest, runs = fixture()
    runs["on"]["results"][0]["output_ids"][1] = 99
    runs["on"]["results"][0]["meta_info"]["output_token_logprobs"][1][1] = 99
    report = {}
    assert any("full output_ids" in error for error in check(manifest, runs, report=report))
    comparison = report["logprobs"]["reuse-r0"]["comparisons"]["off/on"]
    assert comparison == {"status": "not_compared", "reason": "output_ids differ"}


@pytest.mark.parametrize("tokens_differ", [False, True])
def test_cli_writes_logprob_report_on_pass_and_token_failure(tmp_path, monkeypatch, tokens_differ):
    manifest, runs = fixture()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs["on"]["results"][0]["meta_info"]["output_token_logprobs"][1][0] = -1.0
    if tokens_differ:
        runs["on"]["results"][0]["output_ids"][1] = 99
        runs["on"]["results"][0]["meta_info"]["output_token_logprobs"][1][1] = 99
    report_path = tmp_path / "report.json"
    argv = ["checker", "--manifest", str(manifest_path), "--report", str(report_path)]
    for name, run in runs.items():
        run_path = tmp_path / f"{name}.json"
        run_path.write_text(json.dumps({**run, "manifest_sha256": digest}))
        argv.extend((f"--{name.replace('_', '-')}", str(run_path)))
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == int(tokens_differ)
    report = json.loads(report_path.read_text())
    assert report["status"] == ("FAIL" if tokens_differ else "PASS")
    assert report["manifest_sha256"] == digest
    assert bool(report["errors"]) is tokens_differ
    comparison = report["logprobs"]["reuse-r0"]["comparisons"]["off/on"]
    if tokens_differ:
        assert comparison["reason"] == "output_ids differ"
    else:
        assert comparison["different_count"] == 1
        assert comparison["max_abs_diff"] == 0.75
