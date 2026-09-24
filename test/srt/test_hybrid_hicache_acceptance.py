"""Offline contract tests for the manual FULL+SWA L2 serving acceptance."""

import copy

from hybrid_hicache_acceptance import check


def fixture():
    manifest = {
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
        ]
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
    off = {"results": copy.deepcopy(results)}
    on = {"results": copy.deepcopy(results)}
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
    assert check(manifest, runs) == []


def test_abort_requires_cleanup_evidence():
    manifest, runs = fixture()
    runs["on"]["results"][2]["events"].pop()
    assert any("cleanup" in error for error in check(manifest, runs))
