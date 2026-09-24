"""Capture input validation does not need a server."""

import pytest
from hybrid_hicache_acceptance import check
from hybrid_hicache_capture import capture


def manifest():
    return {
        "cases": [
            {
                "rid": "cold",
                "kind": "normal",
                "dp_rank": 0,
                "input_ids": [1, 2],
                "max_new_tokens": 3,
                "expected_finish_reason": "length",
            },
            {
                "rid": "abort",
                "kind": "abort",
                "dp_rank": 0,
                "input_ids": [3],
                "max_new_tokens": 3,
            },
        ]
    }


def test_capture_sends_frozen_normal_request_and_preserves_response():
    sent = []
    response = {"output_ids": [7, 8, 9], "meta_info": {"id": "cold"}}

    def post(payload):
        sent.append(payload)
        return response

    result = capture(manifest(), {"cold": []}, post)
    assert sent == [
        {
            "rid": "cold",
            "input_ids": [1, 2],
            "dp_rank": 0,
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 3,
                "sampling_seed": 3,
            },
        }
    ]
    assert result == [{**response, "events": []}]
    assert "events" not in response


def test_missing_evidence_or_input_fails():
    sent = []

    def post(payload):
        sent.append(payload)
        return {"output_ids": [7], "meta_info": {"id": payload["rid"]}}

    with pytest.raises(ValueError, match="evidence"):
        capture(manifest(), {}, post)
    bad = manifest()
    bad["cases"][0]["input_ids"] = []
    with pytest.raises(ValueError, match="input_ids"):
        capture(bad, {"cold": []}, post)
    assert len(sent) == 1


def test_capture_allows_off_without_l2_events_but_checker_rejects_on():
    data = manifest()
    data["cases"] = data["cases"][:1]
    data["cases"][0]["l2_components"] = ["SWA"]
    response = {
        "output_ids": [7, 8, 9],
        "meta_info": {
            "id": "cold",
            "dp_rank": 0,
            "completion_tokens": 3,
            "finish_reason": {"type": "length"},
        },
    }
    results = capture(data, {"cold": []}, lambda _: response)
    errors = check(
        data, {name: {"results": results} for name in ("off", "off_repeat", "on", "on_repeat")}
    )
    assert len(errors) == 8
    assert all(error.startswith(("on/", "on_repeat/")) for error in errors)
    assert any("missing SWA h2d evidence" in error for error in errors)


def test_duplicate_rid_fails_before_network():
    data = manifest()
    data["cases"].append(dict(data["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        capture(data, {"cold": []}, lambda _: None)
