"""Capture input validation does not need a server."""

import hashlib
import json
import sys

import pytest
from hybrid_hicache_acceptance import check
from hybrid_hicache_capture import capture, main


def manifest():
    return {
        "return_logprob": True,
        "sampling_params": {"no_stop_trim": True, "skip_special_tokens": False},
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
        ],
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
            "return_logprob": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 3,
                "sampling_seed": 3,
                "no_stop_trim": True,
                "skip_special_tokens": False,
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
            "output_token_logprobs": [[-0.5, token, None] for token in [7, 8, 9]],
        },
    }
    results = capture(data, {"cold": []}, lambda _: response)
    errors = check(
        data,
        {
            name: {
                "return_logprob": True,
                "sampling_params": data["sampling_params"],
                "results": results,
            }
            for name in ("off", "off_repeat", "on", "on_repeat")
        },
    )
    assert len(errors) == 8
    assert all(error.startswith(("on/", "on_repeat/")) for error in errors)
    assert any("missing SWA h2d evidence" in error for error in errors)


def test_duplicate_rid_fails_before_network():
    data = manifest()
    data["cases"].append(dict(data["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        capture(data, {"cold": []}, lambda _: None)


@pytest.mark.parametrize("setting", [None, False, 1])
def test_capture_rejects_unfrozen_logprob_setting_before_network(setting):
    data = manifest()
    data["return_logprob"] = setting
    sent = []
    with pytest.raises(ValueError, match="return_logprob"):
        capture(data, {"cold": []}, lambda payload: sent.append(payload))
    assert not sent


@pytest.mark.parametrize("setting", [True, False, None])
@pytest.mark.parametrize(
    "output_params",
    [{"no_stop_trim": True, "skip_special_tokens": False}, None, {"no_stop_trim": True}],
)
def test_offline_capture_preserves_and_checks_frozen_logprob_setting(
    tmp_path, monkeypatch, setting, output_params
):
    data = manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    raw = tmp_path / "raw.json"
    response = {"output_ids": [7], "meta_info": {"id": "cold"}}
    raw.write_text(
        json.dumps(
            {
                "manifest_sha256": digest,
                "return_logprob": setting,
                "sampling_params": output_params,
                "results": [response],
            }
        )
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"cold": []}))
    output = tmp_path / "capture.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--manifest",
            str(manifest_path),
            "--from-raw",
            str(raw),
            "--evidence",
            str(evidence),
            "--output",
            str(output),
        ],
    )
    if setting is not True or output_params != data["sampling_params"]:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert not output.exists()
    else:
        main()
        assert json.loads(output.read_text()) == {
            "manifest_sha256": digest,
            "return_logprob": True,
            "sampling_params": data["sampling_params"],
            "results": [{**response, "events": []}],
        }


@pytest.mark.parametrize("name, value", [("no_stop_trim", False), ("skip_special_tokens", True)])
def test_capture_rejects_token_trimming_before_network(name, value):
    data = manifest()
    data["sampling_params"][name] = value
    sent = []
    with pytest.raises(ValueError, match="sampling_params"):
        capture(data, {"cold": []}, lambda payload: sent.append(payload))
    assert not sent


def test_natural_stop_capture_preserves_eos_for_strict_checker():
    from dataclasses import fields
    from types import SimpleNamespace

    from sgl_jax.srt.managers.detokenizer_manager import DetokenizerManager
    from sgl_jax.srt.managers.io_struct import BatchTokenIDOut

    data = manifest()
    data["cases"] = data["cases"][:1]
    data["cases"][0]["expected_finish_reason"] = "stop"

    def post(payload):
        # Run the actual detokenizer method without creating IPC or loading a
        # tokenizer/model. EOS is both a matched stop and a special token.
        manager = object.__new__(DetokenizerManager)
        manager.decode_status = {}
        manager.tokenizer = SimpleNamespace(
            all_special_ids=[2],
            batch_decode=lambda batches, **_: [" ".join(map(str, ids)) for ids in batches],
        )
        recv = {field.name: None for field in fields(BatchTokenIDOut)}
        recv.update(
            rids=["cold"],
            finished_reasons=[{"type": "stop", "matched": 2}],
            decoded_texts=[""],
            decode_ids=[[11, 2]],
            read_offsets=[0],
            no_stop_trim=[payload["sampling_params"].get("no_stop_trim", False)],
            skip_special_tokens=[payload["sampling_params"].get("skip_special_tokens", True)],
            spaces_between_special_tokens=[True],
            prompt_tokens=[2],
            completion_tokens=[2],
            cached_tokens=[0],
            output_token_logprobs_val=[[-0.25, -0.5]],
            output_token_logprobs_idx=[[11, 2]],
        )
        result = manager.handle_batch_token_id_out(BatchTokenIDOut(**recv))
        return {
            "output_ids": result.output_ids[0],
            "meta_info": {
                "id": "cold",
                "dp_rank": 0,
                "finish_reason": result.finished_reasons[0],
                "completion_tokens": result.completion_tokens[0],
                "output_token_logprobs": [
                    [value, token, None]
                    for value, token in zip(
                        result.output_token_logprobs_val[0],
                        result.output_token_logprobs_idx[0],
                        strict=True,
                    )
                ],
            },
        }

    for params in ({}, {"no_stop_trim": True}):
        trimmed = post({"sampling_params": params})
        assert trimmed["output_ids"] == [11]
        assert trimmed["meta_info"]["completion_tokens"] == 2
        assert len(trimmed["meta_info"]["output_token_logprobs"]) == 2
    results = capture(data, {"cold": []}, post)
    assert results[0]["output_ids"] == [11, 2]
    assert (
        check(
            data,
            {
                name: {
                    "return_logprob": True,
                    "sampling_params": data["sampling_params"],
                    "results": results,
                }
                for name in ("off", "off_repeat", "on", "on_repeat")
            },
        )
        == []
    )
