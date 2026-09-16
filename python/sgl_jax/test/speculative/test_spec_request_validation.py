"""Request-contract tests for greedy-only speculative workers."""

from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.scheduler import (
    validate_dflash_request,
    validate_frozen_kv_mtp_request,
    validate_speculative_request,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _request(
    *,
    return_logprob=False,
    return_output_logprob_only=False,
    **sampling_overrides,
):
    sampling = {
        "top_k": 1,
        "json_schema": None,
        "regex": None,
        "ebnf": None,
        "structural_tag": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_new_tokens": 0,
    }
    sampling.update(sampling_overrides)
    return SimpleNamespace(
        return_logprob=return_logprob,
        return_output_logprob_only=return_output_logprob_only,
        sampling_params=SimpleNamespace(**sampling),
    )


def test_greedy_contract_accepts_supported_request_for_dflash_and_frozen_kv():
    req = _request()

    assert validate_dflash_request(req) is None
    assert validate_frozen_kv_mtp_request(req) is None


@pytest.mark.parametrize("top_k", [-1, 0, 2, 4])
def test_frozen_kv_rejects_non_greedy_sampling(top_k):
    error = validate_frozen_kv_mtp_request(_request(top_k=top_k))

    assert error == "FROZEN_KV_MTP speculative decoding currently only supports greedy sampling."


@pytest.mark.parametrize(
    ("request_update", "message_fragment"),
    [
        ({"return_logprob": True}, "does not support return_logprob"),
        ({"return_output_logprob_only": True}, "does not support return_logprob"),
        ({"json_schema": {}}, "does not support grammar-constrained decoding"),
        ({"regex": "a+"}, "does not support grammar-constrained decoding"),
        ({"frequency_penalty": 0.1}, "does not support frequency, presence"),
        ({"presence_penalty": 0.1}, "does not support frequency, presence"),
        ({"repetition_penalty": 1.1}, "does not support frequency, presence"),
        ({"min_new_tokens": 1}, "does not support frequency, presence"),
    ],
)
def test_frozen_kv_rejects_other_sampling_features_ignored_by_fused_verify(
    request_update, message_fragment
):
    req = _request(**request_update)

    error = validate_frozen_kv_mtp_request(req)

    assert error is not None
    assert error.startswith("FROZEN_KV_MTP speculative decoding")
    assert message_fragment in error


def test_validation_router_leaves_general_spec_algorithms_unchanged():
    req = _request(top_k=4)

    assert validate_speculative_request(req, SpeculativeAlgorithm.FROZEN_KV_MTP) is not None
    assert validate_speculative_request(req, SpeculativeAlgorithm.DFLASH) is not None
    assert validate_speculative_request(req, SpeculativeAlgorithm.DSPARK) is not None
    assert validate_speculative_request(req, SpeculativeAlgorithm.NEXTN) is None
