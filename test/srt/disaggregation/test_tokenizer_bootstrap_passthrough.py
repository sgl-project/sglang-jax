"""Unit tests for the ``bootstrap_{host,port,room}`` passthrough chain.

Verifies the fields land on:

1. ``GenerateReqInput`` (HTTP request body).
2. ``TokenizedGenerateReqInput`` (after tokenization).
3. ``Req`` (scheduler-side handle), via the existing
   ``handle_generate_request`` path.

Also exercises the ``disaggregation_mode=decode`` rejection of
requests missing the fields.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from sgl_jax.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def test_generate_req_input_carries_bootstrap_fields():
    obj = GenerateReqInput(
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=42,
        disagg_transfer_id="wire-1",
    )
    assert obj.bootstrap_host == "10.0.0.1"
    assert obj.bootstrap_port == 8998
    assert obj.bootstrap_room == 42
    assert obj.disagg_transfer_id == "wire-1"


def test_tokenized_generate_req_input_has_bootstrap_fields():
    tokenized = TokenizedGenerateReqInput(rid="r1")
    assert tokenized.bootstrap_host is None
    assert tokenized.bootstrap_port is None
    assert tokenized.bootstrap_room is None
    assert tokenized.disagg_transfer_id is None

    tokenized.bootstrap_host = "10.0.0.1"
    tokenized.bootstrap_port = 8998
    tokenized.bootstrap_room = 42
    tokenized.disagg_transfer_id = "wire-1"
    assert tokenized.bootstrap_host == "10.0.0.1"
    assert tokenized.bootstrap_port == 8998
    assert tokenized.bootstrap_room == 42
    assert tokenized.disagg_transfer_id == "wire-1"


def _make_fake_tokenizer_manager(disaggregation_mode: str):
    """Build a stripped-down TokenizerManager-ish object with just
    enough surface for ``_create_tokenized_object`` to run.
    """

    from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager

    tm = TokenizerManager.__new__(TokenizerManager)
    tm.preferred_sampling_params = None
    tm.model_config = SimpleNamespace(vocab_size=32000)
    tm.tokenizer = SimpleNamespace(
        normalize=lambda x: x,
    )
    tm.server_args = SimpleNamespace(
        disaggregation_mode=disaggregation_mode,
    )
    return tm


def test_tokenizer_passes_bootstrap_fields_through_in_decode_mode():
    tm = _make_fake_tokenizer_manager("decode")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=42,
        disagg_transfer_id="wire-r1",
    )
    # ``_create_tokenized_object`` calls ``SamplingParams.normalize +
    # verify``; SamplingParams.normalize expects a tokenizer, the
    # SimpleNamespace stand-in satisfies the attribute lookup.
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.1"
    assert tokenized.bootstrap_port == 8998
    assert tokenized.bootstrap_room == 42
    assert tokenized.disagg_transfer_id == "wire-r1"


def test_tokenizer_rejects_missing_fields_in_decode_mode():
    tm = _make_fake_tokenizer_manager("decode")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
        pytest.raises(ValueError, match="bootstrap"),
    ):
        tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])


def test_tokenizer_allows_missing_fields_in_null_mode():
    tm = _make_fake_tokenizer_manager("null")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_room is None


def test_tokenizer_allows_missing_fields_in_prefill_mode():
    # Prefill side doesn't receive bootstrap_* from the HTTP request
    # (the decode side does); allow missing here.
    tm = _make_fake_tokenizer_manager("prefill")
    obj = GenerateReqInput(
        rid="r1",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_room is None


def test_req_carries_bootstrap_fields():
    """``Req`` constructor doesn't take these as positional args; the
    scheduler's ``handle_generate_request`` sets them after construction.
    Verify the attribute is present and defaults to None.
    """

    from sgl_jax.srt.managers.schedule_batch import Req

    req = Req(
        rid="r1",
        origin_input_text="hi",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    assert req.bootstrap_host is None
    assert req.bootstrap_port is None
    assert req.bootstrap_room is None
    assert req.disagg_transfer_id is None

    req.bootstrap_host = "10.0.0.1"
    req.bootstrap_port = 8998
    req.bootstrap_room = 42
    req.disagg_transfer_id = "wire-req-1"
    assert req.bootstrap_room == 42
    assert req.disagg_transfer_id == "wire-req-1"


def test_decode_mode_auto_derives_from_bootstrap_url():
    """Stage 3 router integration: when the request body lacks
    bootstrap_*, the tokenizer fills them in from
    ``--disaggregation-bootstrap-url`` so the router doesn't have to.
    """

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://10.0.0.5:8998"
    obj = GenerateReqInput(
        rid="r-auto",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.5"
    assert tokenized.bootstrap_port == 8998
    # Stable CRC32 of "r-auto".
    import zlib

    assert tokenized.bootstrap_room == zlib.crc32(b"r-auto")


def test_decode_mode_auto_derive_brackets_ipv6_host():
    """Stage 3 review I1: urlparse strips IPv6 brackets;
    auto-derive must re-bracket so downstream ``f"{host}:{port}"``
    is a parseable URL.
    """

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://[fe80::1]:8998"
    obj = GenerateReqInput(
        rid="r-v6",
        text="hi",
        sampling_params={"max_new_tokens": 4},
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "[fe80::1]"
    assert tokenized.bootstrap_port == 8998


def test_decode_mode_explicit_values_win_over_auto_derive():
    """Operator-supplied bootstrap_* values must NOT be overwritten
    by the auto-derive."""

    tm = _make_fake_tokenizer_manager("decode")
    tm.server_args.disaggregation_bootstrap_url = "http://10.0.0.5:8998"
    obj = GenerateReqInput(
        rid="r-explicit",
        text="hi",
        sampling_params={"max_new_tokens": 4},
        bootstrap_host="10.0.0.99",
        bootstrap_port=9999,
        bootstrap_room=42,
    )
    with (
        mock.patch.object(SamplingParams, "normalize", lambda self, t: None),
        mock.patch.object(SamplingParams, "verify", lambda self, v: None),
    ):
        tokenized = tm._create_tokenized_object(obj, input_text="hi", input_ids=[1, 2, 3])
    assert tokenized.bootstrap_host == "10.0.0.99"
    assert tokenized.bootstrap_port == 9999
    assert tokenized.bootstrap_room == 42
