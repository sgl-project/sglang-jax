"""Tests for PrefillInfo page_size/kv_dtype compatibility validation.

A decode worker must refuse a prefill peer whose KV layout (page size or
KV cache dtype) differs from its own — otherwise the gathered KV would be
silently misinterpreted on the decode side.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation.bootstrap import (
    PrefillInfo,
    build_app,
    check_prefill_compat,
)


class TestCheckPrefillCompat:
    def test_matching_config_passes(self):
        info = {"page_size": 128, "kv_dtype": "bfloat16"}
        check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_page_size_mismatch_raises(self):
        info = {"page_size": 64, "kv_dtype": "bfloat16"}
        with pytest.raises(ValueError, match="page_size"):
            check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_kv_dtype_mismatch_raises(self):
        info = {"page_size": 128, "kv_dtype": "float16"}
        with pytest.raises(ValueError, match="kv_dtype"):
            check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_peer_missing_fields_is_backward_compatible(self):
        # Older prefill that never reported these fields → skip the check.
        info = {"page_size": 0, "kv_dtype": ""}
        check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_absent_keys_are_backward_compatible(self):
        check_prefill_compat({}, local_page_size=128, local_kv_dtype="bfloat16")

    def test_local_unknown_skips_check(self):
        info = {"page_size": 64, "kv_dtype": "float16"}
        # Decode that doesn't know its own config yet → don't false-reject.
        check_prefill_compat(info, local_page_size=0, local_kv_dtype="")


class TestPrefillInfoFields:
    def test_to_dict_carries_new_fields(self):
        info = PrefillInfo(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            page_size=128,
            kv_dtype="bfloat16",
        )
        d = info.to_dict()
        assert d["page_size"] == 128
        assert d["kv_dtype"] == "bfloat16"

    def test_defaults_are_backward_compatible(self):
        info = PrefillInfo(
            bootstrap_key="h:1", host="h", transfer_port=1, side_channel_port=2
        )
        assert info.page_size == 0
        assert info.kv_dtype == ""


class TestBootstrapRoundTrip:
    def test_register_and_get_preserves_layout_fields(self):
        app, _ = build_app()
        client = TestClient(app)
        resp = client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
                "page_size": 128,
                "kv_dtype": "bfloat16",
            },
        )
        assert resp.status_code == 200
        got = client.get("/get_prefill_info", params={"bootstrap_room": 0})
        assert got.status_code == 200
        body = got.json()
        assert body["page_size"] == 128
        assert body["kv_dtype"] == "bfloat16"

    def test_register_without_layout_fields_defaults(self):
        app, _ = build_app()
        client = TestClient(app)
        resp = client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
            },
        )
        assert resp.status_code == 200
        body = client.get("/get_prefill_info", params={"bootstrap_room": 0}).json()
        assert body["page_size"] == 0
        assert body["kv_dtype"] == ""
