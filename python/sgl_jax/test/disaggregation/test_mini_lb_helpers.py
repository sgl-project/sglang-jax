"""Tests for mini_lb_helpers.py."""


from sgl_jax.srt.disaggregation.mini_lb_helpers import (
    ensure_request_identity_fields,
    generate_bootstrap_room,
    get_request_batch_size,
    inject_bootstrap_fields,
    maybe_wrap_ipv6_address,
)


class TestMaybeWrapIpv6:
    def test_ipv4_unchanged(self):
        assert maybe_wrap_ipv6_address("10.0.0.1") == "10.0.0.1"

    def test_hostname_unchanged(self):
        assert maybe_wrap_ipv6_address("myhost.local") == "myhost.local"

    def test_ipv6_wrapped(self):
        assert maybe_wrap_ipv6_address("::1") == "[::1]"

    def test_full_ipv6_wrapped(self):
        assert maybe_wrap_ipv6_address("2001:db8::1") == "[2001:db8::1]"


class TestGenerateBootstrapRoom:
    def test_room_is_int(self):
        room = generate_bootstrap_room()
        assert isinstance(room, int)

    def test_room_in_range(self):
        for _ in range(100):
            room = generate_bootstrap_room()
            assert 0 <= room < 2**63


class TestGetRequestBatchSize:
    def test_single_text(self):
        assert get_request_batch_size({"text": "hello"}) is None

    def test_batch_text(self):
        assert get_request_batch_size({"text": ["hello", "world"]}) == 2

    def test_single_input_ids(self):
        assert get_request_batch_size({"input_ids": [1, 2, 3]}) is None

    def test_batch_input_ids(self):
        assert get_request_batch_size({"input_ids": [[1, 2], [3, 4], [5, 6]]}) == 3

    def test_no_prompt_field(self):
        assert get_request_batch_size({"other": "value"}) is None


class TestEnsureRequestIdentityFields:
    def test_generates_rid_and_transfer_id(self):
        result = ensure_request_identity_fields({"text": "hello"})
        assert "rid" in result
        assert "disagg_transfer_id" in result
        assert result["rid"] == result["disagg_transfer_id"]
        assert len(result["rid"]) == 32  # uuid4 hex

    def test_preserves_existing_rid(self):
        result = ensure_request_identity_fields({"text": "hello", "rid": "abc123"})
        assert result["rid"] == "abc123"
        assert result["disagg_transfer_id"] == "abc123"

    def test_preserves_existing_transfer_id(self):
        result = ensure_request_identity_fields(
            {"text": "hello", "disagg_transfer_id": "tid123"}
        )
        assert result["rid"] == "tid123"
        assert result["disagg_transfer_id"] == "tid123"

    def test_batch_generates_list(self):
        result = ensure_request_identity_fields({"text": ["a", "b", "c"]})
        assert isinstance(result["rid"], list)
        assert len(result["rid"]) == 3
        assert result["rid"] == result["disagg_transfer_id"]

    def test_does_not_mutate_input(self):
        original = {"text": "hello"}
        result = ensure_request_identity_fields(original)
        assert "rid" not in original
        assert "rid" in result


class TestInjectBootstrapFields:
    def test_single_request(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == "10.0.0.1"
        assert result["bootstrap_port"] == 8998
        assert isinstance(result["bootstrap_room"], int)
        assert "rid" in result

    def test_batch_request(self):
        result = inject_bootstrap_fields(
            {"text": ["a", "b"]},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == ["10.0.0.1", "10.0.0.1"]
        assert result["bootstrap_port"] == [8998, 8998]
        assert len(result["bootstrap_room"]) == 2
        assert result["bootstrap_room"][1] == result["bootstrap_room"][0] + 1

    def test_bootstrap_host_override(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://localhost:30100",
            bootstrap_port=8998,
            bootstrap_host_override="10.31.0.1",
        )
        assert result["bootstrap_host"] == "10.31.0.1"

    def test_ipv6_server(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://[::1]:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == "[::1]"

    def test_none_bootstrap_port(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=None,
        )
        assert result["bootstrap_port"] is None
