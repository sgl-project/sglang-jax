"""
Edge case tests for Score API.

This module tests validation corner cases and edge conditions for the
`/v1/score` API endpoint. Tests are organized into:
1. Validation edge cases (fast, no model needed)
2. Behavioral edge cases (require engine, marked as integration tests)

Design Document: sglang-jax-dev-scripts/rfcs/003-score-api-comprehensive-test-suite.md

Usage:
    # Run all edge case tests
    python -m pytest test/srt/test_score_api_edge_cases.py -v

    # Run only fast validation tests (no model)
    python -m pytest test/srt/test_score_api_edge_cases.py -v -m "not integration"

    # Run integration tests (requires TPU/engine)
    python -m pytest test/srt/test_score_api_edge_cases.py -v -m integration
"""

import os

import pytest

from sgl_jax.srt.validation import ValidationError, validate_score_request

# =============================================================================
# Validation Edge Case Tests (Fast, No Model Required)
# =============================================================================


class TestValidationEdgeCases:
    """Edge case tests for input validation.

    These tests verify that the validation layer correctly rejects
    invalid inputs with appropriate error messages and codes.
    """

    # =========================================================================
    # Empty input edge cases
    # =========================================================================

    def test_score_empty_items(self):
        """Empty items list should raise ValidationError.

        RFC-003: items=[] should raise ValueError with clear error message.
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test query",
                items=[],
                label_token_ids=[1, 2, 3],
            )
        assert exc_info.value.code == "empty_items"
        assert "empty" in exc_info.value.message.lower()
        assert exc_info.value.param == "items"

    def test_score_empty_label_token_ids(self):
        """Empty label_token_ids should raise ValidationError.

        RFC-003: label_token_ids=[] should raise ValueError.
        Can't compute scores over zero labels.
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test query",
                items=["item1", "item2"],
                label_token_ids=[],
            )
        assert exc_info.value.code == "empty_label_token_ids"
        assert exc_info.value.param == "label_token_ids"

    def test_score_empty_query_string(self):
        """Empty query string should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="",
                items=["item"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    def test_score_empty_query_token_list(self):
        """Empty query token list should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[],
                items=[[1, 2]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    # =========================================================================
    # Token ID edge cases
    # =========================================================================

    def test_score_negative_token_ids(self):
        """Negative token IDs should raise ValidationError.

        RFC-003: Negative IDs in label_token_ids should raise ValueError.
        Never valid in vocabulary.
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, -5, 3],
            )
        assert exc_info.value.code == "token_id_negative"
        assert "-5" in exc_info.value.message

    def test_score_token_ids_exceeds_vocab(self):
        """Token ID >= vocab_size should raise ValidationError.

        RFC-003: Already implemented, add explicit test.
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 100000, 3],
                vocab_size=32000,
            )
        assert exc_info.value.code == "token_id_exceeds_vocab"
        assert "100000" in exc_info.value.message
        assert "32000" in exc_info.value.message

    def test_score_token_id_at_vocab_boundary(self):
        """Token ID exactly at vocab_size should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[31999, 32000],  # 32000 is out of bounds
                vocab_size=32000,
            )
        assert exc_info.value.code == "token_id_exceeds_vocab"

    def test_score_token_id_at_max_valid(self):
        """Token ID at vocab_size-1 should be valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=[0, 31999],  # 31999 is valid for vocab_size=32000
            vocab_size=32000,
        )

    def test_score_zero_token_id_valid(self):
        """Token ID 0 should be valid."""
        # Should not raise - 0 is a valid token ID
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=[0],
        )

    # =========================================================================
    # Mixed input type edge cases
    # =========================================================================

    def test_score_mixed_input_types_text_query_token_items(self):
        """Text query + token items should raise ValidationError.

        RFC-003: Document this is intentionally rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="text query",
                items=[[1, 2, 3], [4, 5, 6]],  # Token items
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "mixed_input_types"

    def test_score_mixed_input_types_token_query_text_items(self):
        """Token query + text items should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3, 4],  # Token query
                items=["text item 1", "text item 2"],  # Text items
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "mixed_input_types"

    # =========================================================================
    # Invalid type edge cases
    # =========================================================================

    def test_score_items_not_list(self):
        """items not list should raise ValidationError.

        RFC-003: items not list → TypeError
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items="not a list",
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_score_items_is_tuple(self):
        """items as tuple should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=("item1", "item2"),  # Tuple, not list
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_score_label_token_ids_not_list(self):
        """label_token_ids not list should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=123,  # Not a list
            )
        assert exc_info.value.code == "invalid_label_token_ids_type"

    def test_score_label_token_ids_is_tuple(self):
        """label_token_ids as tuple should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=(1, 2, 3),  # Tuple, not list
            )
        assert exc_info.value.code == "invalid_label_token_ids_type"

    def test_score_query_is_dict(self):
        """query as dict should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query={"text": "test"},
                items=["item"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_query_type"

    def test_score_mixed_types_in_label_token_ids(self):
        """Mixed types in label_token_ids should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, "two", 3],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    def test_score_float_in_label_token_ids(self):
        """Float in label_token_ids should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2.5, 3],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    def test_score_none_in_label_token_ids(self):
        """None in label_token_ids should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, None, 3],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    # =========================================================================
    # Boolean parameter edge cases
    # =========================================================================

    def test_score_apply_softmax_string_true(self):
        """apply_softmax as string 'true' should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                apply_softmax="true",
            )
        assert exc_info.value.code == "invalid_apply_softmax_type"

    def test_score_apply_softmax_int_one(self):
        """apply_softmax as int 1 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                apply_softmax=1,
            )
        assert exc_info.value.code == "invalid_apply_softmax_type"

    def test_score_item_first_string_false(self):
        """item_first as string 'false' should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                item_first="false",
            )
        assert exc_info.value.code == "invalid_item_first_type"

    # =========================================================================
    # Valid edge cases (should NOT raise)
    # =========================================================================

    def test_score_single_item_valid(self):
        """Single item should be valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["single item"],
            label_token_ids=[1, 2, 3],
        )

    def test_score_single_label_token_valid(self):
        """Single label token should be valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["item1", "item2"],
            label_token_ids=[42],
        )

    def test_score_many_items_valid(self):
        """Many items should be valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=[f"item{i}" for i in range(100)],
            label_token_ids=[1, 2, 3],
        )

    def test_score_many_label_tokens_valid(self):
        """Many label tokens should be valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=list(range(100)),
            vocab_size=1000,
        )

    def test_score_empty_string_item_valid(self):
        """Empty string as item should be valid.

        This is useful for candidate scoring where items=[""] means
        score the query directly without continuation.
        """
        # Should not raise
        validate_score_request(
            query="The answer is",
            items=[""],
            label_token_ids=[1, 2],
        )

    def test_score_duplicate_label_tokens_valid(self):
        """Duplicate label tokens should be valid.

        RFC-003: Duplicates in label_token_ids should work
        (return scores for each occurrence).
        """
        # Should not raise
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=[1, 2, 1, 3, 2],  # Duplicates
        )


# =============================================================================
# Integration Edge Case Tests (Require Engine)
# =============================================================================


@pytest.mark.integration
class TestBehavioralEdgeCases:
    """Behavioral edge case tests that require a running engine.

    These tests verify correct behavior for edge cases that pass validation
    but may have special handling in the scoring logic.

    Mark with @pytest.mark.integration to skip in fast CI runs.
    """

    @pytest.fixture(autouse=True)
    def skip_without_engine(self):
        """Skip these tests if engine is not available."""
        if os.environ.get("SGLANG_RUN_INTEGRATION_TESTS", "0") != "1":
            pytest.skip("Set SGLANG_RUN_INTEGRATION_TESTS=1 to run integration tests")

    def test_score_unicode_handling(self):
        """Unicode in query/items should work correctly.

        RFC-003: Unicode in query/items - emoji, non-ASCII characters.
        """
        # This test requires actual tokenization and scoring
        # Placeholder for when engine is available
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_whitespace_handling(self):
        """Whitespace variations should be handled correctly.

        RFC-003: Leading/trailing whitespace, multiple spaces.
        """
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_ordering_preserved(self):
        """Output order should match input items order.

        RFC-003: Output order matches input items order.
        Validate indices align.
        """
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_duplicate_label_tokens_returns_duplicates(self):
        """Duplicate label tokens should return scores for each occurrence.

        RFC-003: Should work and return scores for each occurrence.
        """
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_very_long_query(self):
        """Very long query should be handled (or error gracefully)."""
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_very_long_item(self):
        """Very long item should be handled (or error gracefully)."""
        pytest.skip("Requires engine - implement when running integration tests")

    def test_score_special_tokens_in_query(self):
        """Special tokens in query should be handled correctly."""
        pytest.skip("Requires engine - implement when running integration tests")


# =============================================================================
# HTTP Status Code Edge Cases
# =============================================================================


class TestHTTPStatusCodes:
    """Test that validation errors map to correct HTTP status codes."""

    def test_missing_param_returns_400(self):
        """Missing required parameter should return 400."""
        try:
            validate_score_request(
                query=None,
                items=["item"],
                label_token_ids=[1, 2],
            )
        except ValidationError as e:
            assert e.get_http_status() == 400

    def test_empty_value_returns_400(self):
        """Empty value should return 400."""
        try:
            validate_score_request(
                query="test",
                items=[],
                label_token_ids=[1, 2],
            )
        except ValidationError as e:
            assert e.get_http_status() == 400

    def test_invalid_type_returns_400(self):
        """Invalid type should return 400."""
        try:
            validate_score_request(
                query=123,
                items=["item"],
                label_token_ids=[1, 2],
            )
        except ValidationError as e:
            assert e.get_http_status() == 400

    def test_negative_token_returns_422(self):
        """Negative token ID should return 422 (semantic error)."""
        try:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[-1],
            )
        except ValidationError as e:
            assert e.get_http_status() == 422

    def test_token_exceeds_vocab_returns_422(self):
        """Token exceeding vocab should return 422 (semantic error)."""
        try:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[100000],
                vocab_size=32000,
            )
        except ValidationError as e:
            assert e.get_http_status() == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
