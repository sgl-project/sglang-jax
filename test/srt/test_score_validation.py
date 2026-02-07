"""
Tests for Score API validation module.

These tests validate the input validation logic for the Score API,
ensuring proper error messages and codes for invalid requests.

Design Document: sglang-jax-dev-scripts/rfcs/006-error-handling-api-contract.md

Usage:
    python -m pytest test/srt/test_score_validation.py -v
"""

import pytest

from sgl_jax.srt.validation import (
    ValidationError,
    validate_multi_item_scoring_request,
    validate_score_request,
)


class TestValidationError:
    """Tests for ValidationError exception class."""

    def test_basic_creation(self):
        """Test creating a ValidationError with all parameters."""
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
            param="query",
            code="test_code",
        )
        assert error.message == "test error"
        assert error.error_type == "invalid_request_error"
        assert error.param == "query"
        assert error.code == "test_code"
        assert str(error) == "test error"

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
            param="query",
            code="test_code",
        )
        result = error.to_dict()
        assert result == {
            "error": {
                "message": "test error",
                "type": "invalid_request_error",
                "param": "query",
                "code": "test_code",
            }
        }

    def test_to_dict_minimal(self):
        """Test to_dict with only required fields."""
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
        )
        result = error.to_dict()
        assert result == {
            "error": {
                "message": "test error",
                "type": "invalid_request_error",
            }
        }

    def test_http_status_400(self):
        """Test that most errors return 400."""
        error = ValidationError(
            message="test",
            error_type="invalid_request_error",
            code="empty_query",
        )
        assert error.get_http_status() == 400

    def test_http_status_422_vocab(self):
        """Test that vocab errors return 422."""
        error = ValidationError(
            message="test",
            error_type="invalid_value_error",
            code="token_id_exceeds_vocab",
        )
        assert error.get_http_status() == 422

    def test_http_status_422_negative(self):
        """Test that negative token ID errors return 422."""
        error = ValidationError(
            message="test",
            error_type="invalid_value_error",
            code="token_id_negative",
        )
        assert error.get_http_status() == 422


class TestValidateScoreRequest:
    """Tests for validate_score_request function."""

    # =========================================================================
    # Query validation tests
    # =========================================================================

    def test_query_missing(self):
        """Test error when query is None."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=None,
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "missing_query"
        assert exc_info.value.param == "query"

    def test_query_empty_string(self):
        """Test error when query is empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="",
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    def test_query_empty_list(self):
        """Test error when query is empty token list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[],
                items=[[1, 2]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    def test_query_invalid_type(self):
        """Test error when query is neither string nor list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=123,
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_query_type"

    def test_query_list_with_non_integers(self):
        """Test error when query list contains non-integers."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, "two", 3],
                items=[[1, 2]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    # =========================================================================
    # Items validation tests
    # =========================================================================

    def test_items_missing(self):
        """Test error when items is None."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=None,
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "missing_items"

    def test_items_not_list(self):
        """Test error when items is not a list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items="not a list",
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_empty_list(self):
        """Test error when items is empty list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=[],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_items"

    def test_items_mixed_types_with_query(self):
        """Test error when items type doesn't match query type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="text query",
                items=[[1, 2, 3]],  # Token mode items with text query
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "mixed_input_types"

    def test_items_inconsistent_types_text_mode(self):
        """Test error when items contains non-strings in text mode."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["valid", 123, "also valid"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_inconsistent_types_token_mode(self):
        """Test error when items contains non-lists in token mode."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3],
                items=[[1, 2], "not a list", [3, 4]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_token_mode_non_integer(self):
        """Test error when token mode items contain non-integers."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3],
                items=[[1, 2], [3, "four"]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_token_id_type"


class TestValidateMultiItemScoringRequest:
    """Tests for multi-item scoring specific validation."""

    def test_valid_multi_item_request(self):
        validate_multi_item_scoring_request(
            query_tokens=[10, 11, 12],
            item_tokens=[[20], [21, 22]],
            delimiter_token_id=999,
            max_items=8,
            max_total_seq_len=64,
        )

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_multi_item_scoring_request(
                query_tokens=[],
                item_tokens=[[20]],
                delimiter_token_id=999,
            )
        assert exc_info.value.code == "empty_query"

    def test_item_count_limit_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_multi_item_scoring_request(
                query_tokens=[10],
                item_tokens=[[i] for i in range(9)],
                delimiter_token_id=999,
                max_items=8,
            )
        assert exc_info.value.code == "too_many_items"

    def test_delimiter_collision_in_query_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_multi_item_scoring_request(
                query_tokens=[10, 999],
                item_tokens=[[20]],
                delimiter_token_id=999,
            )
        assert exc_info.value.code == "delimiter_collision"
        assert exc_info.value.param == "query"

    def test_delimiter_collision_in_items_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_multi_item_scoring_request(
                query_tokens=[10],
                item_tokens=[[20], [999, 21]],
                delimiter_token_id=999,
            )
        assert exc_info.value.code == "delimiter_collision"
        assert exc_info.value.param == "items"

    def test_total_sequence_len_limit_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_multi_item_scoring_request(
                query_tokens=[1, 2, 3],
                item_tokens=[[4] * 10, [5] * 10],
                delimiter_token_id=999,
                max_total_seq_len=20,
            )
        assert exc_info.value.code == "multi_item_sequence_too_long"

    def test_total_sequence_len_check_can_be_disabled(self):
        # Same inputs as the rejection case, but skip total-length check.
        validate_multi_item_scoring_request(
            query_tokens=[1, 2, 3],
            item_tokens=[[4] * 10, [5] * 10],
            delimiter_token_id=999,
            max_total_seq_len=20,
            enforce_total_seq_len=False,
        )

    # =========================================================================
    # label_token_ids validation tests
    # =========================================================================

    def test_label_token_ids_missing(self):
        """Test error when label_token_ids is None."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=None,
            )
        assert exc_info.value.code == "missing_label_token_ids"

    def test_label_token_ids_not_list(self):
        """Test error when label_token_ids is not a list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=123,
            )
        assert exc_info.value.code == "invalid_label_token_ids_type"

    def test_label_token_ids_empty(self):
        """Test error when label_token_ids is empty."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[],
            )
        assert exc_info.value.code == "empty_label_token_ids"

    def test_label_token_ids_non_integer(self):
        """Test error when label_token_ids contains non-integer."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, "two", 3],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    def test_label_token_ids_negative(self):
        """Test error when label_token_ids contains negative value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, -5, 3],
            )
        assert exc_info.value.code == "token_id_negative"
        assert exc_info.value.get_http_status() == 422

    def test_label_token_ids_exceeds_vocab(self):
        """Test error when label_token_ids exceeds vocabulary size."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 50000, 3],
                vocab_size=32000,
            )
        assert exc_info.value.code == "token_id_exceeds_vocab"
        assert exc_info.value.get_http_status() == 422

    # =========================================================================
    # Boolean parameter validation tests
    # =========================================================================

    def test_apply_softmax_not_boolean(self):
        """Test error when apply_softmax is not boolean."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                apply_softmax="true",
            )
        assert exc_info.value.code == "invalid_apply_softmax_type"

    def test_item_first_not_boolean(self):
        """Test error when item_first is not boolean."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                item_first=1,
            )
        assert exc_info.value.code == "invalid_item_first_type"

    # =========================================================================
    # Valid request tests
    # =========================================================================

    def test_valid_text_mode_request(self):
        """Test that valid text mode request passes validation."""
        # Should not raise
        validate_score_request(
            query="What is the answer?",
            items=[" yes", " no", " maybe"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
            item_first=False,
        )

    def test_valid_token_mode_request(self):
        """Test that valid token mode request passes validation."""
        # Should not raise
        validate_score_request(
            query=[1, 2, 3, 4],
            items=[[5, 6], [7, 8], [9, 10]],
            label_token_ids=[100, 200, 300],
            apply_softmax=False,
            item_first=True,
        )

    def test_valid_request_with_vocab_size(self):
        """Test that valid request with vocab_size passes."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=[100, 200, 300],
            vocab_size=32000,
        )

    def test_valid_single_item(self):
        """Test that single item request is valid."""
        # Should not raise
        validate_score_request(
            query="test",
            items=["single item"],
            label_token_ids=[1],
        )

    def test_valid_empty_string_item(self):
        """Test that empty string item is valid (for candidate scoring)."""
        # Should not raise - empty items are valid for scoring candidates
        validate_score_request(
            query="The answer is",
            items=[""],  # Empty item is valid
            label_token_ids=[1, 2],
        )
