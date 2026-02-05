"""
Score API request validation.

This module provides validation functions for the `/v1/score` API endpoint,
ensuring all requests meet the API contract before processing.

Design Document: sglang-jax-dev-scripts/rfcs/006-error-handling-api-contract.md

Usage:
    from sgl_jax.srt.validation import validate_score_request, ValidationError

    try:
        validate_score_request(query, items, label_token_ids, apply_softmax, item_first)
    except ValidationError as e:
        return create_error_response(e.message, e.error_type, e.param, e.code)
"""

from typing import Any


class ValidationError(Exception):
    """Exception raised for Score API validation errors.

    This exception provides structured error information that can be
    converted to OpenAI-compatible error responses.

    Attributes:
        message: Human-readable error description
        error_type: Error category (e.g., "invalid_request_error")
        param: Parameter that caused the error (optional)
        code: Machine-readable error code (optional)

    Example:
        raise ValidationError(
            message="query cannot be empty",
            error_type="invalid_value_error",
            param="query",
            code="empty_query"
        )
    """

    def __init__(
        self,
        message: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to OpenAI-compatible error response dict.

        Returns:
            Dictionary with 'error' key containing error details.
        """
        error = {
            "message": self.message,
            "type": self.error_type,
        }
        if self.param is not None:
            error["param"] = self.param
        if self.code is not None:
            error["code"] = self.code
        return {"error": error}

    def get_http_status(self) -> int:
        """Get appropriate HTTP status code for this error.

        Returns:
            400 for most validation errors, 422 for semantic errors.
        """
        # Semantic errors (valid syntax but invalid meaning) get 422
        if self.code in ("token_id_exceeds_vocab", "token_id_negative"):
            return 422
        # All other validation errors get 400
        return 400


def validate_score_request(
    query: Any,
    items: Any,
    label_token_ids: Any,
    apply_softmax: Any = False,
    item_first: Any = False,
    vocab_size: int | None = None,
) -> None:
    """Validate a Score API request.

    Performs comprehensive validation of all Score API parameters,
    raising ValidationError with descriptive messages for any issues.

    Args:
        query: Query text (str) or token IDs (list[int])
        items: Items as text (list[str]) or token IDs (list[list[int]])
        label_token_ids: Token IDs to score (list[int])
        apply_softmax: Whether to apply softmax normalization
        item_first: Whether to place items before query
        vocab_size: Optional vocabulary size for token ID validation

    Raises:
        ValidationError: If any validation check fails.

    Example:
        validate_score_request(
            query="What is 2+2?",
            items=[" 4", " 5"],
            label_token_ids=[220, 221],
            apply_softmax=True,
        )
    """
    # =========================================================================
    # Validate query
    # =========================================================================
    if query is None:
        raise ValidationError(
            message="query is required",
            error_type="missing_parameter_error",
            param="query",
            code="missing_query",
        )

    if isinstance(query, str):
        if len(query) == 0:
            raise ValidationError(
                message="query cannot be empty",
                error_type="invalid_value_error",
                param="query",
                code="empty_query",
            )
    elif isinstance(query, list):
        if len(query) == 0:
            raise ValidationError(
                message="query token list cannot be empty",
                error_type="invalid_value_error",
                param="query",
                code="empty_query",
            )
        # Validate all elements are integers
        non_ints = [x for x in query if not isinstance(x, int)]
        if non_ints:
            raise ValidationError(
                message=(
                    f"query contains non-integer values when using token input mode. "
                    f"All token IDs must be integers. Got types: {[type(x).__name__ for x in non_ints[:3]]}"
                ),
                error_type="invalid_request_error",
                param="query",
                code="invalid_token_id_type",
            )
    else:
        raise ValidationError(
            message=f"query must be a string or list of integers, got {type(query).__name__}",
            error_type="invalid_request_error",
            param="query",
            code="invalid_query_type",
        )

    # =========================================================================
    # Validate items
    # =========================================================================
    if items is None:
        raise ValidationError(
            message="items is required",
            error_type="missing_parameter_error",
            param="items",
            code="missing_items",
        )

    if not isinstance(items, list):
        raise ValidationError(
            message="items must be a list of strings or list of token ID lists",
            error_type="invalid_request_error",
            param="items",
            code="invalid_items_type",
        )

    if len(items) == 0:
        raise ValidationError(
            message="items cannot be empty. At least one item is required.",
            error_type="invalid_value_error",
            param="items",
            code="empty_items",
        )

    # Determine input modes
    query_is_text = isinstance(query, str)
    items_is_text = isinstance(items[0], str) if items else True

    # Check type consistency between query and items
    if query_is_text != items_is_text:
        raise ValidationError(
            message=(
                f"query and items must both be text (str) or both be tokens (list[int]). "
                f"Got query type: {'str' if query_is_text else 'list[int]'}, "
                f"items[0] type: {'str' if items_is_text else 'list[int]'}"
            ),
            error_type="invalid_request_error",
            param="items",
            code="mixed_input_types",
        )

    # Validate all items have consistent types
    if not query_is_text:
        # Token input mode: validate all items are list[int]
        for i, item in enumerate(items):
            if not isinstance(item, list):
                raise ValidationError(
                    message=f"items[{i}] must be a list of integers when using token input mode",
                    error_type="invalid_request_error",
                    param="items",
                    code="invalid_items_type",
                )
            # Check that all elements in the list are integers
            non_ints = [x for x in item if not isinstance(x, int)]
            if non_ints:
                raise ValidationError(
                    message=f"items[{i}] contains non-integer values. All token IDs must be integers.",
                    error_type="invalid_request_error",
                    param="items",
                    code="invalid_token_id_type",
                )
    else:
        # Text input mode: validate all items are strings
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise ValidationError(
                    message=f"items[{i}] must be a string when using text input mode",
                    error_type="invalid_request_error",
                    param="items",
                    code="invalid_items_type",
                )

    # =========================================================================
    # Validate label_token_ids
    # =========================================================================
    if label_token_ids is None:
        raise ValidationError(
            message="label_token_ids is required",
            error_type="missing_parameter_error",
            param="label_token_ids",
            code="missing_label_token_ids",
        )

    if not isinstance(label_token_ids, list):
        raise ValidationError(
            message="label_token_ids must be a list of integers",
            error_type="invalid_request_error",
            param="label_token_ids",
            code="invalid_label_token_ids_type",
        )

    if len(label_token_ids) == 0:
        raise ValidationError(
            message="label_token_ids cannot be empty. At least one token ID is required.",
            error_type="invalid_value_error",
            param="label_token_ids",
            code="empty_label_token_ids",
        )

    # Validate each token ID
    for i, token_id in enumerate(label_token_ids):
        if not isinstance(token_id, int):
            raise ValidationError(
                message=f"label_token_ids[{i}] must be an integer, got {type(token_id).__name__}",
                error_type="invalid_request_error",
                param="label_token_ids",
                code="invalid_token_id_type",
            )
        if token_id < 0:
            raise ValidationError(
                message=f"label_token_ids[{i}] is negative ({token_id}). Token IDs must be non-negative.",
                error_type="invalid_value_error",
                param="label_token_ids",
                code="token_id_negative",
            )
        if vocab_size is not None and token_id >= vocab_size:
            raise ValidationError(
                message=f"label_token_ids[{i}] ({token_id}) exceeds vocabulary size ({vocab_size})",
                error_type="invalid_value_error",
                param="label_token_ids",
                code="token_id_exceeds_vocab",
            )

    # =========================================================================
    # Validate boolean parameters
    # =========================================================================
    if not isinstance(apply_softmax, bool):
        raise ValidationError(
            message=f"apply_softmax must be a boolean, got {type(apply_softmax).__name__}",
            error_type="invalid_request_error",
            param="apply_softmax",
            code="invalid_apply_softmax_type",
        )

    if not isinstance(item_first, bool):
        raise ValidationError(
            message=f"item_first must be a boolean, got {type(item_first).__name__}",
            error_type="invalid_request_error",
            param="item_first",
            code="invalid_item_first_type",
        )
