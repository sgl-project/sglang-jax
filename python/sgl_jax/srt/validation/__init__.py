"""
Validation utilities for SGLang API endpoints.

This module provides validation functions and error classes for API request
validation, ensuring consistent error handling across all endpoints.
"""

from sgl_jax.srt.validation.score_validation import (
    ValidationError,
    validate_score_request,
)

__all__ = [
    "ValidationError",
    "validate_score_request",
]
