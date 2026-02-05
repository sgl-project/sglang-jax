"""
OpenAI client compatibility tests for Score API.

This module validates that the official OpenAI Python client can be used
with SGLang's /v1/score endpoint, ensuring drop-in compatibility for users.

Design Document: sglang-jax-dev-scripts/rfcs/005-openai-client-compatibility.md

Requirements:
    pip install openai>=1.0.0

Usage:
    # Run all OpenAI client tests (requires running server)
    python -m pytest test/srt/test_score_openai_client.py -v

    # Run with custom server URL
    SGLANG_BASE_URL=http://localhost:8000 python -m pytest test/srt/test_score_openai_client.py -v

Note:
    These tests require a running SGLang server. Start one with:
        python -m sgl_jax.launch_server --model meta-llama/Llama-3.2-1B-Instruct
"""

import os

import pytest

# Skip entire module if openai is not installed
openai = pytest.importorskip("openai", reason="openai package required for these tests")
from openai import OpenAI  # noqa: E402

# =============================================================================
# Test Configuration
# =============================================================================


def get_base_url() -> str:
    """Get the SGLang server base URL from environment or default."""
    return os.getenv("SGLANG_BASE_URL", "http://localhost:8000")


def create_client() -> OpenAI:
    """Create an OpenAI client configured for SGLang."""
    return OpenAI(
        base_url=f"{get_base_url()}/v1",
        api_key="test-key",  # SGLang doesn't validate by default
    )


def is_server_available() -> bool:
    """Check if the SGLang server is available."""
    try:
        import httpx

        response = httpx.get(f"{get_base_url()}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip all tests in this module if server is not available
pytestmark = pytest.mark.skipif(
    not is_server_available(),
    reason=f"SGLang server not available at {get_base_url()}. "
    "Start server with: python -m sgl_jax.launch_server --model meta-llama/Llama-3.2-1B-Instruct",
)


# =============================================================================
# Basic OpenAI Client Compatibility Tests
# =============================================================================


@pytest.mark.integration
class TestScoreOpenAIClient:
    """Test Score API compatibility with OpenAI Python client.

    These tests validate that the official OpenAI Python client
    can successfully interact with SGLang's /v1/score endpoint.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up OpenAI client for each test."""
        self.client = create_client()

    def test_score_with_openai_client_post(self):
        """Test Score API using OpenAI client's generic post method.

        The /v1/score endpoint is an SGLang extension, so we use
        the generic .post() method rather than a typed method.

        RFC-005: Basic client usage test.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "The capital of France is",
                "items": [" Paris", " London", " Berlin"],
                # Using common token IDs that should exist in most vocabularies
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
                "item_first": False,
            },
            cast_to=dict,
        )

        # Validate response structure
        assert "scores" in response, "Response must contain 'scores' field"
        assert "object" in response, "Response must contain 'object' field"
        assert response["object"] == "scoring", "Object type must be 'scoring'"

        # Validate scores shape and values
        scores = response["scores"]
        assert len(scores) == 3, f"Expected 3 items, got {len(scores)}"
        for i, score_list in enumerate(scores):
            assert len(score_list) == 3, f"Item {i}: Expected 3 labels, got {len(score_list)}"
            assert all(
                isinstance(s, (int, float)) for s in score_list
            ), f"Item {i}: All scores must be numeric"
            # With softmax, scores should sum to ~1.0
            score_sum = sum(score_list)
            assert abs(score_sum - 1.0) < 1e-5, f"Item {i}: Softmax sum {score_sum} != 1.0"

    def test_score_without_softmax_openai_client(self):
        """Test Score API returning raw logprobs (apply_softmax=False)."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test query",
                "items": [" option1", " option2"],
                "label_token_ids": [319, 350],
                "apply_softmax": False,
            },
            cast_to=dict,
        )

        assert "scores" in response
        scores = response["scores"]
        assert len(scores) == 2

        # Without softmax, scores are logprobs (typically negative)
        for score_list in scores:
            for score in score_list:
                # Logprobs should be finite numbers
                assert isinstance(score, (int, float))
                import math

                assert math.isfinite(score), f"Score {score} is not finite"

    def test_score_response_contains_usage(self):
        """Test that response contains usage information.

        RFC-005: Response should include usage field with token counts.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [" A"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "usage" in response, "Response must contain 'usage' field"
        usage = response["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] >= 0
        assert usage["total_tokens"] >= usage["prompt_tokens"]

    def test_score_response_contains_model(self):
        """Test that response contains model information."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [" A"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "model" in response, "Response must contain 'model' field"
        assert isinstance(response["model"], str)

    def test_score_with_item_first_flag(self):
        """Test Score API with item_first=True ordering."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": " continues the story",
                "items": ["Once upon a time", "In a galaxy far away"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
                "item_first": True,  # Item comes before query
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestScoreOpenAIClientErrors:
    """Test error handling with OpenAI client.

    Validates that errors are returned in OpenAI-compatible format
    that the OpenAI client can properly parse.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up OpenAI client for each test."""
        self.client = create_client()

    def test_empty_items_returns_bad_request(self):
        """Test that empty items returns 400 Bad Request.

        OpenAI client expects errors in specific format.
        RFC-006: Empty items should return 400 with code 'empty_items'.
        """
        from openai import BadRequestError

        with pytest.raises(BadRequestError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "Test",
                    "items": [],  # Empty items
                    "label_token_ids": [123],
                },
                cast_to=dict,
            )

        assert exc_info.value.status_code == 400

    def test_empty_query_returns_bad_request(self):
        """Test that empty query returns 400 Bad Request."""
        from openai import BadRequestError

        with pytest.raises(BadRequestError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "",  # Empty query
                    "items": [" test"],
                    "label_token_ids": [123],
                },
                cast_to=dict,
            )

        assert exc_info.value.status_code == 400

    def test_missing_required_field_returns_bad_request(self):
        """Test that missing required field returns 400."""
        from openai import BadRequestError

        with pytest.raises(BadRequestError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "Test",
                    # Missing 'items' and 'label_token_ids'
                },
                cast_to=dict,
            )

        assert exc_info.value.status_code == 400

    def test_invalid_type_returns_bad_request(self):
        """Test that invalid type returns 400."""
        from openai import BadRequestError

        with pytest.raises(BadRequestError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "Test",
                    "items": "not a list",  # Should be list
                    "label_token_ids": [123],
                },
                cast_to=dict,
            )

        assert exc_info.value.status_code == 400

    def test_negative_token_id_returns_unprocessable(self):
        """Test that negative token ID returns 422 Unprocessable Entity.

        RFC-006: Semantic errors like negative token IDs return 422.
        """
        from openai import UnprocessableEntityError

        with pytest.raises(UnprocessableEntityError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "Test",
                    "items": [" test"],
                    "label_token_ids": [-1],  # Negative token ID
                },
                cast_to=dict,
            )

        assert exc_info.value.status_code == 422


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
class TestScoreOpenAIClientEdgeCases:
    """Edge case tests for OpenAI client compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up OpenAI client for each test."""
        self.client = create_client()

    def test_large_batch_openai_client(self):
        """Test large batch through OpenAI client.

        RFC-005: Validate that large batches work correctly.
        """
        items = [f" item{i}" for i in range(20)]

        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Score these items:",
                "items": items,
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert len(response["scores"]) == 20
        for score_list in response["scores"]:
            assert len(score_list) == 2

    def test_unicode_content_openai_client(self):
        """Test Unicode handling through OpenAI client.

        RFC-005: Unicode in query/items should work correctly.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Translate: こんにちは",
                "items": [" Hello", " Goodbye", " Thanks"],
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 3

    def test_special_characters_openai_client(self):
        """Test special characters in query/items.

        RFC-005: Special characters should be handled correctly.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test with special chars: @#$%^&*()",
                "items": [" option<1>", ' option"2"', " option'3'"],
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 3

    def test_single_item_openai_client(self):
        """Test single item scoring through OpenAI client."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "The answer is",
                "items": [" yes"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 1
        assert len(response["scores"][0]) == 1

    def test_many_label_tokens_openai_client(self):
        """Test many label tokens through OpenAI client."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [" A", " B"],
                "label_token_ids": list(range(100, 150)),  # 50 tokens
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 2
        assert len(response["scores"][0]) == 50

    def test_empty_string_item_openai_client(self):
        """Test empty string item (valid for candidate scoring).

        RFC-005: Empty string items should be valid.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "The answer is",
                "items": [""],  # Empty item is valid
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            cast_to=dict,
        )

        assert "scores" in response
        assert len(response["scores"]) == 1


# =============================================================================
# Token Input Mode Tests
# =============================================================================


@pytest.mark.integration
class TestScoreOpenAIClientTokenMode:
    """Test Score API with token IDs instead of text."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up OpenAI client for each test."""
        self.client = create_client()

    def test_token_input_mode_openai_client(self):
        """Test Score API with token IDs instead of text.

        RFC-005: Token input mode should work through client.
        """
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": [1, 306, 4554, 394],  # Token IDs
                "items": [[311, 278, 7353], [310, 278, 3303]],  # Token ID lists
                "label_token_ids": [311, 310],
                "apply_softmax": False,
            },
            cast_to=dict,
        )

        assert "scores" in response
        scores = response["scores"]
        assert len(scores) == 2


# =============================================================================
# Version Compatibility Tests
# =============================================================================


class TestOpenAIClientVersion:
    """Test OpenAI client version compatibility.

    These tests don't require a running server.
    """

    def test_openai_client_version_minimum(self):
        """Verify minimum OpenAI client version requirement.

        RFC-005: Requires openai>=1.0.0.
        """
        import openai

        version = openai.__version__

        # Parse version
        parts = version.split(".")
        major = int(parts[0])

        assert major >= 1, f"Requires openai>=1.0.0, got {version}"

    def test_openai_client_has_post_method(self):
        """Verify OpenAI client has the post() method we use."""
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="test",
        )

        # Verify the client has the post method
        assert hasattr(client, "post"), "OpenAI client must have post() method"
        assert callable(client.post), "post must be callable"


# =============================================================================
# Alternative Client Tests (httpx/requests)
# =============================================================================


@pytest.mark.integration
class TestScoreAlternativeClients:
    """Test Score API with alternative HTTP clients.

    These tests demonstrate that users can also use httpx or requests
    directly if they prefer not to use the OpenAI client.
    """

    def test_score_with_httpx(self):
        """Test Score API using httpx directly.

        RFC-005: Alternative to OpenAI client using httpx.
        """
        httpx = pytest.importorskip("httpx")

        response = httpx.post(
            f"{get_base_url()}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test query",
                "items": [" A", " B"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            timeout=30.0,
        )

        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == 2

    def test_score_with_requests(self):
        """Test Score API using requests directly.

        RFC-005: Alternative to OpenAI client using requests.
        """
        requests = pytest.importorskip("requests")

        response = requests.post(
            f"{get_base_url()}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test query",
                "items": [" A", " B"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            timeout=30.0,
        )

        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
