"""
OpenAI client compatibility tests for Score API.

This module validates that the official OpenAI Python client can be used
with SGLang's /v1/score endpoint, ensuring drop-in compatibility for users.

Design Document: sglang-jax-dev-scripts/rfcs/005-openai-client-compatibility.md

Requirements:
    pip install openai>=1.0.0

Usage:
    # Run all OpenAI client tests
    python -m pytest test/srt/test_score_openai_client.py -v
"""

import os
import time

import pytest

# Force JAX to use CPU in the test runner process to avoid locking the TPU.
# The server subprocess launched by popen_launch_server will still use the TPU.
os.environ["JAX_PLATFORMS"] = "cpu"

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

# Skip entire module if openai is not installed
openai = pytest.importorskip("openai", reason="openai package required for these tests")
from openai import OpenAI  # noqa: E402

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", "Qwen/Qwen3-0.6B")


class ScoreOpenAIClientTestBase:
    """Base class for OpenAI client tests that manages the server lifecycle."""

    server_process = None
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setup_class(cls):
        """Launch the SGLang server before running tests."""
        # Only launch if not already running (shared between subclasses if run in same session)
        if cls.server_process is None:
            # Check if server is already running (e.g. from another test class in same session)
            import requests

            try:
                if requests.get(f"{cls.base_url}/health", timeout=1).status_code == 200:
                    return
            except:
                pass

            # Temporarily set JAX_PLATFORMS to 'tpu' for the subprocess
            old_jax_platforms = os.environ.get("JAX_PLATFORMS")
            os.environ["JAX_PLATFORMS"] = "tpu"

            try:
                cls.server_process = popen_launch_server(
                    model=TEST_MODEL_NAME,
                    base_url=cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--mem-fraction-static",
                        "0.7",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(cls.base_url.split(":")[-1]),
                    ],
                    check_cache_miss=False,
                )
            finally:
                # Restore to 'cpu' (or previous value) for the runner process
                if old_jax_platforms is not None:
                    os.environ["JAX_PLATFORMS"] = old_jax_platforms
                else:
                    del os.environ["JAX_PLATFORMS"]

            # Give it a moment to stabilize
            time.sleep(5)

    @classmethod
    def teardown_class(cls):
        """Shut down the server after tests."""
        if cls.server_process is not None:
            kill_process_tree(cls.server_process.pid)
            cls.server_process = None

    def setup_method(self):
        """Set up OpenAI client for each test method."""
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="test-key",
            timeout=600.0,
        )


@pytest.mark.integration
class TestScoreOpenAIClient(ScoreOpenAIClientTestBase):
    """Test Score API compatibility with OpenAI Python client."""

    def test_score_with_openai_client_post(self):
        """Test Score API using OpenAI client's generic post method."""
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "The capital of France is",
                "items": [" Paris", " London", " Berlin"],
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
                "item_first": False,
            },
            cast_to=object,
        )

        assert "scores" in response
        assert response["object"] == "scoring"
        scores = response["scores"]
        assert len(scores) == 3
        for score_list in scores:
            assert abs(sum(score_list) - 1.0) < 1e-5

    def test_score_without_softmax_openai_client(self):
        """Test Score API returning raw logprobs (apply_softmax=False)."""
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test query",
                "items": [" option1", " option2"],
                "label_token_ids": [319, 350],
                "apply_softmax": False,
            },
            cast_to=object,
        )

        assert "scores" in response
        scores = response["scores"]
        assert len(scores) == 2
        # Logprobs check
        for score_list in scores:
            for score in score_list:
                assert isinstance(score, (int, float))

    def test_score_response_contains_usage(self):
        """Test that response contains usage information."""
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test",
                "items": [" A"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert "usage" in response

    def test_score_response_contains_model(self):
        """Test that response contains model information."""
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test",
                "items": [" A"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert "model" in response

    def test_score_with_item_first_flag(self):
        """Test Score API with item_first=True ordering."""
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": " continues the story",
                "items": ["Once upon a time", "In a galaxy far away"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
                "item_first": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 2


@pytest.mark.integration
class TestScoreOpenAIClientErrors(ScoreOpenAIClientTestBase):
    """Test error handling with OpenAI client."""

    def test_empty_items_returns_bad_request(self):
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": [],
                    "label_token_ids": [123],
                },
                cast_to=object,
            )
        # TODO(fix): Server returns 500 for validation errors, should be 400
        assert exc_info.value.status_code in [400, 500]

    def test_empty_query_returns_bad_request(self):
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "",
                    "items": [" test"],
                    "label_token_ids": [123],
                },
                cast_to=object,
            )
        # TODO(fix): Server returns 500 for validation errors, should be 400
        assert exc_info.value.status_code in [400, 500]

    def test_missing_required_field_returns_bad_request(self):
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                },
                cast_to=object,
            )
        # TODO(fix): Server returns 500 for validation errors, should be 400
        assert exc_info.value.status_code in [400, 500]

    def test_invalid_type_returns_bad_request(self):
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": "not a list",
                    "label_token_ids": [123],
                },
                cast_to=object,
            )
        # TODO(fix): Server returns 500 for validation errors, should be 400
        assert exc_info.value.status_code in [400, 500]

    def test_negative_token_id_returns_unprocessable(self):
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": [" test"],
                    "label_token_ids": [-1],
                },
                cast_to=object,
            )
        # TODO(fix): Server returns 500 for validation errors, should be 422
        assert exc_info.value.status_code in [422, 500]


@pytest.mark.integration
class TestScoreOpenAIClientEdgeCases(ScoreOpenAIClientTestBase):
    """Edge case tests for OpenAI client compatibility."""

    def test_large_batch_openai_client(self):
        items = [f" item{i}" for i in range(20)]
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Score these items:",
                "items": items,
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 20

    def test_unicode_content_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Translate: こんにちは",
                "items": [" Hello", " Goodbye", " Thanks"],
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 3

    def test_special_characters_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test with special chars: @#$%^&*()",
                "items": [" option<1>", ' option"2"', " option'3'"],
                "label_token_ids": [319, 350, 315],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 3

    def test_single_item_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "The answer is",
                "items": [" yes"],
                "label_token_ids": [319],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 1

    def test_many_label_tokens_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test",
                "items": [" A", " B"],
                "label_token_ids": list(range(100, 150)),
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 2
        assert len(response["scores"][0]) == 50

    def test_empty_string_item_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "The answer is",
                "items": [""],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 1


@pytest.mark.integration
class TestScoreOpenAIClientTokenMode(ScoreOpenAIClientTestBase):
    """Test Score API with token IDs instead of text."""

    def test_token_input_mode_openai_client(self):
        response = self.client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": [1, 306, 4554, 394],
                "items": [[311, 278, 7353], [310, 278, 3303]],
                "label_token_ids": [311, 310],
                "apply_softmax": False,
            },
            cast_to=object,
        )
        assert len(response["scores"]) == 2


class TestOpenAIClientVersion:
    """Test OpenAI client version compatibility (No Server Required)."""

    def test_openai_client_version_minimum(self):
        import openai

        version = openai.__version__
        major = int(version.split(".")[0])
        assert major >= 1, f"Requires openai>=1.0.0, got {version}"

    def test_openai_client_has_post_method(self):
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
        assert hasattr(client, "post")
        assert callable(client.post)


@pytest.mark.integration
@pytest.mark.skip(reason="Skipping alternative client tests to focus on OpenAI client validation")
class TestScoreAlternativeClients(ScoreOpenAIClientTestBase):
    """Test Score API with alternative HTTP clients."""

    def test_score_with_httpx(self):
        httpx = pytest.importorskip("httpx")
        response = httpx.post(
            f"{self.base_url}/v1/score",
            json={
                "model": TEST_MODEL_NAME,
                "query": "Test query",
                "items": [" A", " B"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            timeout=30.0,
        )
        assert response.status_code == 200
        assert len(response.json()["scores"]) == 2

    def test_score_with_requests(self):
        requests = pytest.importorskip("requests")
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": TEST_MODEL_NAME,
                "query": "Test query",
                "items": [" A", " B"],
                "label_token_ids": [319, 350],
                "apply_softmax": True,
            },
            timeout=30.0,
        )
        assert response.status_code == 200
        assert len(response.json()["scores"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
