"""
Shared test utilities for Score API tests.

This module provides common fixtures, helpers, and assertions for testing
the `/v1/score` API endpoint. It reduces code duplication across test files
and ensures consistent patterns.

Design Document: sglang-jax-dev-scripts/rfcs/003-score-api-comprehensive-test-suite.md

Usage:
    from sgl_jax.test.score_test_utils import (
        ScoreTestConfig,
        build_engine,
        get_tokenizer,
        get_single_token_id,
        assert_scores_shape,
        assert_scores_valid,
    )

    config = ScoreTestConfig()
    engine = build_engine(config)
    tokenizer = get_tokenizer(config.model_name)
    token_id = get_single_token_id(tokenizer, " yes")
"""

import math
import os
import unittest
from dataclasses import dataclass, field
from typing import List, Optional

import jax
from transformers import AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import (
    CustomTestCase,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    is_in_ci,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScoreTestConfig:
    """Configuration for Score API tests.

    This dataclass provides a centralized configuration for all Score API
    test parameters. Use this to ensure consistent settings across tests.

    Attributes:
        model_name: HuggingFace model path to use for testing
        device: Device to run tests on ("tpu", "gpu", "cpu")
        tp_size: Tensor parallelism size
        dtype: Model dtype ("bfloat16", "float32")
        download_dir: Directory for model downloads
        random_seed: Random seed for reproducibility
        mem_fraction_static: Fraction of memory for static allocation
        max_running_requests: Maximum concurrent requests
        tolerance: Numerical tolerance for score comparisons (1% default)

    Example:
        config = ScoreTestConfig(model_name="meta-llama/Llama-2-7b")
        engine = build_engine(config)
    """
    model_name: str = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    device: str = "tpu"
    tp_size: int = 1
    dtype: str = "bfloat16"
    download_dir: str = "/dev/shm"
    random_seed: int = 3
    mem_fraction_static: float = 0.6
    max_running_requests: int = 16
    tolerance: float = 0.01  # 1% tolerance for score comparison
    chunked_prefill_size: int = 1024
    precompile_bs_paddings: List[int] = field(default_factory=lambda: [16])
    precompile_token_paddings: List[int] = field(default_factory=lambda: [1024])
    page_size: int = 64


# =============================================================================
# Engine and Tokenizer Factories
# =============================================================================

def build_engine(config: Optional[ScoreTestConfig] = None) -> Engine:
    """Build an Engine instance for Score API testing.

    This factory function creates a properly configured Engine with settings
    optimized for Score API testing. It handles all the boilerplate configuration
    and ensures consistent setup across tests.

    Args:
        config: Optional ScoreTestConfig. Uses defaults if not provided.

    Returns:
        Configured Engine instance ready for scoring.

    Example:
        engine = build_engine()
        scores = engine.score(query="test", items=["a", "b"], ...)
    """
    if config is None:
        config = ScoreTestConfig()

    return Engine(
        model_path=config.model_name,
        trust_remote_code=True,
        tp_size=config.tp_size,
        device=config.device,
        random_seed=config.random_seed,
        node_rank=0,
        mem_fraction_static=config.mem_fraction_static,
        chunked_prefill_size=config.chunked_prefill_size,
        download_dir=config.download_dir,
        dtype=config.dtype,
        precompile_bs_paddings=config.precompile_bs_paddings,
        max_running_requests=config.max_running_requests,
        skip_server_warmup=True,
        attention_backend="fa",
        precompile_token_paddings=config.precompile_token_paddings,
        page_size=config.page_size,
        log_requests=False,
        enable_deterministic_sampling=True,
    )


def get_tokenizer(model_name: Optional[str] = None):
    """Load tokenizer for Score API testing.

    Args:
        model_name: HuggingFace model path. Uses default if not provided.

    Returns:
        AutoTokenizer instance.

    Example:
        tokenizer = get_tokenizer()
        token_ids = tokenizer.encode(" yes", add_special_tokens=False)
    """
    if model_name is None:
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# =============================================================================
# Token ID Helpers
# =============================================================================

def get_single_token_id(tokenizer, text: str) -> int:
    """Get token ID for text that MUST tokenize to exactly one token.

    This is the preferred method for getting label_token_ids in tests.
    It validates that the text tokenizes to exactly one token, which is
    required for the Score API's label_token_ids parameter.

    Args:
        tokenizer: HuggingFace tokenizer instance
        text: Text that should tokenize to exactly one token (e.g., " yes")

    Returns:
        Single token ID as integer.

    Raises:
        ValueError: If text tokenizes to 0 or more than 1 tokens.

    Example:
        yes_id = get_single_token_id(tokenizer, " yes")
        no_id = get_single_token_id(tokenizer, " no")
        label_token_ids = [yes_id, no_id]
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Text '{text}' tokenizes to {len(token_ids)} tokens "
            f"(expected 1): {token_ids}"
        )
    return token_ids[0]


def get_label_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    """Convert multiple label token strings to IDs.

    Convenience wrapper around get_single_token_id for multiple tokens.

    Args:
        tokenizer: HuggingFace tokenizer instance
        tokens: List of token strings (each must tokenize to exactly 1 token)

    Returns:
        List of token IDs.

    Raises:
        ValueError: If any token doesn't tokenize to exactly 1 token.

    Example:
        label_ids = get_label_token_ids(tokenizer, [" yes", " no", " maybe"])
    """
    return [get_single_token_id(tokenizer, token) for token in tokens]


# =============================================================================
# Assertions
# =============================================================================

def assert_scores_shape(
    scores: List[List[float]],
    expected_items: int,
    expected_labels: int,
    test_case: Optional[unittest.TestCase] = None,
) -> None:
    """Assert that scores have the expected shape.

    Validates that the scores matrix has the correct dimensions:
    [num_items x num_labels].

    Args:
        scores: Score matrix from engine.score()
        expected_items: Expected number of items (outer list length)
        expected_labels: Expected number of labels per item (inner list length)
        test_case: Optional TestCase for assertions. Uses plain assert if None.

    Raises:
        AssertionError: If shape doesn't match expected.

    Example:
        scores = engine.score(query="Q", items=["a", "b"], label_token_ids=[1, 2, 3])
        assert_scores_shape(scores, expected_items=2, expected_labels=3)
    """
    if test_case:
        test_case.assertEqual(
            len(scores),
            expected_items,
            f"Expected {expected_items} items, got {len(scores)}"
        )
        for i, score_list in enumerate(scores):
            test_case.assertEqual(
                len(score_list),
                expected_labels,
                f"Item {i}: expected {expected_labels} labels, got {len(score_list)}"
            )
    else:
        assert len(scores) == expected_items, \
            f"Expected {expected_items} items, got {len(scores)}"
        for i, score_list in enumerate(scores):
            assert len(score_list) == expected_labels, \
                f"Item {i}: expected {expected_labels} labels, got {len(score_list)}"


def assert_scores_valid(
    scores: List[List[float]],
    apply_softmax: bool = True,
    test_case: Optional[unittest.TestCase] = None,
    tolerance: float = 1e-6,
) -> None:
    """Assert that scores are valid probabilities or logprobs.

    Validates score values based on the apply_softmax mode:
    - If apply_softmax=True: scores should be in [0, 1] and sum to 1.0
    - If apply_softmax=False: scores should be valid logprobs (finite, typically negative)

    Args:
        scores: Score matrix from engine.score()
        apply_softmax: Whether softmax was applied (affects validation)
        test_case: Optional TestCase for assertions. Uses plain assert if None.
        tolerance: Tolerance for sum-to-one check (default 1e-6)

    Raises:
        AssertionError: If scores don't meet validity criteria.

    Example:
        scores = engine.score(..., apply_softmax=True)
        assert_scores_valid(scores, apply_softmax=True)
    """
    for i, score_list in enumerate(scores):
        for j, score in enumerate(score_list):
            # Check score is finite (no NaN or Inf)
            if test_case:
                test_case.assertTrue(
                    math.isfinite(score),
                    f"Score[{i}][{j}] is not finite: {score}"
                )
            else:
                assert math.isfinite(score), \
                    f"Score[{i}][{j}] is not finite: {score}"

            if apply_softmax:
                # Softmax mode: scores should be probabilities in [0, 1]
                if test_case:
                    test_case.assertGreaterEqual(
                        score, 0.0,
                        f"Score[{i}][{j}] = {score} is negative (expected >= 0)"
                    )
                    test_case.assertLessEqual(
                        score, 1.0,
                        f"Score[{i}][{j}] = {score} exceeds 1 (expected <= 1)"
                    )
                else:
                    assert score >= 0.0, \
                        f"Score[{i}][{j}] = {score} is negative (expected >= 0)"
                    assert score <= 1.0, \
                        f"Score[{i}][{j}] = {score} exceeds 1 (expected <= 1)"
            # For logprobs (apply_softmax=False), values are typically negative
            # but we don't enforce this as -inf is valid for impossible tokens

        if apply_softmax:
            # Check probabilities sum to 1.0
            total = sum(score_list)
            if test_case:
                test_case.assertAlmostEqual(
                    total, 1.0, places=6,
                    msg=f"Item {i}: scores sum to {total}, expected 1.0"
                )
            else:
                assert abs(total - 1.0) < tolerance, \
                    f"Item {i}: scores sum to {total}, expected 1.0"


def assert_scores_match(
    expected: List[List[float]],
    actual: List[List[float]],
    tolerance: float = 0.01,
    test_case: Optional[unittest.TestCase] = None,
    case_name: str = "",
) -> None:
    """Assert that two score matrices match within tolerance.

    Used for comparing SGLang scores against HuggingFace reference scores.

    Args:
        expected: Expected scores (e.g., from HuggingFace reference)
        actual: Actual scores (e.g., from SGLang)
        tolerance: Maximum allowed absolute difference (default 1%)
        test_case: Optional TestCase for assertions
        case_name: Name for error messages

    Raises:
        AssertionError: If scores don't match within tolerance.
    """
    prefix = f"[{case_name}] " if case_name else ""

    if test_case:
        test_case.assertEqual(
            len(expected), len(actual),
            f"{prefix}Expected {len(expected)} items, got {len(actual)}"
        )
    else:
        assert len(expected) == len(actual), \
            f"{prefix}Expected {len(expected)} items, got {len(actual)}"

    for i, (exp_list, act_list) in enumerate(zip(expected, actual)):
        if test_case:
            test_case.assertEqual(
                len(exp_list), len(act_list),
                f"{prefix}Item {i}: expected {len(exp_list)} scores, got {len(act_list)}"
            )
        else:
            assert len(exp_list) == len(act_list), \
                f"{prefix}Item {i}: expected {len(exp_list)} scores, got {len(act_list)}"

        for j, (exp, act) in enumerate(zip(exp_list, act_list)):
            diff = abs(exp - act)
            msg = (
                f"{prefix}Item {i}, Label {j}: "
                f"expected {exp:.6f}, got {act:.6f}, diff {diff:.4f} > {tolerance}"
            )
            if test_case:
                test_case.assertLessEqual(diff, tolerance, msg)
            else:
                assert diff <= tolerance, msg


# =============================================================================
# Test Decorators / Skip Conditions
# =============================================================================

def should_run_hf_reference() -> bool:
    """Check if HuggingFace reference tests should run.

    HF reference tests load the full model in PyTorch, which is slow and
    memory-intensive. They're disabled by default and enabled via environment
    variable for thorough validation.

    Returns:
        True if SGLANG_JAX_RUN_HF_REFERENCE=1 is set.
    """
    return os.environ.get("SGLANG_JAX_RUN_HF_REFERENCE", "0") == "1"


def skip_if_no_hf_reference():
    """Skip test if HuggingFace reference validation is not enabled.

    Use as a decorator or call at the start of a test method.

    Example:
        def test_hf_consistency(self):
            skip_if_no_hf_reference()
            # ... test code that uses HF reference ...
    """
    import pytest
    if not should_run_hf_reference():
        pytest.skip("Set SGLANG_JAX_RUN_HF_REFERENCE=1 to run HF reference tests")


def skip_if_no_multidevice():
    """Skip test if multiple devices are not available.

    Use for tests that require multi-device/multi-host configurations.

    Example:
        def test_sharded_scoring(self):
            skip_if_no_multidevice()
            # ... test code that requires multiple devices ...
    """
    import pytest
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(f"Test requires multiple devices, found {len(devices)}")


def skip_if_cpu():
    """Skip test if running on CPU.

    Some tests require TPU/GPU for meaningful results.

    Example:
        def test_performance(self):
            skip_if_cpu()
            # ... performance test ...
    """
    import pytest
    devices = jax.devices()
    if devices and devices[0].platform == "cpu":
        pytest.skip("Test requires TPU or GPU, running on CPU")


# =============================================================================
# HuggingFace Reference Helpers
# =============================================================================

def compute_hf_reference_scores(
    model_name: str,
    query: str,
    items: List[str],
    label_token_ids: List[int],
    item_first: bool = False,
) -> List[List[float]]:
    """Compute reference scores using HuggingFace model.

    This function loads a HuggingFace model and computes scores for comparison
    against SGLang's implementation. It's used for validation tests.

    Note: This is slow and memory-intensive. Only use for validation, not
    regular testing.

    Args:
        model_name: HuggingFace model path
        query: Query text
        items: List of item texts
        label_token_ids: Token IDs to score
        item_first: If True, construct prompts as item+query

    Returns:
        Score matrix [num_items x num_labels] with softmax probabilities.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    try:
        scores = []
        for item in items:
            full_text = f"{item}{query}" if item_first else f"{query}{item}"
            inputs = tokenizer(full_text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                last_token_logits = outputs.logits[0, -1]

            target_logits = last_token_logits[label_token_ids]
            target_probs = torch.softmax(target_logits, dim=-1)
            probs = [target_probs[i].item() for i in range(len(label_token_ids))]
            scores.append(probs)

        return scores
    finally:
        del model
        del tokenizer


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_test_items(count: int, prefix: str = " item") -> List[str]:
    """Generate test items for batch testing.

    Args:
        count: Number of items to generate
        prefix: Prefix for each item (default " item")

    Returns:
        List of item strings.

    Example:
        items = generate_test_items(10)
        # [" item0", " item1", ..., " item9"]
    """
    return [f"{prefix}{i}" for i in range(count)]


# =============================================================================
# Base Test Class
# =============================================================================

class ScoreAPITestCase(CustomTestCase):
    """Base test class for Score API tests.

    Provides common setup/teardown logic and helper methods for Score API
    testing. Extend this class for new Score API test modules.

    Class Attributes:
        config: ScoreTestConfig for this test class
        engine: Shared Engine instance (initialized in setUpClass)
        tokenizer: Shared tokenizer (initialized in setUpClass)

    Example:
        class TestMyScoreFeature(ScoreAPITestCase):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                # Additional setup...

            def test_my_feature(self):
                scores = self.engine.score(...)
                assert_scores_valid(scores, test_case=self)
    """
    config: ScoreTestConfig = None
    engine: Engine = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        """Initialize shared engine and tokenizer."""
        cls.config = ScoreTestConfig()
        cls.engine = build_engine(cls.config)
        cls.tokenizer = get_tokenizer(cls.config.model_name)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def get_token_id(self, text: str) -> int:
        """Get single token ID for text. Convenience wrapper."""
        return get_single_token_id(self.tokenizer, text)

    def get_token_ids(self, tokens: List[str]) -> List[int]:
        """Get multiple token IDs. Convenience wrapper."""
        return get_label_token_ids(self.tokenizer, tokens)
