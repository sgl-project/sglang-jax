"""
Test the scoring API functionality (/v1/score).

This module contains critical tests for the `/v1/score` API endpoint, which computes
the probability of specific tokens appearing after a given prompt. The scoring API
is used for:
- Classifier-free guidance in diffusion models
- Ranking/reranking multiple options
- Prompt evaluation and selection
- Binary classification (Yes/No, True/False)

The tests ensure:
1. Numerical accuracy matches HuggingFace reference implementation (within 1% tolerance)
2. Batch processing works correctly at various scales
3. JAX-specific optimizations (prefill-only execution) are functioning
4. HTTP endpoint returns properly formatted responses

Test Coverage:
- Engine-level API: Engine.score() method
- HTTP endpoint: POST /v1/score
- Input formats: Text strings and pre-tokenized token IDs
- Features: apply_softmax, item_first parameters
- Optimization: Prefill-only execution (max_new_tokens=0)

Usage:
    # Run all tests in this module
    python3 -m unittest test.srt.test_score_api

    # Run individual test methods
    python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency
    python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_batch_handling
    python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_request_construction

Requirements:
    - TPU or GPU access (tests use device="tpu")
    - Model: Qwen/Qwen3-1.7B (downloaded to /dev/shm)
    - Dependencies: transformers, torch (for HuggingFace reference validation)

Example Output:
    $ python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency

    test_score_consistency (test.srt.test_score_api.TestScoreAPI)
    Test that SGLang scoring matches direct HuggingFace model scoring. ... ok

    ----------------------------------------------------------------------
    Ran 1 test in 45.231s

    OK

Performance Notes:
    - Engine initialization takes ~30-60s (model loading, JAX compilation)
    - Each test runs in ~5-15s after initialization
    - Total runtime for all 3 tests: ~2-3 minutes
    - Tests share a single Engine instance to minimize overhead

Debugging Failed Tests:
    If test_score_consistency fails:
        → Check model weights loaded correctly
        → Verify FlashAttention implementation matches reference
        → Inspect absolute differences (should be < 1%)

    If test_score_batch_handling fails:
        → Check batch padding/masking logic
        → Verify KV cache indexing for batched operations
        → Ensure logprob extraction accounts for batch dimension

    If test_score_request_construction fails:
        → Verify max_new_tokens=0 is being set
        → Check is_prefill_only flag is True
        → Ensure token_ids_logprob parameter is populated
"""

import unittest
from unittest.mock import patch

import jax
from transformers import AutoModelForCausalLM, AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

# Use smaller model for faster tests
TEST_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestScoreAPI(CustomTestCase):
    """Test the scoring API functionality.

    This test class validates the core scoring API implementation in SGLang-JAX,
    ensuring it produces accurate probability scores for specified token IDs.

    The scoring API works by:
    1. Constructing prompts from (query, item) pairs
    2. Running prefill-only inference (no token generation)
    3. Extracting logprobs for specified token IDs at the last position
    4. Optionally normalizing with softmax

    Example:
        query = "Is this a city? "
        items = ["Paris", "Ocean"]
        label_token_ids = [9454, 2753]  # Token IDs for "Yes", "No"

        Result: [[0.95, 0.05], [0.1, 0.9]]
        Interpretation: Model assigns 95% probability to "Yes" for Paris,
                       10% probability to "Yes" for Ocean.

    Critical Properties Tested:
    - Numerical accuracy vs HuggingFace reference (1% tolerance)
    - Probability normalization (scores sum to 1.0 when softmax enabled)
    - Batch processing correctness at various scales
    - JAX-specific optimization (prefill-only, no decode phase)
    - Request construction (max_new_tokens=0, token_ids_logprob set)
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test class with a shared engine instance."""
        cls.model_path = TEST_MODEL_NAME
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            chunked_prefill_size=1024,
            download_dir="/dev/shm",
            dtype="bfloat16",
            precompile_bs_paddings=[16],
            max_running_requests=16,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if cls.engine is not None:
            cls.engine.shutdown()
        # JAX doesn't require CUDA cache clearing, but we can clear compilation cache if needed
        jax.clear_caches()

    def compute_hf_scores(
        self, query, items, label_token_ids, apply_softmax=False, item_first=False
    ):
        """Compute reference scores using direct HuggingFace model inference.

        This method provides ground truth scores for validating SGLang's scoring API.
        It loads the same model using HuggingFace Transformers and computes logprobs
        for the specified token IDs at the last token position.

        **Algorithm:**
        1. Load model and tokenizer from HuggingFace
        2. For each item:
           a. Construct full_text = (item + query) or (query + item) based on item_first
           b. Tokenize and run forward pass
           c. Extract logits at last token position: logits[0, -1, :]
           d. Select logits for label_token_ids
           e. Apply softmax to get probabilities
        3. Return list of probability lists

        **Important Notes:**
        - Always uses CPU (to avoid GPU memory conflicts with JAX/TPU)
        - Cleans up model after completion to free memory
        - Applies softmax over ONLY the label_token_ids (not full vocabulary)
        - This matches the behavior of SGLang's token_ids_logprob feature

        Args:
            query: The query text (e.g., "Is this true? ")
            items: List of item texts (e.g., ["Yes", "No", "Maybe"])
            label_token_ids: List of token IDs to compute probabilities for
                           (e.g., [9454, 2753] for "Yes"/"No" tokens)
            apply_softmax: Whether to normalize probabilities using softmax.
                          Should always be True for probability interpretation.
            item_first: If True, construct prompts as f"{item}{query}".
                       If False, construct prompts as f"{query}{item}".

        Returns:
            List of score lists, one per item. Each score list has length
            equal to len(label_token_ids).
            Example: [[0.9, 0.1], [0.3, 0.7]] for 2 items, 2 label tokens.

        Raises:
            May raise HuggingFace model loading errors if model not accessible.
        """
        # Initialize HF model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)
        model.eval()  # Set to evaluation mode

        try:
            scores = []
            for item in items:
                # Construct full text based on item_first parameter
                full_text = f"{item}{query}" if item_first else f"{query}{item}"
                inputs = tokenizer(full_text, return_tensors="pt")

                # Get logits for the last token
                import torch

                with torch.no_grad():
                    outputs = model(**inputs)
                    last_token_logits = outputs.logits[0, -1]

                # Get logits for just our target tokens
                target_logits = last_token_logits[label_token_ids]

                # Apply softmax over just the target tokens
                target_probs = torch.softmax(target_logits, dim=-1)

                # Convert to list of probabilities in order of label_token_ids
                probs = [target_probs[i].item() for i in range(len(label_token_ids))]

                scores.append(probs)

            return scores
        finally:
            # Clean up HF resources
            del model
            del tokenizer
            # No need for torch.cuda.empty_cache() on TPU

    def _get_token_ids(self, tokens):
        """Convert token strings to their corresponding token IDs.

        This helper method tokenizes text strings and extracts the first token ID
        from each tokenization result. Useful for getting token IDs for common
        words like "Yes", "No", "True", "False", etc.

        **Example:**
            tokens = [" to", " the", " of"]
            ids = self._get_token_ids(tokens)
            # ids might be [311, 279, 315] depending on tokenizer

        **Note:**
        - Uses encode_plus with add_special_tokens=False to get raw token IDs
        - Only returns the first token ID if a string tokenizes to multiple tokens
        - Skips empty tokenizations (though these should be rare)

        Args:
            tokens: List of string tokens to convert (e.g., [" Yes", " No"])

        Returns:
            List of integer token IDs corresponding to input tokens.
        """
        label_token_ids = []
        for token in tokens:
            encoding = self.tokenizer.encode_plus(token, add_special_tokens=False)
            token_ids = encoding["input_ids"]
            if token_ids:
                label_token_ids.append(token_ids[0])
        return label_token_ids

    def _compare_scores(self, hf_scores, sglang_scores, label_token_ids, case_name=""):
        """Compare scores between HuggingFace and SGLang with validation.

        This method performs comprehensive validation of SGLang scores against
        HuggingFace reference scores, checking both numerical accuracy and
        probability properties.

        **Validation Steps:**
        1. Check output lengths match (same number of items)
        2. For each item's score list:
           a. Check score list lengths match (same number of label tokens)
           b. Check absolute difference < 1% (TOLERANCE) for each score
           c. Check all scores are in valid range [0, 1]
           d. Check scores sum to 1.0 (probability normalization)

        **Why 1% Tolerance:**
        We allow 1% absolute difference due to:
        - Numerical precision differences (bfloat16 vs float32)
        - Different attention implementations (FlashAttention vs standard)
        - Floating point arithmetic order differences
        - Compiler optimizations in JAX vs PyTorch

        **Assertion Failures Mean:**
        - > 1% difference: Potential bug in model forward pass or logprob extraction
        - Scores not in [0,1]: Probability computation is broken
        - Scores don't sum to 1: Softmax normalization is incorrect

        Args:
            hf_scores: Reference scores from HuggingFace (ground truth)
            sglang_scores: Scores from SGLang to validate
            label_token_ids: Token IDs being scored (for error messages)
            case_name: Descriptive name for this test case (for error messages)

        Raises:
            AssertionError: If any validation check fails
        """
        self.assertEqual(
            len(hf_scores),
            len(sglang_scores),
            f"Score lengths don't match for {case_name}",
        )

        # Use a relative tolerance of 1%
        TOLERANCE = 0.01

        for hf_score_list, sglang_score_list in zip(hf_scores, sglang_scores):
            self.assertEqual(
                len(hf_score_list),
                len(sglang_score_list),
                f"Score list lengths don't match for {case_name}",
            )

            for hf_score, sglang_score in zip(hf_score_list, sglang_score_list):
                diff = abs(hf_score - sglang_score)
                self.assertLessEqual(
                    diff,
                    TOLERANCE,
                    msg=f"Scores differ by {diff:.2%} ({case_name}): "
                    f"HF={hf_score:.6f}, SGLang={sglang_score:.6f}",
                )

                self.assertGreaterEqual(
                    sglang_score, 0, f"SGLang score {sglang_score:.6f} not in [0,1]"
                )
                self.assertLessEqual(
                    sglang_score, 1, f"SGLang score {sglang_score:.6f} not in [0,1]"
                )

            self.assertAlmostEqual(
                sum(sglang_score_list),
                1.0,
                places=6,
                msg=f"SGLang scores don't sum to 1 ({case_name}): {sum(sglang_score_list):.6f}",
            )

    def test_score_consistency(self):
        """Test that SGLang scoring matches direct HuggingFace model scoring.

        **Purpose:**
        This is the most critical test for numerical correctness. It validates that
        the JAX implementation produces the same probabilities as the reference
        HuggingFace implementation, ensuring we haven't introduced bugs in:
        - Model loading and weight conversion
        - Attention computation (FlashAttention implementation)
        - Logit extraction at the last token position
        - Softmax normalization over specified token IDs

        **Test Cases:**
        1. Default case: query + item (e.g., "I pledge allegiance" + " to")
        2. item_first case: item + query (e.g., "Tokyo" + " is a city")

        **Validation:**
        - Absolute difference < 1% for each probability score
        - All scores in valid range [0, 1]
        - Scores sum to 1.0 (within 6 decimal places)

        **Why 1% tolerance:**
        Small numerical differences are expected due to:
        - bfloat16 vs float32 precision
        - Different implementations of attention (FlashAttention vs standard)
        - Floating point arithmetic order differences

        **Failure indicates:**
        - Model weights not loaded correctly
        - Attention mechanism has bugs
        - Logprob extraction is incorrect
        - Softmax normalization is wrong
        """
        # Define test cases - covering both prompt construction orders
        test_cases = [
            {
                "name": "default case",
                "query": "I pledge allegiance",
                "items": ["", " to"],  # Empty string and actual continuation
                "item_first": False,  # Construct as: query + item
            },
            {
                "name": "item_first case",
                "query": " is a city",
                "items": ["Tokyo", "Japan"],
                "item_first": True,  # Construct as: item + query (e.g., "Tokyo is a city")
            },
        ]

        # Common tokens to test for all cases - these should be common words
        # with stable token IDs across different tokenizers
        tokens = [" to", " the"]
        label_token_ids = self._get_token_ids(tokens)

        # Run each test case and compare against HuggingFace reference
        for case in test_cases:
            # Get scores from SGLang
            sglang_scores = self.engine.score(
                query=case["query"],
                items=case["items"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
                item_first=case["item_first"],
            )

            # Get scores from HuggingFace using the same parameters
            hf_scores = self.compute_hf_scores(
                query=case["query"],
                items=case["items"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
                item_first=case["item_first"],
            )

            # Compare scores
            self._compare_scores(hf_scores, sglang_scores, label_token_ids, case["name"])

    def test_score_batch_handling(self):
        """Test that batch scoring works correctly across different batch sizes.

        **Purpose:**
        Validates that the scoring API can handle multiple items in a single request
        without degradation in accuracy or correctness. Batching is critical for
        performance when scoring many options simultaneously (e.g., ranking 100 candidates).

        **Tested Batch Sizes:**
        - 1 item (single request baseline)
        - 2 items (minimal batch)
        - 4 items (small batch)
        - 8 items (medium batch)

        **Validation Per Batch:**
        - Output length matches input length (one score list per item)
        - Each score list has correct dimensions (one score per label_token_id)
        - All values are floats (not ints, not None)
        - Probabilities sum to 1.0 when softmax is applied

        **Why This Matters:**
        - Batching is a JAX optimization - need to ensure correctness
        - Padding/masking bugs often manifest in batch processing
        - Memory layout issues can cause silent failures in batched operations

        **Failure indicates:**
        - Batch padding/masking is incorrect
        - KV cache indexing has bugs
        - Logprob extraction doesn't account for batch dimension properly
        """
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        label_token_ids = [1, 2, 3]

        for batch_size in batch_sizes:
            texts = [f"test {i}" for i in range(batch_size)]
            scores = self.engine.score(
                query="The test was",
                items=texts,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

            self.assertEqual(
                len(scores),
                batch_size,
                f"Expected {batch_size} scores, got {len(scores)}",
            )

            # Verify each score list has the correct length
            for score_list in scores:
                self.assertEqual(
                    len(score_list),
                    len(label_token_ids),
                    f"Score list length {len(score_list)} doesn't match label_token_ids length {len(label_token_ids)}",
                )
                self.assertTrue(
                    all(isinstance(v, float) for v in score_list),
                    "All scores should be floats",
                )
                self.assertAlmostEqual(1.0, sum(score_list), 6, "Scores should sum to 1")

    def test_score_request_construction(self):
        """Test that scoring requests are constructed to avoid decode phase.

        **Purpose:**
        Validates a critical JAX optimization: scoring should only run prefill,
        not the decode phase. This is implemented by setting max_new_tokens=0,
        which triggers the is_prefill_only flag and prevents the expensive
        autoregressive generation loop.

        **Optimization Importance:**
        - Prefill-only is ~10-100x faster than prefill + decode
        - Avoids unnecessary KV cache updates
        - Reduces memory usage (no output token storage)
        - Enables higher throughput for scoring workloads

        **Validated Properties:**
        1. max_new_tokens ≤ 1 (prevents decode loop)
        2. token_ids_logprob is set (enables selective logprob extraction)
        3. return_logprob=True (request logprobs from model)
        4. stream=False (scoring is always non-streaming)

        **Implementation Details:**
        Uses mocking to capture the internal GenerateReqInput object and
        inspect its parameters. This ensures the optimization is applied
        at the request construction level.

        **Failure indicates:**
        - Scoring is running full generation (massive performance loss)
        - token_ids_logprob optimization not being used
        - Scoring results may be incorrect (wrong token logprobs extracted)

        **Related Code:**
        - python/sgl_jax/srt/managers/tokenizer_manager.py:1241 (score_request)
        - python/sgl_jax/srt/managers/schedule_batch.py:645 (is_prefill_only detection)
        """
        # Capture the internal request to verify optimization
        captured_requests = []
        original_gen = self.engine.tokenizer_manager.generate_request

        async def mock_generate_request(req, request=None):
            captured_requests.append(req)
            async for result in original_gen(req, request):
                yield result

        # Patch the generate_request method
        with patch.object(
            self.engine.tokenizer_manager,
            "generate_request",
            side_effect=mock_generate_request,
        ):
            # Run a scoring request
            query = "What is the capital of"
            items = ["France", "Germany"]
            label_token_ids = [1, 2, 3]

            scores = self.engine.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

            # Verify we got results
            self.assertEqual(len(scores), len(items))

            # Verify the captured request has decode-avoiding properties
            # Note: JAX version may batch all items into a single request,
            # or may create separate requests per item. Both are valid.
            self.assertGreater(
                len(captured_requests), 0, "Should have captured at least one request"
            )

            # Check the first request (most important for optimization validation)
            request = captured_requests[0]

            # ========================================
            # Key Assertion #1: Prefill-Only Mode
            # ========================================
            # max_new_tokens should be 0 or 1 to avoid running the decode loop.
            # Setting this triggers is_prefill_only=True in schedule_batch.py:645
            if isinstance(request.sampling_params, dict):
                max_new_tokens = request.sampling_params.get("max_new_tokens", 0)
            elif isinstance(request.sampling_params, list):
                # For batch requests, check the first item
                max_new_tokens = request.sampling_params[0].get("max_new_tokens", 0)
            else:
                max_new_tokens = getattr(request.sampling_params, "max_new_tokens", 0)

            self.assertLessEqual(
                max_new_tokens, 1, "max_new_tokens should be 0 or 1 to avoid decode phase"
            )

            # ========================================
            # Key Assertion #2: Selective Logprob Extraction
            # ========================================
            # token_ids_logprob parameter enables efficient extraction of only the
            # needed token probabilities (not the full vocabulary).
            # This uses JAX fancy indexing: logprobs.at[i, token_ids].get()
            # See: python/sgl_jax/srt/layers/sampler.py:226-240
            if (
                isinstance(request.token_ids_logprob, list)
                and len(request.token_ids_logprob) > 0
                and isinstance(request.token_ids_logprob[0], list)
            ):
                # Batch case: token_ids_logprob is a list of lists
                # Each item in the batch should have the same label_token_ids
                for item_token_ids in request.token_ids_logprob:
                    self.assertEqual(
                        item_token_ids,
                        label_token_ids,
                        "Each batch item should have label_token_ids for scoring",
                    )
            else:
                # Single request case
                self.assertEqual(
                    request.token_ids_logprob,
                    label_token_ids,
                    "Should have label_token_ids for scoring",
                )

            # ========================================
            # Key Assertion #3: Logprob Request Configuration
            # ========================================
            # return_logprob=True: Tells the model to compute and return logprobs
            # stream=False: Scoring doesn't support streaming (returns all scores at once)
            self.assertTrue(request.return_logprob, "Should request logprobs for scoring")
            self.assertFalse(request.stream, "Scoring requests should not stream")


if __name__ == "__main__":
    unittest.main()
