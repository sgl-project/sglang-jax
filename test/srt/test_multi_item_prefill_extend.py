"""
Test multi-item scoring with prefill+extend strategy (Workstream B).
"""

import os
import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


class TestMultiItemPrefillExtend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = TEST_MODEL_NAME
        # Initialize engine with prefill+extend enabled and radix cache configuration
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=42,
            mem_fraction_static=0.7,
            # Critical flags for prefill+extend
            multi_item_enable_prefill_extend=True,
            enable_scoring_cache=True,
            # We don't need disable_radix_cache=True here because enable_scoring_cache overrides intent
            # but usually multi-item sets it.
            # Let's set it to False (default) but enable_scoring_cache=True should handle it.
            disable_radix_cache=False,
            multi_item_extend_batch_size=4,
            log_requests=True,
            enable_deterministic_sampling=True,
            precompile_bs_paddings=[1, 2, 4],
            # Standard args
            download_dir="/dev/shm",
            dtype="bfloat16",
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_prefill_extend_flow(self):
        """Test the basic flow of prefill+extend scoring."""
        query = "What is the capital of France?"
        items = ["Paris", "London", "Berlin"]
        # Token IDs for " Yes" (placeholder, depends on tokenizer, usually stable)
        # We will just score some common tokens
        label_token_ids = [100, 200]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items))
        for item_scores in scores:
            self.assertEqual(len(item_scores), len(label_token_ids))
            self.assertAlmostEqual(sum(item_scores), 1.0, places=5)

    def test_prefill_extend_batching(self):
        """Test with more items than extend batch size."""
        query = "Rank these numbers:"
        items = [str(i) for i in range(10)]
        label_token_ids = [15, 16]

        # Batch size is 4 (set in setUpClass)
        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 10)


if __name__ == "__main__":
    unittest.main()
