"""
Regression checks for multi-item scoring correctness and performance behavior.
"""

import time
import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase

TEST_MODEL_NAME = "/models/Qwen/Qwen3-0.6B"
DELIMITER_TOKEN_ID = 151643
LABEL_TOKEN_IDS = [9834, 902]
QUERY_IDS = [1957, 1437, 25975, 25]
BASE_ITEMS = [
    [358, 2948, 419, 1985, 13],
    [1096, 374, 17478, 323, 38123, 13],
    [1084, 4278, 438, 3601, 13],
    [56938, 4271, 323, 4937, 9691, 13],
    [2806, 5802, 279, 3349, 13],
]


def _max_abs_diff(vec_a: list[float], vec_b: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(vec_a, vec_b, strict=True))


class TestMultiItemRegression(CustomTestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_MODEL_NAME,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            chunked_prefill_size=-1,
            download_dir="/dev/shm",
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
            disable_radix_cache=True,
            multi_item_scoring_delimiter=DELIMITER_TOKEN_ID,
            multi_item_scoring_chunk_size=2,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_multi_item_isolation_and_speed(self):
        # Warmup compilation for one-item and multi-item paths.
        self.engine.score(
            query=QUERY_IDS,
            items=[BASE_ITEMS[0]],
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.engine.score(
            query=QUERY_IDS,
            items=BASE_ITEMS,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )

        base_scores = self.engine.score(
            query=QUERY_IDS,
            items=BASE_ITEMS,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )

        same_length_items = [x[:] for x in BASE_ITEMS]
        same_length_items[1][0] = same_length_items[1][0] + 1
        same_length_scores = self.engine.score(
            query=QUERY_IDS,
            items=same_length_items,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )

        changed_length_items = [x[:] for x in BASE_ITEMS]
        changed_length_items[1] = changed_length_items[1] + [13]
        changed_length_scores = self.engine.score(
            query=QUERY_IDS,
            items=changed_length_items,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )

        unchanged_indices = [0, 2, 3, 4]
        same_diffs = [
            _max_abs_diff(base_scores[i], same_length_scores[i]) for i in unchanged_indices
        ]
        changed_diffs = [
            _max_abs_diff(base_scores[i], changed_length_scores[i]) for i in unchanged_indices
        ]

        # Same-length mutation should be exact.
        self.assertEqual(max(same_diffs), 0.0)
        # Changed-length mutation should not perturb unchanged items.
        self.assertEqual(max(changed_diffs), 0.0)

        # Throughput regression guard: multi-item scoring should beat serial one-item calls.
        perf_items = [BASE_ITEMS[i % len(BASE_ITEMS)] for i in range(32)]

        self.engine.score(
            query=QUERY_IDS,
            items=perf_items,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        for item in perf_items:
            self.engine.score(
                query=QUERY_IDS,
                items=[item],
                label_token_ids=LABEL_TOKEN_IDS,
                apply_softmax=True,
            )

        t0 = time.perf_counter()
        self.engine.score(
            query=QUERY_IDS,
            items=perf_items,
            label_token_ids=LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        multi_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        for item in perf_items:
            self.engine.score(
                query=QUERY_IDS,
                items=[item],
                label_token_ids=LABEL_TOKEN_IDS,
                apply_softmax=True,
            )
        serial_latency = time.perf_counter() - t0

        speedup = serial_latency / multi_latency
        self.assertGreaterEqual(
            speedup,
            3.0,
            f"Expected >=3.0x speedup for 32 items, got {speedup:.3f}x",
        )


if __name__ == "__main__":
    unittest.main()
