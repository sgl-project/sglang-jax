"""
Test pause_generation and continue_generation using Engine API directly.

Usage:
python3 -m unittest test_engine_pause_continue.TestEnginePauseContinue
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import QWEN3_8B, CustomTestCase


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_test_passed(test_name: str):
    """Print a colored PASS message."""
    print(f"{Colors.BOLD}{Colors.GREEN}✓ PASS: {test_name}{Colors.RESET}")


class TestEnginePauseContinue(CustomTestCase):
    """
    Test pause_generation and continue_generation using Engine API directly.

    Tests include:
    - Retract mode: Verify generation is paused, running batch requests are
      moved to waiting queue, and KV cache is cleared
    - In-place mode: Verify generation is paused, running batch is unchanged,
      and KV cache is preserved
    - Continue generation: Verify that after pause, continue_generation properly
      resumes generation
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = QWEN3_8B
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=4,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.65,
            chunked_prefill_size=1024,
            download_dir="/tmp",
            dtype="bfloat16",
            precompile_bs_paddings=[64],
            max_running_requests=64,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[16384],
            page_size=64,
            log_requests=False,
        )
        cls.tokenizer = get_tokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _run_decode(self, max_new_tokens=1000):
        """Run a decode request that generates many tokens."""
        return self.engine.generate(
            prompt="Write a very long story about a magical forest. Once upon a time, in a land far away,",
            sampling_params={
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        )

    def _get_internal_state(self):
        """Get the internal state of the engine."""
        return self.engine.get_server_info()

    def test_1_pause_generation_retract_mode(self):
        """
        Test pause_generation with retract mode.

        Verify:
        1. Generation is actually paused
        2. Running batch requests are moved to waiting queue
        3. KV cache is cleared (available tokens increase)
        """
        # Get initial state
        initial_state = self._get_internal_state()
        initial_internal = initial_state["internal_states"][0]
        initial_available_tokens = initial_internal["available_kv_tokens"]

        # Start multiple long-running requests
        num_requests = 8
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating (in running batch)
            time.sleep(3)

            # Get state before pause - should have requests in running batch
            state_before_pause = self._get_internal_state()
            internal_before = state_before_pause["internal_states"][0]

            # Verify there are requests in running batch
            running_before_pause = internal_before["running_batch_size"]
            waiting_before_pause = internal_before["waiting_queue_size"]
            self.assertGreater(
                running_before_pause, 0, "Expected requests in running batch before pause"
            )

            # Record available tokens before pause (should be less due to KV cache usage)
            tokens_before_pause = internal_before["available_kv_tokens"]
            self.assertLess(
                tokens_before_pause,
                initial_available_tokens,
                "KV cache should be used before pause",
            )

            # Pause generation with retract mode (returns None)
            self.engine.pause_generation(mode="retract")

            # Wait for pause to take effect
            time.sleep(0.5)

            # Get state after pause
            state_after_pause = self._get_internal_state()
            internal_after = state_after_pause["internal_states"][0]

            # Verify engine is paused
            self.assertTrue(
                internal_after["engine_paused"], "Engine should be paused after pause_generation"
            )

            # Verify running batch is empty (requests retracted)
            running_after_pause = internal_after["running_batch_size"]
            waiting_after_pause = internal_after["waiting_queue_size"]
            self.assertEqual(
                running_after_pause, 0, "Running batch should be empty after retract mode pause"
            )

            # Verify requests are in waiting queue
            self.assertGreater(
                waiting_after_pause, 0, "Requests should be moved to waiting queue after retract"
            )

            # Verify quantity: waiting queue increase == running batch decrease
            waiting_queue_increase = waiting_after_pause - waiting_before_pause
            running_batch_decrease = running_before_pause - running_after_pause
            self.assertEqual(
                waiting_queue_increase,
                running_batch_decrease,
                f"Waiting queue increase ({waiting_queue_increase}) should equal "
                f"running batch decrease ({running_batch_decrease})",
            )

            # Verify KV cache is cleared (available tokens increase back toward initial)
            tokens_after_pause = internal_after["available_kv_tokens"]
            self.assertGreater(
                tokens_after_pause,
                tokens_before_pause,
                "KV cache should be cleared after retract mode pause",
            )

            # Continue generation (returns None)
            self.engine.continue_generation()

            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            # Verify all requests completed successfully
            for result in results:
                self.assertIn("text", result, f"Request should complete with text output: {result}")
                # The finish reason should indicate successful completion (length)
                finish_reason = result.get("meta_info", {}).get("finish_reason", {})
                self.assertEqual(
                    finish_reason.get("type"),
                    "length",
                    f"Request should finish due to length limit: {result}",
                )

        print_test_passed("TestEnginePauseContinue.test_1_pause_generation_retract_mode")

    def test_2_pause_generation_in_place_mode(self):
        """
        Test pause_generation with in_place mode.

        Verify:
        1. Generation is actually paused
        2. Running batch is NOT changed (requests stay in running batch)
        3. KV cache is NOT cleared (available tokens stay the same)
        """
        # Start multiple long-running requests
        num_requests = 8
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating (in running batch)
            time.sleep(3)

            # Get state before pause
            state_before_pause = self._get_internal_state()
            internal_before = state_before_pause["internal_states"][0]

            # Verify there are requests in running batch
            running_before = internal_before["running_batch_size"]
            self.assertGreater(running_before, 0, "Expected requests in running batch before pause")

            # Record running batch rids, waiting queue size, and available tokens
            rids_before = set(internal_before["running_batch_rids"])
            tokens_before_pause = internal_before["available_kv_tokens"]
            waiting_before_pause = internal_before["waiting_queue_size"]

            # Pause generation with in_place mode (returns None)
            self.engine.pause_generation(mode="in_place")

            # Wait for pause to take effect
            time.sleep(0.5)

            # Get state after pause
            state_after_pause = self._get_internal_state()
            internal_after = state_after_pause["internal_states"][0]

            # Verify engine is paused
            self.assertTrue(
                internal_after["engine_paused"], "Engine should be paused after pause_generation"
            )

            # Verify running batch is NOT empty (in_place mode keeps requests)
            running_after = internal_after["running_batch_size"]
            self.assertEqual(
                running_after,
                running_before,
                f"Running batch size should be unchanged in in_place mode: {running_before} -> {running_after}",
            )

            # Verify same requests are still in running batch
            rids_after = set(internal_after["running_batch_rids"])
            self.assertEqual(
                rids_before,
                rids_after,
                "Same requests should be in running batch after in_place pause",
            )

            # Verify waiting queue is unchanged
            waiting_after_pause = internal_after["waiting_queue_size"]
            self.assertEqual(
                waiting_after_pause,
                waiting_before_pause,
                f"Waiting queue should be unchanged in in_place mode: "
                f"{waiting_before_pause} -> {waiting_after_pause}",
            )

            # Verify KV cache is NOT cleared
            tokens_after_pause = internal_after["available_kv_tokens"]
            self.assertEqual(
                tokens_after_pause,
                tokens_before_pause,
                "KV cache should NOT be cleared in in_place mode",
            )

            # Continue generation (returns None)
            self.engine.continue_generation()

            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            # Verify all requests completed successfully
            for result in results:
                self.assertIn("text", result, f"Request should complete with text output: {result}")

        print_test_passed("TestEnginePauseContinue.test_2_pause_generation_in_place_mode")

    def test_3_pause_continue_multiple_cycles(self):
        """
        Test multiple pause/continue cycles.

        Verify that the engine can handle multiple pause/continue cycles
        without issues.
        """
        num_cycles = 3

        with ThreadPoolExecutor(4) as executor:
            futures = [executor.submit(self._run_decode, 3000) for _ in range(4)]

            for cycle in range(num_cycles):
                # Wait for some generation to happen
                time.sleep(2)

                # Pause with alternating modes (returns None)
                mode = "retract" if cycle % 2 == 0 else "in_place"
                self.engine.pause_generation(mode=mode)

                # Verify paused
                state = self._get_internal_state()
                self.assertTrue(
                    state["internal_states"][0]["engine_paused"],
                    f"Cycle {cycle}: engine should be paused",
                )

                # Continue (returns None)
                self.engine.continue_generation()

                # Verify not paused
                state = self._get_internal_state()
                self.assertFalse(
                    state["internal_states"][0]["engine_paused"],
                    f"Cycle {cycle}: engine should not be paused after continue",
                )

            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            # All requests should complete
            for i, result in enumerate(results):
                self.assertIn("text", result, f"Request {i} should complete: {result}")

        print_test_passed("TestEnginePauseContinue.test_3_pause_continue_multiple_cycles")


if __name__ == "__main__":
    unittest.main()
