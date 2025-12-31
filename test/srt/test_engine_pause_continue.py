import asyncio
import time
import unittest

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

    Uses async/await pattern to avoid multi-threading event loop conflicts.
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

    async def _async_get_internal_state(self):
        """Get internal state using Engine's async API."""
        server_info = await self.engine.async_get_server_info()
        return server_info["internal_states"]

    async def _async_pause_generation(self, mode: str):
        """Pause generation using Engine's async API."""
        await self.engine.async_pause_generation(mode=mode)

    async def _async_continue_generation(self):
        """Continue generation using Engine's async API."""
        await self.engine.async_continue_generation()

    async def _async_generate(self, max_new_tokens: int = 2000):
        """Generate using async method."""
        return await self.engine.async_generate(
            prompt="Write a very long story about a magical forest. Once upon a time, in a land far away,",
            sampling_params={
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            stream=False,
        )

    async def _run_test_retract_mode(self):
        """Test pause_generation with retract mode using async."""
        # Get initial state (returns list of dicts, take first one)
        initial_states = await self._async_get_internal_state()
        initial_internal = initial_states[0]
        initial_available_tokens = initial_internal["available_kv_tokens"]

        # Start multiple long-running requests concurrently
        num_requests = 8
        tasks = [
            asyncio.create_task(self._async_generate(max_new_tokens=2000))
            for _ in range(num_requests)
        ]

        # Wait a bit for requests to start generating
        await asyncio.sleep(3)

        # Get state before pause
        states_before = await self._async_get_internal_state()
        internal_before = states_before[0]

        # Verify there are requests in running batch
        running_before_pause = internal_before["running_batch_size"]
        waiting_before_pause = internal_before["waiting_queue_size"]
        self.assertGreater(
            running_before_pause, 0, "Expected requests in running batch before pause"
        )

        # Record available tokens before pause
        tokens_before_pause = internal_before["available_kv_tokens"]
        self.assertLess(
            tokens_before_pause,
            initial_available_tokens,
            "KV cache should be used before pause",
        )

        # Pause generation with retract mode
        await self._async_pause_generation(mode="retract")

        # Wait for pause to take effect
        await asyncio.sleep(0.5)

        # Get state after pause
        states_after = await self._async_get_internal_state()
        internal_after = states_after[0]

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

        # Verify KV cache is cleared
        tokens_after_pause = internal_after["available_kv_tokens"]

        # Print colorful log for retract mode state changes
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Retract Mode State Changes ==={Colors.RESET}")
        print(f"{Colors.YELLOW}Before Pause:{Colors.RESET}")
        print(f"  Running Batch Size: {Colors.GREEN}{running_before_pause}{Colors.RESET}")
        print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_before_pause}{Colors.RESET}")
        print(f"  Available KV Tokens: {Colors.GREEN}{tokens_before_pause}{Colors.RESET}")
        print(f"{Colors.YELLOW}After Pause:{Colors.RESET}")
        print(f"  Running Batch Size: {Colors.GREEN}{running_after_pause}{Colors.RESET}")
        print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_after_pause}{Colors.RESET}")
        print(f"  Available KV Tokens: {Colors.GREEN}{tokens_after_pause}{Colors.RESET}")
        print(f"{Colors.YELLOW}Changes:{Colors.RESET}")
        print(f"  Running Batch Decrease: {Colors.RED}-{running_batch_decrease}{Colors.RESET}")
        print(f"  Waiting Queue Increase: {Colors.GREEN}+{waiting_queue_increase}{Colors.RESET}")
        print(
            f"  KV Tokens Freed: {Colors.GREEN}+{tokens_after_pause - tokens_before_pause}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BLUE}================================={Colors.RESET}\n")

        self.assertEqual(
            waiting_queue_increase,
            running_batch_decrease,
            f"Waiting queue increase ({waiting_queue_increase}) should equal "
            f"running batch decrease ({running_batch_decrease})",
        )

        self.assertGreater(
            tokens_after_pause,
            tokens_before_pause,
            "KV cache should be cleared after retract mode pause",
        )

        # Continue generation
        await self._async_continue_generation()

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed successfully
        for result in results:
            if isinstance(result, Exception):
                self.fail(f"Request failed with exception: {result}")
            self.assertIn("text", result, f"Request should have text output: {result}")
            finish_reason = result.get("meta_info", {}).get("finish_reason", {})
            self.assertEqual(
                finish_reason.get("type"),
                "length",
                f"Request should finish due to length limit: {result}",
            )

    async def _run_test_in_place_mode(self):
        """Test pause_generation with in_place mode using async."""
        # Start multiple long-running requests concurrently
        num_requests = 8
        tasks = [
            asyncio.create_task(self._async_generate(max_new_tokens=2000))
            for _ in range(num_requests)
        ]

        # Wait a bit for requests to start generating
        await asyncio.sleep(3)

        # Get state before pause (returns list, take first)
        states_before = await self._async_get_internal_state()
        internal_before = states_before[0]

        # Verify there are requests in running batch
        running_before = internal_before["running_batch_size"]
        self.assertGreater(running_before, 0, "Expected requests in running batch before pause")

        # Record state
        rids_before = set(internal_before["running_batch_rids"])
        tokens_before_pause = internal_before["available_kv_tokens"]
        waiting_before_pause = internal_before["waiting_queue_size"]

        # Pause generation with in_place mode
        await self._async_pause_generation(mode="in_place")

        # Wait for pause to take effect
        await asyncio.sleep(0.5)

        # Get state after pause
        states_after = await self._async_get_internal_state()
        internal_after = states_after[0]

        # Verify engine is paused
        self.assertTrue(
            internal_after["engine_paused"], "Engine should be paused after pause_generation"
        )

        # Verify running batch is NOT empty (in_place mode keeps requests)
        running_after = internal_after["running_batch_size"]
        self.assertEqual(
            running_after,
            running_before,
            f"Running batch size should be unchanged: {running_before} -> {running_after}",
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
            f"Waiting queue should be unchanged: {waiting_before_pause} -> {waiting_after_pause}",
        )

        # Verify KV cache is NOT cleared
        tokens_after_pause = internal_after["available_kv_tokens"]
        self.assertEqual(
            tokens_after_pause,
            tokens_before_pause,
            "KV cache should NOT be cleared in in_place mode",
        )

        # Continue generation
        await self._async_continue_generation()

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed
        for result in results:
            if isinstance(result, Exception):
                self.fail(f"Request failed with exception: {result}")
            self.assertIn("text", result, f"Request should have text output: {result}")

    async def _run_test_multiple_cycles(self):
        """Test multiple pause/continue cycles using async."""
        num_cycles = 3

        # Start requests concurrently
        tasks = [asyncio.create_task(self._async_generate(max_new_tokens=3000)) for _ in range(4)]

        for cycle in range(num_cycles):
            # Wait for some generation to happen
            await asyncio.sleep(2)

            # Pause with alternating modes
            mode = "retract" if cycle % 2 == 0 else "in_place"
            await self._async_pause_generation(mode=mode)

            # Verify paused
            states = await self._async_get_internal_state()
            self.assertTrue(
                states[0]["engine_paused"],
                f"Cycle {cycle}: engine should be paused",
            )

            # Continue
            await self._async_continue_generation()

            # Verify not paused
            states = await self._async_get_internal_state()
            self.assertFalse(
                states[0]["engine_paused"],
                f"Cycle {cycle}: engine should not be paused after continue",
            )

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.fail(f"Request {i} failed with exception: {result}")
            self.assertIn("text", result, f"Request {i} should have text output: {result}")

    async def _run_test_abort_mode(self):
        """Test pause_generation with abort mode using async."""
        # Start multiple long-running requests concurrently
        num_requests = 8
        tasks = [
            asyncio.create_task(self._async_generate(max_new_tokens=2000))
            for _ in range(num_requests)
        ]

        # Wait a bit for requests to start generating
        await asyncio.sleep(3)

        # Get state before pause
        states_before = await self._async_get_internal_state()
        internal_before = states_before[0]

        # Verify there are requests in running batch or waiting queue
        running_before = internal_before["running_batch_size"]
        waiting_before = internal_before["waiting_queue_size"]
        total_before = running_before + waiting_before
        self.assertGreater(
            total_before, 0, "Expected requests in running batch or waiting queue before pause"
        )

        # Pause generation with abort mode
        await self._async_pause_generation(mode="abort")

        # Wait for abort to complete
        await asyncio.sleep(1)

        # Get state after pause
        states_after = await self._async_get_internal_state()
        internal_after = states_after[0]

        # Verify waiting queue is empty (abort clears waiting queue)
        waiting_after = internal_after["waiting_queue_size"]
        self.assertEqual(waiting_after, 0, "Waiting queue should be empty after abort mode pause")

        # Verify running batch is empty (all requests aborted)
        running_after = internal_after["running_batch_size"]
        self.assertEqual(running_after, 0, "Running batch should be empty after abort mode pause")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed with abort status
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Some requests may raise ValueError due to abort
                self.assertIn(
                    "abort",
                    str(result).lower(),
                    f"Request {i} exception should be abort-related: {result}",
                )
            else:
                self.assertIn("meta_info", result, f"Request {i} should have meta_info: {result}")
                finish_reason = result.get("meta_info", {}).get("finish_reason", {})
                self.assertEqual(
                    finish_reason.get("type"),
                    "abort",
                    f"Request {i} should finish due to abort: {result}",
                )

        # Continue generation to reset state (abort mode sets is_pause=True)
        await self._async_continue_generation()

    def test_1_pause_generation_retract_mode(self):
        """Test pause_generation with retract mode."""
        self.engine.loop.run_until_complete(self._run_test_retract_mode())
        print_test_passed("TestEnginePauseContinue.test_1_pause_generation_retract_mode")

    def test_2_pause_generation_in_place_mode(self):
        """Test pause_generation with in_place mode."""
        self.engine.loop.run_until_complete(self._run_test_in_place_mode())
        print_test_passed("TestEnginePauseContinue.test_2_pause_generation_in_place_mode")

    def test_3_pause_continue_multiple_cycles(self):
        """Test multiple pause/continue cycles."""
        self.engine.loop.run_until_complete(self._run_test_multiple_cycles())
        print_test_passed("TestEnginePauseContinue.test_3_pause_continue_multiple_cycles")

    def test_4_pause_generation_abort_mode(self):
        """Test pause_generation with abort mode."""
        self.engine.loop.run_until_complete(self._run_test_abort_mode())
        print_test_passed("TestEnginePauseContinue.test_4_pause_generation_abort_mode")


if __name__ == "__main__":
    unittest.main()
