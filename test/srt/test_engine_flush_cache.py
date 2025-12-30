import asyncio
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
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_test_passed(test_name: str):
    """Print a colored PASS message."""
    print(f"{Colors.BOLD}{Colors.GREEN}✓ PASS: {test_name}{Colors.RESET}")


def print_state_section(title: str, state: dict, keys: list[str]):
    """Print a section of state with specified keys."""
    print(f"{Colors.YELLOW}{title}:{Colors.RESET}")
    for key in keys:
        value = state.get(key, "N/A")
        print(f"  {key}: {Colors.GREEN}{value}{Colors.RESET}")


class TestEngineFlushCache(CustomTestCase):
    """
    Test flush_cache functionality using Engine API.

    Verifies that flush_cache properly clears:
    - tree_cache (radix/prefix cache)
    - req_to_token_pool
    - token_to_kv_pool_allocator (KV cache)
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
        cls.test_prompt = "Write a story about a brave knight. Once upon a time,"
        cls.max_new_tokens = 100

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    async def _async_get_internal_state(self):
        """Get internal state using Engine's async API."""
        server_info = await self.engine.async_get_server_info()
        return server_info["internal_states"]

    async def _async_flush_cache(self):
        """Flush cache using Engine's async API."""
        return await self.engine.async_flush_cache()

    async def _async_generate(self, prompt: str, max_new_tokens: int):
        """Generate using async method."""
        return await self.engine.async_generate(
            prompt=prompt,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            stream=False,
        )

    async def _run_test_flush_cache_after_generation(self):
        """Test flush_cache restores all states after generation completes."""
        # Get initial state
        initial_states = await self._async_get_internal_state()
        initial = initial_states[0]

        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Flush Cache After Generation ==={Colors.RESET}")
        print(f"{Colors.YELLOW}Initial State:{Colors.RESET}")
        print(
            f"  available_kv_tokens: {Colors.GREEN}{initial['available_kv_tokens']}{Colors.RESET}"
        )
        print(f"  tree_cache_size: {Colors.GREEN}{initial['tree_cache_size']}{Colors.RESET}")
        print(
            f"  req_to_token_pool_used: {Colors.GREEN}{initial['req_to_token_pool_used']}{Colors.RESET}"
        )

        # Generate some text (this will use KV cache and update counters)
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        tasks = [
            asyncio.create_task(self._async_generate(prompt, self.max_new_tokens))
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)

        # Verify all requests completed
        for i, result in enumerate(results):
            self.assertIn("text", result, f"Request {i} should have text output")

        # Get state after generation
        states_after_gen = await self._async_get_internal_state()
        after_gen = states_after_gen[0]

        print(f"{Colors.YELLOW}After Generation ({num_requests} requests):{Colors.RESET}")
        print(
            f"  available_kv_tokens: {Colors.GREEN}{after_gen['available_kv_tokens']}{Colors.RESET}"
        )
        print(f"  tree_cache_size: {Colors.GREEN}{after_gen['tree_cache_size']}{Colors.RESET}")
        print(
            f"  req_to_token_pool_used: {Colors.GREEN}{after_gen['req_to_token_pool_used']}{Colors.RESET}"
        )
        print(f"  forward_ct_decode: {Colors.GREEN}{after_gen['forward_ct_decode']}{Colors.RESET}")

        # Flush cache
        flush_result = await self._async_flush_cache()
        print(f"{Colors.YELLOW}Flush Cache Result:{Colors.RESET}")
        print(f"  Result: {Colors.CYAN}{flush_result}{Colors.RESET}")

        # Get state after flush
        states_after_flush = await self._async_get_internal_state()
        after_flush = states_after_flush[0]

        print(f"{Colors.YELLOW}After Flush:{Colors.RESET}")
        print(
            f"  available_kv_tokens: {Colors.GREEN}{after_flush['available_kv_tokens']}{Colors.RESET}"
        )
        print(f"  tree_cache_size: {Colors.GREEN}{after_flush['tree_cache_size']}{Colors.RESET}")
        print(
            f"  req_to_token_pool_used: {Colors.GREEN}{after_flush['req_to_token_pool_used']}{Colors.RESET}"
        )
        print(
            f"  forward_ct_decode: {Colors.GREEN}{after_flush['forward_ct_decode']}{Colors.RESET}"
        )
        print(f"  new_token_ratio: {Colors.GREEN}{after_flush['new_token_ratio']}{Colors.RESET}")

        print(f"\n{Colors.YELLOW}Verification:{Colors.RESET}")

        # Verify KV cache is restored
        if after_flush["available_kv_tokens"] == initial["available_kv_tokens"]:
            print(f"  ✓ KV cache restored: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ KV cache NOT restored: {Colors.RED}FAIL{Colors.RESET}")

        # Verify tree cache is cleared
        if after_flush["tree_cache_size"] == 0:
            print(f"  ✓ Tree cache cleared: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ Tree cache NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

        # Verify req_to_token_pool is cleared
        if after_flush["req_to_token_pool_used"] == 0:
            print(f"  ✓ req_to_token_pool cleared: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ req_to_token_pool NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

        if after_flush["forward_ct_decode"] == 0:
            print(f"  ✓ forward_ct_decode reset: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ forward_ct_decode NOT reset: {Colors.RED}FAIL{Colors.RESET}")

        # Verify new_token_ratio is reset to init value
        if after_flush["new_token_ratio"] == after_flush["init_new_token_ratio"]:
            print(f"  ✓ new_token_ratio reset: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ new_token_ratio NOT reset: {Colors.RED}FAIL{Colors.RESET}")

        print(f"{Colors.BOLD}{Colors.BLUE}======================================{Colors.RESET}\n")

        # Assertions
        self.assertEqual(
            after_flush["available_kv_tokens"],
            initial["available_kv_tokens"],
            "KV cache should be restored after flush",
        )
        self.assertEqual(after_flush["tree_cache_size"], 0, "Tree cache should be cleared")
        self.assertEqual(
            after_flush["req_to_token_pool_used"], 0, "req_to_token_pool should be cleared"
        )
        self.assertEqual(after_flush["forward_ct_decode"], 0, "forward_ct_decode should be reset")
        self.assertEqual(
            after_flush["new_token_ratio"],
            after_flush["init_new_token_ratio"],
            "new_token_ratio should be reset to init value",
        )

    async def _run_test_flush_cache_clears_scheduling_state(self):
        """Test flush_cache clears all scheduling state."""
        # Get initial state
        initial_states = await self._async_get_internal_state()
        initial = initial_states[0]

        print(
            f"\n{Colors.BOLD}{Colors.BLUE}=== Flush Cache Clears Scheduling State ==={Colors.RESET}"
        )
        print(f"{Colors.YELLOW}Initial State:{Colors.RESET}")
        print(f"  running_batch_size: {Colors.GREEN}{initial['running_batch_size']}{Colors.RESET}")
        print(f"  waiting_queue_size: {Colors.GREEN}{initial['waiting_queue_size']}{Colors.RESET}")
        print(f"  cur_batch_is_none: {Colors.GREEN}{initial['cur_batch_is_none']}{Colors.RESET}")
        print(f"  last_batch_is_none: {Colors.GREEN}{initial['last_batch_is_none']}{Colors.RESET}")
        print(
            f"  chunked_req_is_none: {Colors.GREEN}{initial['chunked_req_is_none']}{Colors.RESET}"
        )

        # Flush cache
        flush_result = await self._async_flush_cache()
        print(f"{Colors.YELLOW}Flush Cache Result:{Colors.RESET}")
        print(f"  Result: {Colors.CYAN}{flush_result}{Colors.RESET}")

        # Get state after flush
        states_after_flush = await self._async_get_internal_state()
        after_flush = states_after_flush[0]

        print(f"{Colors.YELLOW}After Flush:{Colors.RESET}")
        print(
            f"  running_batch_size: {Colors.GREEN}{after_flush['running_batch_size']}{Colors.RESET}"
        )
        print(
            f"  waiting_queue_size: {Colors.GREEN}{after_flush['waiting_queue_size']}{Colors.RESET}"
        )
        print(
            f"  cur_batch_is_none: {Colors.GREEN}{after_flush['cur_batch_is_none']}{Colors.RESET}"
        )
        print(
            f"  last_batch_is_none: {Colors.GREEN}{after_flush['last_batch_is_none']}{Colors.RESET}"
        )
        print(
            f"  chunked_req_is_none: {Colors.GREEN}{after_flush['chunked_req_is_none']}{Colors.RESET}"
        )

        print(f"\n{Colors.YELLOW}Verification:{Colors.RESET}")

        # Verify scheduling state is cleared
        if after_flush["running_batch_size"] == 0:
            print(f"  ✓ running_batch cleared: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ running_batch NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

        if after_flush["waiting_queue_size"] == 0:
            print(f"  ✓ waiting_queue cleared: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ waiting_queue NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

        if after_flush["cur_batch_is_none"]:
            print(f"  ✓ cur_batch is None: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ cur_batch is NOT None: {Colors.RED}FAIL{Colors.RESET}")

        if after_flush["last_batch_is_none"]:
            print(f"  ✓ last_batch is None: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ last_batch is NOT None: {Colors.RED}FAIL{Colors.RESET}")

        if after_flush["chunked_req_is_none"]:
            print(f"  ✓ chunked_req is None: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ chunked_req is NOT None: {Colors.RED}FAIL{Colors.RESET}")

        print(
            f"{Colors.BOLD}{Colors.BLUE}============================================{Colors.RESET}\n"
        )

        # Assertions
        self.assertEqual(after_flush["running_batch_size"], 0, "Running batch should be empty")
        self.assertEqual(after_flush["waiting_queue_size"], 0, "Waiting queue should be empty")
        self.assertTrue(after_flush["cur_batch_is_none"], "cur_batch should be None")
        self.assertTrue(after_flush["last_batch_is_none"], "last_batch should be None")
        self.assertTrue(after_flush["chunked_req_is_none"], "chunked_req should be None")

    async def _run_test_generation_works_after_flush(self):
        """Test that generation still works correctly after flush_cache."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Generation After Flush ==={Colors.RESET}")

        # First generation
        result1 = await self._async_generate(self.test_prompt, self.max_new_tokens)
        text1 = result1.get("text", "")
        print(f"{Colors.YELLOW}First Generation:{Colors.RESET}")
        print(f"  Length: {Colors.GREEN}{len(text1)}{Colors.RESET} chars")
        print(f"  Preview: {Colors.CYAN}{text1[:80]}...{Colors.RESET}")

        # Get state after first generation
        states1 = await self._async_get_internal_state()
        after1 = states1[0]

        # Flush cache
        flush_result = await self._async_flush_cache()
        print(f"{Colors.YELLOW}Flush Cache:{Colors.RESET}")
        print(f"  Result: {Colors.CYAN}{flush_result}{Colors.RESET}")

        # Get state after flush
        states_flush = await self._async_get_internal_state()
        after_flush = states_flush[0]

        # Second generation (should work and produce same result with temperature=0)
        result2 = await self._async_generate(self.test_prompt, self.max_new_tokens)
        text2 = result2.get("text", "")
        print(f"{Colors.YELLOW}Second Generation (after flush):{Colors.RESET}")
        print(f"  Length: {Colors.GREEN}{len(text2)}{Colors.RESET} chars")
        print(f"  Preview: {Colors.CYAN}{text2[:80]}...{Colors.RESET}")

        # Get state after second generation
        states2 = await self._async_get_internal_state()
        after2 = states2[0]

        # Verify both generations produced output
        self.assertGreater(len(text1), 0, "First generation should produce text")
        self.assertGreater(len(text2), 0, "Second generation should produce text")

        # With temperature=0, outputs should be identical (deterministic)
        print(f"\n{Colors.YELLOW}Verification:{Colors.RESET}")
        match = text1 == text2
        if match:
            print(f"  ✓ Texts Match (deterministic): {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ Texts DO NOT Match: {Colors.RED}FAIL{Colors.RESET}")
            print(f"    Text1: {text1[:100]}...")
            print(f"    Text2: {text2[:100]}...")

        print(f"{Colors.BOLD}{Colors.BLUE}==============================={Colors.RESET}\n")

        self.assertEqual(
            text1,
            text2,
            "Generation after flush should be deterministic (same output with temperature=0)",
        )

    async def _run_test_multiple_flush_cycles(self):
        """Test multiple flush cycles work correctly."""
        await self._async_flush_cache()
        num_cycles = 3

        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Flush Cycles ==={Colors.RESET}")

        # Get initial available tokens
        initial_states = await self._async_get_internal_state()
        initial_tokens = initial_states[0]["available_kv_tokens"]
        print(
            f"{Colors.YELLOW}Initial Available KV Tokens: {Colors.GREEN}{initial_tokens}{Colors.RESET}"
        )

        for cycle in range(num_cycles):
            print(f"\n{Colors.CYAN}--- Cycle {cycle + 1}/{num_cycles} ---{Colors.RESET}")

            # Generate
            result = await self._async_generate(
                f"{self.test_prompt} Cycle {cycle}:", self.max_new_tokens
            )
            self.assertIn("text", result, f"Cycle {cycle}: Generation should produce text")

            # Get state after generation
            states_after_gen = await self._async_get_internal_state()
            tokens_after_gen = states_after_gen[0]["available_kv_tokens"]
            print(
                f"  After Generation: {Colors.GREEN}{tokens_after_gen}{Colors.RESET} tokens available"
            )

            # Flush
            await self._async_flush_cache()

            # Get state after flush
            states_after_flush = await self._async_get_internal_state()
            tokens_after_flush = states_after_flush[0]["available_kv_tokens"]
            print(
                f"  After Flush: {Colors.GREEN}{tokens_after_flush}{Colors.RESET} tokens available"
            )

            # Verify tokens restored
            self.assertEqual(
                tokens_after_flush,
                initial_tokens,
                f"Cycle {cycle}: Tokens should be restored after flush",
            )

        print(f"\n{Colors.BOLD}{Colors.BLUE}============================={Colors.RESET}\n")

    def test_1_flush_cache_after_generation(self):
        """Test flush_cache restores KV cache after generation."""
        self.engine.loop.run_until_complete(self._run_test_flush_cache_after_generation())
        print_test_passed("TestEngineFlushCache.test_1_flush_cache_after_generation")

    def test_2_flush_cache_clears_scheduling_state(self):
        """Test flush_cache clears all scheduling state."""
        self.engine.loop.run_until_complete(self._run_test_flush_cache_clears_scheduling_state())
        print_test_passed("TestEngineFlushCache.test_2_flush_cache_clears_scheduling_state")

    def test_3_generation_works_after_flush(self):
        """Test generation works correctly after flush_cache."""
        self.engine.loop.run_until_complete(self._run_test_generation_works_after_flush())
        print_test_passed("TestEngineFlushCache.test_3_generation_works_after_flush")

    def test_4_multiple_flush_cycles(self):
        """Test multiple flush cycles work correctly."""
        self.engine.loop.run_until_complete(self._run_test_multiple_flush_cycles())
        print_test_passed("TestEngineFlushCache.test_4_multiple_flush_cycles")


if __name__ == "__main__":
    unittest.main()
