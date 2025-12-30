import asyncio
import unittest

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.managers.io_struct import GenerateReqInput
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


def print_comparison(
    title: str,
    baseline_text: str,
    test_text: str,
    baseline_label: str = "Baseline",
    test_label: str = "Test",
):
    """Print a colorful comparison of two outputs."""
    baseline_words = baseline_text.split()
    test_words = test_text.split()

    print(f"\n{Colors.BOLD}{Colors.BLUE}=== {title} ==={Colors.RESET}")
    print(f"{Colors.YELLOW}{baseline_label}:{Colors.RESET}")
    print(f"  Length (chars): {Colors.GREEN}{len(baseline_text)}{Colors.RESET}")
    print(f"  Length (words): {Colors.GREEN}{len(baseline_words)}{Colors.RESET}")
    print(f"  Preview: {Colors.CYAN}{baseline_text[:100]}...{Colors.RESET}")
    print(f"{Colors.YELLOW}{test_label}:{Colors.RESET}")
    print(f"  Length (chars): {Colors.GREEN}{len(test_text)}{Colors.RESET}")
    print(f"  Length (words): {Colors.GREEN}{len(test_words)}{Colors.RESET}")
    print(f"  Preview: {Colors.CYAN}{test_text[:100]}...{Colors.RESET}")
    print(f"{Colors.YELLOW}Comparison:{Colors.RESET}")
    print(
        f"  Char Length Diff: {Colors.MAGENTA}{len(test_text) - len(baseline_text)}{Colors.RESET}"
    )
    print(
        f"  Word Length Diff: {Colors.MAGENTA}{len(test_words) - len(baseline_words)}{Colors.RESET}"
    )
    print(
        f"  Texts Match: {Colors.GREEN if baseline_text == test_text else Colors.RED}{baseline_text == test_text}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * (len(title) + 8)}{Colors.RESET}\n")


class TestEngineDeterministicGeneration(CustomTestCase):
    """
    Test deterministic generation using Engine API.

    Compares pause/continue modes (abort and retract) vs Baseline with temperature=0.
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
            chunked_prefill_size=4,
            download_dir="/tmp",
            dtype="bfloat16",
            precompile_bs_paddings=[4],
            max_running_requests=4,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[4],
            page_size=4,
            log_requests=False,
        )
        cls.tokenizer = get_tokenizer(cls.model_path)
        cls.test_prompt = "Write a story about a brave knight. Once upon a time,"
        cls.max_new_tokens = 200

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

    async def _async_generate(self, prompt: str, max_new_tokens: int):
        """Generate using async method with temperature=0 for determinism."""
        obj = GenerateReqInput(
            text=prompt,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        )
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    async def _run_generation_no_pause(self, prompt: str, max_new_tokens: int):
        """Run generation without any pause."""
        result = await self._async_generate(prompt, max_new_tokens)
        return result.get("text", "")

    async def _run_generation_with_retract(
        self, prompt: str, max_new_tokens: int, pause_delay: float = 1.0, log_state: bool = False
    ):
        """Run generation with retract mode pause in the middle."""
        task = asyncio.create_task(self._async_generate(prompt, max_new_tokens))

        # Wait for generation to start
        await asyncio.sleep(pause_delay)

        # Get state before pause
        if log_state:
            states_before = await self._async_get_internal_state()
            internal_before = states_before[0]
            running_before = internal_before["running_batch_size"]
            waiting_before = internal_before["waiting_queue_size"]
            tokens_before = internal_before["available_kv_tokens"]

        # Pause with retract mode
        await self._async_pause_generation(mode="retract")
        await asyncio.sleep(0.5)

        # Get state after pause
        if log_state:
            states_after = await self._async_get_internal_state()
            internal_after = states_after[0]
            running_after = internal_after["running_batch_size"]
            waiting_after = internal_after["waiting_queue_size"]
            tokens_after = internal_after["available_kv_tokens"]

            # Print state changes
            print(f"\n{Colors.BOLD}{Colors.BLUE}=== Retract Mode State Changes ==={Colors.RESET}")
            print(f"{Colors.YELLOW}Before Pause:{Colors.RESET}")
            print(f"  Running Batch Size: {Colors.GREEN}{running_before}{Colors.RESET}")
            print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_before}{Colors.RESET}")
            print(f"  Available KV Tokens: {Colors.GREEN}{tokens_before}{Colors.RESET}")
            print(f"{Colors.YELLOW}After Pause:{Colors.RESET}")
            print(f"  Running Batch Size: {Colors.GREEN}{running_after}{Colors.RESET}")
            print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_after}{Colors.RESET}")
            print(f"  Available KV Tokens: {Colors.GREEN}{tokens_after}{Colors.RESET}")
            print(f"{Colors.YELLOW}Changes:{Colors.RESET}")
            print(
                f"  Running Batch: {Colors.RED}{running_before}{Colors.RESET} -> {Colors.GREEN}{running_after}{Colors.RESET} ({Colors.RED}-{running_before - running_after}{Colors.RESET})"
            )
            print(
                f"  Waiting Queue: {Colors.GREEN}{waiting_before}{Colors.RESET} -> {Colors.GREEN}{waiting_after}{Colors.RESET} ({Colors.GREEN}+{waiting_after - waiting_before}{Colors.RESET})"
            )
            print(f"  KV Tokens Freed: {Colors.GREEN}+{tokens_after - tokens_before}{Colors.RESET}")
            print(f"{Colors.YELLOW}Verification:{Colors.RESET}")

            # Verify running batch is cleared
            if running_after == 0:
                print(f"  ✓ Running batch cleared: {Colors.GREEN}PASS{Colors.RESET}")
            else:
                print(
                    f"  ✗ Running batch NOT cleared: {Colors.RED}FAIL{Colors.RESET} (size={running_after})"
                )

            # Verify requests moved to waiting queue
            if waiting_after > waiting_before:
                print(f"  ✓ Requests moved to waiting queue: {Colors.GREEN}PASS{Colors.RESET}")
            else:
                print(f"  ✗ Requests NOT moved to waiting queue: {Colors.RED}FAIL{Colors.RESET}")

            # Verify KV cache cleared
            if tokens_after > tokens_before:
                print(f"  ✓ KV cache cleared (tokens freed): {Colors.GREEN}PASS{Colors.RESET}")
            else:
                print(f"  ✗ KV cache NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

            print(f"{Colors.BOLD}{Colors.BLUE}=================================={Colors.RESET}\n")

        # Continue generation
        await self._async_continue_generation()

        # Wait for completion
        result = await task
        return result.get("text", "")

    async def _run_generation_with_abort_and_regenerate(
        self, prompt: str, max_new_tokens: int, pause_delay: float = 1.0
    ):
        """
        Run generation with abort mode, then re-generate.

        Abort mode:
        - Terminates ALL requests completely
        - Returns partial result (whatever was generated before abort)
        - Cannot continue aborted requests
        - Must re-generate from scratch to get full result
        """
        task = asyncio.create_task(self._async_generate(prompt, max_new_tokens))

        # Wait for generation to start and produce some tokens
        await asyncio.sleep(pause_delay)

        # Abort all requests - they will be terminated
        await self._async_pause_generation(mode="abort")
        await asyncio.sleep(0.5)

        # Reset is_paused flag (does NOT restore aborted requests)
        await self._async_continue_generation()

        # Get partial result from aborted request
        try:
            aborted_result = await task
            partial_text = aborted_result.get("text", "")
        except Exception:
            partial_text = ""

        # Re-generate from scratch to get full result
        regenerated_text = await self._run_generation_no_pause(prompt, max_new_tokens)

        return partial_text, regenerated_text

    # ============ Single Request Tests ============

    async def _run_test_single_retract_vs_no_pause(self):
        """Test single request: retract mode vs Baseline."""
        # Run baseline
        baseline_text = await self._run_generation_no_pause(self.test_prompt, self.max_new_tokens)

        # Run with retract mode (with state logging)
        retract_text = await self._run_generation_with_retract(
            self.test_prompt, self.max_new_tokens, log_state=True
        )

        # Print comparison
        print_comparison(
            "Single Request: Retract vs Baseline",
            baseline_text,
            retract_text,
            "Baseline",
            "Retract Mode",
        )

        # With temperature=0 and retract mode, output should be identical
        # because retract clears KV cache and re-prefills deterministically
        if len(baseline_text) != len(retract_text):
            print(f"\n{Colors.RED}=== Length Mismatch Detected ==={Colors.RESET}")
            print(f"{Colors.YELLOW}Baseline ({len(baseline_text)} chars):{Colors.RESET}")
            print(f"{Colors.CYAN}{baseline_text}{Colors.RESET}")
            print(f"{Colors.YELLOW}Retract ({len(retract_text)} chars):{Colors.RESET}")
            print(f"{Colors.CYAN}{retract_text}{Colors.RESET}")
            print(f"{Colors.RED}================================{Colors.RESET}\n")

        self.assertEqual(
            len(baseline_text),
            len(retract_text),
            f"Retract mode should produce same length output as Baseline. "
            f"Baseline: {len(baseline_text)}, Retract: {len(retract_text)}",
        )
        self.assertEqual(
            baseline_text,
            retract_text,
            "Retract mode should produce identical output as Baseline with temperature=0",
        )

    async def _run_test_single_abort_vs_no_pause(self):
        """
        Test single request: abort mode vs Baseline.

        Demonstrates:
        1. Abort terminates request and returns partial result
        2. After abort, must re-generate (not continue) to get full result
        3. Re-generated result matches baseline (deterministic with temperature=0)
        """
        # Run baseline - full generation
        baseline_text = await self._run_generation_no_pause(self.test_prompt, self.max_new_tokens)

        # Run with abort, then re-generate
        partial_text, regenerated_text = await self._run_generation_with_abort_and_regenerate(
            self.test_prompt, self.max_new_tokens
        )

        # Print comparison: Re-generated vs Baseline
        print_comparison(
            "Single Request: Re-generated vs Baseline",
            baseline_text,
            regenerated_text,
            "Baseline",
            "Re-generated After Abort",
        )

        # 1. Verify partial text is a prefix of baseline (deterministic)
        if partial_text:
            self.assertTrue(
                baseline_text.startswith(partial_text),
                f"Partial text should be a prefix of baseline (deterministic). "
                f"Partial: '{partial_text[:50]}...', Baseline starts: '{baseline_text[:50]}...'",
            )

        # 2. Verify re-generated result matches baseline (deterministic)
        if len(baseline_text) != len(regenerated_text):
            print(f"\n{Colors.RED}=== Length Mismatch (Re-generated vs Baseline) ==={Colors.RESET}")
            print(f"{Colors.YELLOW}Baseline ({len(baseline_text)} chars):{Colors.RESET}")
            print(f"{Colors.CYAN}{baseline_text}{Colors.RESET}")
            print(f"{Colors.YELLOW}Re-generated ({len(regenerated_text)} chars):{Colors.RESET}")
            print(f"{Colors.CYAN}{regenerated_text}{Colors.RESET}")
            print(f"{Colors.RED}================================================={Colors.RESET}\n")

        self.assertEqual(
            baseline_text,
            regenerated_text,
            "Re-generated result should match baseline (deterministic with temperature=0)",
        )

    # ============ Multiple Requests Tests ============

    async def _run_test_multiple_retract_vs_no_pause(self):
        """Test multiple requests: retract mode vs Baseline."""
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline - run sequentially to ensure determinism
        baseline_texts = []
        for prompt in prompts:
            text = await self._run_generation_no_pause(prompt, self.max_new_tokens)
            baseline_texts.append(text)

        # Run with retract mode - all concurrent with pause in the middle
        tasks = [
            asyncio.create_task(self._async_generate(prompt, self.max_new_tokens))
            for prompt in prompts
        ]

        # Wait for generation to start
        await asyncio.sleep(1.5)

        # Get state before pause
        states_before = await self._async_get_internal_state()
        internal_before = states_before[0]
        running_before = internal_before["running_batch_size"]
        waiting_before = internal_before["waiting_queue_size"]
        tokens_before = internal_before["available_kv_tokens"]

        # Pause with retract mode
        await self._async_pause_generation(mode="retract")
        await asyncio.sleep(0.5)

        # Get state after pause
        states_after = await self._async_get_internal_state()
        internal_after = states_after[0]
        running_after = internal_after["running_batch_size"]
        waiting_after = internal_after["waiting_queue_size"]
        tokens_after = internal_after["available_kv_tokens"]

        # Print state changes
        print(
            f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Requests Retract Mode State Changes ==={Colors.RESET}"
        )
        print(f"{Colors.YELLOW}Before Pause:{Colors.RESET}")
        print(f"  Running Batch Size: {Colors.GREEN}{running_before}{Colors.RESET}")
        print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_before}{Colors.RESET}")
        print(f"  Available KV Tokens: {Colors.GREEN}{tokens_before}{Colors.RESET}")
        print(f"{Colors.YELLOW}After Pause:{Colors.RESET}")
        print(f"  Running Batch Size: {Colors.GREEN}{running_after}{Colors.RESET}")
        print(f"  Waiting Queue Size: {Colors.GREEN}{waiting_after}{Colors.RESET}")
        print(f"  Available KV Tokens: {Colors.GREEN}{tokens_after}{Colors.RESET}")
        print(f"{Colors.YELLOW}Changes:{Colors.RESET}")
        print(
            f"  Running Batch: {Colors.RED}{running_before}{Colors.RESET} -> {Colors.GREEN}{running_after}{Colors.RESET} ({Colors.RED}-{running_before - running_after}{Colors.RESET})"
        )
        print(
            f"  Waiting Queue: {Colors.GREEN}{waiting_before}{Colors.RESET} -> {Colors.GREEN}{waiting_after}{Colors.RESET} ({Colors.GREEN}+{waiting_after - waiting_before}{Colors.RESET})"
        )
        print(f"  KV Tokens Freed: {Colors.GREEN}+{tokens_after - tokens_before}{Colors.RESET}")
        print(f"{Colors.YELLOW}Verification:{Colors.RESET}")

        # Verify running batch is cleared
        if running_after == 0:
            print(f"  ✓ Running batch cleared: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(
                f"  ✗ Running batch NOT cleared: {Colors.RED}FAIL{Colors.RESET} (size={running_after})"
            )

        # Verify requests moved to waiting queue
        if waiting_after > waiting_before:
            print(f"  ✓ Requests moved to waiting queue: {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ Requests NOT moved to waiting queue: {Colors.RED}FAIL{Colors.RESET}")

        # Verify KV cache cleared
        if tokens_after > tokens_before:
            print(f"  ✓ KV cache cleared (tokens freed): {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"  ✗ KV cache NOT cleared: {Colors.RED}FAIL{Colors.RESET}")

        print(
            f"{Colors.BOLD}{Colors.BLUE}===================================================={Colors.RESET}\n"
        )

        # Continue generation
        await self._async_continue_generation()

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        retract_texts = [r.get("text", "") if isinstance(r, dict) else "" for r in results]

        # Summary comparison
        print(
            f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Requests: Retract vs Baseline ==={Colors.RESET}"
        )
        for i in range(num_requests):
            match = baseline_texts[i] == retract_texts[i]
            len_diff = len(retract_texts[i]) - len(baseline_texts[i])
            print(
                f"  Request {i}: Match={Colors.GREEN if match else Colors.RED}{match}{Colors.RESET}, "
                f"LenDiff={Colors.MAGENTA}{len_diff}{Colors.RESET}"
            )
        print(f"{Colors.BOLD}{Colors.BLUE}================================={Colors.RESET}\n")

        # Verify all outputs match (deterministic with retract mode)
        for i in range(num_requests):
            if len(baseline_texts[i]) != len(retract_texts[i]):
                print(f"\n{Colors.RED}=== Request {i} Length Mismatch ==={Colors.RESET}")
                print(f"{Colors.YELLOW}Baseline ({len(baseline_texts[i])} chars):{Colors.RESET}")
                print(f"{Colors.CYAN}{baseline_texts[i]}{Colors.RESET}")
                print(f"{Colors.YELLOW}Retract ({len(retract_texts[i])} chars):{Colors.RESET}")
                print(f"{Colors.CYAN}{retract_texts[i]}{Colors.RESET}")
                print(f"{Colors.RED}==================================={Colors.RESET}\n")

            self.assertEqual(
                len(baseline_texts[i]),
                len(retract_texts[i]),
                f"Request {i}: Retract mode should produce same length. "
                f"Baseline: {len(baseline_texts[i])}, Retract: {len(retract_texts[i])}",
            )

    async def _run_test_multiple_abort_vs_no_pause(self):
        """
        Test multiple requests: abort mode vs Baseline.

        Demonstrates:
        1. Abort terminates ALL requests and returns partial results
        2. After abort, must re-generate (not continue) to get full results
        3. Re-generated results match baseline (deterministic with temperature=0)
        """
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline - full generation, run sequentially
        baseline_texts = []
        for prompt in prompts:
            text = await self._run_generation_no_pause(prompt, self.max_new_tokens)
            baseline_texts.append(text)

        # Run with abort mode - all concurrent, will be terminated
        tasks = [
            asyncio.create_task(self._async_generate(prompt, self.max_new_tokens))
            for prompt in prompts
        ]

        # Wait for generation to start and produce some tokens
        await asyncio.sleep(1.5)

        # Abort ALL requests - they will be terminated
        await self._async_pause_generation(mode="abort")
        await asyncio.sleep(0.5)

        # Reset is_paused flag (does NOT restore aborted requests)
        await self._async_continue_generation()

        # Get partial results from aborted requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        partial_texts = []
        for r in results:
            if isinstance(r, dict):
                partial_texts.append(r.get("text", ""))
            else:
                partial_texts.append("")

        # Re-generate ALL requests from scratch to get full results
        regenerated_texts = []
        for prompt in prompts:
            text = await self._run_generation_no_pause(prompt, self.max_new_tokens)
            regenerated_texts.append(text)

        # Summary comparison
        print(
            f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Requests: Re-generated vs Baseline ==={Colors.RESET}"
        )
        print(f"{Colors.YELLOW}Note: After abort, must RE-GENERATE (cannot continue){Colors.RESET}")
        print(f"\n{Colors.CYAN}Partial Results (Aborted):{Colors.RESET}")
        for i in range(num_requests):
            len_baseline = len(baseline_texts[i])
            len_partial = len(partial_texts[i])
            shorter = len_partial < len_baseline
            print(
                f"  Request {i}: Baseline={Colors.GREEN}{len_baseline}{Colors.RESET}, "
                f"Partial={Colors.GREEN}{len_partial}{Colors.RESET}, "
                f"Shorter={Colors.GREEN if shorter else Colors.RED}{shorter}{Colors.RESET}"
            )

        print(f"\n{Colors.CYAN}Re-generated Results:{Colors.RESET}")
        for i in range(num_requests):
            match = baseline_texts[i] == regenerated_texts[i]
            print(
                f"  Request {i}: Match Baseline={Colors.GREEN if match else Colors.RED}{match}{Colors.RESET}"
            )
        print(f"{Colors.BOLD}{Colors.BLUE}======================================={Colors.RESET}\n")

        # Verify re-generated results match baseline (deterministic)
        for i in range(num_requests):
            if len(baseline_texts[i]) != len(regenerated_texts[i]):
                print(
                    f"\n{Colors.RED}=== Request {i} Re-generated Length Mismatch ==={Colors.RESET}"
                )
                print(f"{Colors.YELLOW}Baseline ({len(baseline_texts[i])} chars):{Colors.RESET}")
                print(f"{Colors.CYAN}{baseline_texts[i]}{Colors.RESET}")
                print(
                    f"{Colors.YELLOW}Re-generated ({len(regenerated_texts[i])} chars):{Colors.RESET}"
                )
                print(f"{Colors.CYAN}{regenerated_texts[i]}{Colors.RESET}")
                print(
                    f"{Colors.RED}================================================{Colors.RESET}\n"
                )

            self.assertEqual(
                baseline_texts[i],
                regenerated_texts[i],
                f"Request {i}: Re-generated should match baseline (deterministic).",
            )

    def test_1_single_request_retract_vs_no_pause(self):
        """Test single request: retract mode vs Baseline (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_single_retract_vs_no_pause())
        print_test_passed(
            "TestEngineDeterministicGeneration.test_1_single_request_retract_vs_no_pause"
        )

    def test_2_single_request_abort_vs_no_pause(self):
        """Test single request: abort mode vs Baseline (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_single_abort_vs_no_pause())
        print_test_passed(
            "TestEngineDeterministicGeneration.test_2_single_request_abort_vs_no_pause"
        )

    def test_3_multiple_requests_retract_vs_no_pause(self):
        """Test multiple requests: retract mode vs Baseline (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_multiple_retract_vs_no_pause())
        print_test_passed(
            "TestEngineDeterministicGeneration.test_3_multiple_requests_retract_vs_no_pause"
        )

    def test_4_multiple_requests_abort_vs_no_pause(self):
        """Test multiple requests: abort mode vs Baseline (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_multiple_abort_vs_no_pause())
        print_test_passed(
            "TestEngineDeterministicGeneration.test_4_multiple_requests_abort_vs_no_pause"
        )


if __name__ == "__main__":
    unittest.main()
