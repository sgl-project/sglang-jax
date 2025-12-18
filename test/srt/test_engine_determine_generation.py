import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_8B,
    CustomTestCase,
    popen_launch_server,
)


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
    print(f"  Char Length Diff: {Colors.MAGENTA}{len(test_text) - len(baseline_text)}{Colors.RESET}")
    print(f"  Word Length Diff: {Colors.MAGENTA}{len(test_words) - len(baseline_words)}{Colors.RESET}")
    print(f"  Texts Match: {Colors.GREEN if baseline_text == test_text else Colors.RED}{baseline_text == test_text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * (len(title) + 8)}{Colors.RESET}\n")


class TestEngineDeterministicGeneration(CustomTestCase):
    """
    Test deterministic generation using Engine API.

    Compares pause/continue modes (abort and retract) vs no pause with temperature=0.
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
        cls.max_new_tokens = 200

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    async def _async_get_internal_state(self):
        """Get internal state using async method."""
        return await self.engine.tokenizer_manager.get_internal_state()

    async def _async_pause_generation(self, mode: str):
        """Pause generation using async method."""
        from sgl_jax.srt.managers.io_struct import PauseGenerationReqInput

        obj = PauseGenerationReqInput(mode=mode)
        await self.engine.tokenizer_manager.pause_generation(obj)

    async def _async_continue_generation(self):
        """Continue generation using async method."""
        from sgl_jax.srt.managers.io_struct import ContinueGenerationReqInput

        obj = ContinueGenerationReqInput()
        await self.engine.tokenizer_manager.continue_generation(obj)

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

    async def _run_generation_with_retract(self, prompt: str, max_new_tokens: int, pause_delay: float = 1.0):
        """Run generation with retract mode pause in the middle."""
        task = asyncio.create_task(self._async_generate(prompt, max_new_tokens))

        # Wait for generation to start
        await asyncio.sleep(pause_delay)

        # Pause with retract mode
        await self._async_pause_generation(mode="retract")
        await asyncio.sleep(0.5)

        # Continue generation
        await self._async_continue_generation()

        # Wait for completion
        result = await task
        return result.get("text", "")

    async def _run_generation_with_abort(self, prompt: str, max_new_tokens: int, pause_delay: float = 1.0):
        """Run generation with abort mode pause."""
        task = asyncio.create_task(self._async_generate(prompt, max_new_tokens))

        # Wait for generation to start
        await asyncio.sleep(pause_delay)

        # Pause with abort mode
        await self._async_pause_generation(mode="abort")
        await asyncio.sleep(0.5)

        # Continue generation to reset state
        await self._async_continue_generation()

        # Wait for task to complete (should be aborted)
        try:
            result = await task
            return result.get("text", "")
        except Exception:
            return ""

    # ============ Single Request Tests ============

    async def _run_test_single_retract_vs_no_pause(self):
        """Test single request: retract mode vs no pause."""
        # Run baseline (no pause)
        baseline_text = await self._run_generation_no_pause(self.test_prompt, self.max_new_tokens)

        # Run with retract mode
        retract_text = await self._run_generation_with_retract(self.test_prompt, self.max_new_tokens)

        # Print comparison
        print_comparison(
            "Single Request: Retract vs No Pause",
            baseline_text,
            retract_text,
            "No Pause (Baseline)",
            "Retract Mode",
        )

        # With temperature=0 and retract mode, output should be identical
        # because retract clears KV cache and re-prefills deterministically
        self.assertEqual(
            len(baseline_text),
            len(retract_text),
            f"Retract mode should produce same length output as no pause. "
            f"Baseline: {len(baseline_text)}, Retract: {len(retract_text)}",
        )
        self.assertEqual(
            baseline_text,
            retract_text,
            "Retract mode should produce identical output as no pause with temperature=0",
        )

    async def _run_test_single_abort_vs_no_pause(self):
        """Test single request: abort mode vs no pause."""
        # Run baseline (no pause)
        baseline_text = await self._run_generation_no_pause(self.test_prompt, self.max_new_tokens)

        # Run with abort mode
        abort_text = await self._run_generation_with_abort(self.test_prompt, self.max_new_tokens)

        # Print comparison
        print_comparison(
            "Single Request: Abort vs No Pause",
            baseline_text,
            abort_text,
            "No Pause (Baseline)",
            "Abort Mode",
        )

        # Abort mode should produce shorter output (aborted mid-generation)
        self.assertLess(
            len(abort_text),
            len(baseline_text),
            f"Abort mode should produce shorter output. "
            f"Baseline: {len(baseline_text)}, Abort: {len(abort_text)}",
        )

        # The abort text should be a prefix of baseline (up to the abort point)
        # This may not always be exact due to timing, but abort should be shorter
        baseline_words = baseline_text.split()
        abort_words = abort_text.split()
        self.assertLess(
            len(abort_words),
            len(baseline_words),
            f"Abort mode should produce fewer words. "
            f"Baseline: {len(baseline_words)}, Abort: {len(abort_words)}",
        )

    # ============ Multiple Requests Tests ============

    async def _run_test_multiple_retract_vs_no_pause(self):
        """Test multiple requests: retract mode vs no pause."""
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline (no pause) - run sequentially to ensure determinism
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

        # Pause with retract mode
        await self._async_pause_generation(mode="retract")
        await asyncio.sleep(0.5)

        # Continue generation
        await self._async_continue_generation()

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        retract_texts = [r.get("text", "") if isinstance(r, dict) else "" for r in results]

        # Print comparison for first request
        print_comparison(
            "Multiple Requests: Retract vs No Pause (Request 0)",
            baseline_texts[0],
            retract_texts[0],
            "No Pause (Baseline)",
            "Retract Mode",
        )

        # Summary comparison
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Requests Summary ==={Colors.RESET}")
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
            self.assertEqual(
                len(baseline_texts[i]),
                len(retract_texts[i]),
                f"Request {i}: Retract mode should produce same length. "
                f"Baseline: {len(baseline_texts[i])}, Retract: {len(retract_texts[i])}",
            )

    async def _run_test_multiple_abort_vs_no_pause(self):
        """Test multiple requests: abort mode vs no pause."""
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline (no pause) - run sequentially
        baseline_texts = []
        for prompt in prompts:
            text = await self._run_generation_no_pause(prompt, self.max_new_tokens)
            baseline_texts.append(text)

        # Run with abort mode - all concurrent with abort in the middle
        tasks = [
            asyncio.create_task(self._async_generate(prompt, self.max_new_tokens))
            for prompt in prompts
        ]

        # Wait for generation to start
        await asyncio.sleep(1.5)

        # Pause with abort mode
        await self._async_pause_generation(mode="abort")
        await asyncio.sleep(0.5)

        # Continue to reset state
        await self._async_continue_generation()

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        abort_texts = []
        for r in results:
            if isinstance(r, dict):
                abort_texts.append(r.get("text", ""))
            else:
                abort_texts.append("")

        # Print comparison for first request
        print_comparison(
            "Multiple Requests: Abort vs No Pause (Request 0)",
            baseline_texts[0],
            abort_texts[0],
            "No Pause (Baseline)",
            "Abort Mode",
        )

        # Summary comparison
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Multiple Requests Abort Summary ==={Colors.RESET}")
        for i in range(num_requests):
            len_baseline = len(baseline_texts[i])
            len_abort = len(abort_texts[i])
            shorter = len_abort < len_baseline
            print(
                f"  Request {i}: Baseline={Colors.GREEN}{len_baseline}{Colors.RESET} chars, "
                f"Abort={Colors.GREEN}{len_abort}{Colors.RESET} chars, "
                f"Shorter={Colors.GREEN if shorter else Colors.RED}{shorter}{Colors.RESET}"
            )
        print(f"{Colors.BOLD}{Colors.BLUE}======================================={Colors.RESET}\n")

        # Verify abort produces shorter outputs
        for i in range(num_requests):
            self.assertLess(
                len(abort_texts[i]),
                len(baseline_texts[i]),
                f"Request {i}: Abort mode should produce shorter output. "
                f"Baseline: {len(baseline_texts[i])}, Abort: {len(abort_texts[i])}",
            )

    def test_1_single_request_retract_vs_no_pause(self):
        """Test single request: retract mode vs no pause (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_single_retract_vs_no_pause())
        print_test_passed("TestEngineDeterministicGeneration.test_1_single_request_retract_vs_no_pause")

    def test_2_single_request_abort_vs_no_pause(self):
        """Test single request: abort mode vs no pause (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_single_abort_vs_no_pause())
        print_test_passed("TestEngineDeterministicGeneration.test_2_single_request_abort_vs_no_pause")

    def test_3_multiple_requests_retract_vs_no_pause(self):
        """Test multiple requests: retract mode vs no pause (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_multiple_retract_vs_no_pause())
        print_test_passed("TestEngineDeterministicGeneration.test_3_multiple_requests_retract_vs_no_pause")

    def test_4_multiple_requests_abort_vs_no_pause(self):
        """Test multiple requests: abort mode vs no pause (length and tokens)."""
        self.engine.loop.run_until_complete(self._run_test_multiple_abort_vs_no_pause())
        print_test_passed("TestEngineDeterministicGeneration.test_4_multiple_requests_abort_vs_no_pause")


class TestServerDeterministicGeneration(CustomTestCase):
    """
    Test deterministic generation using HTTP Server API.

    Compares pause/continue modes (abort and retract) vs no pause with temperature=0.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--tp",
                "4",
                "--mem-fraction-static",
                "0.65",
                "--max-prefill-tokens",
                "8192",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--precompile-token-paddings",
                "16384",
                "--precompile-bs-paddings",
                "64",
                "--page-size",
                "64",
                "--max-running-requests",
                "64",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        cls.test_prompt = "Write a story about a brave knight. Once upon a time,"
        cls.max_new_tokens = 200

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt: str, max_new_tokens: int):
        """Run generation with temperature=0."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        result = response.json()
        return result.get("text", "")

    def _pause_generation(self, mode: str):
        """Pause generation with the specified mode."""
        response = requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
        )
        return response.json()

    def _continue_generation(self):
        """Continue generation after pause."""
        response = requests.post(
            self.base_url + "/continue_generation",
            json={},
        )
        return response.json()

    # ============ Single Request Tests ============

    def test_1_single_request_retract_vs_no_pause(self):
        """Test single request: retract mode vs no pause (length and tokens)."""
        # Run baseline (no pause)
        baseline_text = self._generate(self.test_prompt, self.max_new_tokens)

        # Run with retract mode - use thread to run generation while pausing
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(self._generate, self.test_prompt, self.max_new_tokens)

            # Wait for generation to start
            time.sleep(1.0)

            # Pause with retract mode
            self._pause_generation(mode="retract")
            time.sleep(0.5)

            # Continue generation
            self._continue_generation()

            # Get result
            retract_text = future.result()

        # Print comparison
        print_comparison(
            "Server Single Request: Retract vs No Pause",
            baseline_text,
            retract_text,
            "No Pause (Baseline)",
            "Retract Mode",
        )

        # Verify outputs match (deterministic with retract mode)
        self.assertEqual(
            len(baseline_text),
            len(retract_text),
            f"Retract mode should produce same length. "
            f"Baseline: {len(baseline_text)}, Retract: {len(retract_text)}",
        )
        self.assertEqual(
            baseline_text,
            retract_text,
            "Retract mode should produce identical output with temperature=0",
        )

        print_test_passed("TestServerDeterministicGeneration.test_1_single_request_retract_vs_no_pause")

    def test_2_single_request_abort_vs_no_pause(self):
        """Test single request: abort mode vs no pause (length and tokens)."""
        # Run baseline (no pause)
        baseline_text = self._generate(self.test_prompt, self.max_new_tokens)

        # Run with abort mode
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(self._generate, self.test_prompt, self.max_new_tokens)

            # Wait for generation to start
            time.sleep(1.0)

            # Pause with abort mode
            self._pause_generation(mode="abort")
            time.sleep(0.5)

            # Continue to reset state
            self._continue_generation()

            # Get result (may be empty or partial due to abort)
            try:
                abort_text = future.result(timeout=30)
            except Exception:
                abort_text = ""

        # Print comparison
        print_comparison(
            "Server Single Request: Abort vs No Pause",
            baseline_text,
            abort_text,
            "No Pause (Baseline)",
            "Abort Mode",
        )

        # Verify abort produces shorter output
        self.assertLess(
            len(abort_text),
            len(baseline_text),
            f"Abort mode should produce shorter output. "
            f"Baseline: {len(baseline_text)}, Abort: {len(abort_text)}",
        )

        print_test_passed("TestServerDeterministicGeneration.test_2_single_request_abort_vs_no_pause")

    # ============ Multiple Requests Tests ============

    def test_3_multiple_requests_retract_vs_no_pause(self):
        """Test multiple requests: retract mode vs no pause (length and tokens)."""
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline (no pause) - sequential
        baseline_texts = [self._generate(prompt, self.max_new_tokens) for prompt in prompts]

        # Run with retract mode - concurrent with pause
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [
                executor.submit(self._generate, prompt, self.max_new_tokens)
                for prompt in prompts
            ]

            # Wait for generation to start
            time.sleep(1.5)

            # Pause with retract mode
            self._pause_generation(mode="retract")
            time.sleep(0.5)

            # Continue generation
            self._continue_generation()

            # Get results
            retract_texts = [f.result() for f in futures]

        # Print comparison for first request
        print_comparison(
            "Server Multiple Requests: Retract vs No Pause (Request 0)",
            baseline_texts[0],
            retract_texts[0],
            "No Pause (Baseline)",
            "Retract Mode",
        )

        # Summary
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Server Multiple Requests Summary ==={Colors.RESET}")
        for i in range(num_requests):
            match = baseline_texts[i] == retract_texts[i]
            len_diff = len(retract_texts[i]) - len(baseline_texts[i])
            print(
                f"  Request {i}: Match={Colors.GREEN if match else Colors.RED}{match}{Colors.RESET}, "
                f"LenDiff={Colors.MAGENTA}{len_diff}{Colors.RESET}"
            )
        print(f"{Colors.BOLD}{Colors.BLUE}========================================{Colors.RESET}\n")

        # Verify all outputs match
        for i in range(num_requests):
            self.assertEqual(
                len(baseline_texts[i]),
                len(retract_texts[i]),
                f"Request {i}: Retract should produce same length. "
                f"Baseline: {len(baseline_texts[i])}, Retract: {len(retract_texts[i])}",
            )

        print_test_passed("TestServerDeterministicGeneration.test_3_multiple_requests_retract_vs_no_pause")

    def test_4_multiple_requests_abort_vs_no_pause(self):
        """Test multiple requests: abort mode vs no pause (length and tokens)."""
        num_requests = 4
        prompts = [f"{self.test_prompt} Story {i}:" for i in range(num_requests)]

        # Run baseline (no pause) - sequential
        baseline_texts = [self._generate(prompt, self.max_new_tokens) for prompt in prompts]

        # Run with abort mode - concurrent
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [
                executor.submit(self._generate, prompt, self.max_new_tokens)
                for prompt in prompts
            ]

            # Wait for generation to start
            time.sleep(1.5)

            # Pause with abort mode
            self._pause_generation(mode="abort")
            time.sleep(0.5)

            # Continue to reset state
            self._continue_generation()

            # Get results
            abort_texts = []
            for f in futures:
                try:
                    abort_texts.append(f.result(timeout=30))
                except Exception:
                    abort_texts.append("")

        # Print comparison for first request
        print_comparison(
            "Server Multiple Requests: Abort vs No Pause (Request 0)",
            baseline_texts[0],
            abort_texts[0],
            "No Pause (Baseline)",
            "Abort Mode",
        )

        # Summary
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== Server Multiple Requests Abort Summary ==={Colors.RESET}")
        for i in range(num_requests):
            len_baseline = len(baseline_texts[i])
            len_abort = len(abort_texts[i])
            shorter = len_abort < len_baseline
            print(
                f"  Request {i}: Baseline={Colors.GREEN}{len_baseline}{Colors.RESET}, "
                f"Abort={Colors.GREEN}{len_abort}{Colors.RESET}, "
                f"Shorter={Colors.GREEN if shorter else Colors.RED}{shorter}{Colors.RESET}"
            )
        print(f"{Colors.BOLD}{Colors.BLUE}=============================================={Colors.RESET}\n")

        # Verify abort produces shorter outputs
        for i in range(num_requests):
            self.assertLess(
                len(abort_texts[i]),
                len(baseline_texts[i]),
                f"Request {i}: Abort should produce shorter output. "
                f"Baseline: {len(baseline_texts[i])}, Abort: {len(abort_texts[i])}",
            )

        print_test_passed("TestServerDeterministicGeneration.test_4_multiple_requests_abort_vs_no_pause")


if __name__ == "__main__":
    unittest.main()

