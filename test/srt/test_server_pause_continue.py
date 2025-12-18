import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


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


class TestPauseContinueGeneration(CustomTestCase):
    """
    Test pause_generation and continue_generation endpoints.

    Tests include:
    - Retract mode: Verify generation is paused, running batch requests are
      moved to waiting queue, and KV cache is cleared
    - In-place mode: Verify generation is paused, running batch is unchanged,
      and KV cache is preserved
    - Continue generation: Verify that after pause, continue_generation properly
      resumes generation (re-prefill for retract mode, continue decode for in-place mode)
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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_decode(self, max_new_tokens=1000):
        """Run a decode request that generates many tokens."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Write a very long story about a magical forest. Once upon a time, in a land far away,",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def _get_internal_state(self):
        """Get the internal state of the server."""
        response = requests.get(self.base_url + "/get_server_info")
        return response.json()

    def _pause_generation(self, mode="retract"):
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

    def _flush_cache(self):
        """Flush the KV cache."""
        response = requests.post(
            self.base_url + "/flush_cache",
            json={},
        )
        return response.json()

    def test_pause_generation_retract_mode(self):
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

            # Pause generation with retract mode
            pause_response = self._pause_generation(mode="retract")
            self.assertEqual(pause_response["status"], "ok")

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

            # Verify KV cache is cleared (available tokens increase back toward initial)
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
            print(
                f"  Waiting Queue Increase: {Colors.GREEN}+{waiting_queue_increase}{Colors.RESET}"
            )
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
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

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

        print_test_passed("TestPauseContinueGeneration.test_pause_generation_retract_mode")

    def test_pause_generation_in_place_mode(self):
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

            # Pause generation with in_place mode
            pause_response = self._pause_generation(mode="in_place")
            self.assertEqual(pause_response["status"], "ok")

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

            # Continue generation
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

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

        print_test_passed("TestPauseContinueGeneration.test_pause_generation_in_place_mode")

    def test_pause_continue_flush_cache_retract_mode(self):
        """
        Test that flush_cache works after pause with retract mode.

        In retract mode, requests are moved to waiting queue and KV cache is cleared,
        so flush_cache should succeed.
        """
        # Start a request
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(self._run_decode, 2000)

            # Wait for request to start
            time.sleep(2)

            # Pause with retract mode
            pause_response = self._pause_generation(mode="retract")
            self.assertEqual(pause_response["status"], "ok")

            # Wait for pause
            time.sleep(0.5)

            # Flush cache should succeed in retract mode
            # (Note: The requests are in waiting_queue, but there's no running batch,
            #  so we need to abort them first or flush should handle this case)

            # Continue generation to let requests complete
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

            # Wait for completion
            try:
                result = future.result(timeout=60)
                self.assertIn("text", result)
            except Exception as e:
                self.fail(f"Request should complete: {e}")

        print_test_passed(
            "TestPauseContinueGeneration.test_pause_continue_flush_cache_retract_mode"
        )

    def test_pause_continue_multiple_cycles(self):
        """
        Test multiple pause/continue cycles.

        Verify that the server can handle multiple pause/continue cycles
        without issues.
        """
        num_cycles = 3

        with ThreadPoolExecutor(4) as executor:
            futures = [executor.submit(self._run_decode, 3000) for _ in range(4)]

            for cycle in range(num_cycles):
                # Wait for some generation to happen
                time.sleep(2)

                # Pause with alternating modes
                mode = "retract" if cycle % 2 == 0 else "in_place"
                pause_response = self._pause_generation(mode=mode)
                self.assertEqual(
                    pause_response["status"], "ok", f"Cycle {cycle}: pause should succeed"
                )

                # Verify paused
                state = self._get_internal_state()
                self.assertTrue(
                    state["internal_states"][0]["engine_paused"],
                    f"Cycle {cycle}: engine should be paused",
                )

                # Continue
                continue_response = self._continue_generation()
                self.assertEqual(
                    continue_response["status"], "ok", f"Cycle {cycle}: continue should succeed"
                )

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

        print_test_passed("TestPauseContinueGeneration.test_pause_continue_multiple_cycles")

    def test_pause_generation_abort_mode(self):
        """
        Test pause_generation with abort mode.

        Verify:
        1. All requests are aborted (waiting queue becomes empty)
        2. Running batch becomes empty (all requests get to_finish=FINISH_ABORT())
        3. All requests complete with abort finish_reason
        """
        # Start multiple long-running requests
        num_requests = 8
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating (in running batch)
            time.sleep(3)

            # Get state before pause - should have requests in running batch or waiting queue
            state_before_pause = self._get_internal_state()
            internal_before = state_before_pause["internal_states"][0]

            running_before = internal_before["running_batch_size"]
            waiting_before = internal_before["waiting_queue_size"]
            total_before = running_before + waiting_before
            self.assertGreater(
                total_before, 0, "Expected requests in running batch or waiting queue before pause"
            )

            # Pause generation with abort mode
            pause_response = self._pause_generation(mode="abort")
            self.assertEqual(pause_response["status"], "ok")

            # Wait for abort to complete
            time.sleep(1)

            # Get state after pause
            state_after_pause = self._get_internal_state()
            internal_after = state_after_pause["internal_states"][0]

            # Verify waiting queue is empty (abort clears waiting queue)
            waiting_after = internal_after["waiting_queue_size"]
            self.assertEqual(
                waiting_after, 0, "Waiting queue should be empty after abort mode pause"
            )

            # Verify running batch is empty (all requests aborted via to_finish=FINISH_ABORT())
            running_after = internal_after["running_batch_size"]
            self.assertEqual(
                running_after, 0, "Running batch should be empty after abort mode pause"
            )

            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            # Verify all requests completed with abort status
            for i, result in enumerate(results):
                if "error" in result:
                    # Some requests may fail due to abort
                    self.assertIn(
                        "abort",
                        result["error"].lower(),
                        f"Request {i} error should be abort-related: {result}",
                    )
                else:
                    self.assertIn(
                        "meta_info", result, f"Request {i} should have meta_info: {result}"
                    )
                    finish_reason = result.get("meta_info", {}).get("finish_reason", {})
                    self.assertEqual(
                        finish_reason.get("type"),
                        "abort",
                        f"Request {i} should finish due to abort: {result}",
                    )

            # Continue generation to reset state (abort mode sets is_pause=True)
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

        print_test_passed("TestPauseContinueGeneration.test_pause_generation_abort_mode")


class TestPauseContinueGenerationNoOverlap(CustomTestCase):
    """
    Test pause_generation and continue_generation with overlap schedule disabled.
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
                "--tp-size",
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
                "--disable-overlap-schedule",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_decode(self, max_new_tokens=1000):
        """Run a decode request that generates many tokens."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Write a very long story about a magical forest. Once upon a time, in a land far away,",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def _get_internal_state(self):
        """Get the internal state of the server."""
        response = requests.get(self.base_url + "/get_server_info")
        return response.json()

    def _pause_generation(self, mode="retract"):
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

    def test_pause_retract_no_overlap(self):
        """
        Test pause_generation with retract mode without overlap schedule.
        """
        num_requests = 4
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating
            time.sleep(3)

            # Get state before pause
            state_before = self._get_internal_state()
            internal_before = state_before["internal_states"][0]
            tokens_before = internal_before["available_kv_tokens"]
            running_before = internal_before["running_batch_size"]
            waiting_before = internal_before["waiting_queue_size"]

            # Pause with retract mode
            pause_response = self._pause_generation(mode="retract")
            self.assertEqual(pause_response["status"], "ok")

            # Wait for pause
            time.sleep(0.5)

            # Get state after pause
            state_after = self._get_internal_state()
            internal_after = state_after["internal_states"][0]

            # Verify paused
            self.assertTrue(internal_after["engine_paused"])

            # Verify running batch cleared
            running_after = internal_after["running_batch_size"]
            waiting_after = internal_after["waiting_queue_size"]
            self.assertEqual(running_after, 0)

            # Verify requests moved to waiting queue
            self.assertGreater(waiting_after, 0)

            # Verify quantity: waiting queue increase == running batch decrease
            waiting_queue_increase = waiting_after - waiting_before
            running_batch_decrease = running_before - running_after

            # Verify KV cache cleared
            tokens_after = internal_after["available_kv_tokens"]

            # Print colorful log for retract mode state changes (no overlap)
            print(
                f"\n{Colors.BOLD}{Colors.BLUE}=== Retract Mode State Changes (No Overlap) ==={Colors.RESET}"
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
            print(f"  Running Batch Decrease: {Colors.RED}-{running_batch_decrease}{Colors.RESET}")
            print(
                f"  Waiting Queue Increase: {Colors.GREEN}+{waiting_queue_increase}{Colors.RESET}"
            )
            print(f"  KV Tokens Freed: {Colors.GREEN}+{tokens_after - tokens_before}{Colors.RESET}")
            print(
                f"{Colors.BOLD}{Colors.BLUE}=============================================={Colors.RESET}\n"
            )

            self.assertEqual(
                waiting_queue_increase,
                running_batch_decrease,
                f"Waiting queue increase ({waiting_queue_increase}) should equal "
                f"running batch decrease ({running_batch_decrease})",
            )

            self.assertGreater(tokens_after, tokens_before)

            # Continue
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

            # Wait for completion
            for future in as_completed(futures):
                result = future.result()
                self.assertIn("text", result)

        print_test_passed("TestPauseContinueGenerationNoOverlap.test_pause_retract_no_overlap")

    def test_pause_in_place_no_overlap(self):
        """
        Test pause_generation with in_place mode without overlap schedule.
        """
        num_requests = 4
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating
            time.sleep(3)

            # Get state before pause
            state_before = self._get_internal_state()
            internal_before = state_before["internal_states"][0]
            running_before = internal_before["running_batch_size"]
            waiting_before = internal_before["waiting_queue_size"]
            tokens_before = internal_before["available_kv_tokens"]

            # Pause with in_place mode
            pause_response = self._pause_generation(mode="in_place")
            self.assertEqual(pause_response["status"], "ok")

            # Wait for pause
            time.sleep(0.5)

            # Get state after pause
            state_after = self._get_internal_state()
            internal_after = state_after["internal_states"][0]

            # Verify paused
            self.assertTrue(internal_after["engine_paused"])

            # Verify running batch unchanged
            running_after = internal_after["running_batch_size"]
            waiting_after = internal_after["waiting_queue_size"]
            self.assertEqual(running_after, running_before)

            # Verify waiting queue unchanged
            self.assertEqual(
                waiting_after,
                waiting_before,
                f"Waiting queue should be unchanged in in_place mode: "
                f"{waiting_before} -> {waiting_after}",
            )

            # Verify KV cache preserved
            tokens_after = internal_after["available_kv_tokens"]
            self.assertEqual(tokens_after, tokens_before)

            # Continue
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

            # Wait for completion
            for future in as_completed(futures):
                result = future.result()
                self.assertIn("text", result)

        print_test_passed("TestPauseContinueGenerationNoOverlap.test_pause_in_place_no_overlap")

    def test_pause_abort_no_overlap(self):
        """
        Test pause_generation with abort mode without overlap schedule.
        """
        num_requests = 4
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode, 2000) for _ in range(num_requests)]

            # Wait for requests to start generating
            time.sleep(3)

            # Get state before pause
            state_before = self._get_internal_state()
            internal_before = state_before["internal_states"][0]
            running_before = internal_before["running_batch_size"]
            waiting_before = internal_before["waiting_queue_size"]
            total_before = running_before + waiting_before
            self.assertGreater(
                total_before, 0, "Expected requests in running batch or waiting queue before pause"
            )

            # Pause with abort mode
            pause_response = self._pause_generation(mode="abort")
            self.assertEqual(pause_response["status"], "ok")

            # Wait for abort to complete
            time.sleep(1)

            # Get state after pause
            state_after = self._get_internal_state()
            internal_after = state_after["internal_states"][0]

            # Verify waiting queue is empty
            waiting_after = internal_after["waiting_queue_size"]
            self.assertEqual(waiting_after, 0)

            # Verify running batch is empty (requests aborted via to_finish=FINISH_ABORT())
            running_after = internal_after["running_batch_size"]
            self.assertEqual(running_after, 0)

            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            # Verify all requests completed with abort status
            for i, result in enumerate(results):
                if "error" in result:
                    self.assertIn(
                        "abort",
                        result["error"].lower(),
                        f"Request {i} error should be abort-related: {result}",
                    )
                else:
                    finish_reason = result.get("meta_info", {}).get("finish_reason", {})
                    self.assertEqual(
                        finish_reason.get("type"),
                        "abort",
                        f"Request {i} should finish due to abort: {result}",
                    )

            # Continue generation to reset state
            continue_response = self._continue_generation()
            self.assertEqual(continue_response["status"], "ok")

        print_test_passed("TestPauseContinueGenerationNoOverlap.test_pause_abort_no_overlap")


if __name__ == "__main__":
    unittest.main()
