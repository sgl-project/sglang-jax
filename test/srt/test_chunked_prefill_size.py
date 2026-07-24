import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests
from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestChunkedPrefillSize(CustomTestCase):
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
                "--tp",
                "4",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.8",
                "--download-dir",
                "/dev/shm",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--chunked-prefill-size",
                "2048",
                "--page-size",
                "128",
                "--disable-radix-cache",
                "--enable-mixed-chunk",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestChunkedPrefillAbortPressure(CustomTestCase):
    output_tokens = 8
    page_size = 64

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            check_cache_miss=False,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--tp",
                "4",
                "--dp-size",
                "2",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.8",
                "--download-dir",
                "/dev/shm",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "8",
                "--max-total-tokens",
                "4096",
                "--chunked-prefill-size",
                "128",
                "--page-size",
                str(cls.page_size),
                "--disable-radix-cache",
                "--enable-mixed-chunk",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _state(self):
        response = requests.get(self.base_url + "/get_server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]

    def _generate(self, rid, input_ids):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "rid": rid,
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.output_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def _wait_for_state(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        state = None
        while time.monotonic() < deadline:
            try:
                state = self._state()
            except requests.RequestException:
                time.sleep(0.05)
                continue
            if predicate(state):
                return state
            time.sleep(0.05)
        self.fail(f"scheduler did not reach the expected state: {state}")

    def test_stress_abort_recovers_chunked_owners_and_capacity(self):
        initial = self._wait_for_state(
            lambda state: state["waiting_queue_size"] == 0
            and state["pending_dp_reqs_size"] == 0
            and state["running_batch_size"] == 0
            and state["chunked_req_is_none"]
            and state["req_to_token_pool_available"] == state["req_to_token_pool_total"],
            timeout=30,
        )
        token_capacity = initial["available_kv_tokens_per_dp"]
        paged_prompt_len = (min(token_capacity) // self.page_size - 3) * self.page_size
        prompt_len = paged_prompt_len - self.page_size // 2
        self.assertGreater(prompt_len, 128)
        base = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]
        prompt = (base * ((prompt_len + len(base) - 1) // len(base)))[:prompt_len]

        warm_up = self._generate("chunked-pressure-warm-up", prompt)
        self.assertEqual(warm_up["meta_info"]["finish_reason"]["type"], "length")
        self._wait_for_state(
            lambda state: state["req_to_token_pool_available"] == state["req_to_token_pool_total"]
            and state["available_kv_tokens_per_dp"] == token_capacity,
            timeout=30,
        )

        req_capacity = initial["req_to_token_pool_total"]
        rids = [f"chunked-pressure-{i:02d}" for i in range(12)]

        with ThreadPoolExecutor(max_workers=len(rids)) as executor:
            futures = {rid: executor.submit(self._generate, rid, prompt) for rid in rids}
            pressure_state = self._wait_for_state(
                lambda state: sum(rid is not None for rid in state["chunked_req_rids"]) == 2
                and state["waiting_queue_size"] + state["pending_dp_reqs_size"] > 0
                and all(
                    available <= capacity // 2
                    for available, capacity in zip(
                        state["available_kv_tokens_per_dp"], token_capacity
                    )
                ),
                timeout=120,
            )
            aborted_rids = [rid for rid in pressure_state["chunked_req_rids"] if rid is not None]
            for rid in aborted_rids:
                response = requests.post(
                    self.base_url + "/abort_request", json={"rid": rid}, timeout=10
                )
                response.raise_for_status()

            outputs = {rid: future.result(timeout=180) for rid, future in futures.items()}

        for rid in aborted_rids:
            self.assertEqual(outputs[rid]["meta_info"]["finish_reason"]["type"], "abort")
            self.assertEqual(outputs[rid]["output_ids"], [])

        live_outputs = [outputs[rid] for rid in rids if rid not in aborted_rids]
        self.assertTrue(live_outputs)
        for output in live_outputs:
            self.assertEqual(output["meta_info"]["finish_reason"]["type"], "length")
            self.assertEqual(len(output["output_ids"]), self.output_tokens)

        self._wait_for_state(
            lambda state: state["waiting_queue_size"] == 0
            and state["pending_dp_reqs_size"] == 0
            and state["running_batch_size"] == 0
            and state["chunked_req_is_none"]
            and state["req_to_token_pool_available"] == req_capacity
            and state["available_kv_tokens_per_dp"] == token_capacity,
            timeout=30,
        )

        follow_up = self._generate("chunked-pressure-follow-up", prompt)
        self.assertEqual(follow_up["meta_info"]["finish_reason"]["type"], "length")
        self.assertEqual(len(follow_up["output_ids"]), self.output_tokens)


if __name__ == "__main__":
    unittest.main()
