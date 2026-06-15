"""Server-infrastructure smoke tests for nightly CI.

Three tests verify fundamental server paths using Qwen3-1.7B:
  1. Radix cache consistency: prefix reuse, padding/non-padding, page-crossing
  2. Request logger: sequential + concurrent field completeness, no dropped entries
  3. Bench-serving self-check: queueing under load, variable-length batching
"""

import os
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
_BASE_URL = DEFAULT_URL_FOR_TEST

_COMMON_SERVER_ARGS = [
    "--skip-server-warmup",
    "--dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "16",
    "--page-size",
    "64",
    "--random-seed",
    "42",
]

_SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 32}

_PROMPT_A = "The capital of France is"
_PROMPT_A_EXTENDED = "The capital of France is Paris. The capital of Germany is"
_PROMPT_LONG = (
    "In the field of artificial intelligence, large language models have shown "
    "remarkable capabilities in natural language processing. These models can "
    "perform text generation, translation, summarization, and question answering. "
    "The most important breakthrough in this area is"
)
# max_new_tokens=80 crosses page boundary with page_size=64
_RADIX_SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 80}


def _generate(base_url, prompt, sampling_params):
    response = requests.post(
        f"{base_url}/generate",
        json={"text": prompt, "sampling_params": sampling_params},
    )
    response.raise_for_status()
    return response.json()


def _kill_and_wait(process, timeout=5):
    kill_process_tree(process.pid)
    try:
        process.wait(timeout=timeout)
    except Exception:
        pass


class TestRadixCacheConsistency(CustomTestCase):

    def test_radix_on_off_consistency(self):
        test_prompts = [
            ("short", _PROMPT_A),
            ("prefix_hit", _PROMPT_A_EXTENDED),
            ("long", _PROMPT_LONG),
        ]

        outputs = {}
        for mode, extra_args in [
            ("radix_on", []),
            ("radix_off", ["--disable-radix-cache"]),
        ]:
            process = popen_launch_server(
                _MODEL,
                _BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[*_COMMON_SERVER_ARGS, *extra_args],
            )
            try:
                for name, prompt in test_prompts:
                    result = _generate(_BASE_URL, prompt, _RADIX_SAMPLING_PARAMS)
                    outputs[(mode, name)] = result["text"]

                if mode == "radix_on":
                    result = _generate(_BASE_URL, _PROMPT_A, _RADIX_SAMPLING_PARAMS)
                    outputs[("radix_on", "short_rehit")] = result["text"]
            finally:
                _kill_and_wait(process)
                time.sleep(2)

        for name, _ in test_prompts:
            self.assertEqual(
                outputs[("radix_on", name)],
                outputs[("radix_off", name)],
                f"Radix on/off outputs differ for '{name}' with greedy sampling",
            )

        self.assertEqual(
            outputs[("radix_on", "short")],
            outputs[("radix_on", "short_rehit")],
            "Radix cache hit produced different output for re-sent short prompt",
        )


class TestRequestLogger(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._log_path = tempfile.mkstemp(suffix=".log")
        cls._log_fh = os.fdopen(fd, "w")
        cls.process = popen_launch_server(
            _MODEL,
            _BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_SERVER_ARGS,
                "--disable-radix-cache",
                "--log-requests",
                "--log-requests-level",
                "2",
            ],
            return_stdout_stderr=(subprocess.DEVNULL, cls._log_fh),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            _kill_and_wait(cls.process)
        if getattr(cls, "_log_fh", None) is not None and not cls._log_fh.closed:
            try:
                cls._log_fh.close()
            except Exception:
                pass
        if getattr(cls, "_log_path", None):
            try:
                os.unlink(cls._log_path)
            except FileNotFoundError:
                pass

    def test_request_log_completeness_and_count(self):
        num_requests = 16

        marker_seq = f"SEQ_{time.time_ns()}"
        for i in range(num_requests):
            _generate(
                _BASE_URL,
                f"Prompt {marker_seq} #{i}: {_PROMPT_A}",
                _SAMPLING_PARAMS,
            )

        marker_conc = f"CONC_{time.time_ns()}"
        with ThreadPoolExecutor(max_workers=num_requests) as pool:
            futures = [
                pool.submit(
                    _generate,
                    _BASE_URL,
                    f"Prompt {marker_conc} #{i}: {_PROMPT_A}",
                    _SAMPLING_PARAMS,
                )
                for i in range(num_requests)
            ]
            for f in as_completed(futures):
                f.result()

        time.sleep(2)

        with open(self._log_path) as f:
            log_content = f.read()

        lines = log_content.splitlines()

        for marker, label in [
            (marker_seq, "sequential"),
            (marker_conc, "concurrent"),
        ]:
            receive = [ln for ln in lines if "Receive: obj=" in ln and marker in ln]
            finish = [ln for ln in lines if "Finish: obj=" in ln and marker in ln]

            self.assertEqual(
                len(receive),
                num_requests,
                f"Expected {num_requests} Receive entries for {label}, got {len(receive)}",
            )
            self.assertEqual(
                len(finish),
                num_requests,
                f"Expected {num_requests} Finish entries for {label}, got {len(finish)}",
            )

            for line in finish:
                for field in ("rid=", "text=", "sampling_params="):
                    self.assertIn(
                        field,
                        line,
                        f"Finish log line missing '{field}' in {label} batch: {line[:200]}",
                    )


class TestBenchServingSelf(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = _BASE_URL
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_SERVER_ARGS,
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            _kill_and_wait(cls.process)

    def test_bench_serving_runs_to_completion(self):
        num_prompts = 32
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            device="tpu",
            num_prompts=num_prompts,
            request_rate=float("inf"),
            random_input_len=128,
            random_output_len=16,
            max_concurrency=16,
            random_range_ratio=0.5,
        )
        res = run_benchmark(args)

        self.assertEqual(
            res["completed"],
            num_prompts,
            f"Only {res['completed']}/{num_prompts} completed",
        )
        self.assertGreater(res["output_throughput"], 0)


if __name__ == "__main__":
    unittest.main()
