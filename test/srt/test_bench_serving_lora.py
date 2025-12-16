import unittest

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
)

# model information
BASE_MODEL = "Qwen/Qwen3-32B"
LORA_PATH = "flyfishxu/DeepNews-LoRA-Qwen3-32B"
LORA_NAME = "DeepNews-LoRA-Qwen3-32B"

# benchmark cases
NUM_PROMPTS_PER_CONCURRENCY = 3
INPUT_SEQ_LENS = [4096]
OUTPUT_SEQ_LENS = [1, 1024]
MAX_CONCURRENCIES = [8, 32, 64, 256]
ALLOW_GAP = 0.03
# max_concurrency:input_seq_len:ttft|itl|input_throughput|output_throughput
# Note: ttft is median_ttft_ms, itl is median_itl_ms
# Baseline: fix/perf-reconstruct-model-state
EXPECTED_PERFORMANCE = {
    8: {
        4096: {
            "ttft": 1710.23,
            "itl": 19.37,
            "input_throughput": 19063.81,
            "output_throughput": 378.54,
        }
    },
    16: {
        4096: {
            "ttft": 3420.23,
            "itl": 21.68,
            "input_throughput": 19110.85,
            "output_throughput": 650.64,
        }
    },
    32: {
        4096: {
            "ttft": 6832.81,
            "itl": 25.92,
            "input_throughput": 19138.69,
            "output_throughput": 1003.67,
        }
    },
    64: {
        4096: {
            "ttft": 13712.98,
            "itl": 28.53,
            "input_throughput": 19096.70,
            "output_throughput": 1043.96,
        }
    },
    128: {
        4096: {
            "ttft": 27423.97,
            "itl": 28.66,
            "input_throughput": 19094.30,
            "output_throughput": 1050.62,
        }
    },
    256: {
        4096: {
            "ttft": 54899.37,
            "itl": 28.78,
            "input_throughput": 19083.75,
            "output_throughput": 1073.57,
        }
    },
}


class TestBenchServingLoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BASE_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--device",
                "tpu",
                "--tp-size",
                "4",
                "--random-seed",
                "3",
                "--chunked-prefill-size",
                "2048",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.9",
                "--max-running-requests",
                "256",
                "--page-size",
                "128",
                "--disable-radix-cache",
                "--lora-paths",
                LORA_PATH,
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_r32(self):
        for input_seq_len in INPUT_SEQ_LENS:
            for output_seq_len in OUTPUT_SEQ_LENS:
                for max_concurrency in MAX_CONCURRENCIES:
                    print(f"benchmark on {input_seq_len=}, {output_seq_len=}, {max_concurrency=}")
                    num_prompts = max_concurrency * NUM_PROMPTS_PER_CONCURRENCY
                    args = get_benchmark_args(
                        base_url=self.base_url,
                        dataset_name="random",
                        num_prompts=num_prompts,
                        request_rate=float("inf"),
                        random_input_len=input_seq_len,
                        random_output_len=output_seq_len,
                        max_concurrency=max_concurrency,
                        random_range_ratio=1,
                        disable_ignore_eos=True,
                        lora_name=LORA_NAME,
                        backend="sglang-oai",
                        warmup_requests=0,
                    )
                    res = run_benchmark(args)

                    assert res["completed"] == num_prompts

                    if output_seq_len == 1:
                        self.assertGreater(
                            res["input_throughput"],
                            EXPECTED_PERFORMANCE[max_concurrency][input_seq_len]["input_throughput"]
                            * (1 - ALLOW_GAP),
                        )
                        self.assertLess(
                            res["median_ttft_ms"],
                            EXPECTED_PERFORMANCE[max_concurrency][input_seq_len]["ttft"]
                            * (1 + ALLOW_GAP),
                        )
                    else:
                        self.assertGreater(
                            res["output_throughput"],
                            EXPECTED_PERFORMANCE[max_concurrency][input_seq_len][
                                "output_throughput"
                            ]
                            * (1 - ALLOW_GAP),
                        )
                        self.assertLess(
                            res["median_itl_ms"],
                            EXPECTED_PERFORMANCE[max_concurrency][input_seq_len]["itl"]
                            * (1 + ALLOW_GAP),
                        )


if __name__ == "__main__":
    unittest.main()
