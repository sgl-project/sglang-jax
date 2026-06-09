import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA4_26B_A4B_IT,
    GEMMA4_31B_IT,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

_orig_from_pretrained = AutoTokenizer.from_pretrained


def _patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    if "gemma-4" in str(pretrained_model_name_or_path):
        kwargs["extra_special_tokens"] = {}
    return _orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


AutoTokenizer.from_pretrained = _patched_from_pretrained


class TestGemma4Model(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GEMMA4_31B_IT
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
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.8",
                "--load-format",
                "dummy",
                "--precompile-bs-paddings",
                "32",
                "--precompile-token-paddings",
                "4096",
                "--attention-backend",
                "fa",
                "--page-size",
                "64",
                "--max-running-requests",
                "32",
                "--chunked-prefill-size",
                "4096",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_skip(self):
        pass

    def test_input_throughput_default_tp_4(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            device="tpu",
            num_prompts=500,
            request_rate=float("inf"),
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)

        assert res["completed"] == 500

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default_tp_4\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 64960)

    def test_output_throughput_default_tp_4(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            device="tpu",
            num_prompts=512,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 512

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default_tp_4\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 9866)


class TestGemma4MoEModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GEMMA4_26B_A4B_IT
        # Use a distinct port to avoid TIME_WAIT socket conflicts between test classes
        pfx, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
        cls.base_url = f"{pfx}:{int(port) + 10}"
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
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.8",
                "--load-format",
                "dummy",
                "--precompile-bs-paddings",
                "32",
                "--precompile-token-paddings",
                "4096",
                "--attention-backend",
                "fa",
                "--page-size",
                "64",
                "--max-running-requests",
                "32",
                "--chunked-prefill-size",
                "4096",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_skip(self):
        pass

    def test_input_throughput_default_tp_4(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            device="tpu",
            num_prompts=500,
            request_rate=float("inf"),
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)

        assert res["completed"] == 500

        if is_in_ci():
            write_github_step_summary(
                f"### TestGemma4MoEModel test_input_throughput_default_tp_4\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 50000)

    def test_output_throughput_default_tp_4(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            device="tpu",
            num_prompts=512,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
        )
        res = run_benchmark(args)
        assert res["completed"] == 512

        if is_in_ci():
            write_github_step_summary(
                f"### TestGemma4MoEModel test_output_throughput_default_tp_4\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 8000)


if __name__ == "__main__":
    unittest.main()
