import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_5_27B,
    QWEN3_5_35B_A3B,
    CustomTestCase,
    popen_launch_server,
)


def _run_mmlu_smoke(test, base_url, model):
    # Thinking mode + Qwen3 card sampling.
    # enable_thinking must be explicit (qwen3 parser defaults it ON).
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=100,
        num_threads=16,
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        seed=17,
        chat_template_kwargs={"enable_thinking": True},
    )
    metrics = run_eval(args)
    test.assertGreater(metrics["score"], 0.70)


class TestQwen35MoeModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_5_35B_A3B
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
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--nnodes",
                "1",
                "--ep-size",
                "4",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--mem-fraction-static",
                "0.8",
                "--chunked-prefill-size",
                "512",
                "--page-size",
                "64",
                "--max-running-requests",
                "16",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "16",
                "512",
                "1024",
            ],
            env={"JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu_smoke(self):
        _run_mmlu_smoke(self, self.base_url, self.model)


class TestQwen35DenseModel(CustomTestCase):
    """Dense Qwen3.5 (validated on 27B). Same hybrid backbone as the MoE class,
    but a plain SwiGLU FFN, so no expert-parallel args: drop ``--ep-size`` (no
    experts) and ``--moe-backend`` (never exercised). tp4 puts ~13.5 GB/chip of
    weights on a single 4-chip host; mem-fraction 0.8 leaves ample headroom
    (the tp16/dp4 e2e needed 0.90 only because dp replicates the weights)."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_5_27B
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
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--nnodes",
                "1",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--mem-fraction-static",
                "0.8",
                "--chunked-prefill-size",
                "512",
                "--page-size",
                "64",
                "--max-running-requests",
                "16",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "16",
                "512",
                "1024",
            ],
            env={"JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu_smoke(self):
        _run_mmlu_smoke(self, self.base_url, self.model)


if __name__ == "__main__":
    unittest.main()
