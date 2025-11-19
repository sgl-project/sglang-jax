import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_32B,
    QWEN3_32B_EAGLE3,
    CustomTestCase,
    popen_launch_server,
)


class TestSpeculativeDecoding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--mem-fraction-static",
                "0.8",
                "--download-dir",
                "/dev/shm",
                "--max-running-requests",
                "256",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "4096",
                "--context-length",
                "4096",
                "--speculative-draft-model-path",
                QWEN3_32B_EAGLE3,
                "--speculative-draft-model-revision",
                "67caf31f9062d7ab64872e0a111d499bc16cd205",  # this model revision has .safetensor model file, which is converted by huggingface official
                # FIXME(pc) topk > 1 has poor performance now, change it when build_tree_mask_for_draft_decode kernel is  implemented
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-steps",
                "3",
                "--speculative-num-draft-tokens",
                "4",
                # FIXME(pc) currently, spec decode is not fully compatible with scheduler overlap, rm this when fix it
                "--disable-overlap-schedule",
                "--speculative-algorithm",
                "EAGLE3",
                "--page-size",
                "64",
                "--attention-backend",
                "fa",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=512,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
