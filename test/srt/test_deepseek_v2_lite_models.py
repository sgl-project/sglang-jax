import unittest
from types import SimpleNamespace

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEEPSEEK_CODER_V2_LITE_INSTRUCT,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestDeepSeekCoderV2LiteInstruct(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_CODER_V2_LITE_INSTRUCT
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
                "--mem-fraction-static",
                "0.8",
                "--max-prefill-tokens",
                "8192",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--nnodes",
                "1",
                "--dist-init-addr",
                "0.0.0.0:10011",
                # MLA backend asserts page_size > 1 (see
                # MLAAttentionBackend.get_max_running_reqests); 128 keeps
                # max_running_requests > 0 with context_len 8192.
                "--page-size",
                "128",
                "--context-length",
                "8192",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Matches sglang nightly threshold (test/manual/nightly/test_text_models_gsm8k_eval.py).
        # Local 200-example smoke test landed at 0.88, so 0.85 is realistic.
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.85)

        if is_in_ci():
            write_github_step_summary(f"### test_gsm8k\n" f'{metrics["score"]=:.4f}\n')


if __name__ == "__main__":
    unittest.main()
