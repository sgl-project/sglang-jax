import os
import unittest

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEEPSEEK_R1_QWEN_1_5B

# JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --dist-init-addr=0.0.0.0:10011 --nnodes=1 --tp-size=1 --device=tpu --random-seed=27 --node-rank=0 --mem-fraction-static=0.8 --chunked-prefill-size=8192 --download-dir=/tmp --dtype=bfloat16 --precompile-bs-paddings 1 64 --max-running-requests 64 --max-total-tokens 257536 --skip-server-warmup --attention-backend=fa --precompile-token-paddings 8192 --page-size=64 --disable-overlap-schedule --log-requests --log-requests-level=3 --enable-precision-tracer --use-sort-for-toppk-minp

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"


print("Running on Google TPU")
# Default engine configuration
DEFAULT_ENGINE_CONFIG = {
    "model_path": DEEPSEEK_R1_QWEN_1_5B,
    "random_seed": 27,
    "device": "tpu",
    "chunked_prefill_size": 8192,
    "dtype": "bfloat16",
    "max_running_requests": 64,
    "page_size": 64,
    "max_total_tokens": 257536,
    "precompile_token_paddings": [8192],
    "precompile_bs_paddings": [1, 64],
    "use_sort_for_toppk_minp": True,
    "mem_fraction_static": 0.8,
    "disable_overlap_schedule": True,
    "trust_remote_code": True,
    "skip_server_warmup": True,
    "tp_size": 1,
    "enable_precision_tracer": True,
    "log_level": "info",
}


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang-Jax Engine with {DEEPSEEK_R1_QWEN_1_5B}...")
        cls.engine = Engine(**DEFAULT_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()

    def test_logprobs(self):
        ## prompt = "please introduce yourself"
        input_ids = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]

        sampling_params = {"n": 1, "top_k": 1, "max_new_tokens": 3}
        start_len = 1
        top_logprobs_num = 2
        token_ids_logprob = [10]

        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        output_meta = output["meta_info"]
        ## number check
        self.assertEqual(
            len(output_meta["input_token_logprobs"]),
            len(input_ids) - start_len,
            "input_token_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["output_token_logprobs"]),
            len(output["output_ids"]),
            "output_token_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["input_top_logprobs"]),
            len(input_ids) - start_len,
            "intput_top_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["output_top_logprobs"]),
            len(output["output_ids"]),
            "output_top_logprobs is invalid",
        )

        for i, (output_top_logprob, output_id) in enumerate(
            zip(output_meta["output_top_logprobs"], output["output_ids"])
        ):
            self.assertEqual(
                len(output_top_logprob),
                top_logprobs_num,
                f"output_top_logprobs at {i} is invalid",
            )
            self.assertEqual(
                output_top_logprob[0][1],
                output_id,
                "output id is is not the top logprob",
            )
            max_logprobs = output_top_logprob[0][0]
            for j, logprob in enumerate(output_top_logprob):
                self.assertGreaterEqual(max_logprobs, logprob[0], "the logprob is not the max")

        self.assertEqual(
            len(output_meta["input_token_ids_logprobs"]),
            len(input_ids) - start_len,
            "input_token_ids_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["output_token_ids_logprobs"]),
            len(output["output_ids"]),
            "output_token_ids_logprobs is invalid",
        )

        expected_output_logprobs = [
            [-0.9453125, 32313, "Okay"],
            [0.0, 11, ","],
            [-0.3515625, 773, " so"],
        ]
        for i, logprob in enumerate(output_meta["output_token_logprobs"]):
            self.assertEqual(
                logprob[0], expected_output_logprobs[i][0], f"{logprob[0]} logprob is invalid"
            )
            self.assertEqual(
                logprob[1], expected_output_logprobs[i][1], f"{logprob[1]} output id is invalid"
            )
            self.assertEqual(
                logprob[2], expected_output_logprobs[i][2], f"{logprob[2]} token is invalid"
            )

        sampling_params = {"n": 1, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 3}

        expected_output_logprobs = [
            [-0.8046875, 32313, "Okay"],  ## todo use output compute is -0.79296875
            [0.0, 11, ","],
            [-0.1650390625, 773, " so"],
        ]

        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        output_meta = output["meta_info"]
        for i, logprob in enumerate(output_meta["output_token_logprobs"]):
            self.assertEqual(
                logprob[0], expected_output_logprobs[i][0], f"{logprob[0]} logprob is invalid"
            )
            self.assertEqual(
                logprob[1], expected_output_logprobs[i][1], f"{logprob[1]} output id is invalid"
            )
            self.assertEqual(
                logprob[2], expected_output_logprobs[i][2], f"{logprob[2]} token is invalid"
            )


if __name__ == "__main__":
    unittest.main()
