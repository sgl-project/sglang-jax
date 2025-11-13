import os
import unittest

from transformers import AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
# Configuration
DENSE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


print("Running on Google TPU")
# Default engine configuration
DEFAULT_ENGINE_CONFIG = {
    "model_path": DENSE_MODEL_NAME,
    "random_seed": 42,
    "device": "tpu",
    "chunked_prefill_size": 8192,
    "dtype": "bfloat16",
    "max_running_requests": 64,
    "page_size": 64,
    "max_total_tokens": 257536,
    "precompile_token_paddings": [8192],
    "precompile_bs_paddings": [64],
    "disable_overlap_schedule": False,
    "use_sort_for_toppk_minp": True,
    "mem_fraction_static": 0.8,
    "trust_remote_code": True,
}


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang-Jax Engine with {DENSE_MODEL_NAME}...")
        cls.engine = Engine(**DEFAULT_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()

    def test_logprobs(self):
        prompt = "please introduce yourself"
        tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME, truction_remote_code=True)
        input_ids = tokenizer.encode(prompt)
        sampling_params = {
            "n": 1,
            "temperature": 0.7,
            "top_k": 1,
        }
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


if __name__ == "__main__":
    unittest.main()
