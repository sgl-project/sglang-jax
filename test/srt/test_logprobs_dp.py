import math
import os
import unittest

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEEPSEEK_R1_DISTILL_QWEN_1_5B

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"


print("Running on Google TPU")
# tp_size=4 with dp_size=2 builds a [2, 2] mesh in scheduler.create_device_mesh
# (ici_parallelism=[dp_size, tp_size // dp_size]), i.e. 4 devices total — fits the
# 4-chip TPU runner. Cannot be merged into test_logprobs.py because that file
# runs on the 1-chip runner.
DP_REGRESSION_ENGINE_CONFIG = {
    "model_path": DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    "random_seed": 27,
    "device": "tpu",
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "skip_server_warmup": True,
    "mem_fraction_static": 0.8,
    "disable_overlap_schedule": True,
    "use_sort_for_toppk_minp": True,
    "log_level": "info",
    "tp_size": 4,
    "dp_size": 2,
    "chunked_prefill_size": 128,
    "max_running_requests": 64,
    "page_size": 64,
    "max_total_tokens": 32768,
    "precompile_token_paddings": [128, 256, 512, 1024],
    "precompile_bs_paddings": [1, 4, 8],
}


class TestLogprobsDpChunkedPrefill(unittest.TestCase):
    """Regression for the dp>1 chunked-prefill skip-tracking bug.

    Pre-fix, `process_batch_result_prefill` used `skip_stream_req: Req | None`,
    a single slot. On dp>1, each dp rank can have its own chunked-in-flight req,
    so all but the last-assigned one leaked into `stream_output` with
    `input_token_logprobs_val == None`. TokenizerManager then either crashed
    (`'NoneType' object is not iterable`) or, if coerced to `[]`, returned
    truncated logprobs that produced inf PPL downstream.

    This test forces multiple in-flight chunked reqs across dp ranks by
    submitting prompts longer than chunked_prefill_size, with dp_size=2, and
    asserts every req returns full, finite, scalar input_token_logprobs.
    """

    @classmethod
    def setUpClass(cls):
        print(f"Launching dp=2 tp=2 Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        cls.engine = Engine(**DP_REGRESSION_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_dp2_multi_req_chunked_prefill_logprobs(self):
        # 4 prompts of varying lengths, all > chunked_prefill_size=128 so each
        # spans multiple chunks. Different lengths ensure chunk boundaries
        # don't coincide, maximizing the chance multiple reqs are mid-prefill
        # in the same batch step (the dp>1 leak condition).
        base = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]
        prompts = [
            base * 20,  # 160 tokens
            base * 25,  # 200 tokens
            base * 30,  # 240 tokens
            base * 35,  # 280 tokens
        ]
        sampling_params = [{"n": 1, "top_k": 1, "max_new_tokens": 2}] * len(prompts)

        output = self.engine.generate(
            input_ids=prompts,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=[0] * len(prompts),
        )

        self.assertEqual(len(output), len(prompts), "must return one result per req")

        for i, (out, prompt) in enumerate(zip(output, prompts)):
            meta = out["meta_info"]
            self.assertIsNotNone(
                meta.get("input_token_logprobs"),
                f"req[{i}]: input_token_logprobs is None — chunked-skip leak",
            )
            self.assertEqual(
                len(meta["input_token_logprobs"]),
                len(prompt),
                f"req[{i}]: expected {len(prompt)} input logprobs, got "
                f"{len(meta['input_token_logprobs'])} — truncated by chunked-skip leak",
            )
            for j, (logprob, token_id, _) in enumerate(meta["input_token_logprobs"]):
                # With logprob_start_len=0, the first token has no preceding
                # context to score against — it MUST be None. Every other
                # position MUST be a finite scalar; partial chunked-skip leaks
                # show up as scattered None/inf in the middle of the sequence,
                # which a length-only check would miss.
                if j == 0:
                    self.assertIsNone(
                        logprob,
                        f"req[{i}][0]: expected None (no prior context), got {logprob}",
                    )
                    self.assertEqual(
                        token_id,
                        prompt[j],
                        f"req[{i}][0]: token_id mismatch {token_id} vs prompt {prompt[j]}",
                    )
                    continue
                self.assertIsNotNone(
                    logprob,
                    f"req[{i}][{j}]: logprob is None mid-sequence — chunked-skip leak",
                )
                shape = getattr(logprob, "shape", ())
                self.assertEqual(
                    shape,
                    (),
                    f"req[{i}][{j}]: logprob must be scalar, got shape {shape}",
                )
                self.assertFalse(
                    isinstance(logprob, (list, tuple)),
                    f"req[{i}][{j}]: logprob must be scalar, got {type(logprob)}",
                )
                self.assertTrue(
                    math.isfinite(float(logprob)),
                    f"req[{i}][{j}]: logprob is non-finite ({logprob}) — "
                    f"likely empty-logprobs→inf-PPL from chunked-skip leak",
                )
                self.assertEqual(
                    token_id,
                    prompt[j],
                    f"req[{i}][{j}]: token_id mismatch {token_id} vs prompt "
                    f"{prompt[j]} — index alignment regression",
                )


if __name__ == "__main__":
    unittest.main()
