import math
import os
import unittest

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEEPSEEK_R1_DISTILL_QWEN_1_5B

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"


print("Running on Google TPU")
# tp_size=4 / dp_size=2 -> [dp, tp//dp] = [2, 2] mesh = 4 devices (the TPU runner
# size). Separate from test_logprobs.py, which runs on the 1-chip runner.
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
    """dp>1 chunked-prefill leak: process_batch_result_prefill tracked a single
    skip_stream_req slot, so on dp>1 all but one mid-chunk req leaked
    None/truncated input_token_logprobs. Asserts every req returns full, finite,
    scalar logprobs.
    """

    @classmethod
    def setUpClass(cls):
        print(f"Launching dp=2 tp=2 Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        cls.engine = Engine(**DP_REGRESSION_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_dp2_multi_req_chunked_prefill_logprobs(self):
        # Prompts all > chunked_prefill_size with distinct lengths -> several reqs
        # mid-prefill across dp ranks in the same step (the leak condition).
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
                # logprob_start_len=0: position 0 has no prior context -> None;
                # every other position must be a finite scalar (partial leaks
                # scatter None/inf mid-sequence, which a length check would miss).
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

    def test_dp2_top_logprobs(self):
        # #1261 overlap-off: top_logprobs + dp>1. Uneven prompts make per-dp
        # token totals indivisible by dp, which crashed the legacy sharded-slice
        # fallback.
        _assert_top_logprobs(self, self.engine)


class TestLogprobsDpOverlap(unittest.TestCase):
    """#1261 / #1276 on the default path (overlap on, dp=4 = the issue's config).
    dp=4 strictly supersedes dp=2 for the per-DP padded layout.
    """

    @classmethod
    def setUpClass(cls):
        # tp_size=4 / dp_size=4 -> [4, 1] mesh, the issue's exact tp4/dp4 layout.
        config = dict(DP_REGRESSION_ENGINE_CONFIG)
        config["dp_size"] = 4
        config["disable_overlap_schedule"] = False
        print(f"Launching dp=4 tp=1 overlap-on Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        cls.engine = Engine(**config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_overlap_top_logprobs(self):
        # top_logprobs on the overlap-on default path (pre-#1261 it crashed on the
        # legacy per-req hidden_states slice).
        _assert_top_logprobs(self, self.engine)

    def test_overlap_decode_residual_logprobs(self):
        # dp=4 with overlap, two regressions:
        # 1. decode crash: the speculated decode batch reuses extend reqs
        #    (extend_input_logprob_token_ids not cleared); the else-branch concat'd
        #    their residual and sharded it P("data") -> crash when len % dp != 0.
        #    Needs n not a multiple of dp: n=3 x (8-1) = 21, 21 % 4 != 0 (the old
        #    dp=2 test's 4 identical reqs gave 4*r, divisible by 4 too, so it never
        #    tripped this). max_new_tokens=2 forces the decode step; fix leaves the
        #    field None for decode.
        # 2. Identical prompts must give identical scalar logprobs (overlap must
        #    compact the per-DP padding, else rank>=1 reads padding rows).
        prompt = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]  # 8 tokens
        n = 3
        prompts = [list(prompt) for _ in range(n)]
        start = 1
        sampling_params = [{"n": 1, "top_k": 1, "max_new_tokens": 2}] * n

        output = self.engine.generate(
            input_ids=prompts,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=[start] * n,
        )
        self.assertEqual(len(output), n, "must return one result per req")

        def scalar_logprobs(meta):
            # entries are (logprob, token_id, _); index 0 is the None placeholder.
            tok = meta.get("input_token_logprobs")
            self.assertIsNotNone(tok, "input_token_logprobs is None")
            return [e[0] for e in tok]

        ref = scalar_logprobs(output[0]["meta_info"])
        self.assertEqual(len(ref), len(prompt) - start, "ref length mismatch")
        self.assertIsNone(ref[0], "index 0 must be the None placeholder")
        for j in range(1, len(ref)):
            self.assertTrue(math.isfinite(float(ref[j])), f"ref[{j}] non-finite")

        for i in range(1, n):
            vals = scalar_logprobs(output[i]["meta_info"])
            self.assertEqual(len(vals), len(ref), f"req[{i}] length mismatch")
            for j in range(1, len(ref)):
                self.assertAlmostEqual(
                    float(vals[j]),
                    float(ref[j]),
                    delta=1e-2,
                    msg=f"req[{i}][{j}]={vals[j]} != req[0][{j}]={ref[j]} — dp=4 overlap "
                    f"per-DP padded scalar input_token_logprobs mismatch",
                )


def _assert_top_logprobs(test, engine):
    """Shared #1261 check: uneven prompts + top_logprobs, every req returns
    `len(prompt) - start` rows of finite width-`k` top logprobs."""
    base = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]
    prompts = [base * 2, base * 3, base * 1, base * 4]  # 16, 24, 8, 32 tokens
    start = 1
    top_logprobs_num = 3
    sampling_params = [{"n": 1, "top_k": 1, "max_new_tokens": 2}] * len(prompts)

    output = engine.generate(
        input_ids=prompts,
        sampling_params=sampling_params,
        return_logprob=True,
        logprob_start_len=[start] * len(prompts),
        top_logprobs_num=[top_logprobs_num] * len(prompts),
    )

    test.assertEqual(len(output), len(prompts), "must return one result per req")
    for i, (out, prompt) in enumerate(zip(output, prompts)):
        meta = out["meta_info"]
        top = meta.get("input_top_logprobs")
        test.assertIsNotNone(top, f"req[{i}]: input_top_logprobs is None")
        test.assertEqual(
            len(top),
            len(prompt) - start,
            f"req[{i}]: expected {len(prompt) - start} input_top_logprobs rows, "
            f"got {len(top)} — per-dp padded split misaligned",
        )
        for j, row in enumerate(top):
            # index 0 is the None placeholder; later rows are finite width-k lists.
            if j == 0:
                test.assertIsNone(row, f"req[{i}][0]: expected None placeholder, got {row}")
                continue
            test.assertIsNotNone(row, f"req[{i}][{j}]: unexpected None row mid-sequence")
            test.assertEqual(
                len(row),
                top_logprobs_num,
                f"req[{i}][{j}]: expected {top_logprobs_num} top logprobs, got {len(row)}",
            )
            for entry in row:
                logprob = entry[0]
                test.assertTrue(
                    math.isfinite(float(logprob)),
                    f"req[{i}][{j}]: non-finite top logprob {logprob}",
                )


if __name__ == "__main__":
    unittest.main()
