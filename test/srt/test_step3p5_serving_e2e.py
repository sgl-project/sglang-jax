"""Step 3.5 serving-level self-consistency e2e (TPU, real server, dummy weights).

These are the three Task C serving-mechanism invariants that CANNOT be unit-tested
without "测假" (mocking the scheduler/cache). They run against a REAL launched
server, but they are SELF-CONSISTENCY checks (output A == output B for the same
model), so they need only a microscale model with RANDOM weights — NOT the 398GB
real checkpoint. We launch with:

  --model-path <microscale config dir>   (HF config.json only, no weights)
  --load-format dummy                     (engine random-initialises weights)
  --skip-tokenizer-init                   (no tokenizer; we send raw input_ids)

and compare greedy (temperature=0) output_ids, which are deterministic regardless
of the (random) weight values.

  * cache_hit==miss : a prefix-cache hit must not change the greedy output.
  * chunked==full   : chunked prefill must match single-shot full prefill.
  * SWA==full       : full MHATokenToKVPool (--disable-hybrid-swa-memory) must
                      match the default windowed SWAKVPool.

Accuracy (gsm8k/mmlu) is a SEPARATE concern that genuinely needs the real
checkpoint — it is NOT covered here.

Mirrors the launch pattern of test_qwen1_5_models_dummy.py and the /generate
pattern of test_unified_radix_cache_serving.py. TPU-only; the smoke test gates
the rest (it validates the whole launch chain: config schema, dummy load with
stacked-MoE, tokenizer skip, kernel paddings).
"""

import json
import os
import tempfile
import unittest

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Microscale Step 3.5 — identical proportions to test_step3p5_engine._make_config
# (5 layers: full/sliding mix, dense 0-1, MoE 2-4, sliding_window=16), shrunk so a
# dummy-weight server fits on a single chip.
_HEAD_DIM = 128
_VOCAB = 64
_SLIDING_WIN = 16

_MODEL_CONFIG = {
    "architectures": ["Step3p5ForCausalLM"],
    "model_type": "step3p5",
    "torch_dtype": "bfloat16",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_attention_groups": 2,
    "head_dim": _HEAD_DIM,
    "vocab_size": _VOCAB,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 128,
    "rope_theta": [5000000.0, 10000.0, 10000.0, 5000000.0, 5000000.0],
    "rope_scaling": None,
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ],
    "partial_rotary_factors": [0.5, 1.0, 1.0, 0.5, 0.5],
    "attention_other_setting": {
        "attention_type": "sliding_attention",
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": _HEAD_DIM,
    },
    "swiglu_limits": [0.0, 0.0, 0.0, 0.0, 7.0],
    "swiglu_limits_shared": [0.0, 0.0, 0.0, 0.0, 16.0],
    "moe_layers_enum": "2,3,4",
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_intermediate_size": 32,
    "share_expert_dim": 32,
    "moe_router_activation": "sigmoid",
    "moe_router_scaling_factor": 3.0,
    "norm_expert_weight": True,
    "use_moe_router_bias": True,
    "use_qk_norm": True,
    "use_head_wise_attn_gate": True,
    "sliding_window": _SLIDING_WIN,
    "yarn_only_types": ["full_attention"],
    "need_fp32_gate": True,
    "zero_centered": True,
    "tie_word_embeddings": False,
}

# A fixed prompt of raw token ids in [0, vocab). Length 40 > sliding_window (16) so
# the window truncates, and long enough to be split by a small chunked-prefill-size.
_INPUT_IDS = [(i * 7 + 3) % _VOCAB for i in range(40)]
_MAX_NEW_TOKENS = 16

# Base server args shared by every launch. Page size 16 == sliding_window keeps the
# SWA paging simple; dummy load + skip-tokenizer-init avoid weights and a tokenizer.
_BASE_ARGS = [
    "--skip-server-warmup",
    "--skip-tokenizer-init",
    "--load-format",
    "dummy",
    "--dtype",
    "bfloat16",
    "--random-seed",
    "3",
    "--page-size",
    "16",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "8",
    "--attention-backend",
    "fa",
    "--max-prefill-tokens",
    "256",
]


def _config_dir():
    """Create (once) a temp HF model dir holding only the microscale config.json."""
    d = tempfile.mkdtemp(prefix="step3p5_microscale_")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(_MODEL_CONFIG, f)
    return d


def _launch(model_dir, extra_args):
    return popen_launch_server(
        model_dir,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        device="tpu",
        other_args=_BASE_ARGS + extra_args,
    )


def _generate(base_url, input_ids):
    """Greedy generate from raw input_ids; return the output token ids."""
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": _MAX_NEW_TOKENS},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["output_ids"]


def _flush(base_url):
    requests.post(f"{base_url}/flush_cache", timeout=30)


class TestStep3p5ServingSmokeAndCache(CustomTestCase):
    """Smoke (launch chain) + cache_hit==miss, on one server (radix cache ON)."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_dir = _config_dir()
        cls.process = _launch(cls.model_dir, [])

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "process", None)
        if proc is not None:
            kill_process_tree(proc.pid)

    def test_smoke_server_generates(self):
        """Gate: the microscale dummy server launches and answers one request.

        If this fails, the launch chain (config schema / dummy load with stacked
        MoE / tokenizer skip / kernel paddings) is broken — fix that before the
        self-consistency tests can mean anything.
        """
        out = _generate(self.base_url, _INPUT_IDS)
        self.assertGreater(len(out), 0, "server returned no output_ids")

    def test_cache_hit_equals_miss(self):
        """A prefix-cache hit must not change the greedy output."""
        _flush(self.base_url)
        out_miss = _generate(self.base_url, _INPUT_IDS)  # cold: cache miss
        out_hit = _generate(self.base_url, _INPUT_IDS)  # warm: prefix cached
        self.assertEqual(
            out_miss,
            out_hit,
            "cache hit changed the greedy output vs cache miss",
        )


class TestStep3p5ChunkedEqualsFull(CustomTestCase):
    """chunked prefill greedy output == single-shot full prefill greedy output."""

    def test_chunked_equals_full(self):
        model_dir = _config_dir()

        # Chunked: force the 40-token prompt to be split (chunk size 8). Disable the
        # radix cache so the only variable is the prefill chunking.
        proc = _launch(model_dir, ["--chunked-prefill-size", "8", "--disable-radix-cache"])
        try:
            out_chunked = _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
        finally:
            kill_process_tree(proc.pid)

        # Full: large chunk size so the prompt is prefilled in one shot.
        proc = _launch(model_dir, ["--chunked-prefill-size", "2048", "--disable-radix-cache"])
        try:
            out_full = _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
        finally:
            kill_process_tree(proc.pid)

        self.assertEqual(
            out_chunked,
            out_full,
            "chunked prefill greedy output differs from full prefill",
        )


class TestStep3p5SWAEqualsFull(CustomTestCase):
    """Default windowed SWAKVPool greedy output == full MHATokenToKVPool output."""

    def test_swa_equals_full(self):
        model_dir = _config_dir()

        # Full MHA: every layer keeps the whole KV (no windowed pool).
        proc = _launch(model_dir, ["--disable-hybrid-swa-memory", "--disable-radix-cache"])
        try:
            out_full = _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
        finally:
            kill_process_tree(proc.pid)

        # Default: hybrid SWAKVPool for the sliding layers.
        proc = _launch(model_dir, ["--disable-radix-cache"])
        try:
            out_swa = _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
        finally:
            kill_process_tree(proc.pid)

        self.assertEqual(
            out_swa,
            out_full,
            "SWAKVPool greedy output differs from full MHATokenToKVPool",
        )


if __name__ == "__main__":
    unittest.main()
