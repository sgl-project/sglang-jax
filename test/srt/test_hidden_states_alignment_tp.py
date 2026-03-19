"""
Multi-node TP hidden states alignment test.

Verifies hidden states remain numerically correct when model weights
and computation are sharded across multiple TPU chips via Tensor Parallelism.

Environment: TPU v6e-8, tp_size=8, nnodes=2 (4 chips per host)
Model: Qwen/Qwen3-32B (64 layers, hidden_dim=5120)
"""

import multiprocessing
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

# Node 1 starts Engine immediately, but node 0 first computes HF hidden states
os.environ.setdefault("JAX_COORDINATOR_STARTUP_TIMEOUT", "600")

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import (
    QWEN3_32B,
    CustomTestCase,
    assert_hidden_states_aligned,
)

ALL_PROMPTS = [
    "The capital of France is",
    "Hello world",
    "The future of AI is",
]

# Multi-node configuration from environment
NODE_RANK = int(os.environ.get("NODE_RANK", "0"))
NNODES = int(os.environ.get("NNODES", "1"))
HEAD_IP = os.environ.get("HEAD_IP", "127.0.0.1")
DIST_INIT_ADDR = f"{HEAD_IP}:29500"
TP_SIZE = NNODES * 4  # 4 chips per v6e host

ENGINE_KWARGS = dict(
    model_path=QWEN3_32B,
    trust_remote_code=True,
    tp_size=TP_SIZE,
    nnodes=NNODES,
    dist_init_addr=DIST_INIT_ADDR,
    device="tpu",
    random_seed=3,
    mem_fraction_static=0.6,
    chunked_prefill_size=1024,
    download_dir="/tmp",
    dtype="bfloat16",
    precompile_bs_paddings=[8],
    max_running_requests=8,
    skip_server_warmup=True,
    attention_backend="fa",
    precompile_token_paddings=[1024],
    page_size=64,
    log_requests=False,
    enable_return_hidden_states=True,
)


def _compute_hf_hidden_states(model_path, prompts, output_path):
    """Run in a subprocess so all memory is freed when the process exits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[HF subprocess] Loading {model_path} on CPU (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    ).eval()
    print("[HF subprocess] Model loaded.")

    results = {}
    for prompt in prompts:
        print(f"[HF subprocess] Computing hidden states for: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        results[prompt] = tuple(h.float().numpy() for h in outputs.hidden_states)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print("[HF subprocess] Done. Exiting (all memory will be freed).")


class TestHiddenStatesAlignmentTP(CustomTestCase):
    """
    Multi-chip TP hidden states alignment test (v6e-8, tp_size=8).

    Only runs on the coordinator node (NODE_RANK=0).
    """

    @classmethod
    def setUpClass(cls):
        # Phase 1: Compute HF reference hidden states in a subprocess.
        # The subprocess exits after completion, freeing all memory.
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        proc = multiprocessing.Process(
            target=_compute_hf_hidden_states,
            args=(QWEN3_32B, ALL_PROMPTS, tmp.name),
        )
        proc.start()
        proc.join()
        assert proc.exitcode == 0, f"HF subprocess failed with exit code {proc.exitcode}"

        with open(tmp.name, "rb") as f:
            cls.hf_hidden_states = pickle.load(f)
        os.unlink(tmp.name)
        print("HF hidden states loaded from subprocess results.")

        # Phase 2: Load SGLang-JAX engine (coordinator node).
        # The worker node's Engine is already waiting in jax.distributed.initialize.
        print(
            f"Loading SGLang-JAX engine: {QWEN3_32B} on TPU "
            f"(tp_size={TP_SIZE}, nnodes={NNODES})..."
        )
        cls.engine = Engine(**ENGINE_KWARGS, node_rank=NODE_RANK)
        print("SGLang-JAX engine loaded.")

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _get_sgl_prefill_hidden_states(self, prompt):
        outputs = self.engine.generate(
            prompt=[prompt],
            sampling_params={"temperature": 0, "max_new_tokens": 1},
            return_hidden_states=True,
        )
        prefill_hs = outputs[0]["meta_info"]["hidden_states"][0]
        if not isinstance(prefill_hs, np.ndarray):
            prefill_hs = np.array(prefill_hs)
        return prefill_hs

    def _compare_per_layer(self, prompt):
        hf_hs = self.hf_hidden_states[prompt]
        sgl_hs = self._get_sgl_prefill_hidden_states(prompt)
        assert_hidden_states_aligned(self, hf_hs, sgl_hs, prompt)

    def test_tp_single_prompt_alignment(self):
        """Per-layer hidden states match HF after TP all-reduce."""
        self._compare_per_layer("The capital of France is")

    def test_tp_multi_prompt_alignment(self):
        """Batch + TP correctness."""
        prompts = ["Hello world", "The future of AI is"]
        for prompt in prompts:
            self._compare_per_layer(prompt)


if __name__ == "__main__":
    if NODE_RANK >= 1:
        # Worker node: create Engine and block until coordinator shuts down.
        print(f"[Worker node {NODE_RANK}] Creating Engine (tp_size={TP_SIZE})...")
        Engine(**ENGINE_KWARGS, node_rank=NODE_RANK)
        print(f"[Worker node {NODE_RANK}] Engine exited.")
        sys.exit(0)

    # Coordinator node (rank 0): run tests.
    unittest.main()
