"""Step 3.5 Flash real-weight gsm8k — validated against UPSTREAM SGLANG, not self.

The earlier version of this file ran a 0-shot chat gsm8k + a no-reference mmlu and
asserted floors derived from our OWN measured scores. That is NOT a correctness
test: with no external reference, "passing" only means "no worse than our own
baseline" — it cannot prove the implementation is correct, and the official
Step-3.5 standard MMLU/GSM8K (85.8 / 88.2) are BASE-model numbers under a
different setup, so they are not comparable either. Those cases were removed.

How real-weight gsm8k correctness IS established now (apples-to-apples vs sglang):

  upstream sglang tests Step-3.5-Flash gsm8k in
    sglang/test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
  via GSM8KMixin with EXACTLY:
    api="completion", num_shots=5, max_tokens=512, num_examples=200,
    num_threads=128, threshold 0.83.

  We run sglang's OWN run_eval (no port, no divergence) with that identical setup
  against the sglang-jax server's OpenAI-compatible /v1/completions endpoint:
    python sgl_jax/test/models/step3p5_align/run_sglang_gsm8k_against_jax.py \
        --base-url http://<node0>:30000/v1 --model <served-model>
  score >= 0.83 with the identical harness/setup => functional equivalence to
  sglang for the gsm8k path.

  Per-element / decision-level real-weight correctness vs the official HF
  reference is the HF<->jax alignment tooling in step3p5_align/.

(An in-repo gsm8k test would require porting sglang's 5-shot completion harness
into sgl_jax's run_eval — sgl_jax run_eval currently only has the 0-shot
ChatCompletionSampler. Deferred; the out-of-repo sglang-eval above is the
authoritative check.)
"""
