"""Run sglang's OWN gsm8k eval against the sglang-jax Step-3.5-Flash server.

This is the TRUEST alignment check: it reuses sglang's run_eval / GSM8KMixin eval
logic verbatim (no port, no divergence) and points it at the sglang-jax server's
OpenAI-compatible /v1/completions endpoint. The parameters replicate exactly how
sglang tests Step-3.5-Flash gsm8k in
  sglang/test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
    -> GSM8KMixin (sglang/python/sglang/test/kits/eval_accuracy_kit.py):
       api="completion", num_shots=5, max_tokens=512, num_examples=200,
       num_threads=128, gsm8k_accuracy_thres=0.83.

PASS (score >= 0.83 with this identical setup) => our implementation is
functionally equivalent to what sglang validates for Step-3.5-Flash gsm8k.

Run on a machine with the `sglang` package installed and network access to the
sglang-jax server (the real-weight server must be running, OpenAI API enabled):

    pip install sglang            # the upstream sglang package (for its eval code)
    python run_sglang_gsm8k_against_jax.py \
        --base-url http://<node0>:30000/v1 \
        --model <served-model-name>      # e.g. the checkpoint path / id /v1/models reports
"""

import argparse
import os
from types import SimpleNamespace

from sglang.test.run_eval import run_eval  # upstream sglang's eval

# sglang's Step-3.5-Flash gsm8k setup (GSM8KMixin defaults + chain_mtp overrides).
SGLANG_STEP35_GSM8K = dict(
    eval_name="gsm8k",
    api="completion",  # raw completion, NOT chat
    num_shots=5,  # 5-shot
    max_tokens=512,
    num_examples=200,
    num_threads=128,
)
SGLANG_STEP35_GSM8K_THRESHOLD = 0.83


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        required=True,
        help="sglang-jax server ROOT, NO /v1 (e.g. http://host:30000). "
        "sglang's run_eval appends /v1 itself — passing /v1 yields /v1/v1 -> 404.",
    )
    ap.add_argument("--model", required=True, help="served model name (see /v1/models)")
    ap.add_argument("--threshold", type=float, default=SGLANG_STEP35_GSM8K_THRESHOLD)
    args = ap.parse_args()

    # The OpenAI client (used by sglang's sampler) requires an api_key; the server
    # ignores it. Set a dummy so client init doesn't fail.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    # Guard against the /v1/v1 404: run_eval will append /v1, so base_url must not.
    base = args.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]

    eval_args = SimpleNamespace(base_url=base, model=args.model, **SGLANG_STEP35_GSM8K)
    print(
        f"[align] running sglang's gsm8k (5-shot/completion/512/200) vs {base}/v1",
        flush=True,
    )
    metrics = run_eval(eval_args)
    score = metrics["score"]
    print(f"\n[align] gsm8k score = {score:.4f}   (sglang Step-3.5 threshold = {args.threshold})")
    if score >= args.threshold:
        print(
            "[align] PASS — matches sglang's Step-3.5-Flash gsm8k test "
            "(same harness/setup/threshold). Implementation functionally equivalent for gsm8k."
        )
    else:
        print(
            f"[align] FAIL — below sglang's Step-3.5 gsm8k threshold {args.threshold}; "
            "real divergence vs sglang for the gsm8k path."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
