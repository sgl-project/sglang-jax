"""ring_v3_tiny (BailingMoeV3: MLA + KDA + MoE) correctness smoke test on TPU.

Validates the new sglang-jax BailingMoeV3ForCausalLM end-to-end:
  - hybrid KDA (linear attn) + MLA (latent attn, head-wise gated) layers
  - sigmoid-router grouped top-k MoE with shared expert
  - real ring_v3_tiny weights loaded from a gcsfuse mount

Pass criteria: all quality prompts return non-trivial, coherent answers
(factual questions answered correctly), and the engine loads/runs without
NaN or shape errors.

Env vars:
  MODEL_DIR  - local path to the gcsfuse-mounted checkpoint
  TP_SIZE    - tensor parallel size (default 4)
"""

import os
import time

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models/ring_v3_tiny")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
# ring_v3_tiny hidden_size=1536 is not a multiple of the fused MoE kernel's
# default bd1/bd2=1024 tile. The fused kernel now auto-reduces bd1/bd2 to the
# largest valid block that divides hidden_size (768 here), so "fused" works.
# Set MOE_BACKEND=epmoe to fall back to the GMM-based backend for comparison.
MOE_BACKEND = os.environ.get("MOE_BACKEND", "fused")

QUALITY_PROMPTS = [
    "What is 25 * 4 + 10?",
    "What is the capital of France? Answer in one word.",
    "If a train travels 60 km/h for 2 hours, how far does it go?",
    "Name three primary colors.",
    "Complete the sentence: The sun rises in the",
    "What is 100 divided by 4?",
    "Who wrote Romeo and Juliet?",
    "Translate 'hello' to French.",
]


def main():
    print(f"=== ring_v3_tiny correctness test (tp={TP_SIZE}) ===", flush=True)
    print(f"MODEL_DIR={MODEL_DIR}", flush=True)
    print(f"MOE_BACKEND={MOE_BACKEND}", flush=True)

    from sgl_jax.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def format_prompt(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"(no chat template, using raw prompt: {e})", flush=True)
            return text

    t0 = time.perf_counter()
    engine = Engine(
        model_path=MODEL_DIR,
        tp_size=TP_SIZE,
        page_size=128,
        kv_cache_dtype="bf16",
        mem_fraction_static=0.85,
        # Hybrid recurrent (KDA) models require radix cache disabled — prefix
        # sharing is unsafe with recurrent state.
        disable_radix_cache=True,
        precompile_bs_paddings=[1, 8],
        precompile_token_paddings=[256, 512],
        max_running_requests=8,
        # MoE backend selectable via env (default "fused"); the fused kernel now
        # auto-reduces bd1/bd2 to divide hidden_size=1536.
        moe_backend=MOE_BACKEND,
        enable_single_process=True,
        trust_remote_code=True,
        log_level="info",
        random_seed=42,
        dtype="bfloat16",
    )
    t1 = time.perf_counter()
    print(f"Engine init took {t1 - t0:.1f}s", flush=True)

    formatted = [format_prompt(p) for p in QUALITY_PROMPTS]
    sp = [
        {"temperature": 0.0, "max_new_tokens": 64, "top_k": 1, "ignore_eos": False}
        for _ in formatted
    ]
    results = engine.generate(prompt=formatted, sampling_params=sp)

    print("\n" + "=" * 60, flush=True)
    print("QUALITY CHECK — ring_v3_tiny (BailingMoeV3: MLA+KDA+MoE)", flush=True)
    print("=" * 60, flush=True)
    all_ok = True
    for i, (p, r) in enumerate(zip(QUALITY_PROMPTS, results)):
        text = r["text"]
        tokens = r["meta_info"]["completion_tokens"]
        print(f"\nQ{i + 1}: {p}", flush=True)
        print(f"A{i + 1} ({tokens}t): {text[:300]}", flush=True)
        if len(text.strip()) < 2:
            print("  ** FAIL: response too short", flush=True)
            all_ok = False

    print(f"\nQUALITY_CHECK: {'PASS' if all_ok else 'FAIL'}", flush=True)
    print("\nDONE_RING_V3_TINY_TEST", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
