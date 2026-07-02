"""sglang-jax benchmark: quality check + throughput matrix.

Supports both single-node (2x2, tp=4) and multi-node (4x4, tp=16) deployments.
All parameters are configurable via environment variables — see README.md.

Usage:
  # Single-node (2x2):
  MODEL_DIR=/path/to/Ling-mini-2.0 python3 -u sglang_bench.py

  # Multi-node (4x4), each node runs this script:
  DIST_INIT_ADDR=worker-0:29500 TP_SIZE=16 NNODES=4 \
      MODEL_DIR=/path/to/Ling-mini-2.0 python3 -u sglang_bench.py
"""

import json
import os
import time

WORKER_ID = int(os.environ.get("TPU_WORKER_ID", "0"))
DIST_INIT_ADDR = os.environ.get("DIST_INIT_ADDR", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models/inclusionAI/Ling-mini-2.0")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
DP_SIZE = int(os.environ.get("DP_SIZE", "1"))
NNODES = int(os.environ.get("NNODES", "1"))
WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", "1"))
TIMED_ROUNDS = int(os.environ.get("TIMED_ROUNDS", "2"))

BS_LIST = [int(x) for x in os.environ.get("BS_LIST", "1,2,4,8,16,32").split(",")]
TOKEN_LIST = [int(x) for x in os.environ.get("TOKEN_LIST", "128,256,512,1024").split(",")]
MOE_BACKEND = os.environ.get("MOE_BACKEND", "fused")
# KDA/recurrent hybrid models (e.g. ring_v3_tiny) need radix cache disabled.
DISABLE_RADIX_CACHE = os.environ.get("DISABLE_RADIX_CACHE", "0") == "1"

SYSTEM_PROMPT = "You are Ling, an assistant created by inclusionAI."

QUALITY_PROMPTS = [
    "What is 25 * 4 + 10?",
    "If a train travels 60 km/h for 2.5 hours, how far does it go?",
    "Explain what machine learning is in one sentence.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What is the capital of France? Answer in one word.",
]

LONG_SUFFIX = "\n\nNow, write a detailed essay about the history of artificial intelligence."
LONG_PREFIXES = [
    "Explain the history of the internet from ARPANET to modern day.",
    "What are the main differences between Python and Java?",
    "Describe the process of photosynthesis in detail.",
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing to a high school student.",
    "What are the causes and effects of climate change?",
    "Describe the architecture of a modern CPU.",
    "Explain how machine learning models are trained.",
]


def main():
    is_multinode = NNODES > 1

    print(f"Worker {WORKER_ID}: starting Engine "
          f"(tp={TP_SIZE}, dp={DP_SIZE}, nnodes={NNODES}, moe_backend={MOE_BACKEND}"
          f"{', dist=' + DIST_INIT_ADDR if DIST_INIT_ADDR else ''})", flush=True)

    from sgl_jax.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def format_prompt(text):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Some checkpoints (e.g. ring_v3_tiny) ship no chat template.
            return text

    engine_kwargs = dict(
        model_path=MODEL_DIR,
        tp_size=TP_SIZE,
        dp_size=DP_SIZE,
        page_size=128,
        kv_cache_dtype="bf16",
        mem_fraction_static=0.88,
        precompile_bs_paddings=BS_LIST,
        precompile_token_paddings=TOKEN_LIST,
        max_running_requests=max(BS_LIST),
        moe_backend=MOE_BACKEND,
        enable_single_process=True,
        trust_remote_code=True,
        log_level="warning",
        random_seed=42,
        dtype="bfloat16",
    )
    if DISABLE_RADIX_CACHE:
        engine_kwargs["disable_radix_cache"] = True
    if is_multinode:
        engine_kwargs.update(
            nnodes=NNODES,
            node_rank=WORKER_ID,
            dist_init_addr=DIST_INIT_ADDR,
        )

    t0 = time.perf_counter()
    engine = Engine(**engine_kwargs)
    t1 = time.perf_counter()
    print(f"Worker {WORKER_ID}: Engine init took {t1 - t0:.1f}s", flush=True)

    if is_multinode and WORKER_ID != 0:
        print(f"Worker {WORKER_ID}: Participating in distributed inference...",
              flush=True)
        while True:
            time.sleep(60)

    # ── Quality check ────────────────────────────────────────────
    formatted = [format_prompt(p) for p in QUALITY_PROMPTS]
    sp = [{"temperature": 0.0, "max_new_tokens": 256, "top_k": 1,
           "ignore_eos": False} for _ in formatted]
    results = engine.generate(prompt=formatted, sampling_params=sp)

    print("\n" + "=" * 60, flush=True)
    print(f"QUALITY CHECK (sglang-jax tp={TP_SIZE} dp={DP_SIZE} nnodes={NNODES})", flush=True)
    print("=" * 60, flush=True)
    all_ok = True
    for i, (p, r) in enumerate(zip(QUALITY_PROMPTS, results)):
        text = r["text"]
        tokens = r["meta_info"]["completion_tokens"]
        print(f"\nQ{i + 1}: {p}", flush=True)
        print(f"A{i + 1} ({tokens}t): {text[:500]}", flush=True)
        if len(text.strip()) < 5:
            print("  ** QUALITY FAIL: response too short", flush=True)
            all_ok = False
    print(f"\nQUALITY_CHECK: {'PASS' if all_ok else 'FAIL'}", flush=True)

    # ── Throughput benchmark ─────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"THROUGHPUT BENCHMARK (sglang-jax tp={TP_SIZE} dp={DP_SIZE} nnodes={NNODES}, "
          f"warmup={WARMUP_ROUNDS} timed={TIMED_ROUNDS})", flush=True)
    print("=" * 60, flush=True)

    def make_long_prompts(bs):
        return [format_prompt(LONG_PREFIXES[i % len(LONG_PREFIXES)] + LONG_SUFFIX)
                for i in range(bs)]

    for max_tokens in TOKEN_LIST:
        for bs in BS_LIST:
            prompts_batch = make_long_prompts(bs)
            sp_batch = [{"temperature": 0.0, "max_new_tokens": max_tokens,
                         "top_k": 1, "ignore_eos": True}
                        for _ in prompts_batch]

            for _ in range(WARMUP_ROUNDS):
                engine.generate(prompt=prompts_batch, sampling_params=sp_batch)

            tps_samples = []
            for _ in range(TIMED_ROUNDS):
                t_start = time.perf_counter()
                results_bench = engine.generate(
                    prompt=prompts_batch, sampling_params=sp_batch)
                t_end = time.perf_counter()

                total_tokens = sum(
                    r["meta_info"]["completion_tokens"] for r in results_bench)
                elapsed = t_end - t_start
                tps_samples.append(total_tokens / elapsed if elapsed > 0 else 0)

            avg_tps = sum(tps_samples) / len(tps_samples)
            variance_pct = ((max(tps_samples) - min(tps_samples)) / avg_tps * 100
                            if avg_tps > 0 and len(tps_samples) > 1 else 0)

            print(json.dumps({
                "moe_backend": MOE_BACKEND,
                "max_tokens": max_tokens,
                "bs": bs,
                "tok_per_s": round(avg_tps, 1),
                "variance_pct": round(variance_pct, 1),
                "samples": [round(t, 1) for t in tps_samples],
            }), flush=True)

    print("\nDONE_SGLANG_BENCH", flush=True)

    if is_multinode:
        os._exit(0)
    else:
        engine.shutdown()


if __name__ == "__main__":
    main()
