"""AInfer TPU benchmark: quality check + throughput matrix.

Each node runs this script independently with tp=4 (single-host mode).
For multi-node deployments, aggregate throughput = N × single-node.

Usage:
  MODEL_DIR=/path/to/Ling-mini-2.0 python3 -u ainfer_bench.py
"""

import json
import os
import time

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models/inclusionAI/Ling-mini-2.0")
WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", "1"))
TIMED_ROUNDS = int(os.environ.get("TIMED_ROUNDS", "2"))

BS_LIST = [int(x) for x in os.environ.get("BS_LIST", "1,2,4,8,16,32").split(",")]
TOKEN_LIST = [int(x) for x in os.environ.get("TOKEN_LIST", "128,256,512,1024").split(",")]

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
    import jax
    print(f"TPU devices: {jax.device_count()}", flush=True)
    assert jax.device_count() == 4, f"Expected 4 devices, got {jax.device_count()}"

    from ainfer.engine.tpu_managers import LLMEngine
    from ainfer.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def format_prompt(text):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    cache_dir = os.environ.get("TPU_COMPILE_CACHE_DIR", "")
    t0 = time.perf_counter()
    engine = LLMEngine(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        tpu_data_parallel_size=1,
        tpu_expert_parallel_size=1,
        max_num_seqs=max(BS_LIST),
        trust_remote_code=True,
        disable_warmup=True,
        tpu_bs_paddings=BS_LIST,
        tpu_token_paddings=TOKEN_LIST,
        tpu_compile_cache_dir=cache_dir,
    )
    t1 = time.perf_counter()
    print(f"Engine init took {t1 - t0:.1f}s", flush=True)

    default_sp = SamplingParams(temperature=0.0, top_k=1, max_tokens=256,
                                ignore_eos=False)

    formatted = [format_prompt(p) for p in QUALITY_PROMPTS]
    results = engine.generate(formatted, default_sp)

    print("\n" + "=" * 60, flush=True)
    print("QUALITY CHECK (AInfer tp=4)", flush=True)
    print("=" * 60, flush=True)
    all_ok = True
    for i, (p, r) in enumerate(zip(QUALITY_PROMPTS, results)):
        text = r["text"] if isinstance(r, dict) else r.text
        tokens = (r["meta_info"]["completion_tokens"] if isinstance(r, dict)
                  else r.meta_info.get("completion_tokens", 0))
        print(f"\nQ{i + 1}: {p}", flush=True)
        print(f"A{i + 1} ({tokens}t): {text[:500]}", flush=True)
        if len(text.strip()) < 5:
            print("  ** QUALITY FAIL: response too short", flush=True)
            all_ok = False
    print(f"\nQUALITY_CHECK: {'PASS' if all_ok else 'FAIL'}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"THROUGHPUT BENCHMARK (AInfer tp=4, "
          f"warmup={WARMUP_ROUNDS} timed={TIMED_ROUNDS})", flush=True)
    print("=" * 60, flush=True)

    def make_long_prompts(bs):
        return [format_prompt(LONG_PREFIXES[i % len(LONG_PREFIXES)] + LONG_SUFFIX)
                for i in range(bs)]

    for max_tokens in TOKEN_LIST:
        for bs in BS_LIST:
            prompts_batch = make_long_prompts(bs)
            sp = SamplingParams(temperature=0.0, top_k=1,
                                max_tokens=max_tokens, ignore_eos=True)

            for _ in range(WARMUP_ROUNDS):
                engine.generate(prompts_batch, sp)

            tps_samples = []
            for _ in range(TIMED_ROUNDS):
                t_start = time.perf_counter()
                results_bench = engine.generate(prompts_batch, sp)
                t_end = time.perf_counter()

                total_tokens = 0
                for r in results_bench:
                    if isinstance(r, dict):
                        total_tokens += r["meta_info"]["completion_tokens"]
                    else:
                        total_tokens += r.meta_info.get("completion_tokens", 0)
                elapsed = t_end - t_start
                tps_samples.append(total_tokens / elapsed if elapsed > 0 else 0)

            avg_tps = sum(tps_samples) / len(tps_samples)
            variance_pct = ((max(tps_samples) - min(tps_samples)) / avg_tps * 100
                            if avg_tps > 0 and len(tps_samples) > 1 else 0)

            print(json.dumps({
                "max_tokens": max_tokens,
                "bs": bs,
                "tok_per_s": round(avg_tps, 1),
                "variance_pct": round(variance_pct, 1),
                "samples": [round(t, 1) for t in tps_samples],
            }), flush=True)

    print("\nDONE_AINFER_BENCH", flush=True)


if __name__ == "__main__":
    main()
