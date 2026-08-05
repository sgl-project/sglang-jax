"""Long-context decode benchmark for ling_v3_flash: bs=1, in=16K, out=2K.

Compares prefill (TTFT) and decode (per-token) latency under a fixed
single-stream long-context workload — the regime AInfer v4 was designed to
shine at on the decode side (assuming the decode kernel path activates).

How the workload is constructed:
- Single request (bs=1).
- Prompt is repeated until tokenizer encodes it to >= TARGET_IN tokens, then
  truncated to exactly TARGET_IN tokens. This avoids any reliance on the chat
  template padding.
- Two rounds:
    (1) max_new_tokens=1 -> measures pure prefill / TTFT.
    (2) max_new_tokens=TARGET_OUT -> measures total; decode_elapsed = total - ttft.
- WARMUP_ROUNDS warmup passes first to settle JIT cache.

This bench reuses the Engine init kwargs of ling_v3_flash_test.py (only
overrides precompile_token_paddings to include the long context and a small
size for decode-time graphs) so the path under test exactly matches the
production path on the same model.

Env vars:
  MODEL_DIR     - gcsfuse mount point (default /tmp/models/ling_v3_flash)
  TP_SIZE       - tensor parallel size (default 4)
  MOE_BACKEND   - "fused" (v1 EP) | "fused_v4" (v4 TP) | "epmoe" | "fused_v2"
  TARGET_IN     - target input token count (default 16384)
  TARGET_OUT    - target output token count (default 2048)
  WARMUP_ROUNDS - warmup rounds (default 1)
  TIMED_ROUNDS  - timed rounds (default 3)
  BS_PADDINGS   - comma-separated precompile bs paddings (default "1").
                  Use "8" for fused_v2 (its kernel rejects bs=1; engine routes
                  bs=1 requests through the bs=8 graph with internal padding).
"""

import json
import os
import time

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models/ling_v3_flash")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
MOE_BACKEND = os.environ.get("MOE_BACKEND", "fused")
TARGET_IN = int(os.environ.get("TARGET_IN", "16384"))
TARGET_OUT = int(os.environ.get("TARGET_OUT", "2048"))
WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", "1"))
TIMED_ROUNDS = int(os.environ.get("TIMED_ROUNDS", "3"))
BS_PADDINGS = [int(x) for x in os.environ.get("BS_PADDINGS", "1").split(",")]


BASE_TEXT = (
    "Explain the history of the internet from ARPANET to the modern day in "
    "rich detail, covering protocols, network topology, key contributors, "
    "and the social impact of the world wide web. Also discuss the rise of "
    "cloud computing, the role of TCP/IP, DNS, BGP, and the eventual shift "
    "toward content-delivery networks and edge computing. Then describe how "
    "machine learning systems exchange gradients across networks of GPUs and "
    "TPUs, why interconnect bandwidth and latency are the dominant cost in "
    "large model training, and what hardware-software co-design tricks are "
    "used to hide that cost. "
)


def build_long_prompt(tokenizer, target_tokens: int) -> tuple[str, list[int]]:
    """Repeat BASE_TEXT until tokenized length >= target_tokens, then trim
    token-exact. Returns (prompt_text, token_ids)."""
    text = BASE_TEXT
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text = text + BASE_TEXT
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids), ids


def main():
    print(
        f"=== ling_v3_flash long-ctx bench (tp={TP_SIZE} moe={MOE_BACKEND} "
        f"in={TARGET_IN} out={TARGET_OUT}) ===",
        flush=True,
    )

    from sgl_jax.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    prompt_text, prompt_ids = build_long_prompt(tokenizer, TARGET_IN)
    real_in = len(prompt_ids)
    print(f"Prompt: {real_in} input tokens (target {TARGET_IN})", flush=True)
    assert real_in == TARGET_IN, f"prompt token mismatch: {real_in} vs {TARGET_IN}"

    # Precompile token paddings: cover the long prompt + a small decode-step
    # padding. precompile_bs_paddings=[1] matches the single-stream workload.
    # The engine still synthesizes a few intermediate paddings internally
    # (saw [256,512,4096] for the default test), but we set [128, TARGET_IN]
    # explicitly so the 16K graph is definitely compiled.
    precompile_tokens = sorted({128, TARGET_IN})

    t0 = time.perf_counter()
    engine = Engine(
        model_path=MODEL_DIR,
        tp_size=TP_SIZE,
        page_size=128,
        kv_cache_dtype="bf16",
        mem_fraction_static=0.85,
        disable_radix_cache=True,
        # bs=1 alone is fine for v1 (wrapper self-pads bs=1 -> bs=8) and v4
        # (TP, no padding needed). fused_v2 kernel rejects bs=1 outright
        # (requires num_tokens % ep_size == 0); set BS_PADDINGS=8 for v2 so the
        # engine compiles only the bs=8 graph and routes bs=1 requests through
        # it (auto-padded). The runtime payload is still bs=1 in both cases.
        precompile_bs_paddings=BS_PADDINGS,
        precompile_token_paddings=precompile_tokens,
        max_running_requests=1,
        device_indexes=list(range(TP_SIZE)),
        moe_backend=MOE_BACKEND,
        enable_single_process=True,
        trust_remote_code=True,
        log_level="info",
        random_seed=42,
        dtype="bfloat16",
    )
    t1 = time.perf_counter()
    print(f"Engine init took {t1 - t0:.1f}s", flush=True)

    # Sampling params: greedy decode, ignore EOS so we hit TARGET_OUT exactly.
    sp_prefill = [
        {"temperature": 0.0, "max_new_tokens": 1, "top_k": 1, "ignore_eos": True}
    ]
    sp_full = [
        {
            "temperature": 0.0,
            "max_new_tokens": TARGET_OUT,
            "top_k": 1,
            "ignore_eos": True,
        }
    ]

    # ── Warmup ──
    print(
        f"Warmup {WARMUP_ROUNDS} round(s): full decode to settle JIT cache.",
        flush=True,
    )
    for w in range(WARMUP_ROUNDS):
        wt0 = time.perf_counter()
        engine.generate(prompt=[prompt_text], sampling_params=sp_full)
        print(f"  warmup#{w} took {time.perf_counter() - wt0:.1f}s", flush=True)

    # ── Timed rounds ──
    print(f"Timed {TIMED_ROUNDS} round(s).", flush=True)
    ttft_samples = []
    decode_per_tok_samples_ms = []
    completion_token_samples = []

    for r in range(TIMED_ROUNDS):
        # Prefill only (max_new_tokens=1).
        t = time.perf_counter()
        rp = engine.generate(prompt=[prompt_text], sampling_params=sp_prefill)
        ttft = time.perf_counter() - t
        ttft_samples.append(ttft)

        # Full generation (TARGET_OUT new tokens). Use decode_elapsed = total - ttft
        # to isolate the decode-step cost; this matches the v1 baseline bench script.
        t = time.perf_counter()
        rd = engine.generate(prompt=[prompt_text], sampling_params=sp_full)
        total = time.perf_counter() - t
        decode_tokens = rd[0]["meta_info"]["completion_tokens"]
        decode_elapsed = total - ttft  # subtract the matching prefill cost
        per_tok_ms = 1000.0 * decode_elapsed / max(decode_tokens, 1)
        decode_per_tok_samples_ms.append(per_tok_ms)
        completion_token_samples.append(decode_tokens)

        print(
            f"  round#{r}: ttft={ttft:.3f}s decode_tokens={decode_tokens} "
            f"decode_per_tok={per_tok_ms:.2f}ms total={total:.2f}s",
            flush=True,
        )

    avg_ttft = sum(ttft_samples) / len(ttft_samples)
    avg_per_tok = sum(decode_per_tok_samples_ms) / len(decode_per_tok_samples_ms)

    def spread_pct(xs):
        if len(xs) < 2 or sum(xs) == 0:
            return 0.0
        return 100.0 * (max(xs) - min(xs)) / (sum(xs) / len(xs))

    summary = {
        "moe_backend": MOE_BACKEND,
        "in_tokens": TARGET_IN,
        "out_tokens": TARGET_OUT,
        "tp_size": TP_SIZE,
        "ttft_s_avg": round(avg_ttft, 3),
        "ttft_samples_s": [round(t, 3) for t in ttft_samples],
        "ttft_spread_pct": round(spread_pct(ttft_samples), 1),
        "decode_per_tok_ms_avg": round(avg_per_tok, 3),
        "decode_per_tok_ms_samples": [round(t, 3) for t in decode_per_tok_samples_ms],
        "decode_per_tok_spread_pct": round(spread_pct(decode_per_tok_samples_ms), 1),
        "decode_tokens_per_round": completion_token_samples,
        # Throughputs.
        "prefill_tps": round(TARGET_IN / avg_ttft, 1) if avg_ttft > 0 else 0,
        "decode_tps": round(1000.0 / avg_per_tok, 2) if avg_per_tok > 0 else 0,
    }
    print("\nBENCH_RESULT " + json.dumps(summary), flush=True)
    print(f"DONE_LING_V3_FLASH_LONG_CTX_BENCH backend={MOE_BACKEND}", flush=True)

    engine.shutdown()


if __name__ == "__main__":
    main()
