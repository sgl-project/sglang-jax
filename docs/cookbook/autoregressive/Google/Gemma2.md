---
title: "Gemma 2"
---

# Gemma 2 on SGL-JAX

> **27B-it: Validated** on TPU v6e-4 (build `de29d9f0`, 2026-05-25). See §4 for measured numbers. The 9B / 9B-it sizes below remain Starter — same launch path with the same flags, unmeasured.

## 1. Model Introduction

[**google/gemma-2**](https://huggingface.co/google) is Google's second-generation open Gemma series — dense decoder models with **hybrid attention** (alternating global 8K and sliding-window 4K layers) and soft-cap attention/logits. Both released sizes (9B and 27B) fit on a single TPU v6e-4 host.

**Variants** (pick by size / fine-tune):

- [**google/gemma-2-9b**](https://huggingface.co/google/gemma-2-9b) — base 9B pre-trained.
- [**google/gemma-2-9b-it**](https://huggingface.co/google/gemma-2-9b-it) — 9B instruction-tuned.
- [**google/gemma-2-27b**](https://huggingface.co/google/gemma-2-27b) — base 27B pre-trained.
- [**google/gemma-2-27b-it**](https://huggingface.co/google/gemma-2-27b-it) — 27B instruction-tuned; default chat choice.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `top_k=64`, `max_tokens=1024` (Gemma 2 model-card defaults).

**License**: see the [Gemma model card](https://huggingface.co/google/gemma-2-9b) for the authoritative Gemma Terms of Use.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | Gemma 2 9B / 9B-it | v6e-4 | 2x2 | 4 | 4 | BF16 ~18 GB — fits with headroom; default `--mem-fraction-static 0.88` |
| Minimum runnable | Gemma 2 27B / 27B-it | v6e-4 | 2x2 | 4 | 4 | BF16 ~54 GB — fits with `--mem-fraction-static 0.85` (validated 2026-05-25, ~13.5 GB weights/chip + dual KV pools) |
| Recommended production (27B) | Gemma 2 27B-it | v6e-8 | 2x4 | 8 | 8 | More HBM headroom so the global + sliding KV pools both have room; raises `--max-running-requests` ceiling |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Gemma 2 9B-it)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path google/gemma-2-9b-it \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Single-host (Docker) — TPU v6e-4 (Gemma 2 27B-it)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path google/gemma-2-27b-it \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

The 27B variant uses a lower `--mem-fraction-static` than 9B to leave room for the dual (global + sliding) KV pools.

#### Multi-host

Not needed at the 9B / 27B scale — both fit single-host on v6e-4.

### 2.4 Configuration Tips

**Memory Management:**
- 9B: `--mem-fraction-static 0.88` (TPU default). Raise to `0.9` for higher concurrency on a dedicated host.
- 27B: `--mem-fraction-static 0.85` validated; ~13.5 GB weights/chip leaves enough HBM for both KV pools at `--max-running-requests 64`. Drop to `0.8` if you raise concurrency further and hit OOM.

**Paging / concurrency (mandatory):**
- `--page-size 128` is required. Without it (default = 1) the scheduler silently caps `Final max_running_requests: 1`, serializing all requests. `--max-running-requests 64` then sets the actual concurrency ceiling.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.

**Hybrid Attention (Gemma-specific):**
- Gemma 2 alternates global (8K context) and sliding-window (4K) attention layers. SGL-JAX manages two KV pools — global and sliding — which is more memory-sensitive than uniform-attention models at the same parameter count.
- `--swa-full-tokens-ratio` (default 0.8) controls the per-layer ratio of sliding-window vs full-attention layers and gates pool sizing. If you see sliding-window pool exhaustion at high concurrency, lower this ratio to give the sliding pool more capacity. See [troubleshooting §SWA pool exhaustion](../../troubleshooting.md#swa-pool-exhaustion-mimo-hybrid-attention-models).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, or `--swa-full-tokens-ratio` invalidates cached entries.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="google/gemma-2-9b-it"` (or your chosen variant) with the §1 recommended sampling parameters — pass `top_k` via `extra_body={"top_k": 64}` since the OpenAI schema does not include it.

> Gemma 2 is non-reasoning and has no native tool-call format. For those workloads choose a model with `--reasoning-parser` / `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — measured baseline.** TPU v6e-4 (single host, 4 chips, TP=4), build `de29d9f0` (2026-05-25). sgl-jax-only; no vLLM-on-TPU comparison.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | google/gemma-2-27b-it (BF16) |
| Tensor Parallelism | 4 |
| Tested build | `de29d9f0` (2026-05-25) |

**Benchmark Command**

```bash
python3 -m sgl_jax.bench_serving \
  --backend sglang \
  --model google/gemma-2-27b-it \
  --tokenizer google/gemma-2-27b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

```
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  53.81
Total input tokens:                      50561
Total generated tokens:                  52444
Request throughput (req/s):              1.86
Input token throughput (tok/s):          939.58
Output token throughput (tok/s):         974.57
Peak output token throughput (tok/s):    1159.00
Total token throughput (tok/s):          1914.15
Mean E2E Latency (ms):                   7654.91
Mean TTFT (ms):                          77.57
Mean TPOT (ms):                          14.47
Median TPOT (ms):                        14.53
Mean ITL (ms):                           14.48
==================================================
```

For Gemma 2 9B on the same host, expect roughly 2-3x higher output throughput at the same concurrency — PR back the measured block to upgrade.

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | google/gemma-2-27b-it (BF16) |
| Tensor Parallelism | 4 |
| Tested build | `de29d9f0` (2026-05-25) |

**Deployment Command** — same as [§2.3 27B-it](#single-host-docker--tpu-v6e-4-gemma-2-27b-it).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model google/gemma-2-27b-it \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 16 \
  --limit 200
```

Recommended additional datasets: MMLU, GPQA Diamond.

**Test Results**

| Dataset | Subset | Samples | Score |
|---|---|---|---|
| gsm8k | main | 200 | **0.865** |

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Gemma 2 27B OOM at startup | `--mem-fraction-static 0.88` (default) too high for dual KV pools | Lower to 0.85 or 0.8. Verify `--tp-size 4` matches v6e-4 chip count. |
| Output throughput stuck near single-stream (~150 tok/s); TTFT > 5 s under concurrent load | `--page-size` defaulted to 1 → scheduler logs `Final max_running_requests: 1` and serializes requests | Add `--page-size 128 --max-running-requests 64` (see §2.4 Paging / concurrency). |
| Sliding-window pool exhaustion at high concurrency | `--swa-full-tokens-ratio` mismatches workload sequence-length distribution | Lower the ratio to give the sliding pool more capacity. See [troubleshooting §SWA pool exhaustion](../../troubleshooting.md#swa-pool-exhaustion-mimo-hybrid-attention-models). |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [Gemma model collection](https://huggingface.co/google)
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues including the SWA pool exhaustion note.
