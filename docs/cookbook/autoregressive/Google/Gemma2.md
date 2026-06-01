---
title: "Gemma 2"
---

# Gemma 2 on SGL-JAX

> **Validated recipe** — Gemma 2 27B-it validated on TPU v6e-4 with sglang-jax 0.1.0; see §4 for measured numbers.

## 1. Model Introduction

[**google/gemma-2-27b-it**](https://huggingface.co/google/gemma-2-27b-it) is Google's instruction-tuned 27B Gemma 2 — a dense decoder with **hybrid attention** (alternating global 8K and sliding-window 4K layers) and soft-cap attention/logits. Fits on a single TPU v6e-4 host (BF16 ~54 GB).

**Key Features**:

- **Dense decoder, single-host fit**: 27B in BF16 (~54 GB) on a single TPU v6e-4 host. No multi-host complexity.
- **Hybrid Attention**: Alternating global (8K) and sliding-window (4K) layers — SGL-JAX manages two KV pools (global + sliding), giving stronger long-context behavior than uniform-attention models at the same parameter count.
- **Soft-cap attention/logits**: Gemma 2's signature stability tweak — caps attention logits and output logits to reduce extreme activations.
- **Instruction-tuned for chat**: `27b-it` is the post-trained chat model — non-reasoning, no native tool-call format.
- **Open weights under Gemma Terms of Use**: see [Gemma model card](https://huggingface.co/google/gemma-2-27b-it).

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `top_k=64`, `max_tokens=1024` (Gemma 2 model-card defaults).

**License**: see the [Gemma model card](https://huggingface.co/google/gemma-2-27b-it) for the authoritative Gemma Terms of Use.

## 2. Deployment

### 2.1 Hardware Matrix

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | Gemma 2 27B-it | v6e-4 | 2x2 | 4 | 4 | BF16 ~54 GB — fits with `--mem-fraction-static 0.85` (validated 2026-05-25, ~13.5 GB weights/chip + dual KV pools) |
| Recommended production | Gemma 2 27B-it | v6e-8 | 2x4 | 8 | 8 | More HBM headroom so the global + sliding KV pools both have room; raises `--max-running-requests` ceiling |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

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

The 27B-it `--mem-fraction-static 0.85` leaves room for the dual (global + sliding) KV pools.

#### Multi-host

Not needed at the 27B scale — fits single-host on v6e-4.

### 2.4 Configuration Tips

**Memory Management:**
- 27B-it: `--mem-fraction-static 0.85` validated; ~13.5 GB weights/chip leaves enough HBM for both KV pools at `--max-running-requests 64`. Drop to `0.8` if you raise concurrency further and hit OOM.

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

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Pass `top_k` via `extra_body={"top_k": 64}` since the OpenAI schema does not include it.

Short Python OpenAI client example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="google/gemma-2-27b-it",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"top_k": 64},
)
print(resp.choices[0].message.content)
```

> Gemma 2 is non-reasoning and has no native tool-call format. For those workloads choose a model with `--reasoning-parser` / `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | google/gemma-2-27b-it (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax 0.1.0 |

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

### 4.2 Speed

> **Layout B — measured baseline.** TPU v6e-4 (single host, 4 chips, TP=4), sglang-jax 0.1.0. sgl-jax-only; no vLLM-on-TPU comparison.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | google/gemma-2-27b-it (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax 0.1.0 |

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

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Gemma 2 27B OOM at startup | `--mem-fraction-static 0.88` (default) too high for dual KV pools | Lower to 0.85 or 0.8. Verify `--tp-size 4` matches v6e-4 chip count. |
| Output throughput stuck near single-stream (~150 tok/s); TTFT > 5 s under concurrent load | `--page-size` defaulted to 1 → scheduler logs `Final max_running_requests: 1` and serializes requests | Add `--page-size 128 --max-running-requests 64` (see §2.4 Paging / concurrency). |
| Sliding-window pool exhaustion at high concurrency | `--swa-full-tokens-ratio` mismatches workload sequence-length distribution | Lower the ratio to give the sliding pool more capacity. See [troubleshooting §SWA pool exhaustion](../../troubleshooting.md#swa-pool-exhaustion-mimo-hybrid-attention-models). |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [Gemma 2 27B-it model card](https://huggingface.co/google/gemma-2-27b-it)
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues including the SWA pool exhaustion note.
