---
title: "Gemma 2"
description: "Gemma 2 9B and 27B hybrid-attention decoders serving on TPU v6e-4 with SGL-JAX."
---

# Gemma 2 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

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

| Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Gemma 2 9B / 9B-it  | v6e-4 | 2x2 | 4 | 4 | BF16 ~18 GB — fits with headroom |
| Gemma 2 27B / 27B-it | v6e-4 | 2x2 | 4 | 4 | BF16 ~54 GB — fits; lower `--mem-fraction-static` |

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
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

The 27B variant uses a lower `--mem-fraction-static` than 9B to leave room for the dual (global + sliding) KV pools.

#### Multi-host

Not needed at the 9B / 27B scale — both fit single-host on v6e-4.

### 2.4 Configuration Tips

**Memory Management:**
- 9B: `--mem-fraction-static 0.88` (TPU default). Raise to `0.9` for higher concurrency on a dedicated host.
- 27B: start at `--mem-fraction-static 0.85`. Drop to `0.8` if you hit OOM at startup with high `--max-running-requests`.

**Hybrid Attention (Gemma-specific):**
- Gemma 2 alternates global (8K context) and sliding-window (4K) attention layers. SGL-JAX manages two KV pools — global and sliding — which is more memory-sensitive than uniform-attention models at the same parameter count.
- `--swa-full-tokens-ratio` (default 0.8) controls the per-layer ratio of sliding-window vs full-attention layers and gates pool sizing. If you see sliding-window pool exhaustion at high concurrency, lower this ratio to give the sliding pool more capacity. See [troubleshooting §SWA pool exhaustion](../../troubleshooting.md#swa-pool-exhaustion-mimo-hybrid-attention-models).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, or `--swa-full-tokens-ratio` invalidates cached entries.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion) for the curl / Python pattern. Substitute `model="google/gemma-2-9b-it"` (or your chosen variant).

> Gemma 2 does not ship with native hybrid reasoning or a built-in tool-call format. For reasoning / tool-call workloads use a model with `--reasoning-parser` / `--tool-call-parser` support (see [`Qwen3.md` §3.2 / §3.3](../Qwen/Qwen3.md) or [`MiMo-V2.5-Pro.md`](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | google/gemma-2-9b-it or gemma-2-27b-it (BF16) |
| Tensor Parallelism | 4 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-gemma-2-9b-it).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model google/gemma-2-9b-it \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.2](../Qwen/Qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Gemma checkpoint).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Gemma 2 27B OOM at startup | `--mem-fraction-static 0.88` (default) too high for dual KV pools | Lower to 0.85 or 0.8. Verify `--tp-size 4` matches v6e-4 chip count. |
| Sliding-window pool exhaustion at high concurrency | `--swa-full-tokens-ratio` mismatches workload sequence-length distribution | Lower the ratio to give the sliding pool more capacity. See [troubleshooting §SWA pool exhaustion](../../troubleshooting.md#swa-pool-exhaustion-mimo-hybrid-attention-models). |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [Gemma model collection](https://huggingface.co/google)
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues including the SWA pool exhaustion note.
