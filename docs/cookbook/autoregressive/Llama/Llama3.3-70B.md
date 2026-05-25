---
title: "Llama 3.3 70B"
---

# Llama 3.3 70B on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**meta-llama/Llama-3.3-70B-Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) is Meta's 70B dense decoder from the Llama 3.3 release — multi-host serving required at BF16.

For the 8B size (single host + Phi-3 / InternLM3 alias support) see [`Llama3.1.md`](Llama3.1.md). For Llama 4 see the upstream sgl-cookbook (`Llama/Llama4.md`).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.9`, `max_tokens=1024` (Llama 3 Instruct defaults).

**License**: see the [Llama model card](https://huggingface.co/meta-llama) for the authoritative Meta Llama Community License terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|---|
| Minimum runnable | Llama 3.3 70B | v6e-32 | 4x8 | 8 | 32 | 32 | BF16 ~140 GB — multi-host required to fit weights + KV |
| Recommended production | Llama 3.3 70B | v6e-64 | 8x8 | 16 | 64 | 64 | More HBM per chip → higher `--max-running-requests` and longer context budget |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-32

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=llama-70b`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tp-size 32 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.9` leaves ~4 GB per chip for KV at TP=32. Lower to 0.85 if you hit OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at 70B scale.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.
- `--max-running-requests 256` caps concurrent decodes; lower for tighter latency tails.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- The cache is per-node; mount a shared PVC at the cache directory to amortize compilation across all 8 nodes.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="meta-llama/Llama-3.3-70B-Instruct"` with the §1 recommended sampling parameters.

> Llama 3 Instruct is non-reasoning and has no native tool-call format. For those workloads choose a model with `--reasoning-parser` / `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `meta-llama/Llama-3.3-70B-Instruct`, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-32 (8 nodes × 4 chips) |
| Model | meta-llama/Llama-3.3-70B-Instruct (BF16) |
| Tensor Parallelism | 32 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-32).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, IFEval.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Weights + KV exceed budget at chosen `--mem-fraction-static` | Lower to 0.85. Verify `--tp-size 32` matches v6e-32 chip count (4 × 8 = 32). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across all 8 nodes for amortized compilation. |

## Additional Resources

- [Llama 3.3 70B model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [`Llama3.1.md`](Llama3.1.md) — 8B single-host sibling.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
