---
title: "DeepSeek V3"
---

# DeepSeek V3 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-V3**](https://huggingface.co/deepseek-ai/DeepSeek-V3) is DeepSeek's 671B / 37B-activated MoE flagship — built on **MLA** (Multi-head Latent Attention). Multi-host serving required at BF16.

For the V2 generation (V2 / V2-Lite) see [`DeepSeek-V2.md`](DeepSeek-V2.md). For the reasoning-tuned R1 derivative see [`DeepSeek-R1.md`](DeepSeek-R1.md).

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — `--moe-backend fused` is the right choice at V3 scale (see §2.4).
- **DSA** (DeepSeek Sparse Attention) on V3.2 — activated by model config; no extra launch flag.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V3) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|
| v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~1.3 TB — full v6e-64 slice |
| v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-64

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=deepseek-v3`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, and `completions: 16`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --tp-size 64 --ep-size 64 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v7x-16

Use `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`; change the launch flags above to:

```text
  --tp-size 32 --ep-size 32 \
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The default SkyPilot template is v6e-only; use GKE for v7x.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` for V3 (EP ≥ 32). `epmoe` is only for EP ≤ 8.

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if OOM at startup with high `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at trillion-parameter scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC at the cache directory to amortize compilation across all 16 nodes.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="deepseek-ai/DeepSeek-V3"` with the §1 recommended sampling parameters.

> DeepSeek V3 is non-reasoning and has no native tool-call format. For reasoning use [`DeepSeek-R1.md`](DeepSeek-R1.md); for tool-call workloads choose a model with `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `deepseek-ai/DeepSeek-V3`, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) or v7x-16 |
| Model | deepseek-ai/DeepSeek-V3 (BF16) |
| Tensor Parallelism | 64 (v6e) / 32 (v7x) |
| Expert Parallelism | 64 (v6e) / 32 (v7x) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-64).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model deepseek-ai/DeepSeek-V3 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, HumanEval, LiveCodeBench.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau | Wrong `--moe-backend` | Use `--moe-backend fused` at V3 scale. `epmoe` is only for EP ≤ 8. |
| OOM at startup | `--mem-fraction-static 0.92` too high for this slice | Lower to 0.9. Verify `--tp-size` matches chip count (v6e-64 → 64; v7x-16 → 32, since v7x exposes 2 JAX devices per chip). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across all 16 nodes for amortized compilation. |

## Additional Resources

- [DeepSeek-V3 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [`DeepSeek-V2.md`](DeepSeek-V2.md) — V2 / V2-Lite generation.
- [`DeepSeek-R1.md`](DeepSeek-R1.md) — reasoning-tuned V3 derivative.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
