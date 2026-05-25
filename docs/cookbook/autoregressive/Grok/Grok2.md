---
title: "Grok-2"
---

# Grok-2 on SGL-JAX

> **Starter recipe** — derived from the Grok-2 release format and SGL-JAX multi-host launch path; not yet empirically validated on TPU. Tune values for your hardware and PR back tested numbers.

## 1. Model Introduction

[**xai-org/grok-2**](https://huggingface.co/xai-org/grok-2) is xAI's open-weight **314B-parameter dense** model — one of the largest open dense LLMs available. SGL-JAX serves it on TPU v6e-32 (8 nodes × 4 chips) with tensor parallelism. The primary user-facing deployment path is GKE Indexed Job; SkyPilot is an advanced v6e experiment alternative.

**Key Features**:

- **314B dense parameters** — large flagship model from xAI; ~628 GB BF16 weights spread across 32 chips.
- **Open weights** — community-licensed for self-hosted serving.
- **Long context** — supports extended context windows (verify on the model card for the exact length).

**Recommended Generation Parameters** (matching the official xAI defaults): `temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=0.5`.

**Tokenizer note**: Grok-2 weights ship **without a tokenizer file** — use the community tokenizer `alvarobartt/grok-2-tokenizer` via `--tokenizer-path`. See [Community tokenizer card](https://huggingface.co/alvarobartt/grok-2-tokenizer).

**License**: see the [HuggingFace model card](https://huggingface.co/xai-org/grok-2) for the authoritative xAI Community License terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips per node | Total chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| **v6e-32** (minimum, required) | 4x8 | 8 | 4 | 32 | 32 | v6e is 1:1 chip↔device; `--tp-size=32` saturates the slice |

Grok-2 314B requires the full v6e-32 slice — no smaller config fits. See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). For multi-node launches, use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

The community tokenizer is downloaded on first launch — no extra pip needed.

### 2.3 Launch

Grok-2 314B is multi-host only; cannot fit single-host.

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (8 nodes)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=grok-2`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path /models/xai-grok-2 \
  --trust-remote-code \
  --tokenizer-path alvarobartt/grok-2-tokenizer \
  --tp-size 32 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --download-dir /dev/shm \
  --random-seed 3 \
  --skip-server-warmup
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.9` is appropriate for a dedicated serving slice (default 0.88 leaves more headroom for unexpected allocations). At 314B BF16 weights split across 32 chips, ~20 GB per chip for weights leaves room for KV cache.
- `--download-dir /dev/shm` stages weights on tmpfs for fast load (~50% faster cold start than `/tmp`). Switch back to `/tmp` if shared memory is constrained on your host.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at 314B scale. Default `1` is much slower for this model size.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill. Larger values (4096) reduce TTFT on long prompts but risk prefill-time OOM.
- `--max-running-requests 256` is the concurrent decode bound; raise for higher throughput, lower for tighter latency tails.

**Tokenizer:**
- Without `--tokenizer-path alvarobartt/grok-2-tokenizer`, the server fails at startup since Grok-2 weights ship without a tokenizer file.
- The tokenizer requires an HF token if your network has rate limits — set `HF_TOKEN` env if needed.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters, the cache is per-node. Mount a shared PVC if you want compilation to amortize across nodes.

For full flag definitions and defaults see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="xai-org/grok-2"` and replace `127.0.0.1` with your rank-0 internal IP; pass `top_k`/`presence_penalty` from §1 via `extra_body={"top_k": 20, "presence_penalty": 0.5}` since the OpenAI schema does not include them.

> Grok-2 has no hybrid reasoning or native tool-calling format. For those workloads use a model with `--reasoning-parser` / `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Speed — single workload (low-concurrency latency baseline)

**Test Environment** — same as §4.2.

**Deployment Command** — same as [§2.3 Multi-host (GKE Indexed Job)](#multi-host-gke-indexed-job--tpu-v6e-32-8-nodes).

**Benchmark Command** — adapt the driver script from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `xai-org/grok-2`, remove the vLLM half).

**Test Results** — _Pending. Run and PR back the full `============ Serving Benchmark Result ============` block._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-32 (8 nodes × 4 chips) |
| Model | xai-org/grok-2 (BF16, local path `/models/xai-grok-2`) |
| Tokenizer | alvarobartt/grok-2-tokenizer |
| Tensor Parallelism | 32 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3 Multi-host (GKE Indexed Job)](#multi-host-gke-indexed-job--tpu-v6e-32-8-nodes).

**Benchmark Command — GSM8K** (math reasoning):

```bash
evalscope eval \
  --model /models/xai-grok-2 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 64 \
  --generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.5}'
```

**Benchmark Command — GPQA Diamond** (graduate-level science QA):

```bash
evalscope eval \
  --model /models/xai-grok-2 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gpqa_diamond \
  --eval-batch-size 198 \
  --dataset-args '{"gpqa_diamond": {"few_shot_num": 4}}' \
  --generation-config '{"temperature": 0.5, "top_p": 0.8, "top_k": 40, "max_tokens": 4096}'
```

**Test Results** — _Pending. Run the commands above and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tokenizer load fails at startup | `--tokenizer-path` missing or unreachable | Add `--tokenizer-path alvarobartt/grok-2-tokenizer`; if HF rate-limited, set `HF_TOKEN` env. |
| Multi-node hang at `jax.distributed.initialize` | `--dist-init-addr` unreachable from non-rank-0 nodes | `sky status -a ${CLUSTER_NAME}` to verify rank-0 internal IP; check firewall on the chosen port. |
| OOM at startup | `--mem-fraction-static 0.9` too high for shared host | Lower to 0.85; verify `--tp-size 32` matches v6e-32 chip count (4 × 8 = 32). |
| Slow cold start (~4 min per node) on every launch | JIT cache not persisted across launches | Mount a persistent volume at `/tmp/jit_cache` (or a shared PVC across all 8 nodes for amortized compilation). |

## Additional Resources

- [Grok-2 Model Card](https://huggingface.co/xai-org/grok-2)
- [Community tokenizer](https://huggingface.co/alvarobartt/grok-2-tokenizer)
- [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) — primary multi-host launcher template.
- [`../deployment/skypilot.md`](../../deployment/skypilot.md) — advanced v6e experiment alternative.
- [`MiMo-V2.5-Pro.md`](../Xiaomi/MiMo-V2.5-Pro.md) — GKE Indexed Job manifest reference (adapt for Grok-2).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
