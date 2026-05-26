---
title: "Llama 3.3 70B"
---

# Llama 3.3 70B on SGL-JAX

> **Validated** on TPU v6e-16 (build `b2daa46d`, 2026-05-25). See §4 for measured numbers. v6e-32 / v6e-64 production tiers below remain Starter — same launch path with larger `--tp-size`, unmeasured.

## 1. Model Introduction

[**meta-llama/Llama-3.3-70B-Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) is Meta's 70B dense decoder from the Llama 3.3 release — multi-host serving required at BF16.

For the 8B size (single host + Phi-3 / InternLM3 alias support) see [`Llama3.1.md`](Llama3.1.md). For Llama 4 see the upstream sgl-cookbook (`Llama/Llama4.md`).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.9`, `max_tokens=1024` (Llama 3 Instruct defaults).

**License**: see the [Llama model card](https://huggingface.co/meta-llama) for the authoritative Meta Llama Community License terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|---|
| Minimum runnable | Llama 3.3 70B | v6e-16 | 4x4 | 4 | 16 | 16 | BF16 ~140 GB — fits with `--mem-fraction-static 0.85` (validated 2026-05-25, ~8.75 GB weights/chip + ample KV headroom) |
| Recommended production | Llama 3.3 70B | v6e-32 | 4x8 | 8 | 32 | 32 | More HBM per chip → higher `--max-running-requests` and longer context budget |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-16 (minimum)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=llama-70b`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tp-size 16 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (recommended production)

Same as above but `<TOPOLOGY>=4x8`, `parallelism: 8`, `completions: 8`, and bump `--tp-size 32 --mem-fraction-static 0.9`.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Memory Management:**
- v6e-16 (TP=16): `--mem-fraction-static 0.85` validated; ~8.75 GB weights per chip leaves ~18 GB headroom for KV at 32 GB HBM.
- v6e-32 (TP=32): bump to `--mem-fraction-static 0.9` — weights drop to ~4.4 GB per chip, more room for higher `--max-running-requests`. Lower to 0.85 if you hit OOM at startup.

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

> **Layout B — measured baseline.** TPU v6e-16 (4 nodes × 4 chips, TP=16), build `b2daa46d` (2026-05-25). sgl-jax-only; no vLLM-on-TPU comparison.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | meta-llama/Llama-3.3-70B-Instruct (BF16) |
| Tensor Parallelism | 16 |
| Tested build | `b2daa46d` (2026-05-25) |

**Benchmark Command**

```bash
python3 -m sgl_jax.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

```
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  59.45
Total input tokens:                      50561
Total generated tokens:                  52444
Request throughput (req/s):              1.68
Input token throughput (tok/s):          850.49
Output token throughput (tok/s):         882.17
Peak output token throughput (tok/s):    1040.00
Total token throughput (tok/s):          1732.66
Mean E2E Latency (ms):                   8647.20
Mean TTFT (ms):                          113.86
Mean TPOT (ms):                          16.33
Median TPOT (ms):                        16.47
Mean ITL (ms):                           16.30
==================================================
```

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | meta-llama/Llama-3.3-70B-Instruct (BF16) |
| Tensor Parallelism | 16 |
| Tested build | `b2daa46d` (2026-05-25) |

**Deployment Command** — same as [§2.3 v6e-16](#multi-host-gke-indexed-job--tpu-v6e-16-minimum).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 16 \
  --limit 200
```

Recommended additional datasets: MMLU, GPQA Diamond, IFEval.

**Test Results**

| Dataset | Subset | Samples | Score |
|---|---|---|---|
| gsm8k | main | 200 | **0.950** |

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Weights + KV exceed budget at chosen `--mem-fraction-static` | Lower to 0.85 (or 0.8). Verify `--tp-size` matches the chip count (v6e-16 → 16, v6e-32 → 32). |
| `Internal error when accessing libtpu multi-process lockfile` on rank 0 | Stale `/tmp/libtpu_lockfile` from a prior failed launch on the same pod | `rm -f /tmp/libtpu_lockfile` on every rank before relaunching. Worth scripting into the launch wrapper since multi-host failures often leave stale locks. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across all 8 nodes for amortized compilation. |

## Additional Resources

- [Llama 3.3 70B model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [`Llama3.1.md`](Llama3.1.md) — 8B single-host sibling.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
