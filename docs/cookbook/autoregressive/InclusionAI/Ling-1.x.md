---
title: "Ling 1.x"
---

# Ling 1.x on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**inclusionAI/Ling**](https://huggingface.co/inclusionAI) is InclusionAI's first-generation Ling MoE family — instruction-tuned MoE decoders ranging from a 16.8B "lite" tier to a 290B "plus" tier. All Ling 1.x checkpoints share the same runtime path on SGL-JAX.

**Variants** (pick by size / fine-tune):

- [**inclusionAI/Ling-lite**](https://huggingface.co/inclusionAI/Ling-lite) — 16.8B total / 2.75B activated; chat-tuned single-host fit.
- [**inclusionAI/Ling-plus**](https://huggingface.co/inclusionAI/Ling-plus) — 290B total / 28.8B activated; multi-host required.
- [**inclusionAI/Ling-Coder-lite**](https://huggingface.co/inclusionAI/Ling-Coder-lite) — 16.8B / 2.75B, further pre-trained on 3T tokens of coding data; same launch path as Ling-lite.

For the Ling 2.0 generation (Ling-mini-2.0 / Ling-flash-2.0 / Ling-1T) see [`Ling-2.md`](Ling-2.md). For the linear-attention 2.6 generation see [`Ling-2.6.md`](Ling-2.6.md). For the reasoning-tuned Ring 2.0 series see [`Ring-2.md`](Ring-2.md).

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024` (Ling-lite chat defaults). For Ling-Coder-lite raise `max_tokens` to 2048+ to give room for code blocks.

**License**: see each model card on the [InclusionAI HF collection](https://huggingface.co/inclusionAI) for the authoritative MIT terms applied to Ling 1.x.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Ling-lite / Ling-Coder-lite (16.8B / 2.75B) | v6e-4  | 2x2 | 1 | 4  | 4  | 4  | BF16 ~34 GB — single host |
| Ling-plus (290B / 28.8B)                    | v6e-32 | 4x8 | 8 | 32 | 32 | 32 | BF16 ~580 GB — multi-host required |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For Ling-lite / Ling-Coder-lite use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md); for Ling-plus multi-host use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Ling-lite / Ling-Coder-lite)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-lite \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `inclusionAI/Ling-Coder-lite` for the coding-tuned variant.

#### Multi-host (SkyPilot) — TPU v6e-32 (Ling-plus)

**Step 1** — provision the cluster:

```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-32 main
```

**Step 2** — launch the server:

```bash
CLUSTER_NAME=$(cat .cluster_name)
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-plus \
  --trust-remote-code \
  --tp-size 32 --ep-size 32 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --dist-init-addr <NODE_0_IP_ADDRESS>:5000 \
  --nnodes 8 --node-rank \${SKYPILOT_NODE_RANK} \
  --host 0.0.0.0 --port 30000"
```

For GKE, adapt the manifest pattern from [`../Xiaomi/MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=ling-plus`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8` / `completions: 8`, and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` at `--ep-size ≤ 8` (Ling-lite / Ling-Coder-lite single-host).
- `--moe-backend fused` at `--ep-size ≥ 16` (Ling-plus multi-host).

**Memory Management:**
- Ling-lite / Coder-lite: `--mem-fraction-static 0.88` (TPU default). Raise to `0.9` for higher concurrency on a dedicated host.
- Ling-plus: `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if OOM at startup with high `--max-running-requests`.

**Throughput vs Latency (Ling-plus):**
- `--page-size 128` reduces KV page-table overhead at MoE scale. Default `1` is fine for Ling-lite low-concurrency mixed traffic.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- On multi-node Ling-plus clusters the cache is per-node; mount a shared PVC to amortize compilation across all 8 nodes.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`../Qwen/Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion) for the curl / Python pattern. Substitute `model="inclusionAI/Ling-lite"` (or your chosen variant).

> Ling 1.x is an instruct/chat baseline without native hybrid reasoning or built-in tool-call format. For reasoning workloads use [`Ring-2.md`](Ring-2.md) (the reasoning-tuned sibling line). For tool-call workloads use a model with `--tool-call-parser` support (see [`../Qwen/Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`../Xiaomi/MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`../Qwen/Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Ling checkpoint, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (lite / coder-lite) or v6e-32 (plus) |
| Model | inclusionAI/Ling-lite / Ling-Coder-lite / Ling-plus (BF16) |
| Tensor Parallelism | 4 (lite) / 32 (plus) |
| Expert Parallelism | 4 (lite) / 32 (plus) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-ling-lite--ling-coder-lite).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model inclusionAI/Ling-lite \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU (general); for Ling-Coder-lite use HumanEval / MBPP / LiveCodeBench.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau on Ling-plus | Wrong `--moe-backend` for the EP size | Use `--moe-backend fused` at EP ≥ 16; `epmoe` only at EP ≤ 8. |
| Ling-plus OOM at startup | `--mem-fraction-static 0.92` too high | Lower to 0.9. Verify `--tp-size 32` matches v6e-32 chip count (4 × 8 = 32). |
| Multi-node Ling-plus hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker; shared PVC across multi-node). |

## Additional Resources

- [InclusionAI HF collection](https://huggingface.co/inclusionAI)
- [`Ling-2.md`](Ling-2.md) — Ling 2.0 generation (1/32 sparsity, MTP).
- [`Ling-2.6.md`](Ling-2.6.md) — Ling 2.6 (linear / delta attention).
- [`Ring-2.md`](Ring-2.md) — reasoning-tuned Ring 2.0 sibling line.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
