---
title: "Ling 2.0"
description: "InclusionAI Ling 2.0 family (mini / flash / 1T) — 1/32 sparsity MoE with MTP serving on TPU v6e-4 to v6e-64 with SGL-JAX."
---

# Ling 2.0 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**inclusionAI/Ling-2.0**](https://huggingface.co/inclusionAI) is InclusionAI's second-generation Ling MoE family — a redesigned MoE architecture with **1/32 activation sparsity**, **MTP (Multi-Token Prediction) layers**, aux-loss-free + sigmoid routing, QK-Norm, and half-RoPE. Ling 2.0 spans from a 16B mini tier to the 1T flagship `Ling-1T`.

**Variants** (pick by size):

- [**inclusionAI/Ling-mini-2.0**](https://huggingface.co/inclusionAI/Ling-mini-2.0) — 16B total / 1.4B activated; comfortable single-host fit, ~7× equivalent dense performance per activated parameter.
- [**inclusionAI/Ling-flash-2.0**](https://huggingface.co/inclusionAI/Ling-flash-2.0) — 100B total / 6.1B activated; multi-host MoE.
- [**inclusionAI/Ling-1T**](https://huggingface.co/inclusionAI/Ling-1T) — 1T total / ~50B activated per token; FP8-native flagship non-thinking model with 128K context.

> **FP8 weights note (Ling-1T):** Ling-1T ships natively as FP8 (the largest FP8-trained foundation model to date). For SGL-JAX BF16 serving you'll need to convert weights via InclusionAI's `convert_dcp_to_safe_tensors.py` (see the model card). The hardware sizing below assumes BF16 weights.

For the first-generation Ling 1.x family see [`Ling-1.x.md`](Ling-1.x.md). For the linear-attention 2.6 generation see [`Ling-2.6.md`](Ling-2.6.md). For the reasoning-tuned Ring 2.0 sibling line (same architecture, RL post-training) see [`Ring-2.md`](Ring-2.md).

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048` (give room for Ling-1T's long reasoning chains even in non-thinking mode).

**License**: see each model card on the [InclusionAI HF collection](https://huggingface.co/inclusionAI) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Ling-mini-2.0 (16B / 1.4B)  | v6e-4  | 2x2 | 1  | 4  | 4  | 4  | BF16 ~32 GB — single host |
| Ling-flash-2.0 (100B / 6.1B) | v6e-16 | 4x4 | 4  | 16 | 16 | 16 | BF16 ~200 GB |
| Ling-1T (1T / 50B)           | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~2 TB — multi-host mandatory |
| Ling-1T (1T / 50B)           | v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For Ling-mini-2.0 single-host use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md); for Ling-flash-2.0 / Ling-1T multi-host use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Ling-mini-2.0)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-mini-2.0 \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Multi-host (SkyPilot) — TPU v6e-16 (Ling-flash-2.0)

**Step 1** — provision the cluster:

```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-16 main
```

**Step 2** — launch the server:

```bash
CLUSTER_NAME=$(cat .cluster_name)
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-flash-2.0 \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --dist-init-addr <NODE_0_IP_ADDRESS>:5000 \
  --nnodes 4 --node-rank \${SKYPILOT_NODE_RANK} \
  --host 0.0.0.0 --port 30000"
```

#### Multi-host (SkyPilot) — TPU v6e-64 (Ling-1T)

Swap the topology to `tpu-v6e-64`, the model path to `inclusionAI/Ling-1T`, and use:

```text
  --tp-size 64 --ep-size 64 \
  --mem-fraction-static 0.92 \
  --nnodes 16 --node-rank \${SKYPILOT_NODE_RANK} \
```

For GKE, adapt the manifest pattern from [`../Xiaomi/MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=ling-2-0`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), the corresponding topology, and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (Ling-mini-2.0).
- `--moe-backend fused` for `--ep-size ≥ 16` (Ling-flash-2.0 / Ling-1T).

**Memory Management:**
- Ling-mini-2.0: `--mem-fraction-static 0.88` (TPU default). Raise to `0.9` for dedicated single-host serving.
- Ling-flash-2.0: `--mem-fraction-static 0.9` for dedicated multi-host serving.
- Ling-1T: `--mem-fraction-static 0.92` for the full v6e-64 slice. Drop to `0.9` if OOM at startup.

**FP8 Weights (Ling-1T):**
- Convert FP8 weights to BF16 with InclusionAI's `convert_dcp_to_safe_tensors.py` before serving — SGL-JAX consumes BF16 on TPU. The model card has the conversion command.
- Once converted, `--dtype bfloat16` is the right choice (per the standard launch above).

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale. Default `1` is fine for Ling-mini low-concurrency mixed traffic.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill (Ling-1T supports up to 128K context).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node Ling-flash / Ling-1T clusters, mount a shared PVC at the cache directory to amortize compilation across nodes.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`../Qwen/Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute `model="inclusionAI/Ling-mini-2.0"` (or your chosen variant).

> Ling 2.0 is a non-thinking general-purpose family. For chain-of-thought workloads use the reasoning-tuned [`Ring-2.md`](Ring-2.md) sibling (same architecture, RL-trained). For tool-call workloads use a model with `--tool-call-parser` support (see [`../Qwen/Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`../Xiaomi/MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (mini) / v6e-16 (flash) / v6e-64 (1T) |
| Model | inclusionAI/Ling-mini-2.0 / Ling-flash-2.0 / Ling-1T (BF16) |
| Tensor Parallelism | 4 / 16 / 64 |
| Expert Parallelism | 4 / 16 / 64 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-ling-mini-20).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model inclusionAI/Ling-mini-2.0 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, HumanEval (general); for Ling-1T also IFEval (long-context instruction following).

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`../Qwen/Qwen3.md` §4.2](../Qwen/Qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Ling 2.0 checkpoint, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau (flash / 1T) | Wrong `--moe-backend` for the EP size | Use `--moe-backend fused` at EP ≥ 16; `epmoe` only at EP ≤ 8. |
| Ling-1T weights fail to load | Loaded raw FP8 weights without conversion | Run InclusionAI's `convert_dcp_to_safe_tensors.py` to produce BF16 safetensors first. |
| OOM at startup (1T) | `--mem-fraction-static 0.92` too high for this slice | Lower to 0.9. Verify `--tp-size` matches chip count (v6e-64 → 64; v7x-16 → 32, since v7x exposes 2 JAX devices per chip). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [InclusionAI HF collection](https://huggingface.co/inclusionAI)
- [`Ling-1.x.md`](Ling-1.x.md) — first-generation Ling family.
- [`Ling-2.6.md`](Ling-2.6.md) — Ling 2.6 (linear / delta attention).
- [`Ring-2.md`](Ring-2.md) — reasoning-tuned Ring 2.0 sibling line.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
