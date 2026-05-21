---
title: "DeepSeek V2"
description: "DeepSeek V2 / V2-Lite MoE with MLA serving on TPU v6e-4 (Lite) or v6e-32 (V2) with SGL-JAX."
---

# DeepSeek V2 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-V2**](https://huggingface.co/deepseek-ai/DeepSeek-V2) is DeepSeek's second-generation MoE decoder built on **MLA** (Multi-head Latent Attention). The V2 generation ships in two sizes — a single-host "Lite" tier and a 236B multi-host flagship.

**Variants** (pick by size):

- [**deepseek-ai/DeepSeek-V2-Lite**](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) — 15.7B total / 2.4B activated; minimal MoE that fits single-host.
- [**deepseek-ai/DeepSeek-V2**](https://huggingface.co/deepseek-ai/DeepSeek-V2) — 236B total / 21B activated; multi-host on v6e-32.

For the 671B V3 flagship see [`DeepSeek-V3.md`](DeepSeek-V3.md). For the reasoning-tuned R1 see [`DeepSeek-R1.md`](DeepSeek-R1.md).

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — `--moe-backend` choice matters (see §2.4).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V2) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | v6e-4  | 2x2 | 1 | 4  | 4  | 4  | BF16 ~32 GB — single host |
| DeepSeek-V2      | v6e-32 | 4x8 | 8 | 32 | 32 | 32 | BF16 ~470 GB |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For V2-Lite single-host use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md); for V2 multi-host use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (DeepSeek-V2-Lite)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Multi-host (SkyPilot) — TPU v6e-32 (DeepSeek-V2)

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
  --model-path deepseek-ai/DeepSeek-V2 \
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

For GKE, adapt the manifest pattern from [`MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=deepseek-v2`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8` / `completions: 8`, and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (V2-Lite).
- `--moe-backend fused` for `--ep-size ≥ 16` (V2 multi-host).

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.

**Memory Management:**
- V2-Lite: `--mem-fraction-static 0.88` (TPU default).
- V2: start at `0.92` for dedicated multi-host serving; drop to `0.9` if OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Multi-node clusters: mount a shared PVC at the cache directory to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute `model="deepseek-ai/DeepSeek-V2"` (or `DeepSeek-V2-Lite`).

> DeepSeek V2 does not ship with a native tool-call format. For tool-call workloads use [`Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling). For reasoning workloads use [`DeepSeek-R1.md`](DeepSeek-R1.md).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (V2-Lite) / v6e-32 (V2) |
| Model | deepseek-ai/DeepSeek-V2-Lite or DeepSeek-V2 (BF16) |
| Tensor Parallelism | 4 (Lite) / 32 (V2) |
| Expert Parallelism | 4 (Lite) / 32 (V2) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-deepseek-v2-lite).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model deepseek-ai/DeepSeek-V2 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, HumanEval.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.2](../Qwen/Qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the DeepSeek-V2 checkpoint, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau (V2) | Wrong `--moe-backend` for EP size | Use `--moe-backend fused` at EP ≥ 16; `epmoe` only at EP ≤ 8 (V2-Lite). |
| OOM at startup (V2) | `--mem-fraction-static 0.92` too high | Lower to 0.9. Verify `--tp-size 32` matches v6e-32 chip count. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [DeepSeek-V2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V2)
- [DeepSeek-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [`DeepSeek-V3.md`](DeepSeek-V3.md) — 671B V3 flagship.
- [`DeepSeek-R1.md`](DeepSeek-R1.md) — reasoning-tuned V3 derivative.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
