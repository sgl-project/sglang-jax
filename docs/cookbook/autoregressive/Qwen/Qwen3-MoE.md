---
title: "Qwen3-MoE"
description: "Qwen3-30B-A3B and Qwen3-235B-A22B MoE variants serving on TPU v6e-16 / v6e-64 with SGL-JAX."
---

# Qwen3-MoE on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**Qwen/Qwen3-MoE**](https://huggingface.co/Qwen) is Alibaba's MoE variant of the Qwen3 series — sparse mixture-of-experts decoders with the same hybrid reasoning + tool-call format as dense Qwen3. Two released sizes: 30B-A3B (3B active) and 235B-A22B (22B active).

**Variants** (pick by size):

- [**Qwen/Qwen3-30B-A3B**](https://huggingface.co/Qwen/Qwen3-30B-A3B) — 30B total / 3B activated; multi-host on v6e-16.
- [**Qwen/Qwen3-235B-A22B**](https://huggingface.co/Qwen/Qwen3-235B-A22B) — 235B total / 22B activated; multi-host on v6e-64.

For the dense Qwen3 variants (8B / 32B) see [`Qwen3.md`](Qwen3.md).

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+`.
- Thinking-off (instant): `temperature=0.7`, `top_p=0.8`, `max_tokens=512`.

**License**: see the [Qwen model cards](https://huggingface.co/Qwen) for authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B   | v6e-16 | 4x4 | 4  | 16 | 16 | 16 | BF16 ~60 GB  |
| Qwen3-235B-A22B | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~470 GB |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). For multi-host launches use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

Qwen3-MoE is multi-host only at both released sizes.

#### Multi-host (SkyPilot) — TPU v6e-16 (Qwen3-30B-A3B)

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
  --model-path Qwen/Qwen3-30B-A3B \
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

#### Multi-host (SkyPilot) — TPU v6e-64 (Qwen3-235B-A22B)

Swap the topology to `tpu-v6e-64`, the model path to `Qwen/Qwen3-235B-A22B`, and use:

```text
  --tp-size 64 --ep-size 64 \
  --mem-fraction-static 0.92 \
  --nnodes 16 --node-rank \${SKYPILOT_NODE_RANK} \
```

For GKE, adapt the manifest pattern from [`MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=qwen3-moe`, `<ACCELERATOR>=tpu-v6e-slice`, the corresponding topology (`4x4` or `8x8`), and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` is the recommended choice at EP ≥ 16 (both 30B and 235B configs above). Switch to `epmoe` only when EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.9` is appropriate for the 30B-A3B config; raise to `0.92` for 235B-A22B to make room for the much larger weight set.
- Lower by 0.02 increments if you hit OOM at startup with high `--max-running-requests`.

**Hybrid Reasoning / Tool Calling:**
- Qwen3-MoE shares the hybrid reasoning (`--reasoning-parser qwen3`) and tool-call (`--tool-call-parser qwen25`) format with dense Qwen3. Append these flags to the launch command — full streaming Python examples in [`Qwen3.md` §3.2](Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) and [§3.3](Qwen3.md#33-tool-calling) apply directly.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale. Default `1` is much slower at high concurrency.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters the cache is per-node. Mount a shared PVC to amortize compilation across all nodes.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](Qwen3.md#31-basic-chat-completion). Substitute `model="Qwen/Qwen3-30B-A3B"` (or `Qwen3-235B-A22B`).

### 3.2 Reasoning / Tool Calling

Qwen3-MoE inherits the dense Qwen3 hybrid-reasoning and tool-call format — the full streaming Python clients and multi-turn `Handling Tool Call Results` pattern in [`Qwen3.md` §3.2](Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) and [§3.3](Qwen3.md#33-tool-calling) apply directly. Substitute the model path and the §2.3 launch flags above.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (30B-A3B) / v6e-64 (235B-A22B) |
| Model | Qwen/Qwen3-30B-A3B or Qwen3-235B-A22B (BF16) |
| Tensor Parallelism | 16 / 64 |
| Expert Parallelism | 16 / 64 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-skypilot--tpu-v6e-16-qwen3-30b-a3b).

**Benchmark Command** — example for GSM8K (with thinking-on for reasoning):

```bash
evalscope eval \
  --model Qwen/Qwen3-30B-A3B \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --generation-config '{"chat_template_kwargs": {"enable_thinking": true}, "temperature": 0.7, "top_p": 0.95}'
```

Recommended additional datasets: AIME 2025, MATH, GPQA Diamond.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.2](Qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Qwen3-MoE checkpoint, remove the vLLM half if not comparing).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. `epmoe` is for EP ≤ 8. |
| OOM at startup (235B-A22B) | `--mem-fraction-static` too high | Lower to 0.9. Verify `--tp-size 64` matches v6e-64 chip count (8 × 8 = 64). |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser qwen25`. |
| No `reasoning_content` in response | `--reasoning-parser` not set | Add `--reasoning-parser qwen3`. |
| Multi-node hang at init | `--dist-init-addr` unreachable | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [Qwen3 model collection](https://huggingface.co/Qwen)
- [`Qwen3.md`](Qwen3.md) — dense Qwen3-8B / 32B recipe (same reasoning/tool-call format).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
