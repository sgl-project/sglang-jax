---
title: "Ring 2.0"
---

# Ring 2.0 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**inclusionAI/Ring-2.0**](https://huggingface.co/inclusionAI) is InclusionAI's reasoning-tuned MoE series — built on the same Ling 2.0 architecture (1/32 sparsity MoE + MTP layers) but post-trained with the `icepop` RL algorithm to address training instability after cold-start long-CoT SFT. Ring shares the SGL-JAX runtime path with Ling 2.0; the difference is checkpoint + reasoning parser at launch.

**Variants** (pick by size):

- [**inclusionAI/Ring-mini-2.0**](https://huggingface.co/inclusionAI/Ring-mini-2.0) — 16B / 1.4B; 128K long context, 300+ tok/s generation, reasoning-comparable to dense models under 10B.
- [**inclusionAI/Ring-flash-2.0**](https://huggingface.co/inclusionAI/Ring-flash-2.0) — 100B / 6.1B; high-throughput reasoning derived from Ling-flash-2.0-base.
- [**inclusionAI/Ring-1T-preview**](https://huggingface.co/inclusionAI/Ring-1T-preview) — 1T total parameters; flagship reasoning preview, RLVR-trained on 20T tokens.

For the non-reasoning Ling 2.0 base family see [`Ling-2.md`](Ling-2.md). For the Ling 1.x first-generation family see [`Ling-1.x.md`](Ling-1.x.md). For the linear-attention 2.6 generation (both Ling and Ring linear variants) see [`Ling-2.6.md`](Ling-2.6.md).

**Recommended Generation Parameters** (reasoning workloads): `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+`. Give the model room to think — Ring's value is in the chain-of-thought.

**License**: see each model card on the [InclusionAI HF collection](https://huggingface.co/inclusionAI) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Ring-mini-2.0 (16B / 1.4B)   | v6e-4  | 2x2 | 1  | 4  | 4  | 4  | BF16 ~32 GB — single host |
| Ring-flash-2.0 (100B / 6.1B) | v6e-16 | 4x4 | 4  | 16 | 16 | 16 | BF16 ~200 GB |
| Ring-1T-preview (1T)         | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~2 TB — multi-host mandatory |
| Ring-1T-preview (1T)         | v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For Ring-mini-2.0 single-host use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md); for Ring-flash-2.0 / Ring-1T-preview multi-host use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Ring-mini-2.0)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path inclusionAI/Ring-mini-2.0 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
  --tp-size 4 --ep-size 4 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

> **`--reasoning-parser deepseek-r1`**: Ring 2.0 emits `<think>...</think>` blocks the same way DeepSeek-R1 does — no `ring` parser key is registered, so reuse `deepseek-r1` (the generic `<think>` block parser). Without this flag, thinking content stays inline in `content` instead of being split into `reasoning_content`.

#### Multi-host (SkyPilot) — TPU v6e-16 (Ring-flash-2.0)

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
  --model-path inclusionAI/Ring-flash-2.0 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
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

#### Multi-host (SkyPilot) — TPU v6e-64 (Ring-1T-preview)

Swap the topology to `tpu-v6e-64`, the model path to `inclusionAI/Ring-1T-preview`, and use:

```text
  --tp-size 64 --ep-size 64 \
  --mem-fraction-static 0.92 \
  --nnodes 16 --node-rank \${SKYPILOT_NODE_RANK} \
```

For GKE, adapt the manifest pattern from [`../Xiaomi/MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=ring-2-0`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), the corresponding topology, and the launch flags above.

### 2.4 Configuration Tips

**Reasoning Parser:**
- Ring 2.0 is reasoning-tuned and emits `<think>...</think>` blocks. Launch with `--reasoning-parser deepseek-r1` (the generic `<think>` parser — no Ring-specific key is registered). Without it, thinking content stays inline in `content` instead of being split into `reasoning_content`.
- The streaming Python client pattern from [`../Qwen/Qwen3.md` §3.2](../Qwen/Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) applies directly once the parser is set.

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (Ring-mini-2.0).
- `--moe-backend fused` for `--ep-size ≥ 16` (Ring-flash-2.0 / Ring-1T-preview).

**Memory Management:**
- Ring-mini-2.0: `--mem-fraction-static 0.88` (TPU default). Reasoning workloads tend to generate longer outputs — keep some KV cache headroom rather than maxing out.
- Ring-flash-2.0: `--mem-fraction-static 0.9` for dedicated multi-host serving.
- Ring-1T-preview: `--mem-fraction-static 0.92` for the full v6e-64 slice. Drop to `0.9` if OOM at startup.

**Throughput vs Latency (reasoning trade-off):**
- `--page-size 128` reduces KV page-table overhead, important when reasoning outputs grow long.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.
- `--max-running-requests 256` is a starter cap; reasoning workloads usually run with fewer concurrent decodes than chat because per-request token budgets are larger.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters, mount a shared PVC across nodes to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`../Qwen/Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute `model="inclusionAI/Ring-mini-2.0"` (or your chosen variant).

### 3.2 Reasoning (thinking-on streaming)

Once you launch with `--reasoning-parser deepseek-r1` (see §2.3 — Ring 2.0 emits `<think>...</think>` blocks; no `ring` parser key is registered, so reuse the generic `deepseek-r1` parser), use the streaming Python client from [`../Qwen/Qwen3.md` §3.2](../Qwen/Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) to separate `reasoning_content` from `content`. The pattern applies directly — only the model path changes.

> Ring 2.0 does not ship with a native tool-call format. For tool-call workloads see [`../Qwen/Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`../Xiaomi/MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** Cells below are command templates only; no measured numbers yet. PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`../Qwen/Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Ring checkpoint, raise `--random-output` to 2048+ to reflect reasoning token budgets, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (mini) / v6e-16 (flash) / v6e-64 (1T-preview) |
| Model | inclusionAI/Ring-mini-2.0 / Ring-flash-2.0 / Ring-1T-preview (BF16) |
| Tensor Parallelism | 4 / 16 / 64 |
| Expert Parallelism | 4 / 16 / 64 |
| Reasoning Parser | `deepseek-r1` (generic `<think>` parser; no `ring` key registered) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-ring-mini-20).

**Benchmark Command** — reasoning-heavy datasets (pass `enable_thinking=true` if the model card requires per-request opt-in):

```bash
evalscope eval \
  --model inclusionAI/Ring-mini-2.0 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime_2025 \
  --eval-batch-size 4 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}'
```

Recommended primary datasets: **AIME 2025**, **GPQA Diamond**, **LiveCodeBench**, **MATH** — these are where Ring's reasoning advantage shows up vs the Ling 2.0 base.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Response contains raw `<think>` text instead of `reasoning_content` | `--reasoning-parser` not set | Add `--reasoning-parser deepseek-r1` (the generic `<think>` parser; no `ring` key is registered today). |
| MoE throughput plateau (flash / 1T-preview) | Wrong `--moe-backend` for EP size | Use `--moe-backend fused` at EP ≥ 16; `epmoe` only at EP ≤ 8. |
| Decode tail latency spikes at high concurrency | Reasoning outputs exceed KV budget at chosen `--max-running-requests` | Lower `--max-running-requests` to 128 or 64; reasoning workloads need fewer in-flight requests than chat. |
| OOM at startup (1T-preview) | `--mem-fraction-static 0.92` too high | Lower to 0.9. Verify `--tp-size` matches chip count (v6e-64 → 64; v7x-16 → 32). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [InclusionAI HF collection](https://huggingface.co/inclusionAI)
- [Ring-V2 GitHub repo](https://github.com/inclusionAI/Ring-V2)
- [`Ling-2.md`](Ling-2.md) — non-reasoning Ling 2.0 base family (same architecture).
- [`Ling-1.x.md`](Ling-1.x.md) — first-generation Ling.
- [`Ling-2.6.md`](Ling-2.6.md) — Ling 2.6 (linear / delta attention; the linear-attention Ring variants share the same runtime path).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
