---
title: "Kimi-Linear"
---

# Kimi-Linear on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**moonshotai/Kimi-Linear**](https://huggingface.co/moonshotai) is Moonshot AI's linear-attention decoder series built on **Kimi Delta Attention** with a hybrid recurrent state pool. The currently released checkpoint is a 48B MoE with 3B activated parameters.

**Variants**:

- [**moonshotai/Kimi-Linear-48B-A3B-Instruct**](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) — 48B total / 3B activated, instruction-tuned.

For other linear-attention models in the cookbook see [`Ling-2.6.md`](../InclusionAI/Ling-2.6.md) (InclusionAI's trillion-scale linear-attention MoE).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024` (Kimi defaults — verify against the model card).

**License**: see the [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|---|
| Minimum runnable | Kimi-Linear-48B-A3B | v6e-16 | 4x4 | 4 | 16 | 16 | BF16 ~96 GB — multi-host required to fit weights + recurrent state pool |
| Recommended production | Kimi-Linear-48B-A3B | v6e-32 | 4x8 | 8 | 32 | 32 | More HBM per active expert and larger recurrent state budget for long-prompt linear-attention workloads |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). Multi-host recommended at this size — use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (SkyPilot) — TPU v6e-16

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
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --tp-size 16 \
  --recurrent-state-memory-ratio 0.9 \
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

For GKE, adapt the manifest pattern from [`MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=kimi-linear`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4` / `completions: 4`, and the launch flags above.

### 2.4 Configuration Tips

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.

**Memory Management:**
- `--mem-fraction-static 0.9` for dedicated multi-host serving. Drop to `0.88` (TPU default) if you hit OOM at startup with high `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill — Kimi-Linear's linear-attention benefit is most visible on long prompts.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's 4 nodes to amortize compilation.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute `model="moonshotai/Kimi-Linear-48B-A3B-Instruct"`.

> Kimi-Linear-Instruct does not ship with a built-in tool-call format. For tool-call workloads use a model with `--tool-call-parser` support (see [`Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `moonshotai/Kimi-Linear-48B-A3B-Instruct`, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | moonshotai/Kimi-Linear-48B-A3B-Instruct (BF16) |
| Tensor Parallelism | 16 |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-skypilot--tpu-v6e-16).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, RULER (to exercise long-context linear-attention).

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Recurrent state + KV exceed budget | Lower `--recurrent-state-memory-ratio` (e.g. to 0.7) and/or `--mem-fraction-static` to 0.88. |
| Long-prompt requests stall | KV cache exhausted before recurrent state | Lower `--recurrent-state-memory-ratio` to give the KV cache more headroom. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [`Ling-2.6.md`](../InclusionAI/Ling-2.6.md) — InclusionAI's trillion-scale linear-attention MoE.
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
