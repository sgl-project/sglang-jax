# Bailing MoE Linear on SGL-JAX

> **Starter recipe** — derived from upstream model cards; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.
>
> **Architecture-family view.** This recipe documents the **linear-attention** Bailing MoE variant. The most prominent model using it today is Ling-2.6-1T — for the model-card view of that specific deployment see [`ling-2.6.md`](ling-2.6.md). For the dense-attention Bailing MoE family see [`bailing-moe.md`](bailing-moe.md).

## 1. Model Introduction

The linear-attention Bailing MoE family is InclusionAI's hybrid recurrent-state MoE lineage — adds **linear / delta attention** in place of standard softmax attention and a **hybrid recurrent state** that shares HBM with the KV cache. Two distinguishers vs the dense-attention family:

- **Linear / delta attention**: drops the quadratic softmax for a recurrent-style aggregation.
- **Hybrid recurrent state pool**: budgeted against the KV cache via `--recurrent-state-memory-ratio` (default `0.9`).

**Common upstream models**:

- [**inclusionAI Ling 2.5 / 2.6 series**](https://huggingface.co/inclusionAI) — including the trillion-scale Ling-2.6-1T.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the upstream model card on the [InclusionAI HF collection](https://huggingface.co/inclusionAI).

## 2. Deployment

### 2.1 Hardware Matrix (starter target — Ling-2.6-1T)

The table below covers the most widely deployed instance. Smaller Bailing-Linear variants (e.g. `Ling-2.6-flash`) fit on smaller slices — adapt the launch command after picking a checkpoint from the [InclusionAI HF collection](https://huggingface.co/inclusionAI).

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Ling-2.6-1T | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | Trillion-scale; multi-host mandatory |
| Ling-2.6-1T | v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../get_started/install.md). Multi-host required — use [`../deployment/gke-indexed-job.md`](../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (SkyPilot) — TPU v6e-64 (Ling-2.6-1T)

**Step 1** — provision the cluster:

```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-64 main
```

**Step 2** — launch the server:

```bash
CLUSTER_NAME=$(cat .cluster_name)
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-2.6-1T \
  --trust-remote-code \
  --tp-size 64 --ep-size 64 \
  --moe-backend fused \
  --recurrent-state-memory-ratio 0.9 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --dist-init-addr <NODE_0_IP_ADDRESS>:5000 \
  --nnodes 16 --node-rank \${SKYPILOT_NODE_RANK} \
  --host 0.0.0.0 --port 30000"
```

#### Multi-host (SkyPilot) — TPU v7x-16 (Ling-2.6-1T)

Swap the topology to `tpu-v7x-16` and use:

```text
  --tp-size 32 --ep-size 32 \
  --nnodes 4 --node-rank \${SKYPILOT_NODE_RANK} \
```

For GKE, adapt the manifest pattern from [`mimo-v2.5-pro.md` §2.3 Multi-host](mimo-v2.5-pro.md#23-launch) with `<JOB>=ling-2-6`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), and the launch flags above.

### 2.4 Configuration Tips

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.

**MoE Backend:**
- `--moe-backend fused` for `--ep-size ≥ 16` (both configs above). Switch to `epmoe` only at EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if you hit OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at trillion-scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's 16 nodes to amortize compilation.

For full flag definitions see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`qwen3.md` §3.1](qwen3.md#31-basic-chat-completion). Substitute `model="inclusionAI/Ling-2.6-1T"` (or your chosen Bailing-Linear checkpoint).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) or v7x-16 |
| Model | inclusionAI/Ling-2.6-1T (BF16) |
| Tensor Parallelism | 64 (v6e) / 32 (v7x) |
| Expert Parallelism | 64 (v6e) / 32 (v7x) |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-skypilot--tpu-v6e-64-ling-26-1t).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model inclusionAI/Ling-2.6-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, RULER (to exercise long-context linear-attention).

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`qwen3.md` §4.2](qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Ling-2.6-1T checkpoint, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Recurrent state + KV exceed budget | Lower `--recurrent-state-memory-ratio` (e.g. to 0.7) and/or `--mem-fraction-static` to 0.9. |
| Long-prompt requests stall | KV cache exhausted before recurrent state | Lower `--recurrent-state-memory-ratio` to give the KV cache more headroom. |
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [InclusionAI HF collection](https://huggingface.co/inclusionAI)
- [`bailing-moe.md`](bailing-moe.md) — dense-attention Bailing MoE family.
- [`ling-2.6.md`](ling-2.6.md) — model-card view for Ling-2.6.
- [`kimi-linear.md`](kimi-linear.md) — another linear-attention model in the cookbook.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
