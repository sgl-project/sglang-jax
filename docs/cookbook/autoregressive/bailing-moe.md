# Bailing MoE on SGL-JAX

> **Starter recipe** — derived from upstream model cards; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.
>
> **Architecture-family view.** This recipe documents the Bailing MoE architecture (non-linear attention) shared across InclusionAI's Ling V2 and Ring families. For the **linear-attention** variant used by Ling-2.6 see [`bailing-moe-linear.md`](bailing-moe-linear.md); for a specific Ling-2.6 deployment see [`ling-2.6.md`](ling-2.6.md).

## 1. Model Introduction

The Bailing MoE architecture family (non-linear attention) is InclusionAI's MoE decoder lineage shared across multiple upstream model families:

- [**InclusionAI Ling V2**](https://huggingface.co/inclusionAI) — V2 generation, before the linear-attention variant.
- [**InclusionAI Ring**](https://huggingface.co/inclusionAI) — reasoning-focused sibling line.

The runtime path is the same for both families — only the model checkpoint and (occasionally) parser flags change. Pick a specific checkpoint from the [InclusionAI HF collection](https://huggingface.co/inclusionAI).

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024`. Use longer `max_tokens` for Ring (reasoning-tuned) checkpoints.

**License**: see each upstream model card on the [InclusionAI HF collection](https://huggingface.co/inclusionAI) for authoritative license terms — they vary per release.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

Bailing MoE checkpoints span a wide size range. Choose the slice that fits the activated-parameter count of your specific checkpoint:

| Activated params | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` |
|---|---|---|---|---|---|---|
| ~30B  | v6e-16 | 4x4 | 4  | 16 | 16 | 16 |
| ~200B+ | v6e-64 | 8x8 | 16 | 64 | 64 | 64 |
| ~200B+ | v7x-16 | 4x4 | 4  | 16 | 32 | 32 (v7x exposes 2 JAX devices per chip) |

See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../get_started/install.md). Multi-host required at typical sizes — use [`../deployment/gke-indexed-job.md`](../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (template)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path <HF_REPO> \
  --trust-remote-code \
  --tp-size <N> --ep-size <N> \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --dist-init-addr <NODE_0_IP_ADDRESS>:5000 \
  --nnodes <NODES> --node-rank ${NODE_RANK} \
  --host 0.0.0.0 --port 30000
```

Fill in `<HF_REPO>`, `<N>` (= chip count), and `<NODES>` from §2.1. For SkyPilot, wrap this in `sky exec ${CLUSTER_NAME} -- "..."` per the [`grok2.md` §2.3 SkyPilot pattern](grok2.md#multi-host-skypilot-recommended--tpu-v6e-32-8-nodes).

For GKE, adapt the manifest pattern from [`mimo-v2.5-pro.md` §2.3 Multi-host](mimo-v2.5-pro.md#23-launch) with `<JOB>=bailing-moe`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` for `--ep-size ≥ 16`. Switch to `epmoe` only at EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` (or 0.88) if you hit OOM at startup with the larger checkpoints.

**Reasoning / Tool Calling:**
- The Bailing base architecture does not ship with a built-in reasoning or tool-call parser. For Ring (reasoning-tuned) checkpoints, check the upstream model card to see if its `<think>` format matches an existing parser key (e.g. `deepseek-r1`); if so, add `--reasoning-parser <key>` to the launch command.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across nodes to amortize compilation.

For full flag definitions see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`qwen3.md` §3.1](qwen3.md#31-basic-chat-completion). Substitute the chosen Ling / Ring model path.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | _Pending_ (matches §2.1 row) |
| Model | _Pending_ (BF16) |
| Tensor Parallelism | _Pending_ |
| Expert Parallelism | _Pending_ |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-template).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model <HF_REPO> \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond; for Ring checkpoints also AIME 2025 and MATH.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`qwen3.md` §4.2](qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Bailing checkpoint, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. `epmoe` is for EP ≤ 8. |
| OOM at startup on large checkpoints | `--mem-fraction-static 0.92` too high for this slice | Lower to 0.9 or 0.88; verify `--tp-size` matches the chip count (and that v7x is counted as 2 JAX devices per chip). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [InclusionAI HF collection](https://huggingface.co/inclusionAI)
- [`bailing-moe-linear.md`](bailing-moe-linear.md) — linear-attention variant used by Ling-2.6.
- [`ling-2.6.md`](ling-2.6.md) — model-card view for the Ling-2.6 deployment.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
