---
title: "GLM-5"
---

# GLM-5 MoE on SGL-JAX

> **Placeholder recipe (📝 planned)** — GLM-5 is supported in the runtime (`Glm5ForCausalLM` / `GlmMoeDsaForCausalLM` are registered), but the public model release pins are not yet known. Hardware Matrix and benchmark cells are placeholders. Once Zhipu publishes the checkpoint, fill in the model path, sizes, and TPU sizing from the model card, then promote to 🚧 Starter.

## 1. Model Introduction

[**zai-org/GLM-5**](https://huggingface.co/zai-org) is Zhipu AI's GLM-5 series — successor to GLM-4.5 — adding **DSA** (DeepSeek Sparse Attention) on selected sizes for cheaper long-context serving. Tool calling uses the newer `glm47` parser format.

**Variants**:

- **GLM-5** — dense-attention MoE; pick this if you don't need long-context savings.
- **GLM-5 DSA** — DeepSeek Sparse Attention variant; same launch path, selected automatically by the model config when the DSA checkpoint is loaded.

For the GLM-4.5 family (without DSA, `glm45` tool-call format) see [`GLM-4.5.md`](GLM-4.5.md).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024` (general); raise `max_tokens` for reasoning workloads.

**License**: see the [GLM-5 model card](https://huggingface.co/zai-org) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| GLM-5 (dense attn)  | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | Confirm against HF card when public |
| GLM-5 DSA            | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | DSA cuts long-context HBM via sparse attention |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host likely required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

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

Fill in `<HF_REPO>`, `<N>` (= chip count), and `<NODES>` from the hardware matrix once the release pins them. For SkyPilot, wrap this in `sky exec ${CLUSTER_NAME} -- "..."` per the [`Grok2.md` §2.3 SkyPilot pattern](../Grok/Grok2.md#multi-host-skypilot-recommended--tpu-v6e-32-8-nodes).

For GKE, adapt the manifest pattern from [`MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=glm-5`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), and the launch flags above.

### 2.4 Configuration Tips

**DSA Variant (long-context):**
- No extra flag is required — the DSA path is selected by the model config when you load the DSA checkpoint.
- DSA's benefit is most pronounced at long context (≥ 32K tokens). Raise `--context-length` to the workload's actual upper bound to make the savings count; default keeps the model config value.

**MoE Backend:**
- `--moe-backend fused` for `--ep-size ≥ 16`; `--moe-backend epmoe` for `≤ 8`.

**Tool Calling (GLM-5 / GLM-4.7 format):**
- Add `--tool-call-parser glm47` to the launch command. The streaming Python tool-call client from [`Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) applies directly — only the server-side parser name differs from Qwen3.
- For the GLM-4.5 / 4.6 family use `glm45` instead (see [`GLM-4.5.md`](GLM-4.5.md)).

**Reasoning (if GLM-5 emits `<think>` blocks):**
- The reasoning parser key for GLM-5 is not separately registered; if the public release retains the GLM-4.5 reasoning format, `--reasoning-parser glm45` should apply. Verify against the model card on release.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if you hit OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute the GLM-5 model path.

### 3.2 Tool Calling

Launch with `--tool-call-parser glm47` and reuse the streaming Python client from [`Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) — only the server-side parser name changes.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; hardware sizing pending GLM-5 public release.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the GLM-5 checkpoint, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | _Pending_ |
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

For DSA variants, add a long-context dataset (e.g. RULER 32K / 128K) to exercise the sparse-attention path.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tool calls return empty arguments | `--tool-call-parser` not set or wrong key | Add `--tool-call-parser glm47` (not `glm45`, which is GLM-4.5 / 4.6). |
| DSA variant gives no long-context savings | Loaded the dense-attention checkpoint | Verify the model path points to a DSA checkpoint and `--context-length` is raised. |
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [GLM-5 / GLM-4.5 model collection](https://huggingface.co/zai-org)
- [`GLM-4.5.md`](GLM-4.5.md) — predecessor GLM-4.5 family (uses `glm45` parsers).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
