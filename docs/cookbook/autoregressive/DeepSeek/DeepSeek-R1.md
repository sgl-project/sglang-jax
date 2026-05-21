---
title: "DeepSeek R1"
---

# DeepSeek R1 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) is DeepSeek's reasoning-tuned derivative of V3 — RL-trained on long chain-of-thought for math, code, and graduate-level reasoning. Emits `<think>` blocks that SGL-JAX exposes as `reasoning_content` via the `deepseek-r1` parser. Multi-host serving required at BF16.

For the V3 non-reasoning base see [`DeepSeek-V3.md`](DeepSeek-V3.md). For the V2 generation see [`DeepSeek-V2.md`](DeepSeek-V2.md).

**Architectural notes**:

- Same MoE + MLA backbone as V3 (`Glm5ForCausalLM` / `DeepseekV3ForCausalLM` runtime path).
- Reasoning surface needs `--reasoning-parser deepseek-r1` at launch — see §2.4.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+` (give room for thinking).

**License**: see the [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|
| v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~1.3 TB — full v6e-64 slice |
| v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (SkyPilot) — TPU v6e-64

**Step 1** — provision the cluster:

```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-64 main
```

**Step 2** — launch the server (note `--reasoning-parser deepseek-r1`):

```bash
CLUSTER_NAME=$(cat .cluster_name)
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
  --tp-size 64 --ep-size 64 \
  --moe-backend fused \
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

#### Multi-host (SkyPilot) — TPU v7x-16

Swap the topology to `tpu-v7x-16` and use:

```text
  --tp-size 32 --ep-size 32 \
  --nnodes 4 --node-rank \${SKYPILOT_NODE_RANK} \
```

For GKE, adapt the manifest pattern from [`MiMo-V2.5-Pro.md` §2.3 Multi-host](../Xiaomi/MiMo-V2.5-Pro.md#23-launch) with `<JOB>=deepseek-r1`, `<ACCELERATOR>=tpu-v6e-slice` (or `tpu7x` for v7x), the corresponding topology, and the launch flags above (keep `--reasoning-parser deepseek-r1`).

### 2.4 Configuration Tips

**Reasoning Parser:**
- `--reasoning-parser deepseek-r1` is **required** for R1 — without it, `<think>` content stays inline in `content` instead of being split into `reasoning_content`. See [§3.2](#32-reasoning-thinking-enabled-streaming) for the streaming pattern.

**MoE Backend:**
- `--moe-backend fused` for R1 (EP ≥ 32). `epmoe` is only for EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if OOM at startup.
- Reasoning workloads generate longer outputs — keep some KV cache headroom rather than maxing out.

**Throughput vs Latency (reasoning trade-off):**
- `--page-size 128` reduces KV page-table overhead, important when reasoning outputs grow long.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.
- `--max-running-requests 256` is a starter cap; reasoning workloads usually run with fewer concurrent decodes than chat because per-request token budgets are larger.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC at the cache directory to amortize compilation across all 16 nodes.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`Qwen3.md` §3.1](../Qwen/Qwen3.md#31-basic-chat-completion). Substitute `model="deepseek-ai/DeepSeek-R1"`.

### 3.2 Reasoning (thinking-enabled streaming)

R1 emits a thinking block before the final answer. Launch with `--reasoning-parser deepseek-r1` (see §2.4), then stream both `reasoning_content` and `content` deltas:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Solve step by step: What is 15% of 240?"}],
    temperature=0.6,
    max_tokens=4096,
    stream=True,
)

thinking_started = False
content_started = False
for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not thinking_started:
            print("=============== Thinking =================", flush=True)
            thinking_started = True
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        if thinking_started and not content_started:
            print("\n=============== Content =================", flush=True)
            content_started = True
        print(delta.content, end="", flush=True)
print()
```

**Output Example** (shape; actual reasoning trace will vary):

```text
=============== Thinking =================
To find 15% of 240, convert 15% to 0.15 and multiply: 0.15 × 240 = 36.
=============== Content =================

15% of 240 is **36**.
```

For non-streaming requests, the field appears on `response.choices[0].message.reasoning_content`.

> R1 does not ship with a native tool-call format. For tool-call workloads use [`Qwen3.md` §3.3](../Qwen/Qwen3.md#33-tool-calling) or [`MiMo-V2.5-Pro.md` §3.3](../Xiaomi/MiMo-V2.5-Pro.md#33-tool-calling) with a model that has built-in tool-call support.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `deepseek-ai/DeepSeek-R1`, raise `--random-output` to 2048+ to reflect reasoning token budgets, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) or v7x-16 |
| Model | deepseek-ai/DeepSeek-R1 (BF16) |
| Tensor Parallelism | 64 (v6e) / 32 (v7x) |
| Expert Parallelism | 64 (v6e) / 32 (v7x) |
| Reasoning Parser | deepseek-r1 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-skypilot--tpu-v6e-64).

**Benchmark Command** — reasoning-heavy datasets:

```bash
evalscope eval \
  --model deepseek-ai/DeepSeek-R1 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime_2025 \
  --eval-batch-size 4 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}'
```

Recommended primary datasets: **AIME 2025**, **MATH**, **GPQA Diamond**, **LiveCodeBench** — these are where R1's reasoning advantage shows up vs V3.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Response contains raw `<think>` text instead of `reasoning_content` | `--reasoning-parser` not set | Add `--reasoning-parser deepseek-r1` to the launch command. |
| MoE throughput plateau | Wrong `--moe-backend` | Use `--moe-backend fused` at R1 scale. `epmoe` is only for EP ≤ 8. |
| Decode tail latency spikes at high concurrency | Reasoning outputs exceed KV budget | Lower `--max-running-requests` to 128 or 64; reasoning workloads need fewer in-flight requests than chat. |
| OOM at startup | `--mem-fraction-static 0.92` too high | Lower to 0.9. Verify `--tp-size` matches chip count (v6e-64 → 64; v7x-16 → 32). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across all 16 nodes for amortized compilation. |

## Additional Resources

- [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [`DeepSeek-V3.md`](DeepSeek-V3.md) — V3 non-reasoning base.
- [`DeepSeek-V2.md`](DeepSeek-V2.md) — V2 / V2-Lite generation.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
