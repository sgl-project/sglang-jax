# DeepSeek V2 / V3 / R1 on SGL-JAX

> **Starter recipe** — derived from HuggingFace model cards; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**deepseek-ai/DeepSeek**](https://huggingface.co/deepseek-ai) is DeepSeek's family of MoE decoder models built on **MLA** (Multi-head Latent Attention). SGL-JAX serves the V2 / V3 / R1 families and the V2-Lite single-host variant through one runtime path.

**Variants** (pick by capability / size):

- [**deepseek-ai/DeepSeek-V2-Lite**](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) — 15.7B total / 2.4B activated; minimal MoE that fits single-host.
- [**deepseek-ai/DeepSeek-V2**](https://huggingface.co/deepseek-ai/DeepSeek-V2) — 236B total / 21B activated; multi-host on v6e-32.
- [**deepseek-ai/DeepSeek-V3**](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 671B total / 37B activated; large multi-host slice required.
- [**deepseek-ai/DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) — reasoning-tuned V3 derivative; default choice for chain-of-thought workloads.

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — `--moe-backend` choice matters (see §2.4).
- **DSA** (DeepSeek Sparse Attention) on V3.2 — activated by model config; no extra launch flag.

**Recommended Generation Parameters**:

- General (V2 / V3): `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.
- Reasoning (R1): `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+` (give room for thinking).

**License**: see the [DeepSeek model cards](https://huggingface.co/deepseek-ai) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite     | v6e-4  | 2x2 | 1  | 4  | 4  | 4  | BF16 ~32 GB — single host |
| DeepSeek-V2          | v6e-32 | 4x8 | 8  | 32 | 32 | 32 | BF16 ~470 GB |
| DeepSeek-V3 / R1     | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~1.3 TB |
| DeepSeek-V3 / R1     | v7x-16 | 4x4 | 4  | 16 | 32 | 32 | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../get_started/install.md). For V2-Lite single-host use [`../deployment/single-host-docker.md`](../deployment/single-host-docker.md); for V2 / V3 / R1 multi-host use [`../deployment/gke-indexed-job.md`](../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../deployment/skypilot.md). The required JAX TPU container image:

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

#### Multi-host (SkyPilot) — TPU v6e-64 (DeepSeek-V3 / R1)

Swap the topology to `tpu-v6e-64`, the model path to `deepseek-ai/DeepSeek-V3` (or `DeepSeek-R1`), and use:

```text
  --tp-size 64 --ep-size 64 \
  --nnodes 16 --node-rank \${SKYPILOT_NODE_RANK} \
```

For R1, also add `--reasoning-parser deepseek-r1` so the API splits `<think>` blocks into `reasoning_content`.

For GKE, adapt the manifest pattern from [`mimo-v2.5-pro.md` §2.3 Multi-host](mimo-v2.5-pro.md#23-launch) with `<JOB>=deepseek-v3`, `<ACCELERATOR>=tpu-v6e-slice`, the corresponding topology (`4x8` or `8x8`), and the launch flags above.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (V2-Lite).
- `--moe-backend fused` for `--ep-size ≥ 16` (V2 / V3 / R1).

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.

**Memory Management:**
- V2-Lite: `--mem-fraction-static 0.88` (TPU default).
- V2 / V3 / R1: start at `0.92` for dedicated multi-host serving; drop to `0.9` if OOM at startup.

**Reasoning (R1 only):**
- Launch with `--reasoning-parser deepseek-r1`. Without it, `<think>` content stays inline in `content` instead of being split into `reasoning_content`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Multi-node clusters: mount a shared PVC at the cache directory to amortize compilation.

For full flag definitions see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`qwen3.md` §3.1](qwen3.md#31-basic-chat-completion). Substitute `model="deepseek-ai/DeepSeek-V3"` (or your chosen variant).

### 3.2 Reasoning (R1 — thinking-enabled streaming)

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

> DeepSeek V2 / V3 do not ship with a native tool-call format. For tool-call workloads use [`qwen3.md` §3.3](qwen3.md#33-tool-calling) or [`mimo-v2.5-pro.md` §3.3](mimo-v2.5-pro.md#33-tool-calling) with a model that has built-in tool-call support.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (V2-Lite) / v6e-32 (V2) / v6e-64 (V3 / R1) |
| Model | DeepSeek-V2-Lite / V2 / V3 / R1 (BF16) |
| Tensor Parallelism | matches chip count |
| Expert Parallelism | matches chip count |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-deepseek-v2-lite).

**Benchmark Command** — example for GSM8K (R1 should pass `enable_thinking=true`):

```bash
evalscope eval \
  --model deepseek-ai/DeepSeek-V3 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, HumanEval; for R1 also AIME 2025 and MATH.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`qwen3.md` §4.2](qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the DeepSeek checkpoint, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. `epmoe` is for EP ≤ 8. |
| R1 returns no `reasoning_content` | `--reasoning-parser` not set | Add `--reasoning-parser deepseek-r1` to the launch command. |
| OOM at startup (V3 / R1) | `--mem-fraction-static 0.92` too high for this slice | Lower to 0.9. Verify `--tp-size` matches the chip count (and remember v7x exposes 2 JAX devices per chip). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [DeepSeek model collection](https://huggingface.co/deepseek-ai)
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
