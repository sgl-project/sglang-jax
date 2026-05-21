# Llama on SGL-JAX

> **Starter recipe** — derived from HuggingFace model cards; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

The Llama family is Meta's open-weight dense decoder series. SGL-JAX serves the most common Llama 3 sizes on TPU; the same runtime path also serves Phi-3 and InternLM3 checkpoints that share the Llama architecture.

**Variants** (pick by size / fine-tune):

- [**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — 8B dense; comfortable single-host fit.
- [**meta-llama/Llama-3.3-70B-Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) — 70B dense; multi-host required.
- [**microsoft/Phi-3.5-mini-instruct**](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) — Llama-compatible 3.8B Phi-3 variant.
- [**internlm/internlm3-8b-instruct**](https://huggingface.co/internlm/internlm3-8b-instruct) — Llama-compatible 8B InternLM3 variant.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.9`, `max_tokens=1024` (Llama 3 Instruct defaults).

**License**: see the [Llama model card](https://huggingface.co/meta-llama) for the authoritative Meta Llama Community License terms. Phi-3 / InternLM3 follow their own model-card licenses.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Llama 3.1 8B / Phi-3.5 / InternLM3-8B | v6e-4 | 2x2 | 4 | 4 | BF16 ~16 GB — fits with headroom |
| Llama 3.3 70B | v6e-32 | 4x8 | 32 | 32 | BF16 ~140 GB — multi-host required |

See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../get_started/install.md). For 8B use [`../deployment/single-host-docker.md`](../deployment/single-host-docker.md); for 70B use [`../deployment/gke-indexed-job.md`](../deployment/gke-indexed-job.md) or [`../deployment/skypilot.md`](../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Llama 3.1 8B)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `microsoft/Phi-3.5-mini-instruct` or `internlm/internlm3-8b-instruct` for the aliased variants.

#### Multi-host (SkyPilot) — TPU v6e-32 (Llama 3.3 70B)

**Step 1** — provision the cluster:

```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-32 main
```

**Step 2** — launch the server (one `sky exec` fans out to all 8 nodes):

```bash
CLUSTER_NAME=$(cat .cluster_name)
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tp-size 32 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --dist-init-addr <NODE_0_IP_ADDRESS>:5000 \
  --nnodes 8 --node-rank \${SKYPILOT_NODE_RANK} \
  --host 0.0.0.0 --port 30000"
```

For GKE, adapt the manifest pattern from [`mimo-v2.5-pro.md` §2.3 Multi-host](mimo-v2.5-pro.md#23-launch) with `<JOB>=llama-70b`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8` / `completions: 8`, and the launch flags above.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.88` is the TPU default for 8B. Raise to `0.9` for dedicated serving / higher concurrency.
- For 70B at TP=32, `--mem-fraction-static 0.9` leaves ~4 GB per chip for KV — lower to 0.85 if you hit OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at 70B scale. Default `1` is fine for 8B low-concurrency mixed traffic.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.
- `--max-running-requests 256` caps concurrent decodes; lower for tighter latency tails.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- On multi-node 70B clusters the cache is per-node; mount a shared PVC to amortize compilation across nodes.

For full flag definitions see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`qwen3.md` §3.1](qwen3.md#31-basic-chat-completion) for the curl / Python pattern. Substitute `model="meta-llama/Llama-3.1-8B-Instruct"` (or your chosen variant).

> Llama 3 Instruct does not ship with native hybrid reasoning or a built-in tool-call format. For reasoning / tool-call workloads use a model with `--reasoning-parser` / `--tool-call-parser` support (see [`qwen3.md` §3.2 / §3.3](qwen3.md) or [`mimo-v2.5-pro.md`](mimo-v2.5-pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (8B) / v6e-32 (70B) |
| Model | meta-llama/Llama-3.1-8B-Instruct or Llama-3.3-70B-Instruct (BF16) |
| Tensor Parallelism | 4 (8B) / 32 (70B) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-llama-31-8b).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, HumanEval, IFEval.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`qwen3.md` §4.2](qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Llama checkpoint).

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Llama 3.3 70B OOM at startup | Weights + KV exceed budget at chosen `--mem-fraction-static` | Lower to 0.85; verify `--tp-size 32` matches v6e-32 chip count (4 × 8 = 32). |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |
| Phi-3 / InternLM3 fails to load | Missing `--trust-remote-code` | Add it to the launch command; both aliases ship custom modeling code. |
| Multi-node 70B hangs at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |

## Additional Resources

- [Llama model collection](https://huggingface.co/meta-llama)
- [Phi-3 model card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [InternLM3 model card](https://huggingface.co/internlm/internlm3-8b-instruct)
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
