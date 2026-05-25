---
title: "Grok-1"
---

# Grok-1 on SGL-JAX

> **Starter recipe** — derived from the Grok-1 release format and SGL-JAX multi-host launch path; not yet empirically validated on TPU. Tune values for your hardware and PR back tested numbers.

## 1. Model Introduction

[**xai-org/grok-1**](https://huggingface.co/xai-org/grok-1) is xAI's first open-weight release (2024-03-17) — a **314B-parameter Mixture-of-Experts** model with 8 experts (2 active per token, ~86B active parameters). Released under the Apache 2.0 license and intended as a base model — **not fine-tuned for instruction following or chat**. SGL-JAX serves it on TPU v6e-32 (8 nodes × 4 chips) with tensor + expert parallelism.

For the chat-tuned successor see [`Grok2.md`](Grok2.md).

**Key Features**:

- **314B MoE / 86B active** — 8 experts, 2 active per token (25% active fraction); served via SGL-JAX `Grok1ForCausalLM` runtime with `FusedEPMoE` / `EPMoE` backend.
- **Apache 2.0 license** — fully open weights and code, no restrictions on derivatives.
- **Base model, not chat-tuned** — has no chat template; use the raw `/v1/completions` endpoint (see [§3.1](#31-basic-text-completion-base-model)), not `/v1/chat/completions`.

**Recommended Generation Parameters** (base-model text completion): `temperature=0.7`, `top_p=0.95`, `max_tokens=256`. For deterministic eval use `temperature=0, top_k=1`.

**Tokenizer note**: Grok-1's HF release ships a SentencePiece-format weights checkpoint (`ckpt-0/*`) without a standard HuggingFace tokenizer config. You will need to point `--tokenizer-path` to a community tokenizer that mirrors the Grok-1 vocab — verify against the model card and the [GitHub repo](https://github.com/xai-org/grok-1) for the current recommendation.

**License**: see the [HuggingFace model card](https://huggingface.co/xai-org/grok-1) for the authoritative Apache 2.0 license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips per node | Total chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| **v6e-32** (minimum, required) | 4x8 | 8 | 4 | 32 | 32 | 8 | v6e is 1:1 chip↔device; `--tp-size=32` saturates the slice; `--ep-size=8` matches the 8 experts |

Grok-1 314B requires the full v6e-32 slice — no smaller config fits at BF16 (~628 GB total). See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For multi-node launches, use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

**Weights preparation** (one-time before first launch): Grok-1 ships as a multi-shard `ckpt-0/*` checkpoint, not standard safetensors. Download via:

```bash
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include 'ckpt-0/*' \
  --local-dir /models/xai-grok-1 \
  --local-dir-use-symlinks False
```

Stage on tmpfs (`/dev/shm`) or local SSD on each serving node before launch for fast cold-start.

### 2.3 Launch

Grok-1 314B is multi-host only; cannot fit single-host.

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (8 nodes)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=grok-1`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path /models/xai-grok-1 \
  --trust-remote-code \
  --tokenizer-path <community-grok-1-tokenizer> \
  --tp-size 32 --ep-size 8 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --download-dir /dev/shm \
  --random-seed 3 \
  --skip-server-warmup
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` is the right choice at `--ep-size 8` on a 32-chip slice. The fused kernel keeps all 8 experts together per device group; use `--moe-backend epmoe` only when you want to test the non-fused reference implementation.

**Memory Management:**
- `--mem-fraction-static 0.9` is appropriate for a dedicated serving slice. At 314B BF16 weights (~628 GB) split across 32 chips, ~20 GB per chip for weights leaves room for KV cache + MoE expert activations.
- `--download-dir /dev/shm` stages weights on tmpfs for fast load (~50% faster cold start than `/tmp`). Switch back to `/tmp` if shared memory is constrained on your host.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at 314B scale. Default `1` is much slower for this model size.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill. Larger values (4096) reduce TTFT on long prompts but risk prefill-time OOM.
- `--max-running-requests 256` is the concurrent decode bound; raise for higher throughput, lower for tighter latency tails.

**Tokenizer:**
- Grok-1 weights ship without a HuggingFace-compatible tokenizer config — `--tokenizer-path` is **required**, pointing at a community tokenizer repo.
- If the chosen tokenizer requires an HF token, set `HF_TOKEN` env on every node.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters, the cache is per-node. Mount a shared PVC if you want compilation to amortize across nodes.

For full flag definitions and defaults see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Text Completion (base model)

For the standard cURL / Python `requests` / OpenAI client / native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Grok-1 is a base model with no chat template — use the raw `/v1/completions` endpoint, not `/v1/chat/completions`:

```bash
curl -X POST http://<rank0-ip>:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai-org/grok-1",
    "prompt": "The capital city of France is",
    "max_tokens": 16,
    "temperature": 0,
    "top_k": 1
  }'
```

Python OpenAI client equivalent:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.completions.create(
    model="xai-org/grok-1",
    prompt="The capital city of France is",
    max_tokens=16,
    temperature=0,
)
print(resp.choices[0].text)
```

> **Why not `/v1/chat/completions`?** Grok-1 has no chat template — sending `messages` would either fail (no template registered) or produce garbage (raw `system`/`user`/`assistant` tokens fed in without the model knowing what they mean). For chat-style usage, fine-tune Grok-1 on instruction data or use [`Grok2.md`](Grok2.md) (chat-tuned) instead.

> Grok-1 has no native reasoning or tool-calling formats. For those workloads use a model with `--reasoning-parser` / `--tool-call-parser` support (see [`MiMo-V2.5-Pro.md` §3.2 / §3.3](../Xiaomi/MiMo-V2.5-Pro.md) or [`Qwen3.md` §3.2 / §3.3](../Qwen/Qwen3.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Speed — single workload (low-concurrency latency baseline)

**Test Environment** — same as §4.2.

**Deployment Command** — same as [§2.3 Multi-host (GKE Indexed Job)](#multi-host-gke-indexed-job--tpu-v6e-32-8-nodes).

**Benchmark Command** — adapt the driver script from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `xai-org/grok-1`, remove the vLLM half).

**Test Results** — _Pending. Run and PR back the full `============ Serving Benchmark Result ============` block._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-32 (8 nodes × 4 chips) |
| Model | xai-org/grok-1 (BF16, local path `/models/xai-grok-1`) |
| Tokenizer | _community Grok-1 tokenizer_ (verify against model card) |
| Tensor Parallelism | 32 |
| Expert Parallelism | 8 |
| MoE Backend | fused |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3 Multi-host (GKE Indexed Job)](#multi-host-gke-indexed-job--tpu-v6e-32-8-nodes).

**Benchmark Command — MMLU** (base-model knowledge, completion-style):

```bash
evalscope eval \
  --model /models/xai-grok-1 \
  --api-url http://127.0.0.1:30000/v1/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets mmlu \
  --eval-batch-size 64 \
  --generation-config '{"temperature": 0, "max_tokens": 4, "top_k": 1}'
```

**Benchmark Command — HellaSwag** (commonsense, completion-style):

```bash
evalscope eval \
  --model /models/xai-grok-1 \
  --api-url http://127.0.0.1:30000/v1/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets hellaswag \
  --eval-batch-size 64 \
  --generation-config '{"temperature": 0, "max_tokens": 4, "top_k": 1}'
```

> Use `/v1/completions` (not chat) for evals — base model has no chat template. Pick completion-style datasets (MMLU, HellaSwag, BBH) rather than chat datasets (GSM8K few-shot prompt, MT-Bench).

**Test Results** — _Pending. Run the commands above and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tokenizer load fails at startup | `--tokenizer-path` missing or unreachable | Add `--tokenizer-path <community-grok-1-tokenizer>`; if HF rate-limited, set `HF_TOKEN` env on every node. |
| `/v1/chat/completions` returns empty / nonsense | No chat template on base model | Use `/v1/completions` with raw prompt instead; see [§3.1](#31-basic-text-completion-base-model). |
| Weights load fails — checkpoint format | `ckpt-0/*` not a recognized safetensors layout | Verify you downloaded with `--include 'ckpt-0/*'` per §2.2 and pointed `--model-path` at the parent directory. |
| MoE throughput plateau | Wrong `--moe-backend` | Stay on `--moe-backend fused` for v6e-32 / EP=8. `epmoe` is the reference path and is slower in practice. |
| Multi-node hang at `jax.distributed.initialize` | `--dist-init-addr` unreachable from non-rank-0 nodes | `sky status -a ${CLUSTER_NAME}` to verify rank-0 internal IP; check firewall on the chosen port. |
| OOM at startup | `--mem-fraction-static 0.9` too high for shared host | Lower to 0.85; verify `--tp-size 32 --ep-size 8` matches v6e-32 chip count (4 × 8 = 32 chips, 8 experts). |
| Slow cold start (~4 min per node) on every launch | JIT cache not persisted across launches | Mount a persistent volume at `/tmp/jit_cache` (or a shared PVC across all 8 nodes for amortized compilation). |

## Additional Resources

- [Grok-1 Model Card on HuggingFace](https://huggingface.co/xai-org/grok-1)
- [Grok-1 GitHub repo (xai-org/grok-1)](https://github.com/xai-org/grok-1) — reference Python implementation
- [xAI Open Release announcement](https://x.ai/news/grok-os)
- [`Grok2.md`](Grok2.md) — chat-tuned successor (different license, dense not MoE)
- [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) — primary multi-host launcher template.
- [`../../deployment/skypilot.md`](../../deployment/skypilot.md) — advanced v6e experiment alternative.
- [`MiMo-V2.5-Pro.md`](../Xiaomi/MiMo-V2.5-Pro.md) — GKE Indexed Job manifest reference (adapt for Grok-1).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
