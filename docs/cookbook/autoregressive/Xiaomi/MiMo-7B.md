---
title: "MiMo-7B"
---

# MiMo-7B on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**XiaomiMiMo/MiMo-7B**](https://huggingface.co/XiaomiMiMo) is Xiaomi's 7B-parameter dense decoder model trained with reasoning-oriented objectives — built on the Qwen 2 base architecture. Fits comfortably on a single TPU v6e-4 host.

**Variants** (pick by training objective):

- [**XiaomiMiMo/MiMo-7B-Base**](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base) — base pre-trained.
- [**XiaomiMiMo/MiMo-7B-SFT**](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT) — supervised fine-tuned for instruction following.
- [**XiaomiMiMo/MiMo-7B-RL**](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) — RL-tuned for reasoning; default choice for chain-of-thought workloads.

For the larger Xiaomi MoE models, see [`MiMo-V2-Flash.md`](MiMo-V2-Flash.md) and [`MiMo-V2.5-Pro.md`](MiMo-V2.5-Pro.md) — these are different architectures (256-expert MoE with hybrid attention), not just larger MiMo-7B variants.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` for RL/SFT variants (give room for reasoning).

**License**: see the [HuggingFace model card](https://huggingface.co/XiaomiMiMo) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | MiMo-7B (any variant) | v6e-4 | 2x2 | 4 | 4 | BF16 weights ~14 GB — fits with headroom; lowest-cost single-host serving |
| Recommended production | MiMo-7B (any variant) | v6e-8 | 2x4 | 8 | 8 | More HBM headroom for higher `--max-running-requests` and longer reasoning outputs on RL variant |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-7B-RL \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `MiMo-7B-Base` or `MiMo-7B-SFT` as needed.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.88` is the TPU default. Raise to `0.9` for dedicated serving / higher concurrency.

**Tool Calling:**
- MiMo-7B shares the `mimo` tool-call parser format with MiMo-V2.5-Pro. Add `--tool-call-parser mimo` when using the OpenAI tools API. See [`MiMo-V2.5-Pro.md` §3.3](MiMo-V2.5-Pro.md#33-tool-calling) for the request/response pattern.

**Reasoning (RL / SFT variants):**
- Pass `extra_body={"chat_template_kwargs": {"enable_thinking": true}}` per-request to unlock chain-of-thought outputs (verify support per checkpoint via model card).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`MiMo-V2.5-Pro.md` §3.1](MiMo-V2.5-Pro.md#31-basic-chat-completion) for the curl / Python pattern. Substitute `model="XiaomiMiMo/MiMo-7B-RL"` (or your chosen variant).

### 3.2 Reasoning / Tool Calling

MiMo-7B shares the `mimo` reasoning and tool-call parser formats with MiMo-V2.5-Pro — the full streaming Python client + multi-turn `Handling Tool Call Results` pattern in [`MiMo-V2.5-Pro.md` §3.2](MiMo-V2.5-Pro.md#32-reasoning-thinking-on-default-thinking-off-optional) and [§3.3](MiMo-V2.5-Pro.md#33-tool-calling) applies directly. Substitute the model path and the §2.3 launch flags above.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `XiaomiMiMo/MiMo-7B-RL`).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | XiaomiMiMo/MiMo-7B-RL (BF16) |
| Tensor Parallelism | 4 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4).

**Benchmark Command** — example for GSM8K (reasoning variants should add `chat_template_kwargs.enable_thinking=true` to `generation-config`):

```bash
evalscope eval \
  --model XiaomiMiMo/MiMo-7B-RL \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets for reasoning variants: AIME 2025, MATH.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser mimo` to the launch command. |
| No `reasoning_content` in response on RL/SFT variant | `--reasoning-parser` not set, or `enable_thinking` not passed | Add `--reasoning-parser mimo` to launch; pass `extra_body={"chat_template_kwargs":{"enable_thinking":true}}` per request. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [MiMo-7B family on HuggingFace](https://huggingface.co/XiaomiMiMo)
- [`MiMo-V2-Flash.md`](MiMo-V2-Flash.md) and [`MiMo-V2.5-Pro.md`](MiMo-V2.5-Pro.md) — larger Xiaomi MoE models (different architecture).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
