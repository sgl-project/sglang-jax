# Ling-2.6 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.
>
> **Model-card view.** Concrete deployment guide for the Ling-2.6 checkpoint, which uses the **Bailing MoE Linear** architecture. For the architecture-family view (flag semantics, recurrent-state notes, alternative checkpoints) see [`bailing-moe-linear.md`](bailing-moe-linear.md).

## 1. Model Introduction

[**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) is the 1T-parameter Ling 2.6 release — InclusionAI's trillion-scale MoE built on linear / delta attention with a hybrid recurrent state. Smaller siblings (e.g. `Ling-2.6-flash`) are released under the same [InclusionAI HF collection](https://huggingface.co/inclusionAI).

**Variants**:

- [**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) — full trillion-scale flagship; default focus of this page.
- Smaller Ling-2.6 variants — adapt the §2.3 launch command after picking a checkpoint.

**Recommended Generation Parameters**: see the Ling-2.6 model card for authoritative defaults. As a starter: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` (give room if you enable reasoning mode).

**License**: see the [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

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

Use the launch commands in [`bailing-moe-linear.md` §2.3](bailing-moe-linear.md#23-launch) with `--model-path inclusionAI/Ling-2.6-1T`. Flag semantics (recurrent state, MoE backend, memory tuning) are owned by that page — this page intentionally does not duplicate them.

### 2.4 Configuration Tips (Ling-2.6 specifics)

**Reasoning Mode:**
- If the Ling-2.6 checkpoint supports `<think>` blocks, add `--reasoning-parser <key>` to the launch command — run `python -m sgl_jax.launch_server --help` to see registered parser keys. The streaming Python client from [`qwen3.md` §3.2](qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) applies directly once the parser is set.

**Context Length:**
- Pin via `--context-length` to your workload's longest prompt + output. Smaller values reduce KV cache footprint at trillion-scale.

**All other tuning** (recurrent state ratio, MoE backend, memory fraction, page size, chunked prefill, compilation cache) — see [`bailing-moe-linear.md` §2.4](bailing-moe-linear.md#24-configuration-tips).

For full flag definitions see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

Standard OpenAI-compatible request — see [`qwen3.md` §3.1](qwen3.md#31-basic-chat-completion). Substitute `model="inclusionAI/Ling-2.6-1T"`.

### 3.2 Reasoning (if supported by the checkpoint)

If you launched with `--reasoning-parser <key>`, mirror the thinking-on streaming pattern from [`qwen3.md` §3.2](qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) with `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 or v7x-16 |
| Model | inclusionAI/Ling-2.6-1T (BF16) |
| Tensor Parallelism | 64 (v6e) / 32 (v7x) |
| Expert Parallelism | 64 (v6e) / 32 (v7x) |
| Tested build | _Pending_ |

**Deployment Command** — same as [`bailing-moe-linear.md` §2.3](bailing-moe-linear.md#23-launch) with `--model-path inclusionAI/Ling-2.6-1T`.

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

Recommended additional datasets: AIME 2025, GPQA Diamond (reasoning); MMLU (general); RULER (long-context linear-attention).

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

**Benchmark Command** — adapt the driver from [`qwen3.md` §4.2](qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `inclusionAI/Ling-2.6-1T`, remove the vLLM half).

**Test Results** — _Pending._

## 5. Troubleshooting

See [`bailing-moe-linear.md` §5](bailing-moe-linear.md#5-troubleshooting) for the full troubleshooting matrix — the symptoms / fixes apply directly to Ling-2.6 since the runtime path is shared.

## Additional Resources

- [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T)
- [InclusionAI HF collection](https://huggingface.co/inclusionAI) — sibling checkpoints.
- [`bailing-moe-linear.md`](bailing-moe-linear.md) — architecture-family view (flags, semantics, recurrent state).
- [`bailing-moe.md`](bailing-moe.md) — non-linear Bailing MoE (older Ling variants).
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
