---
title: "Ling 2.6"
---

# Ling-2.6 on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) is InclusionAI's 1T-parameter Ling 2.6 release — a trillion-scale MoE built on **linear / delta attention** with a **hybrid recurrent state** pool that shares HBM with the KV cache. Smaller siblings (e.g. `Ling-2.6-flash`) are released under the same [InclusionAI HF collection](https://huggingface.co/inclusionAI).

**Architectural distinguishers**:

- **Linear / delta attention** in place of standard softmax attention — most of the long-context benefit shows up here.
- **Hybrid recurrent state pool** — budgeted against the KV cache via `--recurrent-state-memory-ratio` (default `0.9`).

**Variants**:

- [**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) — full trillion-scale flagship; default focus of this page.
- Smaller Ling-2.6 variants — adapt the §2.3 launch command after picking a checkpoint.

For the previous Ling 2.5 hybrid linear-attention generation see [`Ling2.5.md`](Ling2.5.md). For Moonshot AI's separate linear-attention model see [`Kimi-Linear.md`](../Moonshotai/Kimi-Linear.md).

**Recommended Generation Parameters**: see the Ling-2.6 model card for authoritative defaults. As a starter: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` (give room if you enable reasoning mode).

**License**: see the [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| Ling-2.6-1T | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | 🚧 starter | Trillion-scale; multi-host mandatory |
| Ling-2.6-1T | v7x-16 | 4x4 | 4  | 16 | 32 | 32 | 🚧 starter | v7x exposes 2 JAX devices per chip → `--tp-size 32` |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-64

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=ling-2-6`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, and parallelism/completions set to 16. Add these model flags to the job command:

```text
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
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v7x-16

Use GKE with `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=4x4`, and parallelism/completions set to 4. Change the model flags to:

```text
  --tp-size 32 --ep-size 32
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.

**MoE Backend:**
- `--moe-backend fused` for `--ep-size ≥ 16` (both configs above). Switch to `epmoe` only at EP ≤ 8.

**Reasoning Mode:**
- If the Ling-2.6 checkpoint emits `<think>...</think>` blocks (verify per model card; some reasoning-tuned variants do, base instruct variants do not), add `--reasoning-parser deepseek-r1` to the launch command — that's the generic `<think>` parser, since no `ling-2-6` or `bailing` parser key is registered. The streaming Python client from [`Qwen3.md` §3.2](../Qwen/Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) applies directly once the parser is set.

**Context Length:**
- Pin via `--context-length` to your workload's longest prompt + output. Smaller values reduce KV cache footprint at trillion-scale.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if you hit OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at trillion-scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's nodes to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="inclusionAI/Ling-2.6-1T"` with the §1 recommended sampling parameters; for thinking + content streaming see §3.2.

### 3.2 Reasoning (if supported by the checkpoint)

Ling-2.6 emits `<think>...</think>` blocks; reuse the generic `deepseek-r1` reasoning parser. Append `--reasoning-parser deepseek-r1` to the §2.3 launch command (see §2.4 Reasoning Mode). If your checkpoint supports the per-request thinking toggle, set `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`; stream `reasoning_content` separately from `content`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="inclusionAI/Ling-2.6-1T",
    messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    temperature=0.7,
    top_p=0.95,
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

For non-streaming requests, the field appears on `response.choices[0].message.reasoning_content`. To see the full set of `--reasoning-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `inclusionAI/Ling-2.6-1T`, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) or v7x-16 |
| Model | inclusionAI/Ling-2.6-1T (BF16) |
| Tensor Parallelism | 64 (v6e) / 32 (v7x) |
| Expert Parallelism | 64 (v6e) / 32 (v7x) |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-64).

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

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Recurrent state + KV exceed budget | Lower `--recurrent-state-memory-ratio` (e.g. to 0.7) and/or `--mem-fraction-static` to 0.9. |
| Long-prompt requests stall | KV cache exhausted before recurrent state | Lower `--recurrent-state-memory-ratio` to give the KV cache more headroom. |
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC for amortized compilation. |

## Additional Resources

- [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T)
- [InclusionAI HF collection](https://huggingface.co/inclusionAI) — sibling checkpoints.
- [`Ling2.5.md`](Ling2.5.md) — Ling / Ring 2.5 hybrid linear-attention generation.
- [`Kimi-Linear.md`](../Moonshotai/Kimi-Linear.md) — Moonshot AI's separate linear-attention model.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
