---
title: "Ling 2.5"
---

# Ling 2.5 on SGL-JAX

> **Starter recipe** — derived from the public HuggingFace model cards and the SGL-JAX `BailingMoeV2_5ForCausalLM` runtime path; not yet empirically validated on TPU. Treat hardware rows as candidates until benchmark data is added.

## 1. Model Introduction

[**inclusionAI/Ling-2.5-1T**](https://huggingface.co/inclusionAI/Ling-2.5-1T) and [**inclusionAI/Ring-2.5-1T**](https://huggingface.co/inclusionAI/Ring-2.5-1T) are InclusionAI's trillion-parameter Ling 2.5 generation:

- **Ling-2.5-1T** — non-reasoning / instant model; 1T total parameters, 63B active parameters, 256K context extendable to 1M with YaRN.
- **Ring-2.5-1T** — thinking / reasoning model on the same Ling 2.5 architecture; 1T total parameters, 128K context extendable to 256K with YaRN.

SGL-JAX serves this family through `BailingMoeV2_5ForCausalLM` (`python/sgl_jax/srt/models/bailing_moe_linear.py`). The architecture is a hybrid linear-attention MoE path: Ling 2.5 upgrades the Ling 2.0 GQA stack to a **1:7 MLA + Lightning Linear Attention** mix, reducing KV-cache pressure for long-context decode while keeping full-attention layers for expressiveness.

**Runtime architecture fields to check before launch:**

- `architectures`: must include `BailingMoeV2_5ForCausalLM`.
- `model_type`: usually `bailing_hybrid`.
- MoE: 256 experts, 8 experts per token in the default SGL-JAX config path.
- Attention: `layers_block_type` selects `linear_attention` vs `attention` / `full_attention`; MLA layers use the DeepSeek-style MLA patch path.

For the newer 2.6 generation see [`Ling-2.6.md`](Ling-2.6.md).

**Recommended Generation Parameters**:

- Ling-2.5-1T: use model-card defaults first; starter values `temperature=0.7`, `top_p=0.95`, `max_tokens=4096`.
- Ring-2.5-1T: reasoning workloads usually need larger budgets; starter values `temperature=0.6`, `top_p=0.95`, `max_tokens=8192+`.

**License**: see each HuggingFace model card for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | Minimum candidate | Validated / recommended | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|
| Ling-2.5-1T | v6e-64 | _Pending validation_; prefer v7x-16 for HBM headroom if available | 64 (v6e-64) / 32 (v7x-16) | same as TP | BF16 1T weights are ~2 TB total; v6e-64 is tight before KV + recurrent state |
| Ring-2.5-1T | v6e-64 | _Pending validation_; prefer v7x-16 for reasoning workloads | 64 (v6e-64) / 32 (v7x-16) | same as TP | Reasoning output lengths increase KV / recurrent-state pressure |

> v6e has 32 GiB per JAX device. v7x exposes 2 JAX devices per chip with 96 GiB per JAX device, so `tpu-v7x-16` is 32 JAX devices. See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md).

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For multi-host serving, use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md).

| Hardware Platform | Docker Image |
|---|---|
| TPU v6e (Trillium) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (Ling-2.5-1T)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=ling-2-5`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, and parallelism/completions set to 16. Add these model flags to the job command:

```text
  --model-path inclusionAI/Ling-2.5-1T \
  --trust-remote-code \
  --tp-size 64 --ep-size 64 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 128 \
  --disable-radix-cache \
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (Ring-2.5-1T)

Ring-2.5 is the thinking variant. Use the same topology, add the generic `<think>` parser, and switch the model path:

```text
  --model-path inclusionAI/Ring-2.5-1T \
  --reasoning-parser deepseek-r1 \
```

#### TPU v7x-16 candidate

For v7x-16, use GKE with `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=4x4`, and parallelism/completions set to 4. Change the model flags to `--tp-size 32 --ep-size 32`. Keep `--disable-radix-cache`; hybrid recurrent state models are unsafe with prefix sharing.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Hybrid recurrent state:**
- Keep `--disable-radix-cache`. The runtime requires this for hybrid recurrent-state models because prefix sharing is unsafe with recurrent state.
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory; first compile on 1T hybrid models can be long.

**MoE backend and EP sizing:**
- Use `--moe-backend fused` for the 1T variants.
- Keep `--ep-size == --tp-size` unless you have explicitly validated a different mesh. SGL-JAX fused MoE uses the data × tensor mesh as the EP group; mismatched EP semantics can silently invalidate the recipe.
- Confirm `num_experts % ep_size == 0`. The default config path has 256 experts, so EP 64 and EP 32 both divide cleanly.

**Memory management:**
- Start at `--mem-fraction-static 0.9`. Raise only after the model loads and a long-context request succeeds.
- Keep `--max-running-requests` conservative (`64`-`128`) for Ring reasoning workloads.
- Long contexts can move the bottleneck from weights to KV + recurrent state; validate with realistic prompt and decode lengths before increasing concurrency.

**Reasoning parser:**
- Ling-2.5-1T is the instant variant and should not need a reasoning parser by default.
- Ring-2.5-1T is the thinking variant. Start with `--reasoning-parser deepseek-r1` because no `ring` / `ling` parser key is registered in SGL-JAX today.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="inclusionAI/Ling-2.5-1T"` (instant) or `inclusionAI/Ring-2.5-1T` (thinking) with the §1 recommended sampling parameters; for Ring-2.5 reasoning streaming see §3.2.

### 3.2 Reasoning (Ring-2.5)

Launch Ring-2.5 with `--reasoning-parser deepseek-r1`, then stream `reasoning_content` separately from final `content`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="inclusionAI/Ring-2.5-1T",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

### 3.3 Tool Calling

The Ling-2.5 model card advertises agent/tool-calling strength, but SGL-JAX currently has no Ling-specific `--tool-call-parser` key documented in the cookbook. Do not claim native tool-calling until a parser key and end-to-end response shape are validated. For tool-calling examples with known parser support, see [`../Qwen/Qwen3.md`](../Qwen/Qwen3.md) or [`../Xiaomi/MiMo-V2.5-Pro.md`](../Xiaomi/MiMo-V2.5-Pro.md).

## 4. Benchmark

> Benchmark data below is a placeholder. Add real `bench_serving` and `evalscope` output before upgrading this recipe from Starter.

### 4.1 Speed

**Benchmark Command** — adapt the driver from [`../Qwen/Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm). For Ling-2.5:

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --dataset-name random \
  --num-prompts 100 \
  --random-input 2048 --random-output 2048 \
  --random-range-ratio 1 \
  --max-concurrency 16 \
  --warmup-requests 0 \
  --tokenizer inclusionAI/Ling-2.5-1T
```

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | v6e-64 candidate / v7x-16 candidate |
| Model | inclusionAI/Ling-2.5-1T / inclusionAI/Ring-2.5-1T |
| Tensor Parallelism | 64 (v6e-64) / 32 (v7x-16) |
| Expert Parallelism | 64 (v6e-64) / 32 (v7x-16) |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-64-ling-25-1t).

**Benchmark Command** — example for Ling-2.5:

```bash
evalscope eval \
  --model inclusionAI/Ling-2.5-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 4
```

For Ring-2.5, run reasoning-heavy datasets with larger output budgets:

```bash
evalscope eval \
  --model inclusionAI/Ring-2.5-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime_2025 \
  --eval-batch-size 4 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}'
```

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Startup asserts that hybrid recurrent state requires disabling radix cache | Prefix sharing was left enabled | Add `--disable-radix-cache`. |
| Model architecture is unsupported | HF config `architectures` does not resolve to `BailingMoeV2_5ForCausalLM` | Verify config and use a registered architecture override only after checking the model card. |
| Response contains raw `<think>` text for Ring-2.5 | Missing reasoning parser | Add `--reasoning-parser deepseek-r1`. |
| OOM at startup on v6e-64 | BF16 weights + KV + recurrent state leave little headroom | Lower `--mem-fraction-static`, lower concurrency, or prefer v7x-16. |
| MoE throughput plateau | EP mesh does not match fused MoE assumptions | Keep `--ep-size == --tp-size` and ensure `num_experts % ep_size == 0`. |
| First request takes several minutes per node | Empty JIT cache | Persist `JAX_COMPILATION_CACHE_DIR`; mount shared cache storage across nodes. |

## Additional Resources

- [Ling-2.5-1T model card](https://huggingface.co/inclusionAI/Ling-2.5-1T)
- [Ring-2.5-1T model card](https://huggingface.co/inclusionAI/Ring-2.5-1T)
- [`Ling-2.6.md`](Ling-2.6.md) — newer Ling 2.6 generation.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
