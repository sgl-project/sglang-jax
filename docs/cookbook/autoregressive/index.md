---
title: "Autoregressive Models"
description: "Text-only autoregressive LLM serving recipes organized by vendor — Qwen / DeepSeek / GLM / Llama / InclusionAI / MiMo and more."
---

# Autoregressive Model Recipes

End-to-end serving recipes for text-only autoregressive LLMs on SGL-JAX, organized by vendor.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — empirically tuned on hardware with reference benchmark numbers in §4 |
| 🚧 | **Starter** — launch command derived from HF model card; not yet measured. PR back tested numbers to upgrade to ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet |

## Recipes by vendor

### DeepSeek — `DeepSeek/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | DeepSeek-V2 / V2-Lite | [`DeepSeek/DeepSeek-V2.md`](DeepSeek/DeepSeek-V2.md) | v6e-4 (Lite) / v6e-32 (V2) | MoE + MLA | `DeepseekV2ForCausalLM` |
| 🚧 | DeepSeek-V3 | [`DeepSeek/DeepSeek-V3.md`](DeepSeek/DeepSeek-V3.md) | v6e-64 / v7x-16 | MoE + MLA | `DeepseekV3ForCausalLM` |
| 🚧 | DeepSeek-R1 | [`DeepSeek/DeepSeek-R1.md`](DeepSeek/DeepSeek-R1.md) | v6e-64 / v7x-16 | MoE + MLA + reasoning | `DeepseekV3ForCausalLM` |

### GLM — `GLM/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | GLM-4.5 / GLM-4.5-Air | [`GLM/GLM-4.5.md`](GLM/GLM-4.5.md) | v6e-32 / v6e-64 | MoE | `Glm4MoeForCausalLM` |
| 🚧 | GLM-5 | [`GLM/GLM-5.md`](GLM/GLM-5.md) | _pending_ | MoE (+ DSA variant) | `Glm5ForCausalLM` / `GlmMoeDsaForCausalLM` |

### Google — `Google/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | Gemma 2 (9B / 27B) | [`Google/Gemma2.md`](Google/Gemma2.md) | v6e-4 | dense (hybrid attn) | `Gemma2ForCausalLM` |

### Grok — `Grok/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| ✅ | Grok-2 | [`Grok/Grok2.md`](Grok/Grok2.md) | v6e-32 | dense | `Grok1ForCausalLM` |

### InclusionAI — `InclusionAI/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | Ling 1.x (lite / plus / Coder-lite) | [`InclusionAI/Ling-1.x.md`](InclusionAI/Ling-1.x.md) | v6e-4 / v6e-32 | MoE | `BailingMoeForCausalLM` |
| 🚧 | Ling 2.0 (mini / flash / 1T) | [`InclusionAI/Ling-2.md`](InclusionAI/Ling-2.md) | v6e-4 / v6e-16 / v6e-64 | MoE (1/32 sparsity + MTP) | `BailingMoeV2ForCausalLM` |
| 🚧 | Ring 2.0 (mini / flash / 1T-preview) | [`InclusionAI/Ring-2.md`](InclusionAI/Ring-2.md) | v6e-4 / v6e-16 / v6e-64 | MoE + reasoning | `BailingMoeV2ForCausalLM` |
| 🚧 | Ling 2.6 (1T / flash) | [`InclusionAI/Ling-2.6.md`](InclusionAI/Ling-2.6.md) | v6e-64 / v7x-16 | MoE + linear attn | `BailingMoeV2_5ForCausalLM` |

### Llama — `Llama/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | Llama 3.1 8B (+ Phi-3 / InternLM3 aliases) | [`Llama/Llama3.1.md`](Llama/Llama3.1.md) | v6e-4 | dense | `LlamaForCausalLM` |
| 🚧 | Llama 3.3 70B | [`Llama/Llama3.3-70B.md`](Llama/Llama3.3-70B.md) | v6e-32 | dense | `LlamaForCausalLM` |
| 📝 | Phi-3 | _alias under Llama3.1.md_ | — | dense (Llama alias) | `Phi3ForCausalLM` |
| 📝 | InternLM 3 | _alias under Llama3.1.md_ | — | dense (Llama alias) | `InternLM3ForCausalLM` |

### Moonshotai — `Moonshotai/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| 🚧 | Kimi-Linear (48B-A3B) | [`Moonshotai/Kimi-Linear.md`](Moonshotai/Kimi-Linear.md) | v6e-16 | dense + linear attn | `KimiLinearForCausalLM` |

### Qwen — `Qwen/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| ✅ | Qwen-7B-Chat | [`Qwen/Qwen.md`](Qwen/Qwen.md) | v6e-4 | dense | `QWenLMHeadModel` |
| ✅ | Qwen3-8B / Qwen3-32B | [`Qwen/Qwen3.md`](Qwen/Qwen3.md) | v6e-4 | dense | `Qwen3ForCausalLM` |
| 🚧 | Qwen3-MoE (30B-A3B / 235B-A22B) | [`Qwen/Qwen3-MoE.md`](Qwen/Qwen3-MoE.md) | v6e-16 / v6e-64 | MoE | `Qwen3MoeForCausalLM` |
| 📝 | Qwen2 | _no recipe_ | — | dense | `Qwen2ForCausalLM` |
| 📝 | Qwen2-MoE | _no recipe_ | — | MoE | `Qwen2MoeForCausalLM` |

### Xiaomi — `Xiaomi/`

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| ✅ | MiMo-V2-Flash | [`Xiaomi/MiMo-V2-Flash.md`](Xiaomi/MiMo-V2-Flash.md) | v7x-8 or v6e-16 | MoE (`fused` / `epmoe`) | `MiMoV2FlashForCausalLM` |
| ✅ | MiMo-V2.5-Pro | [`Xiaomi/MiMo-V2.5-Pro.md`](Xiaomi/MiMo-V2.5-Pro.md) | v7x-16 or v6e-64 | MoE (`fused`) | `MiMoV2ForCausalLM` |
| 🚧 | MiMo-7B | [`Xiaomi/MiMo-7B.md`](Xiaomi/MiMo-7B.md) | v6e-4 | dense | `MiMoForCausalLM` |

> Upgrade path: 🚧 → ✅ requires real `evalscope` (accuracy) or `bench_serving` (throughput) output in §4, structured as **Test Environment → Deployment Command → Benchmark Command → Test Results**. See [`Xiaomi/MiMo-V2-Flash.md` §4](Xiaomi/MiMo-V2-Flash.md#4-benchmark) for the canonical four-section form.

## What "autoregressive" means here

A model that generates text one token at a time, conditioning on its own previous outputs. Includes:

- **Dense LLMs** — Qwen / Qwen3 / Llama / Gemma 2 / MiMo-7B / Grok-2.
- **MoE LLMs** — Qwen3-MoE / DeepSeek V2/V3/R1 / GLM-4.5 / GLM-5 / Ling 1.x / Ling 2.0 / Ring 2.0 / Ling 2.6 / MiMo-V2-Flash / MiMo-V2.5-Pro / Kimi-Linear.

Diffusion / TTS / VLM models live in [`../multimodal/`](../multimodal/index.md).

## Picking a starting recipe to clone for a new model

| Goal | Clone from |
|---|---|
| Single-host dense model | [`Qwen/Qwen.md`](Qwen/Qwen.md) ✅ or [`Llama/Llama3.1.md`](Llama/Llama3.1.md) 🚧 |
| Dense model with benchmark comparison | [`Qwen/Qwen3.md`](Qwen/Qwen3.md) ✅ |
| Single-host MoE with backend choice | [`Xiaomi/MiMo-V2-Flash.md`](Xiaomi/MiMo-V2-Flash.md) ✅ |
| Large multi-node MoE with GKE manifest | [`Xiaomi/MiMo-V2.5-Pro.md`](Xiaomi/MiMo-V2.5-Pro.md) ✅ |
| SkyPilot multi-node dense | [`Grok/Grok2.md`](Grok/Grok2.md) ✅ (+ [`../deployment/skypilot.md`](../deployment/skypilot.md)) |
| Linear-attention model with recurrent state | [`InclusionAI/Ling-2.6.md`](InclusionAI/Ling-2.6.md) 🚧 or [`Moonshotai/Kimi-Linear.md`](Moonshotai/Kimi-Linear.md) 🚧 |
| Reasoning model (RL-tuned, `<think>` blocks) | [`InclusionAI/Ring-2.md`](InclusionAI/Ring-2.md) 🚧 or [`DeepSeek/DeepSeek-R1.md`](DeepSeek/DeepSeek-R1.md) 🚧 |

## Architecture coverage vs codebase

This index lists models with a recipe (✅ / 🚧) plus models the runtime supports but where no curated recipe exists yet (📝). The complete set of registered architectures lives under [`python/sgl_jax/srt/models/`](https://github.com/sgl-project/sglang-jax/tree/main/python/sgl_jax/srt/models). 📝 entries are still served correctly by the runtime — they just don't have a deployment guide.
