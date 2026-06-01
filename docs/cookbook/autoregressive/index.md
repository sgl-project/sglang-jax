---
title: "Autoregressive Models"
---

# Autoregressive Model Recipes

End-to-end serving recipes for autoregressive models on SGL-JAX, organized by vendor. This includes text-only LLMs and vision-language decoders whose output is still generated token by token.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — primary model / hardware path empirically tuned with reference benchmark numbers in §4 |
| 🧪 | **Partially validated** — at least one variant / hardware path has real benchmark output; other variants, matrix cells, or current-build reruns are still pending |
| 🚧 | **Starter** — launch command derived from HF model card; not yet measured. PR back tested numbers to upgrade to 🧪 or ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet (or model release pending) |
| 🚫 | **Blocked** — runnable path blocked by an upstream weight format / HBM / runtime constraint; banner cites the root cause and unblocking plan |

## Recipes by vendor

### DeepSeek — `DeepSeek/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | DeepSeek-V2-Lite | [`DeepSeek/DeepSeek-V2.md`](DeepSeek/DeepSeek-V2.md) | v6e-4 | MoE + MLA |
| ✅ | DeepSeek-V3 | [`DeepSeek/DeepSeek-V3.md`](DeepSeek/DeepSeek-V3.md) | v6e-64 / v7x-16 | MoE + MLA |
| ✅ | DeepSeek-R1 | [`DeepSeek/DeepSeek-R1.md`](DeepSeek/DeepSeek-R1.md) | v6e-64 / v7x-16 | MoE + MLA + reasoning (`deepseek-r1`) |

### GLM — `GLM/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| 🧪 | GLM-4.5-Air (106B) | [`GLM/GLM-4.5.md`](GLM/GLM-4.5.md) | v6e-32 | MoE + reasoning/tool (`glm45`) |

### Google — `Google/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Gemma 2 27B-it | [`Google/Gemma2.md`](Google/Gemma2.md) | v6e-4 | dense (hybrid attn) |

### Grok — `Grok/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Grok-2 (base) | [`Grok/Grok2.md`](Grok/Grok2.md) | v6e-32 | MoE (8 experts, 2 active) |

### InclusionAI — `InclusionAI/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Ling 2.6-1T | [`InclusionAI/Ling-2.6.md`](InclusionAI/Ling-2.6.md) | v6e-64 | MoE + linear attn |

### Llama — `Llama/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Llama 3.1 8B-Instruct | [`Llama/Llama3.1.md`](Llama/Llama3.1.md) | v6e-4 | dense |
| ✅ | Llama 3.3 70B | [`Llama/Llama3.3-70B.md`](Llama/Llama3.3-70B.md) | v6e-32 | dense |

### Moonshotai — `Moonshotai/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Kimi-Linear (48B-A3B) | [`Moonshotai/Kimi-Linear.md`](Moonshotai/Kimi-Linear.md) | v6e-16 | dense + linear attn |

### Qwen — `Qwen/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | Qwen-7B-Chat | [`Qwen/Qwen.md`](Qwen/Qwen.md) | v6e-4 | dense |
| ✅ | Qwen3-8B / Qwen3-32B | [`Qwen/Qwen3.md`](Qwen/Qwen3.md) | v6e-4 | dense + reasoning (`qwen3`) + tool (`qwen25`) |
| 🧪 | Qwen3-30B-A3B | [`Qwen/Qwen3-MoE.md`](Qwen/Qwen3-MoE.md) | v6e-16 | MoE + reasoning (`qwen3`) + tool (`qwen25`) |
| ✅ | Qwen2.5-VL (3B / 7B / 32B / 72B) | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) | v6e-4 for 3B/7B/32B; 72B pending | vision-language autoregressive decoder |
| 📝 | Qwen2 / Qwen2-MoE | _no recipe — same family runtime path_ | — | dense / MoE |

### Xiaomi — `Xiaomi/`

| Status | Model | Recipe | Min TPU | Backend |
|---|---|---|---|---|
| ✅ | MiMo-V2-Flash | [`Xiaomi/MiMo-V2-Flash.md`](Xiaomi/MiMo-V2-Flash.md) | v7x-8 or v6e-16 | MoE + reasoning/tool (`mimo`) |
| ✅ | MiMo-V2.5-Pro | [`Xiaomi/MiMo-V2.5-Pro.md`](Xiaomi/MiMo-V2.5-Pro.md) | v6e-64 (v7x-16 alternative) | MoE + reasoning/tool (`mimo`) |
| ✅ | MiMo-7B-RL | [`Xiaomi/MiMo-7B.md`](Xiaomi/MiMo-7B.md) | v6e-4 | dense + reasoning/tool (`mimo`) |

> Upgrade path: 🚧 → 🧪 requires real `evalscope` (accuracy) or `bench_serving` (throughput) output for at least one variant / hardware path. 🧪 → ✅ requires the recipe's claimed primary path to have complete **Test Environment → Deployment Command → Benchmark Command → Test Results** evidence without unresolved required cells. See [`Xiaomi/MiMo-V2-Flash.md` §4](Xiaomi/MiMo-V2-Flash.md#4-benchmark) for the canonical four-section form.

## What "autoregressive" means here

A model that generates text one token at a time, conditioning on its own previous outputs. Includes:

- **Dense LLMs** — Qwen / Qwen3 / Llama / Gemma 2 / MiMo-7B.
- **MoE LLMs** — Qwen3-MoE / DeepSeek V2/V3/R1 / GLM-4.5 / Ling 2.6 / MiMo-V2-Flash / MiMo-V2.5-Pro / Kimi-Linear / Grok-2 (base).
- **Vision-language decoders** — Qwen2.5-VL ingests image / video inputs, but the answer is still produced by an autoregressive generation stage.

Diffusion image/video generation models live in [`../diffusion/`](../diffusion/index.md).

## Picking a starting recipe to clone for a new model

| Goal | Clone from |
|---|---|
| Single-host dense model | [`Qwen/Qwen.md`](Qwen/Qwen.md) ✅ or [`Llama/Llama3.1.md`](Llama/Llama3.1.md) ✅ |
| Dense model with benchmark comparison | [`Qwen/Qwen3.md`](Qwen/Qwen3.md) ✅ |
| Single-host MoE with backend choice | [`Xiaomi/MiMo-V2-Flash.md`](Xiaomi/MiMo-V2-Flash.md) ✅ |
| Vision-language chat | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) ✅ |
| Large multi-node MoE with GKE manifest | [`Xiaomi/MiMo-V2.5-Pro.md`](Xiaomi/MiMo-V2.5-Pro.md) ✅ |
| Multi-node dense | [`Llama/Llama3.3-70B.md`](Llama/Llama3.3-70B.md) ✅ (+ [`../deployment/gke-indexed-job.md`](../deployment/gke-indexed-job.md)) |
| Base model (no chat template, `/v1/completions` flow) | [`Grok/Grok2.md`](Grok/Grok2.md) ✅ |
| Linear-attention model with recurrent state | [`Moonshotai/Kimi-Linear.md`](Moonshotai/Kimi-Linear.md) ✅ or [`InclusionAI/Ling-2.6.md`](InclusionAI/Ling-2.6.md) ✅ |
| Reasoning model (RL-tuned, `<think>` blocks) | [`DeepSeek/DeepSeek-R1.md`](DeepSeek/DeepSeek-R1.md) ✅ |

## Parser key reference

Reasoning models that emit `<think>` blocks need `--reasoning-parser <key>` at launch; tool-calling models need `--tool-call-parser <key>`. The cookbook recipes pick the key that matches each model's `<think>` / tool-call format:

| Parser key | Reasoning | Tool-call | Where used in cookbook |
|---|---|---|---|
| `deepseek-r1` | ✓ (`<think>...</think>`) | — | [DeepSeek-R1](DeepSeek/DeepSeek-R1.md), Ling 2.6 reasoning variants (use as `<think>` parser) |
| `qwen3` | ✓ (`<think>...</think>` + `enable_thinking` switch) | — | [Qwen3](Qwen/Qwen3.md), [Qwen3-MoE](Qwen/Qwen3-MoE.md) |
| `qwen25` | — | ✓ | Qwen3 / Qwen3-MoE tool-calling |
| `qwen3_coder` | — | ✓ | Qwen3-Coder variants |
| `mimo` | ✓ (alias of `qwen3` parser) | ✓ | [MiMo-V2-Flash](Xiaomi/MiMo-V2-Flash.md), [MiMo-V2.5-Pro](Xiaomi/MiMo-V2.5-Pro.md), [MiMo-7B](Xiaomi/MiMo-7B.md) |
| `glm45` | ✓ (`<think>...</think>`) | ✓ | [GLM-4.5 / 4.5-Air](GLM/GLM-4.5.md) |
| `kimi` | ✓ (`◁think▷...◁/think▷`) | — | Reserved for Kimi reasoning variants; Kimi-Linear-Instruct is not reasoning |

Run `python -m sgl_jax.launch_server --help` against your checkout to see the full registered set — these keys are the cookbook-relevant subset.

## Architecture coverage vs codebase

This index lists models with a recipe (✅ / 🧪 / 🚧) plus models the runtime supports but where no curated recipe exists yet (📝). 📝 entries are still served correctly by the runtime — they just don't have a deployment guide.
