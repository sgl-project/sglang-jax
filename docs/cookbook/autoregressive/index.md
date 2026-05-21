# Autoregressive Model Recipes

End-to-end serving recipes for text-only autoregressive LLMs on SGL-JAX.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — empirically tuned on hardware with reference benchmark numbers in §6 |
| 🚧 | **Starter** — launch command derived from code + HF model card; not yet measured. PR back tested numbers to upgrade to ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet |

## Recipes

| Status | Model | Recipe | Min TPU | Backend | Architecture class |
|---|---|---|---|---|---|
| ✅ | Qwen-7B-Chat | [`qwen.md`](qwen.md) | v6e-4 | dense | `QWenLMHeadModel` |
| ✅ | Qwen3-8B / Qwen3-32B | [`qwen3.md`](qwen3.md) | v6e-4 | dense | `Qwen3ForCausalLM` |
| ✅ | MiMo-V2-Flash | [`mimo-v2-flash.md`](mimo-v2-flash.md) | v7x-8 or v6e-16 | MoE (`fused` / `epmoe`) | `MiMoV2FlashForCausalLM` |
| ✅ | MiMo-V2.5-Pro | [`mimo-v2.5-pro.md`](mimo-v2.5-pro.md) | v7x-16 or v6e-64 | MoE (`fused`) | `MiMoV2ForCausalLM` |
| ✅ | Grok-2 | [`grok2.md`](grok2.md) | v6e-32 | dense | `Grok1ForCausalLM` |
| 🚧 | Llama 3.x | [`llama.md`](llama.md) | v6e-4 (8B) / v6e-32 (70B) | dense | `LlamaForCausalLM` |
| 🚧 | Gemma 2 | [`gemma2.md`](gemma2.md) | v6e-4 | dense (hybrid attn) | `Gemma2ForCausalLM` |
| 🚧 | MiMo-7B | [`mimo-7b.md`](mimo-7b.md) | v6e-4 | dense | `MiMoForCausalLM` |
| 🚧 | Qwen3-MoE | [`qwen3-moe.md`](qwen3-moe.md) | v6e-16 (30B-A3B) / v6e-64 (235B) | MoE | `Qwen3MoeForCausalLM` |
| 🚧 | DeepSeek V2 / V3 / R1 | [`deepseek-v3.md`](deepseek-v3.md) | v6e-4 (Lite) / v6e-64 (V3) | MoE + MLA | `DeepseekV3ForCausalLM` / `DeepseekV2ForCausalLM` |
| 🚧 | GLM-4.5 MoE | [`glm4-moe.md`](glm4-moe.md) | v6e-32 / v6e-64 | MoE | `Glm4MoeForCausalLM` |
| 🚧 | GLM-5 MoE | [`glm5-moe.md`](glm5-moe.md) | TODO | MoE (+ DSA variant) | `Glm5ForCausalLM` / `GlmMoeDsaForCausalLM` |
| 🚧 | Bailing MoE (family) | [`bailing-moe.md`](bailing-moe.md) | v6e-16+ | MoE | `BailingMoeForCausalLM` / `BailingMoeV2ForCausalLM` |
| 🚧 | Bailing MoE Linear (family) | [`bailing-moe-linear.md`](bailing-moe-linear.md) | v6e-64 / v7x-16 | MoE + linear attn | `BailingMoeV2_5ForCausalLM` |
| 🚧 | Ling-2.6 (Bailing-Linear instance) | [`ling-2.6.md`](ling-2.6.md) | v6e-64 / v7x-16 | MoE + linear attn | `BailingMoeV2_5ForCausalLM` |
| 🚧 | Kimi-Linear | [`kimi-linear.md`](kimi-linear.md) | v6e-16 | dense + linear attn | `KimiLinearForCausalLM` |
| 📝 | Qwen2 | _no recipe_ | — | dense | `Qwen2ForCausalLM` |
| 📝 | Qwen2-MoE | _no recipe_ | — | MoE | `Qwen2MoeForCausalLM` |
| 📝 | Phi-3 | _no recipe_ | — | dense (Llama alias) | `Phi3ForCausalLM` |
| 📝 | InternLM 3 | _no recipe_ | — | dense (Llama alias) | `InternLM3ForCausalLM` |

> Upgrade path: 🚧 → ✅ requires real `evalscope` (accuracy) or `bench_serving` (throughput) output in §6, structured as **Test Environment → Deployment Command → Benchmark Command → Test Results**. See [`mimo-v2-flash.md` §6](mimo-v2-flash.md#6-benchmark) for the canonical four-section form.

## What "autoregressive" means here

A model that generates text one token at a time, conditioning on its own previous outputs. Includes:

- **Dense LLMs** — Qwen / Qwen3 / Llama / Gemma 2 / MiMo-7B / Grok-2.
- **MoE LLMs** — Qwen2-MoE / Qwen3-MoE / DeepSeek V2/V3 / GLM-4 MoE / GLM-5 MoE / Bailing MoE (+ Linear) / MiMo-V2-Flash / MiMo-V2.5-Pro / Kimi-Linear.

Diffusion / TTS / VLM models live in [`../multimodal/`](../multimodal/index.md).

## Picking a starting recipe to clone for a new model

| Goal | Clone from |
|---|---|
| Single-host dense model | [`qwen.md`](qwen.md) ✅ or [`llama.md`](llama.md) 🚧 |
| Dense model with benchmark comparison | [`qwen3.md`](qwen3.md) ✅ |
| Single-host MoE with backend choice | [`mimo-v2-flash.md`](mimo-v2-flash.md) ✅ |
| Large multi-node MoE with GKE manifest | [`mimo-v2.5-pro.md`](mimo-v2.5-pro.md) ✅ |
| SkyPilot multi-node dense | [`grok2.md`](grok2.md) ✅ (+ [`../deployment/skypilot.md`](../deployment/skypilot.md)) |
| Linear-attention model with recurrent state | [`bailing-moe-linear.md`](bailing-moe-linear.md) 🚧 or [`kimi-linear.md`](kimi-linear.md) 🚧 |
| Reasoning / DSA variant | [`deepseek-v3.md`](deepseek-v3.md) 🚧 (V3 / R1) |

## Architecture coverage vs codebase

This index lists models with a recipe (✅ / 🚧) plus models the runtime supports but where no curated recipe exists yet (📝). The complete set of registered architectures lives under [`python/sgl_jax/srt/models/`](https://github.com/sgl-project/sglang-jax/tree/main/python/sgl_jax/srt/models). 📝 entries are still served correctly by the runtime — they just don't have a deployment guide.
