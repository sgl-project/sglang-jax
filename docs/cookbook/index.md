---
title: "SGL-JAX Cookbook"
---

# SGL-JAX Cookbook

End-to-end recipes for serving specific models on specific TPU (or GPU) topologies with SGL-JAX.

> Where the rest of the docs (`get_started/`, `basic_usage/features/`, `architecture/`) explain *how SGL-JAX works*, the cookbook answers a single question per page: **"How do I run model X on hardware Y?"**

## Layout

| Section | What's inside |
|---|---|
| [`autoregressive/`](autoregressive/index.md) | LLM recipes organized by vendor subdir (DeepSeek/ · GLM/ · Google/ · Grok/ · InclusionAI/ · Llama/ · Moonshotai/ · Qwen/ · Xiaomi/). One page per model, following sgl-cookbook vendor-naming convention. |
| [`multimodal/`](multimodal/index.md) | Vision-language, audio, and text-to-video recipes (Qwen2.5-VL, Qwen3-Omni MoE, Wan 2.1/2.2 T2V, MiMo Audio). |
| [`base/`](base/) | Cross-cutting references shared by every recipe: TPU topology table, launch flag reference. |
| [`deployment/`](deployment/) | Launcher templates referenced by recipes: single-host Docker, primary multi-host GKE Indexed Job, and advanced SkyPilot v6e experiments. |
| [`troubleshooting.md`](troubleshooting.md) | Common startup / multi-node / runtime failure modes across all recipes. |

## Conventions

Every recipe follows the **same five-section template** so you can scan any page the same way:

1. **Model Introduction** — variants, parameter sizes, HuggingFace links, license, recommended sampling params.
2. **Deployment** — 2.1 Hardware Matrix · 2.2 Environment · 2.3 Launch (single-host + multi-host) · 2.4 Configuration Tips.
3. **Invocation** — basic `curl` / OpenAI-compatible request examples; reasoning / tool-call patterns where applicable.
4. **Benchmark** — accuracy (`evalscope`) and throughput (`bench_serving`) commands with reference numbers.
5. **Troubleshooting** — model-specific symptom → cause → fix table. Generic issues go to [`troubleshooting.md`](troubleshooting.md).

Recipes carry one of three status banners at the top:

- **Validated** — empirically tuned on hardware, with reference benchmark numbers.
- **Partially validated** — at least one variant / hardware path has real benchmark output; other variants, matrix cells, or current-build reruns are still pending.
- **Starter** — derived from HF model card, **not yet validated**. Treat as a starting command, tune and PR back tested values.

## Hardware coverage at a glance

Status prefix: ✅ validated · 🧪 partially validated · 🚧 starter (not yet measured). See [`autoregressive/index.md`](autoregressive/index.md) for the full status legend and vendor breakdown.

### Single-host (1 node)

| TPU | Topology | Recipes |
|---|---|---|
| v6e-4 | 2x2 | ✅ [Qwen-7B-Chat](autoregressive/Qwen/Qwen.md) · 🧪 [Qwen3-8B / 32B](autoregressive/Qwen/Qwen3.md) · 🚧 [Llama 3.1 8B](autoregressive/Llama/Llama3.1.md) · 🚧 [Gemma 2 9B / 27B](autoregressive/Google/Gemma2.md) · 🧪 [MiMo-7B](autoregressive/Xiaomi/MiMo-7B.md) · 🧪 [DeepSeek-V2-Lite](autoregressive/DeepSeek/DeepSeek-V2.md) |
| v7x-8 | single host (4 chips × 2 devices) | 🧪 [MiMo-V2-Flash](autoregressive/Xiaomi/MiMo-V2-Flash.md) |

### Multi-host

| TPU | Topology | Nodes | Recipes |
|---|---|---|---|
| v6e-16 | 4x4 | 4 | 🧪 [MiMo-V2-Flash](autoregressive/Xiaomi/MiMo-V2-Flash.md) · 🚧 [Qwen3-30B-A3B MoE](autoregressive/Qwen/Qwen3-MoE.md) · ✅ [Kimi-Linear](autoregressive/Moonshotai/Kimi-Linear.md) |
| v6e-32 | 4x8 | 8 | 🚧 [Grok-2](autoregressive/Grok/Grok2.md) · 🚧 [Grok-1](autoregressive/Grok/Grok1.md) · 🚧 [Llama 3.3 70B](autoregressive/Llama/Llama3.3-70B.md) · 🚧 [DeepSeek-V2](autoregressive/DeepSeek/DeepSeek-V2.md) · 🚧 [GLM-4.5-Air](autoregressive/GLM/GLM-4.5.md) |
| v6e-64 | 4x4x4 | 16 | 🧪 [MiMo-V2.5-Pro](autoregressive/Xiaomi/MiMo-V2.5-Pro.md) · 🚧 [Qwen3-235B MoE](autoregressive/Qwen/Qwen3-MoE.md) · 🚧 [DeepSeek-V3](autoregressive/DeepSeek/DeepSeek-V3.md) · 🚧 [DeepSeek-R1](autoregressive/DeepSeek/DeepSeek-R1.md) · 🚧 [GLM-4.5](autoregressive/GLM/GLM-4.5.md) · 🚧 [Ling / Ring 2.5](autoregressive/InclusionAI/Ling2.5.md) · 🚧 [Ling-2.6](autoregressive/InclusionAI/Ling-2.6.md) |
| v7x-16 | 2x2x4 | 4 | 🧪 [MiMo-V2.5-Pro](autoregressive/Xiaomi/MiMo-V2.5-Pro.md) · 🚧 [Ling / Ring 2.5](autoregressive/InclusionAI/Ling2.5.md) · 🚧 [Ling-2.6](autoregressive/InclusionAI/Ling-2.6.md) · 🚧 [DeepSeek-V3](autoregressive/DeepSeek/DeepSeek-V3.md) · 🚧 [DeepSeek-R1](autoregressive/DeepSeek/DeepSeek-R1.md) |

### Multimodal (`--multimodal` server)

| TPU | Topology | Recipes |
|---|---|---|
| v6e-4 | 2x2 | 🚧 [Qwen2.5-VL 3B / 7B candidates](multimodal/Qwen/Qwen2.5-VL.md) (`--tp-size 1`) · 🚧 [Qwen2.5-VL 32B](multimodal/Qwen/Qwen2.5-VL.md) (`--tp-size 4`) · 🚧 [Wan 2.1 T2V 1.3B / 14B](multimodal/Wan/Wan-2.x.md) (`--tp-size 2`) · 🚧 [Wan 2.2 T2V A14B](multimodal/Wan/Wan-2.x.md) (`--tp-size 1`) |
| Pending | _varies_ | 🚧 Qwen2.5-VL 72B multi-host (needs a matching staged path + scheduler fix) |

For TPU generation/HBM/per-chip-device specs, see [`base/tpu-topology-reference.md`](base/tpu-topology-reference.md).

Multimodal rows are constrained by SGL-JAX's built-in staged runtime. A larger TPU slice is not automatically used unless the selected model path supports that placement.

## Adding a new recipe

1. Pick the most similar existing recipe as a starting point:
   - Validated single-host dense: [`autoregressive/Qwen/Qwen.md`](autoregressive/Qwen/Qwen.md) is the smallest fully measured pattern.
   - Validated multi-host reference: [`autoregressive/Moonshotai/Kimi-Linear.md`](autoregressive/Moonshotai/Kimi-Linear.md) shows the complete multi-host benchmark / accuracy structure.
   - Partially validated large-MoE pattern: [`autoregressive/Xiaomi/MiMo-V2.5-Pro.md`](autoregressive/Xiaomi/MiMo-V2.5-Pro.md) is the broadest GKE / multi-host reference, with some benchmark cells still pending.
   - Starter pattern: [`autoregressive/Llama/Llama3.1.md`](autoregressive/Llama/Llama3.1.md) is the smallest skeleton.
2. Copy its five-section structure.
3. Place the new file under `autoregressive/<Vendor>/` (PascalCase, matching upstream [sgl-cookbook](https://github.com/sgl-project/sgl-cookbook/tree/main/docs/autoregressive) naming). Create a new vendor subdir if needed.
4. Mark the top banner **Starter** until you have measured numbers; then upgrade to **Partially validated** or **Validated** based on the status legend.
5. Fill in real `evalscope` and `bench_serving` outputs — do not leave `_Pending_` cells in a Validated recipe's claimed primary path.
6. Add an entry to this index's *Hardware coverage* table and to [`autoregressive/index.md`](autoregressive/index.md).
7. Cross-link tunable flags in §2.4 to [`base/launch-flags-reference.md`](base/launch-flags-reference.md) for non-obvious flags.
