# SGL-JAX Cookbook

End-to-end recipes for serving specific models on specific TPU (or GPU) topologies with SGL-JAX.

> Where the rest of the docs (`get_started/`, `basic_usage/features/`, `architecture/`) explain *how SGL-JAX works*, the cookbook answers a single question per page: **"How do I run model X on hardware Y?"**

## Layout

| Section | What's inside |
|---|---|
| [`autoregressive/`](autoregressive/index.md) | LLM recipes (Qwen / Llama / Gemma / GLM / DeepSeek / Grok / MiMo / Bailing / Kimi / ...). One page per model. |
| [`multimodal/`](multimodal/index.md) | Vision-language, audio, and text-to-video recipes (Qwen2.5-VL, Qwen3-Omni MoE, Wan 2.1/2.2 T2V, MiMo Audio). |
| [`base/`](base/) | Cross-cutting references shared by every recipe: TPU topology table, launch flag reference. |
| [`deployment/`](deployment/) | Launcher templates (single-host Docker / GKE Indexed Job / SkyPilot) referenced by recipes. |
| [`troubleshooting.md`](troubleshooting.md) | Common startup / multi-node / runtime failure modes across all recipes. |

## Conventions

Every recipe follows the **same six-section template** so you can scan any page the same way:

1. **Model Introduction** — variants, parameter sizes, HuggingFace links, license, recommended sampling params.
2. **Hardware Requirements** — TPU generation × topology × node count × chip count.
3. **Environment Setup** — JAX image tag + install command (links to [`../get_started/install.md`](../get_started/install.md)).
4. **Server Launch** — single-host / multi-host / GKE / SkyPilot blocks plus a Key Flags table.
5. **Invocation** — basic `curl` / OpenAI-compatible request examples.
6. **Benchmark** — accuracy (evalscope) and throughput (`bench_serving`) commands with reference numbers.

Optional **Troubleshooting** appendix for model-specific gotchas. Generic issues go to [`troubleshooting.md`](troubleshooting.md).

Recipes carry one of two banners at the top:

- **Validated** — empirically tuned on hardware, with reference benchmark numbers.
- **Starter** — derived from code + HF model card, **not yet validated**. Treat as a starting command, tune and PR back tested values.

## Hardware coverage at a glance

Status prefix: ✅ validated · 🚧 starter (not yet measured). See [`autoregressive/index.md`](autoregressive/index.md) for the full status legend.

### Single-host (1 node)

| TPU | Topology | Recipes |
|---|---|---|
| v6e-4 | 2x2 | ✅ [Qwen-7B-Chat](autoregressive/qwen.md) · ✅ [Qwen3-8B / 32B](autoregressive/qwen3.md) · 🚧 [Llama 3.1 8B](autoregressive/llama.md) · 🚧 [Gemma 2 9B / 27B](autoregressive/gemma2.md) · 🚧 [MiMo-7B](autoregressive/mimo-7b.md) · 🚧 [DeepSeek V2-Lite](autoregressive/deepseek-v3.md) |
| v7x-8 | single host (4 chips × 2 devices) | ✅ [MiMo-V2-Flash](autoregressive/mimo-v2-flash.md) |

### Multi-host

| TPU | Topology | Nodes | Recipes |
|---|---|---|---|
| v6e-16 | 4x4 | 4 | ✅ [MiMo-V2-Flash](autoregressive/mimo-v2-flash.md) · 🚧 [Qwen3-30B-A3B MoE](autoregressive/qwen3-moe.md) · 🚧 [Kimi-Linear](autoregressive/kimi-linear.md) · 🚧 [Bailing MoE (small)](autoregressive/bailing-moe.md) |
| v6e-32 | 4x8 | 8 | ✅ [Grok-2](autoregressive/grok2.md) · 🚧 [Llama 3.3 70B](autoregressive/llama.md) · 🚧 [DeepSeek V2](autoregressive/deepseek-v3.md) · 🚧 [GLM-4.5-Air](autoregressive/glm4-moe.md) |
| v6e-64 | 4x4x4 | 16 | ✅ [MiMo-V2.5-Pro](autoregressive/mimo-v2.5-pro.md) · 🚧 [Qwen3-235B MoE](autoregressive/qwen3-moe.md) · 🚧 [DeepSeek V3 / R1](autoregressive/deepseek-v3.md) · 🚧 [GLM-4.5](autoregressive/glm4-moe.md) · 🚧 [Bailing MoE Linear](autoregressive/bailing-moe-linear.md) · 🚧 [Ling-2.6](autoregressive/ling-2.6.md) |
| v7x-16 | 2x2x4 | 4 | ✅ [MiMo-V2.5-Pro](autoregressive/mimo-v2.5-pro.md) · 🚧 [Ling-2.6](autoregressive/ling-2.6.md) · 🚧 [DeepSeek V3 / R1](autoregressive/deepseek-v3.md) |

For TPU generation/HBM/per-chip-device specs, see [`base/tpu-topology-reference.md`](base/tpu-topology-reference.md).

## Adding a new recipe

1. Pick the most similar existing recipe as a starting point:
   - Validated: [`mimo-v2.5-pro.md`](autoregressive/mimo-v2.5-pro.md) is the most complete reference (covers single-host, multi-host, GKE, key flags, benchmark).
   - Starter pattern: [`llama.md`](autoregressive/llama.md) is the smallest skeleton.
2. Copy its six-section structure.
3. Mark the top banner **Starter** until you have measured numbers; then upgrade to **Validated**.
4. Fill in real `evalscope` and `bench_serving` outputs — do not leave placeholder numbers in a Validated recipe.
5. Add an entry to this index's *Hardware coverage* table and to [`autoregressive/index.md`](autoregressive/index.md).
6. Cross-link `Key Flags` table cells to [`base/launch-flags-reference.md`](base/launch-flags-reference.md) for non-obvious flags.
