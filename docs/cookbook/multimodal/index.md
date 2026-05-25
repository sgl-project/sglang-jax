---
title: "Multimodal Models"
---

# Multimodal Recipes

End-to-end serving recipes for non-text-only models served by SGL-JAX: vision-language, text-to-video, and audio. All multimodal recipes launch with `--multimodal`, which enables the staged scheduler (text encoder / ViT / DiT / VAE / AR) and the diffusion-style HTTP endpoints (`/api/v1/videos/generation`, `/api/v1/images/generation`) in addition to OpenAI-compatible `/v1/chat/completions`.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — primary model / hardware path empirically tuned with reference numbers in §4 |
| 🧪 | **Partially validated** — at least one variant / hardware path has real benchmark output; other variants, matrix cells, or current-build reruns are still pending |
| 🚧 | **Starter** — launch command derived from HF model card; not yet measured. PR back tested numbers to upgrade to 🧪 or ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet |

## Recipes by vendor

### Qwen — `Qwen/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 🚧 | Qwen2.5-VL (3B / 7B / 32B / 72B) | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) | v6e-4 for 3B/7B/32B; 72B pending | Vision-language | `/v1/chat/completions` (image_url + video_url content blocks) |
| 📝 | Qwen3-Omni MoE | _planned_ | _Pending_ | Vision + audio + text | `/v1/chat/completions` (multimodal content blocks) |

### Wan-AI — `Wan/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 🚧 | Wan 2.1 / 2.2 T2V (1.3B / 14B / A14B) | [`Wan/Wan-2.x.md`](Wan/Wan-2.x.md) | v6e-4 | Text-to-video diffusion | `/api/v1/videos/generation`, `/api/v1/images/generation` |

### Xiaomi — `Xiaomi/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 📝 | MiMo Audio | _planned_ | _Pending_ | Audio | `/v1/audio/speech`, `/v1/audio/transcriptions` |

> Upgrade path: 🚧 → 🧪 requires real benchmark output for at least one variant / hardware path. 🧪 → ✅ requires complete evidence for the recipe's claimed primary path. Vision-language recipes use `evalscope` MMMU-class datasets; diffusion recipes report wall-clock per `(size × num_frames × num_inference_steps)` triple.

## Supported model families

The model families below have dedicated SGL-JAX multimodal implementations:

| Architecture family | Recipe coverage |
|---|---|
| Qwen2.5-VL | 🚧 [Qwen2.5-VL recipe](Qwen/Qwen2.5-VL.md) |
| Qwen3-Omni MoE | 📝 planned |
| Wan 2.1 / 2.2 T2V | 🚧 [Wan 2.x T2V recipe](Wan/Wan-2.x.md) |
| MiMo Audio | 📝 planned |

## Multimodal Staging Constraint

Multimodal serving is not plug-and-play for arbitrary HF checkpoints. SGL-JAX must already know how to stage the selected model family. The cookbook therefore lists model-specific TPU and `--tp-size` values instead of asking users to edit source files.

A larger TPU slice is not automatically used by increasing only `--tp-size`; the model's built-in multimodal staging path must support that placement.

Current cookbook-facing configuration summary:

| Family | User-facing TPU config | Cookbook implication |
|---|---|---|
| Qwen2.5-VL 3B | v6e-4, `--tp-size 1` | candidate only until validated |
| Qwen2.5-VL 7B | v6e-4, `--tp-size 1` | candidate only until validated |
| Qwen2.5-VL 32B | v6e-4, `--tp-size 4` | current starter target |
| Qwen2.5-VL 72B | pending | needs a multi-host path and scheduler fix |
| Wan 2.1 | v6e-4, `--tp-size 2` | current starter target |
| Wan 2.2 | v6e-4, `--tp-size 1` | current starter target |
| Qwen3-Omni | planned | expected to need a model-specific staged path |
| MiMo Audio | planned | expected to need a model-specific staged path |

## Existing material (pre-cookbook)

- [`../../mutlimodal/multimodal_usage.md`](../../mutlimodal/multimodal_usage.md) — generic multimodal overview (note: the directory name has a `mutlimodal` typo predating the cookbook reorganisation; fixing the path is out of scope for now).
- [`../../mutlimodal/design/[RFC]multimodal_architechure.md`](../../mutlimodal/design/) — RFC for the multimodal subsystem architecture (developer-facing, not user-facing).

## Shared references

All multimodal recipes reuse the same base material as autoregressive:

- [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) — TPU generation / HBM / topology.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md) — launch flag full table; multimodal-specific flags (`--multimodal`, `--precompile-width-heights`, `--vae-tiling`, `--text-encoder-precisions`, ...) appear when you run `python -m sgl_jax.launch_server --multimodal --help`.
- [`../deployment/`](../deployment/) — launcher templates (single-host Docker, GKE Indexed Job, and advanced SkyPilot v6e experiments).
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.

## Picking a starting recipe to clone for a new multimodal model

| Goal | Clone from |
|---|---|
| Vision-language chat (image / multi-image / video → text) | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) 🚧 |
| Text-to-video diffusion (prompt → MP4) | [`Wan/Wan-2.x.md`](Wan/Wan-2.x.md) 🚧 |
| Audio (TTS / ASR) | Wait for [`Xiaomi/MiMo-Audio.md`](Xiaomi/MiMo-Audio.md) (planned) |
