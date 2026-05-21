---
title: "Multimodal Models"
---

# Multimodal Recipes

End-to-end serving recipes for non-text-only models served by SGL-JAX: vision-language, text-to-video, and audio. All multimodal recipes launch with `--multimodal`, which enables the staged scheduler (text encoder / ViT / DiT / VAE / AR) and the diffusion-style HTTP endpoints (`/api/v1/videos/generation`, `/api/v1/images/generation`) in addition to OpenAI-compatible `/v1/chat/completions`.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — empirically tuned on hardware with reference numbers in §4 |
| 🚧 | **Starter** — launch command derived from HF model card; not yet measured. PR back tested numbers to upgrade to ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet |

## Recipes by vendor

### Qwen — `Qwen/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 🚧 | Qwen2.5-VL (3B / 7B / 32B / 72B) | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) | v6e-4 (3B/7B) / v6e-32 (72B) | Vision-language | `/v1/chat/completions` (image_url + video_url content blocks) |
| 📝 | Qwen3-Omni MoE | _planned_ | _Pending_ | Vision + audio + text | `/v1/chat/completions` (multimodal content blocks) |

### Wan-AI — `Wan/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 🚧 | Wan 2.1 / 2.2 T2V (1.3B / 14B / A14B) | [`Wan/Wan-2.x.md`](Wan/Wan-2.x.md) | v6e-4 (1.3B/14B) / v6e-8 (A14B) | Text-to-video diffusion | `/api/v1/videos/generation`, `/api/v1/images/generation` |

### Xiaomi — `Xiaomi/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| 📝 | MiMo Audio | _planned_ | _Pending_ | Audio | `/v1/audio/speech`, `/v1/audio/transcriptions` |

> Upgrade path: 🚧 → ✅ requires real benchmark output in §4. Vision-language recipes use `evalscope` MMMU-class datasets; diffusion recipes report wall-clock per `(size × num_frames × num_inference_steps)` triple.

## Supported architectures (per runtime registry)

The four model families above match what's registered under `python/sgl_jax/srt/multimodal/models/`:

| Architecture family | Recipe coverage |
|---|---|
| `qwen2_5VL` (Qwen2.5-VL) | ✅ [Qwen2.5-VL recipe](Qwen/Qwen2.5-VL.md) |
| `qwen3_omni_moe` (Qwen3-Omni MoE) | 📝 planned |
| `wan` (Wan 2.1 / 2.2 T2V — dual transformer in 2.2) | ✅ [Wan 2.x T2V recipe](Wan/Wan-2.x.md) |
| `mimo_audio` | 📝 planned |

## Existing material (pre-cookbook)

- [`../../mutlimodal/multimodal_usage.md`](../../mutlimodal/multimodal_usage.md) — generic multimodal overview (note: the directory name has a `mutlimodal` typo predating the cookbook reorganisation; fixing the path is out of scope for now).
- [`../../mutlimodal/design/[RFC]multimodal_architechure.md`](../../mutlimodal/design/) — RFC for the multimodal subsystem architecture (developer-facing, not user-facing).

## Shared references

All multimodal recipes reuse the same base material as autoregressive:

- [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) — TPU generation / HBM / topology.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md) — launch flag full table; multimodal-specific flags (`--multimodal`, `--precompile-width-heights`, `--vae-tiling`, `--text-encoder-precisions`, ...) appear when you run `python -m sgl_jax.launch_server --multimodal --help`.
- [`../deployment/`](../deployment/) — launcher templates (single-host Docker, GKE Indexed Job, SkyPilot).
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.

## Picking a starting recipe to clone for a new multimodal model

| Goal | Clone from |
|---|---|
| Vision-language chat (image / multi-image / video → text) | [`Qwen/Qwen2.5-VL.md`](Qwen/Qwen2.5-VL.md) 🚧 |
| Text-to-video diffusion (prompt → MP4) | [`Wan/Wan-2.x.md`](Wan/Wan-2.x.md) 🚧 |
| Audio (TTS / ASR) | Wait for [`Xiaomi/MiMo-Audio.md`](Xiaomi/MiMo-Audio.md) (planned) |
