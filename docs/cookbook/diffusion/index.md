---
title: "Diffusion Models"
---

# Diffusion Model Recipes

End-to-end serving recipes for diffusion-style image and video generation models on SGL-JAX. These models do not decode text one token at a time; they generate media by iterative denoising, usually behind `/api/v1/images/generation` or `/api/v1/videos/generation`.

## Status legend

| Emoji | Meaning |
|---|---|
| ✅ | **Validated** — primary model / hardware path empirically tuned with reference benchmark numbers in §4 |
| 🧪 | **Partially validated** — at least one variant / hardware path has real benchmark output; other variants, matrix cells, or current-build reruns are still pending |
| 🚧 | **Starter** — launch command derived from HF model card; not yet measured. PR back tested numbers to upgrade to 🧪 or ✅ |
| 📝 | **Planned** — architecture supported by the runtime but no recipe yet (or model release pending) |
| 🚫 | **Blocked** — runnable path blocked by an upstream weight format / HBM / runtime constraint; banner cites the root cause and unblocking plan |

## Recipes by vendor

### Wan-AI — `Wan/`

| Status | Model | Recipe | Min TPU | Modality | Endpoint |
|---|---|---|---|---|---|
| ✅ | Wan 2.1 T2V-14B | [`Wan/Wan2.1.md`](Wan/Wan2.1.md) | v6e-4 | Text-to-video diffusion | `/api/v1/videos/generation`, `/api/v1/images/generation` |
| ✅ | Wan 2.2 T2V A14B | [`Wan/Wan2.2.md`](Wan/Wan2.2.md) | v6e-4 | Text-to-video diffusion | `/api/v1/videos/generation`, `/api/v1/images/generation` |

> Upgrade path: 🚧 → 🧪 requires measured launch / quality / wall-clock output for at least one variant and hardware path. 🧪 → ✅ requires complete evidence for the recipe's claimed primary path.

## What "diffusion" means here

A diffusion recipe covers models where the expensive generation loop is denoising, not autoregressive token decoding. For Wan, SGL-JAX stages a text encoder, a DiT denoiser, and a VAE decoder; the endpoint returns an image or video artifact rather than chat tokens.

Autoregressive text and vision-language decoders, including Qwen2.5-VL, live in [`../autoregressive/`](../autoregressive/index.md).

## Built-in Staging Constraint

Diffusion serving is constrained by SGL-JAX's built-in staged runtime. The cookbook lists model-specific TPU and `--tp-size` values instead of asking users to edit source files.

A larger TPU slice is not automatically used by increasing only `--tp-size`; the model's built-in staged path must support that placement.

Current cookbook-facing configuration summary:

| Family | User-facing TPU config | Cookbook implication |
|---|---|---|
| Wan 2.1 | v6e-4, `--tp-size 2` | validated path (T2V-14B) |
| Wan 2.2 | v6e-4, `--tp-size 1` | validated path (T2V A14B) |

## Shared references

- [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) — TPU generation / HBM / topology.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md) — launch flag full table; diffusion-specific flags appear when you run `python -m sgl_jax.launch_server --multimodal --help`.
- [`../deployment/`](../deployment/) — launcher templates (single-host Docker, GKE Indexed Job, and advanced SkyPilot v6e experiments).
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.

## Picking a starting recipe to clone

| Goal | Clone from |
|---|---|
| Text-to-video diffusion with Wan 2.1 | [`Wan/Wan2.1.md`](Wan/Wan2.1.md) ✅ |
| Text-to-video diffusion with Wan 2.2 | [`Wan/Wan2.2.md`](Wan/Wan2.2.md) ✅ |
| Text-to-image diffusion (prompt → image) | Use [`Wan/Wan2.1.md`](Wan/Wan2.1.md) or [`Wan/Wan2.2.md`](Wan/Wan2.2.md) as the current staged runtime pattern, then replace the model path and endpoint examples as needed |
