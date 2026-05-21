---
title: "Wan 2.x T2V"
---

# Wan 2.x Text-to-Video on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card and the SGL-JAX multimodal pipeline; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**Wan-AI/Wan2.x-T2V**](https://huggingface.co/Wan-AI) is Alibaba's open text-to-video diffusion family — generates short video clips from a text prompt by running a UMT5 text encoder, a 3D DiT (Diffusion Transformer) denoiser, and a 3D VAE in sequence. SGL-JAX serves it through the multimodal pipeline (`--multimodal`), exposing the `POST /api/v1/videos/generation` endpoint.

**Variants** (pick by size and noise-stage architecture):

- [**Wan-AI/Wan2.1-T2V-1.3B-Diffusers**](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) — 1.3B; comfortable single-host fit on v6e-4. Default starter target.
- [**Wan-AI/Wan2.1-T2V-14B-Diffusers**](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) — 14B; single-host on v6e-4 with reduced batch, or v6e-8 for headroom.
- [**Wan-AI/Wan2.2-T2V-A14B-Diffusers**](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) — 14B "MoE-style" dual-transformer (high-noise + low-noise experts run at different denoising stages); needs more HBM than the 2.1-14B for the second transformer.

**Architectural notes**:

- **Three-stage SGL-JAX pipeline** — Wan 2.x runs as a `text_encoder` stage (UMT5) producing prompt embeddings + a `diffusion` stage (Wan DiT) doing iterative denoising + a `vae` stage (AutoencoderKLWan) decoding latents to RGB frames. Each stage is scheduled independently and overlapped where possible.
- **Wan 2.2 dual-transformer** — high-noise transformer denoises the early/coarse stages; low-noise transformer takes over for the late/fine stages. Both load into HBM at startup — plan capacity accordingly.

**Recommended Generation Parameters** (request body, not launch flags):

- `size` — `720x1280` (default), `480x832` (lower-cost), or `1024x576` etc. Must match a precompiled bucket — see [§2.4 Precompiled resolutions](#24-configuration-tips).
- `num_frames` — defaults to model card recommended count (e.g. 41 for Wan 2.1).
- `num_inference_steps` — defaults to model card recommended count.
- `seconds` / `fps` — alternative way to specify frame count; the server resolves to `num_frames`.

**License**: see the [Wan-AI model collection](https://huggingface.co/Wan-AI) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | Wan 2.1 1.3B | v6e-4 | 2x2 | 4 | 4 | Smallest model; text encoder + DiT + VAE fit comfortably |
| Recommended production | Wan 2.1 14B | v6e-4 | 2x2 | 4 | 4 | Fits with `--mem-fraction-static 0.85`; lower `--precompile-width-heights` count to keep cache size manageable |
| Recommended production | Wan 2.2 A14B | v6e-8 | 2x4 | 8 | 8 | Dual transformers ≈ 2× DiT HBM vs Wan 2.1 14B; v6e-8 leaves headroom |

> Wan 2.x runs as three SGL-JAX stages (text encoder + DiT + VAE). The bundled stage YAMLs already pin `runtime.num_tpus` per stage — the starter command does not need a custom stage config.

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md) and use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

No extra pip is needed for video generation — the response returns a video URL/path, and any post-processing (ffmpeg, MP4 transcoding) happens client-side.

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Wan 2.1 14B)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --precompile-width-heights 720*1280 480*832 \
  --precompile-frame-paddings 41 \
  --vae-precision bf16 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` for the smaller variant (raise `--mem-fraction-static` to 0.88) or `Wan-AI/Wan2.2-T2V-A14B-Diffusers` for the dual-transformer variant (lower to 0.8 on v6e-4, or use v6e-8 below).

#### Single-host (Docker) — TPU v6e-8 (Wan 2.2 A14B, recommended)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --trust-remote-code \
  --tp-size 8 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --precompile-width-heights 720*1280 480*832 \
  --precompile-frame-paddings 41 \
  --vae-precision bf16 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

VAE tiling is enabled by default in the multimodal server (`vae_tiling=True` in `MultimodalServerArgs`) and there is no `--no-vae-tiling` flag to disable it today — tiling stays on for all Wan workloads regardless of how this launch is invoked.

> `--multimodal` is required — without it, the text-only launcher boots and the `/api/v1/videos/generation` endpoint is not registered.

### 2.4 Configuration Tips

**Precompiled resolutions:**
- `--precompile-width-heights 720*1280 480*832` tells the launcher to JIT-precompile the DiT and VAE for both `1280×720` and `832×480` output sizes. Requests sized outside this list will trigger a fresh ~4 min compilation and may stall other in-flight requests.
- Pin **only** the resolutions you actually serve — each entry multiplies the JIT cache size and prolongs cold-start compilation.
- Format is `WIDTH*HEIGHT` (asterisk-separated); the launcher validates the format at startup.

**Frame count buckets:**
- `--precompile-frame-paddings 41` precompiles the 41-frame bucket; add additional values (e.g. `--precompile-frame-paddings 41 81`) if you serve multiple `num_frames` values.
- Default is `[1]`, which is **only correct for image generation** — T2V workloads must include the actual frame counts they serve, otherwise the first video request stalls on per-frame-count JIT compilation.

**VAE Precision and Tiling:**
- `--vae-precision bf16` is the default for TPU; `fp16` is unsupported on TPU and `fp32` doubles VAE memory with no quality benefit for these checkpoints.
- **VAE tiling is always on** — `MultimodalServerArgs.vae_tiling` defaults to `True` and there is no CLI toggle to disable it today (no `--no-vae-tiling` flag exists). The server always decodes VAE output in spatial tiles, which keeps peak HBM bounded for high-resolution / long-frame outputs at a small latency cost. Disabling would require source-code patching.
- `--vae-sp` enables VAE spatial parallelism across `--tp-size` devices — useful only when VAE is the bottleneck (rarely; the DiT dominates).

**Text Encoder Precision:**
- `--text-encoder-precisions fp32` (default) for UMT5 — text encoder is small, fp32 keeps prompt embedding quality high.
- Lowering to `bf16` saves a small amount of HBM but rarely matters compared to the DiT footprint.

**Memory Management:**
- `--mem-fraction-static 0.85` on Wan 2.1 14B (v6e-4) leaves room for the DiT activation cache and VAE intermediates.
- Wan 2.2 A14B's dual transformer **doubles** the DiT weight footprint vs Wan 2.1 14B — use v6e-8 or accept lower `--mem-fraction-static 0.8` on v6e-4.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min **per (resolution × frame-count) bucket** while XLA/Pallas compiles the DiT and VAE kernels.
- The cache keys on full kernel shape: changing `--precompile-width-heights`, `--precompile-frame-paddings`, `--tp-size`, or `--dit-precision` invalidates cached entries. Give each tuning experiment its own cache dir.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md) and the multimodal-specific options (run `python -m sgl_jax.launch_server --multimodal --help`).

## 3. Invocation

### 3.1 Basic Video Generation

```bash
curl -X POST http://127.0.0.1:30000/api/v1/videos/generation \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A curious raccoon peeks through a wooden fence into a sunny garden",
    "size": "480x832",
    "num_frames": 41
  }'
```

**Output Example:**

```text
{
  "id": "vid_a1b2c3d4...",
  "path": "/tmp/sglang-jax-videos/vid_a1b2c3d4.mp4"
}
```

The server writes the generated MP4 to disk and returns the `path`. Open the file directly, or mount the output directory as a shared volume if the client runs on a different host.

Python equivalent:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/api/v1/videos/generation",
    json={
        "prompt": "A curious raccoon peeks through a wooden fence into a sunny garden",
        "size": "480x832",
        "num_frames": 41,
    },
    timeout=600,  # diffusion takes much longer than text generation
)
resp.raise_for_status()
print(resp.json())
```

### 3.2 Negative Prompt and Resolution Control

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/api/v1/videos/generation",
    json={
        "prompt": "An astronaut riding a horse on the moon, cinematic lighting",
        "neg_prompt": "blurry, low quality, watermark, text overlay",
        "size": "720x1280",
        "num_frames": 81,
        "num_inference_steps": 50,
    },
    timeout=900,
)
resp.raise_for_status()
result = resp.json()
print(f"Generated video: {result['path']}")
```

> **`size` must match a precompiled bucket** — pin every resolution you intend to serve via `--precompile-width-heights` at launch (see §2.4). Off-bucket requests trigger a fresh ~4 min JIT compilation.

### 3.3 Image Generation (T2I)

The same `--multimodal` server also exposes `POST /api/v1/images/generation` for single-frame image generation. The schema is the same minus the frame parameters:

```bash
curl -X POST http://127.0.0.1:30000/api/v1/images/generation \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene Japanese garden in autumn, watercolor style",
    "size": "1024x1024",
    "n": 1
  }'
```

**Output Example:**

```text
{
  "id": "img_a1b2c3d4...",
  "url": "http://127.0.0.1:30000/static/img_a1b2c3d4.png"
}
```

For T2I-only deployments, the same launch command works — Wan can render single frames (`num_frames=1`) via the videos endpoint, or you can serve a dedicated image-only diffusion checkpoint through the same `--multimodal` flag.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Accuracy / Quality

Video generation quality is typically evaluated subjectively (FVD, motion smoothness, prompt fidelity) rather than via a numeric `evalscope` dataset. There is no canonical automated accuracy benchmark for Wan 2.x in this cookbook today — see the model card for the reference quality numbers.

### 4.2 Speed

> **Layout B — methodology + command template.** Video diffusion latency is dominated by `num_inference_steps × DiT step time × num_frames`; benchmark each (resolution, frame count, step count) triple separately.

**Test Environment** — same as the §2.3 launch command for the checkpoint you measure.

**Benchmark Command** — `bench_serving` does not support the videos endpoint today; use a custom load-test driver that issues `POST /api/v1/videos/generation` at varying concurrency. Report wall-clock per request along with the `(size, num_frames, num_inference_steps)` triple. PR back the raw wall-clock numbers.

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/v1/videos/generation` returns 404 | `--multimodal` not set at launch | Add `--multimodal` to the launch command. |
| First request blocks ~4 min on every new resolution | Resolution not in `--precompile-width-heights` | Add the size you serve (`WIDTH*HEIGHT`) to `--precompile-width-heights` and relaunch; subsequent requests at that size will be cache hits. |
| First request blocks ~4 min on every new frame count | Frame count not in `--precompile-frame-paddings` | Add the frame count to `--precompile-frame-paddings` and relaunch. |
| OOM during VAE decode at high resolution | Full VAE decode too large for HBM despite tiling | Lower the request `size` or split the request; VAE tiling is already on by default. Tile size is not currently CLI-tunable — patch the source if you need to shrink tiles further. |
| Wan 2.2 A14B OOM at startup on v6e-4 | Dual transformer doubles DiT footprint | Move to v6e-8, or lower `--mem-fraction-static` to 0.8 on v6e-4 (accept lower concurrency). |
| Video response `path` not accessible to client | Server-written file lives on TPU host only | Mount a shared output volume (`-v /shared/videos:/tmp/sglang-jax-videos` in `docker run`) and serve via a separate file server, or stream the file back from the TPU host. |
| Slow throughput at moderate concurrency | DiT is the bottleneck; concurrency doesn't help if each step saturates compute | Lower request `num_inference_steps` to trade quality for throughput; the DiT stage cannot batch arbitrary concurrent requests at this scale. |

## Additional Resources

- [Wan-AI model collection](https://huggingface.co/Wan-AI)
- [Wan 2.1 / 2.2 paper / blog](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (model card links to the upstream blog and tech report)
- [`Qwen/Qwen2.5-VL.md`](../Qwen/Qwen2.5-VL.md) — companion vision-language recipe (image-in, text-out).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
