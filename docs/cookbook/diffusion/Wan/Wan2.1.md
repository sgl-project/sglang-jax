---
title: "Wan 2.1 T2V"
---

# Wan 2.1 Text-to-Video on SGL-JAX

> **Partially validated recipe** — Wan 2.1 14B-Diffusers validated on TPU v6e-4 with sglang-jax 0.1.0: §3.1 basic-usage smoke path (480*832 / 41 frames / default 50 steps, ~4 min wall-clock). Wan models do not have a numeric `evalscope` accuracy benchmark or a `bench_serving` driver — §4 is intentionally narrative. The 1.3B variant remains Starter — same launch path, smaller checkpoint, unmeasured.

## 1. Model Introduction

[**Wan-AI/Wan2.1-T2V**](https://huggingface.co/Wan-AI) is Alibaba's open text-to-video diffusion family. It generates short video clips from a text prompt by running a UMT5 text encoder, a 3D DiT denoiser, and a 3D VAE in sequence. SGL-JAX serves it through the multimodal pipeline (`--multimodal`), exposing the `POST /api/v1/videos/generation` endpoint.

**Variants**:

- [**Wan-AI/Wan2.1-T2V-1.3B-Diffusers**](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) - 1.3B; comfortable single-host fit on v6e-4.
- [**Wan-AI/Wan2.1-T2V-14B-Diffusers**](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) - 14B; current starter path runs on one v6e-4 host with `--tp-size 2`.

**Architectural notes**:

- **Three-stage SGL-JAX pipeline** - Wan 2.1 runs as a `text_encoder` stage (UMT5) producing prompt embeddings, a `diffusion` stage (Wan DiT) doing iterative denoising, and a `vae` stage (AutoencoderKLWan) decoding latents to RGB frames.
- **Built-in staged runtime** - the cookbook exposes only the user-facing model path, TPU slice, and `--tp-size`. The internal stage placement is selected by SGL-JAX.

**Recommended Generation Parameters** (request body, not launch flags):

- `size` - `720*1280` (default), `480*832` (lower-cost), or another precompiled bucket. **Format is `WIDTH*HEIGHT` (asterisk-separated), the same syntax as `--precompile-width-heights`.** Sending `WIDTHxHEIGHT` (lowercase `x`) raises `ValueError: invalid literal for int() with base 10: ...` and crashes the GlobalScheduler — see [§5 Troubleshooting](#5-troubleshooting). Must match a precompiled bucket; see [§2.4 Configuration Tips](#24-configuration-tips).
- `num_frames` - defaults to the model-card recommended count, commonly 41 for Wan 2.1.
- `num_inference_steps` - defaults to the model-card recommended count.
- `seconds` / `fps` - alternative way to specify frame count; the server resolves to `num_frames`.

**License**: see the [Wan-AI model collection](https://huggingface.co/Wan-AI) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Tier | Model | TPU | Topology | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Starter target | Wan 2.1 1.3B / 14B | v6e-4 | 2x2 | 2 | Current built-in staging uses part of the host for text encoding and the rest for video stages |

> Wan 2.1 runs through SGL-JAX's built-in staged multimodal runtime. Use the `--tp-size` shown for this model; moving to a larger TPU host or slice does not automatically make every stage use more devices.

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md) and use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform | Docker Image |
|---|---|
| TPU v5e / v5p / v6e (Trillium) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

No extra pip package is needed for video generation. The response returns a video path or URL, and any post-processing such as MP4 transcoding happens client-side.

### 2.3 Launch

#### Single-host (Docker) - TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --trust-remote-code \
  --tp-size 2 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --precompile-width-heights 720*1280 480*832 \
  --precompile-frame-paddings 41 \
  --vae-precision bf16 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` for the smaller Wan 2.1 variant. Do not reuse this command for Wan 2.2 because its built-in stage layout and `--tp-size` are different.

VAE tiling is enabled by default in the multimodal server (`vae_tiling=True` in `MultimodalServerArgs`) and there is no `--no-vae-tiling` flag to disable it today.

> `--multimodal` is required. Without it, the text-only launcher boots and the `/api/v1/videos/generation` endpoint is not registered.

### 2.4 Configuration Tips

**Precompiled resolutions:**

- `--precompile-width-heights 720*1280 480*832` tells the launcher to JIT-precompile the DiT and VAE for both `1280x720` and `832x480` output sizes. Requests sized outside this list trigger a fresh compilation and may stall other in-flight requests.
- Pin only the resolutions you actually serve. Each entry multiplies JIT cache size and prolongs cold-start compilation.
- Format is `WIDTH*HEIGHT` (asterisk-separated); the launcher validates the format at startup.

**Frame count buckets:**

- `--precompile-frame-paddings 41` precompiles the 41-frame bucket; add additional values such as `--precompile-frame-paddings 41 81` if you serve multiple `num_frames` values.
- Default is `[1]`, which is only correct for image generation. T2V workloads must include the actual frame counts they serve.

**Built-in multimodal staging:**

- Wan 2.1 uses the built-in staged runtime path associated with `--tp-size 2` on v6e-4.
- Do not scale Wan 2.1 by changing only `--tp-size`. Larger TPU slices require SGL-JAX support for a matching staged runtime path.

**VAE precision and tiling:**

- `--vae-precision bf16` is the default for TPU. `fp16` is unsupported on TPU and `fp32` doubles VAE memory with no quality benefit for these checkpoints.
- VAE tiling is always on today. The server decodes VAE output in spatial tiles, which keeps peak HBM bounded for high-resolution or long-frame outputs at a small latency cost.
- `--vae-sp` should be considered only after validating that the chosen Wan path can actually use VAE spatial parallelism.

**Text encoder precision:**

- `--text-encoder-precisions fp32` is the default for UMT5.
- Lowering to `bf16` saves a small amount of HBM but rarely matters compared to the DiT footprint.

**Memory management:**

- `--mem-fraction-static 0.85` on Wan 2.1 14B (v6e-4) leaves room for the DiT activation cache and VAE intermediates.
- If startup or the first request OOMs, lower resolution or frame-count buckets before increasing concurrency.

**Compilation cache hygiene:**

- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` avoids recompiling the same `(resolution x frame-count)` buckets across server restarts.
- The cache keys on full kernel shape. Changing `--precompile-width-heights`, `--precompile-frame-paddings`, `--tp-size`, or `--dit-precision` invalidates cached entries.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md) and the multimodal-specific options (`python -m sgl_jax.launch_server --multimodal --help`).

## 3. Invocation

### 3.1 Basic Video Generation

```bash
curl -X POST http://127.0.0.1:30000/api/v1/videos/generation \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A neon-lit city street after rain, cinematic camera movement",
    "size": "480*832",
    "num_frames": 41
  }'
```

**Output Example:**

```text
{"success": true, "meta_info": {}}
```

The HTTP response only confirms success — it does **not** carry a video URL or path. The server writes the generated MP4 to its **process working directory** (`cwd`) as `<uuid>.mp4`. The corresponding line in the server log looks like `Saved output to <uuid>.mp4`. To collect the file from a remote client, either run the server with a known `cwd` (e.g., `cd /var/sglang-jax-videos && python -m sgl_jax.launch_server ...`) or mount that directory as a shared volume; otherwise locate the file by its UUID through the server log.

Python equivalent:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/api/v1/videos/generation",
    json={
        "prompt": "A neon-lit city street after rain, cinematic camera movement",
        "size": "480*832",
        "num_frames": 41,
    },
    timeout=600,
)
resp.raise_for_status()
print(resp.json())  # {"success": True, "meta_info": {}}
```

### 3.2 Negative Prompt and Resolution Control

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/api/v1/videos/generation",
    json={
        "prompt": "A glass sculpture slowly rotating under studio lights",
        "neg_prompt": "blurry, low quality, watermark, text overlay",
        "size": "720*1280",
        "num_frames": 81,
        "num_inference_steps": 50,
    },
    timeout=900,
)
resp.raise_for_status()
print(resp.json())  # {"success": True, "meta_info": {}} — see §3.1 for where the MP4 lands
```

> `size` must match a precompiled bucket and must use `WIDTH*HEIGHT` (asterisk). Pin every resolution you intend to serve via `--precompile-width-heights` at launch.

### 3.3 Image Generation (T2I)

The same `--multimodal` server also exposes `POST /api/v1/images/generation` for single-frame image generation. The schema is the same minus the frame parameters:

```bash
curl -X POST http://127.0.0.1:30000/api/v1/images/generation \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A quiet mountain observatory at sunrise, watercolor style",
    "size": "1024*1024",
    "n": 1
  }'
```

`size` follows the same `WIDTH*HEIGHT` rule as the videos endpoint. The image is written to the server's process `cwd` (`<uuid>.png`); the response only confirms `success`. See §3.1 for how to locate the file.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Accuracy / Quality

Video generation quality is typically evaluated subjectively (FVD, motion smoothness, prompt fidelity) rather than via a numeric `evalscope` dataset. There is no canonical automated accuracy benchmark for Wan 2.1 in this cookbook today; see the model card for reference quality numbers.

### 4.2 Speed

> **Layout B — methodology + single-shot smoke datapoint.** Latency dominated by `num_inference_steps x DiT step time x num_frames`; benchmark each `(resolution, frame count, step count)` triple separately. One smoke wall-clock recorded; throughput sweeps require a custom driver (`bench_serving` does not cover the videos endpoint).

Video diffusion latency is dominated by `num_inference_steps x DiT step time x num_frames`. Benchmark each `(resolution, frame count, step count)` triple separately.

**Test Environment** - same as the §2.3 launch command for the checkpoint you measure.

**Benchmark Command** - `bench_serving` does not support the videos endpoint today. Use a custom load-test driver that issues `POST /api/v1/videos/generation` at varying concurrency. Report wall-clock per request along with the `(size, num_frames, num_inference_steps)` triple.

**Test Results**

| Build | Variant | Hardware | `size` | `num_frames` | `num_inference_steps` | Concurrency | Wall-clock per request |
|---|---|---|---|---|---|---|---|
| sglang-jax 0.1.0 | Wan2.1-T2V-14B-Diffusers | TPU v6e-4 (TP=2) | `480*832` | 41 | default (50) | 1 | ~4 min 19 s |

This is a single-shot smoke datapoint, not a throughput sweep. Throughput sweeps require the custom driver above.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/v1/videos/generation` returns 404 | `--multimodal` not set at launch | Add `--multimodal` to the launch command. |
| `ValueError: invalid literal for int() with base 10: '480x832'` and `GlobalScheduler hit an exception` (server crashes) | Request body used `WIDTHxHEIGHT` (lowercase `x`); server parses `size` as `WIDTH*HEIGHT` (asterisk, see `multimodal/manager/global_scheduler.py:242-243`) | Send `"size": "480*832"` (asterisk-separated). Also restart the server — the GlobalScheduler exits on this exception. |
| Response body is `{"success": true, "meta_info": {}}` with no `path`/`url` | Expected behavior — the videos / images endpoints persist the file to the server's `cwd` and only acknowledge success in the response | Locate the MP4 by the `Saved output to <uuid>.mp4` line in the server log, or run the server with a known `cwd` and mount that directory. |
| First request blocks on every new resolution | Resolution not in `--precompile-width-heights` | Add the size you serve (`WIDTH*HEIGHT`) to `--precompile-width-heights` and relaunch. |
| First request blocks on every new frame count | Frame count not in `--precompile-frame-paddings` | Add the frame count to `--precompile-frame-paddings` and relaunch. |
| OOM during VAE decode at high resolution | Full VAE decode too large for HBM despite tiling | Lower the request `size` or split the request; VAE tiling is already on by default. |
| Video response `path` not accessible to client | Server-written file lives on TPU host only | Mount a shared output volume and serve via a separate file server, or stream the file back from the TPU host. |
| Slow throughput at moderate concurrency | DiT is the bottleneck | Lower request `num_inference_steps` to trade quality for throughput. |

## Additional Resources

- [Wan-AI model collection](https://huggingface.co/Wan-AI)
- [Wan2.1-T2V-14B-Diffusers model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- [`Wan2.2.md`](Wan2.2.md) - companion Wan 2.2 recipe.
- [`../../autoregressive/Qwen/Qwen2.5-VL.md`](../../autoregressive/Qwen/Qwen2.5-VL.md) - companion vision-language autoregressive recipe.
