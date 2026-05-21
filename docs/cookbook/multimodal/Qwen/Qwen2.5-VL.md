---
title: "Qwen2.5-VL"
---

# Qwen2.5-VL on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card and the SGL-JAX multimodal pipeline; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**Qwen/Qwen2.5-VL**](https://huggingface.co/Qwen) is Alibaba's second-generation Qwen vision-language family — multimodal decoders that ingest images / video frames and emit text, with the same chat interface as text-only Qwen2.5. SGL-JAX serves it through the multimodal pipeline (`--multimodal`), which runs a separate ViT stage that produces vision embeddings and an autoregressive stage that does prefill/decode.

**Variants** (pick by size):

- [**Qwen/Qwen2.5-VL-3B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B parameters; comfortable v6e-4 single-host fit.
- [**Qwen/Qwen2.5-VL-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — 7B; single-host on v6e-4.
- [**Qwen/Qwen2.5-VL-32B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) — 32B; single-host on v6e-4 with reduced `--mem-fraction-static`, or v6e-8 for headroom.
- [**Qwen/Qwen2.5-VL-72B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) — 72B; multi-host on v6e-32.

For the text-only Qwen3 dense recipes see [`../../autoregressive/Qwen/Qwen3.md`](../../autoregressive/Qwen/Qwen3.md).

**Key Features**:

- **Multi-image and video input** — single chat request can mix any number of `image_url` and `video_url` content blocks alongside the text prompt; the OpenAI Vision API schema is used directly.
- **Long-context VL** — supports the underlying Qwen2.5 32K context window (extendable to 128K with rope scaling on supported checkpoints).
- **Instruction-tuned** — default chat behaviour; no per-request `enable_thinking` toggle (Qwen2.5 is non-reasoning; for reasoning use Qwen3).
- **Two-stage SGL-JAX pipeline** — Qwen2.5-VL runs as a `vit` scheduler stage producing vision embeddings + an `auto_regressive` stage doing prefill/decode; `--multimodal` enables both.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024` (verify defaults against each variant's model card).

**License**: see the [Qwen model cards](https://huggingface.co/Qwen) for the authoritative Tongyi Qianwen License terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Tier | Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|---|
| Minimum runnable | Qwen2.5-VL-3B / 7B | v6e-4 | 2x2 | 1 | 4 | 4 | BF16 ~6 GB (3B) / ~16 GB (7B) — comfortable headroom for ViT + AR stages |
| Recommended production | Qwen2.5-VL-32B | v6e-4 | 2x2 | 1 | 4 | 4 | BF16 ~64 GB — fits with `--mem-fraction-static 0.8`; raise to v6e-8 for higher concurrency |
| Recommended production | Qwen2.5-VL-72B | v6e-32 | 4x8 | 8 | 32 | 32 | BF16 ~140 GB — multi-host required; aligns with the [`Llama3.3-70B.md`](../../autoregressive/Llama/Llama3.3-70B.md) deployment pattern |

> Qwen2.5-VL runs as two SGL-JAX stages (ViT + AR). Both stages share the same TPU slice — there is no separate flag to size them independently in the starter form; stage CPU/TPU allocation comes from the bundled stage config (see §2.4 Stage Configuration).

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For single-host VL launches use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md); for the 72B multi-host config use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) or [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

Extra pip for accuracy benchmarking only:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Qwen2.5-VL-7B-Instruct)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `Qwen/Qwen2.5-VL-3B-Instruct` (raise `--mem-fraction-static` to 0.88) or `Qwen/Qwen2.5-VL-32B-Instruct` (lower to 0.8, lower `--max-running-requests` to 32).

> `--multimodal` is required — without it, the launcher boots the text-only HTTP server which has no ViT stage and cannot consume `image_url` / `video_url` content blocks.

#### Multi-host (GKE 或 SkyPilot) — TPU v6e-32 (Qwen2.5-VL-72B-Instruct)

The launch command is the same on every node — only `${NODE_RANK}` and `${MASTER_ADDR}` vary. `${NODE_RANK}` ranges from `0` to `7`.

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Qwen/Qwen2.5-VL-72B-Instruct \
  --trust-remote-code \
  --tp-size 32 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup \
  --nnodes 8 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

**Launcher** — wrap the above into either:

- **GKE Indexed Job + headless Service** — adapt [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md). Differences from the template: `<JOB>=qwen25-vl-72b`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `<N>=8`, plus the launch flags above. `${NODE_RANK}` comes from `${JOB_COMPLETION_INDEX}`.
- **SkyPilot** — adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with `tpu-v6e-32` accelerator. `${NODE_RANK}` comes from `${SKYPILOT_NODE_RANK}`.

For an end-to-end GKE manifest with the same template applied, see [`../../autoregressive/Xiaomi/MiMo-V2.5-Pro.md` §2.3 Multi-host](../../autoregressive/Xiaomi/MiMo-V2.5-Pro.md#23-launch) — substitute the model path, the `--multimodal` flag, and the TP from above.

### 2.4 Configuration Tips

**Memory Management:**
- VL workloads use HBM for both KV cache **and** vision embeddings (ViT output). The 32B/72B variants run with lower `--mem-fraction-static` (0.8 / 0.9) than the equivalent text-only Qwen3 to leave room for vision tensors that scale with input image count.
- Lower `--max-running-requests` (64 for VL vs 256 for text-only Qwen3 at the same size) — each VL request can carry multiple high-resolution images that explode KV demand. Raise only if you measure HBM headroom.

**Stage Configuration:**
- Qwen2.5-VL ships a bundled 2-stage config (ViT + autoregressive). The default config is loaded automatically from the multimodal `static_configs` directory when you pass `--multimodal --model-path Qwen/Qwen2.5-VL-...`.
- Custom stage YAMLs can override per-stage `num_tpus` and `tp_size`. For the starter command you do not need to author a stage YAML — the bundled default works.

**Chunked Prefill (image embeddings):**
- `--chunked-prefill-size 2048` bounds peak HBM during prefill. Vision-language prefills include both text tokens and vision embeddings — raising this past 4096 risks prefill-time OOM on 32B/72B variants.

**Multimodal Attention Backend:**
- The vision-language attention path runs on the default `--attention-backend fa` (FlashAttention on Pallas) — no override needed.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per stage while XLA/Pallas re-compiles every kernel (ViT and AR each compile independently).
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, image resolution buckets, or `--context-length` invalidates cached entries.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md) and the multimodal-specific options in the multimodal `ServerArgs` (run `python -m sgl_jax.launch_server --multimodal --help` to see).

## 3. Invocation

### 3.1 Basic Chat Completion (text only)

Qwen2.5-VL accepts plain-text requests on the same OpenAI-compatible `/v1/chat/completions` endpoint — useful for sanity-checking the server before sending images:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(resp.choices[0].message.content)
```

### 3.2 Multimodal Input

Vision-language input uses the OpenAI Vision API schema — each `messages[i].content` is a **list** of content blocks mixing `image_url`, `video_url`, and `text`. SGL-JAX accepts both `https://` URLs and local files via the `file://` protocol.

#### Single Image

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"},
            },
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

**Output Example:**

```text
A wooden boardwalk stretches through a vibrant green wetland under a clear blue sky with scattered clouds.
```

#### Multi-Image

Stack multiple `image_url` blocks into the same `content` list followed by a single text prompt:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}},
            {"type": "text", "text": "Compare these two images and describe what changed in 50 words or less."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

**Output Example:**

```text
The 'before' shot shows an empty workshop floor; the 'after' shot shows the same space populated with assembled chairs and a long workbench, suggesting the workspace was cleaned, organized, and turned into a production area.
```

#### Video

Use a `video_url` content block — same schema as `image_url`. The server samples frames from the video and feeds them through the ViT stage:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
            {"type": "text", "text": "Describe what happens in this video in 3 bullet points."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

**Output Example:**

```text
- A small group of seagulls gathers on a wet rocky beach at the water's edge.
- A wave rolls in and partially submerges the rocks, scattering the birds briefly.
- The birds return and resume foraging in the shallow tide as the wave recedes.
```

> **Long video / large image set:** Make sure `--context-length` is large enough to fit the vision token count plus the text prompt and response. Each high-resolution image and each sampled video frame contributes a non-trivial number of vision tokens to the prefill.

> Qwen2.5-VL is non-reasoning (no `<think>` blocks) and does not ship a native tool-call format. For reasoning workloads use [Qwen3](../../autoregressive/Qwen/Qwen3.md); for tool-calling workloads use a model with `--tool-call-parser` support (see [`../../autoregressive/Qwen/Qwen3.md` §3.3](../../autoregressive/Qwen/Qwen3.md#33-tool-calling)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Accuracy — MMMU

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | Qwen/Qwen2.5-VL-7B-Instruct (BF16) |
| Tensor Parallelism | 4 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3 Single-host](#single-host-docker--tpu-v6e-4-qwen25-vl-7b-instruct).

**Benchmark Command**

```bash
evalscope eval \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets mmmu \
  --eval-batch-size 4
```

Recommended additional vision datasets: **MMMU Pro Vision**, **DocVQA**, **ChartQA**.

**Test Results** — _Pending. Run and PR back._

### 4.2 Speed

> **Layout B — methodology + command template.** Vision-language speed depends heavily on image resolution and frame count per request, so picking representative workloads is more important than a single number.

**Test Environment** — same as §4.1.

**Deployment Command** — same as [§2.3 Single-host](#single-host-docker--tpu-v6e-4-qwen25-vl-7b-instruct).

**Benchmark Command** — `bench_serving` does not have native multimodal input support today; use a custom OpenAI-client load test that sends the §3.2 multi-image / video patterns at varying concurrency. PR back full TTFT / ITL / output tok/s numbers along with the image resolution and prompt template used.

**Test Results** — _Pending._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Server rejects `image_url` / `video_url` content blocks | `--multimodal` not set | Add `--multimodal` to the launch command; the text-only server cannot parse vision content blocks. |
| OOM at startup (32B / 72B) | Default `--mem-fraction-static` too high once ViT stage weights load | Lower to 0.8 (32B single-host) or 0.85 (72B multi-host); verify the bundled stage config did not bump per-stage `num_tpus` past your chip count. |
| OOM at request-time on multi-image input | Vision embeddings push KV demand past budget at chosen `--max-running-requests` | Lower `--max-running-requests` to 16 or 8; raise `--context-length` if available to spread KV per request. |
| First request takes 6+ min (vs 4 min text-only) | Both ViT and AR stages JIT-compile separately on first request | Persist `JAX_COMPILATION_CACHE_DIR` across restarts — both stages share the same cache dir. |
| Video URL works in browser but server rejects it | URL not directly fetchable from the TPU host (auth / region / firewall) | Stage the video on a mounted volume and pass `file:///path/to/video.mp4` instead, or use a publicly reachable URL. |

## Additional Resources

- [Qwen2.5-VL model collection](https://huggingface.co/Qwen)
- [`../../autoregressive/Qwen/Qwen3.md`](../../autoregressive/Qwen/Qwen3.md) — text-only Qwen3 dense recipe (Qwen3 series is the reasoning generation).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
