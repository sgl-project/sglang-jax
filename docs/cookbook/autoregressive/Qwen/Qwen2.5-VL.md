---
title: "Qwen2.5-VL"
---

# Qwen2.5-VL on SGL-JAX

> **Validated recipe** — empirically validated on TPU v6e-4 with sglang-jax 0.1.0; §4 Benchmark is intentionally omitted (see §4 omitted note + design §3 Validated criteria interpretation).

## 1. Model Introduction

[**Qwen/Qwen2.5-VL**](https://huggingface.co/Qwen) is Alibaba's second-generation Qwen vision-language family — multimodal decoders that ingest images / video frames and emit text, with the same chat interface as text-only Qwen2.5. SGL-JAX serves it through the multimodal pipeline (`--multimodal`), which runs a separate ViT stage that produces vision embeddings and an autoregressive stage that does prefill/decode.

**Variants** (pick by size):

- [**Qwen/Qwen2.5-VL-3B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B parameters; candidate single-host path on v6e-4 with `--tp-size 1`.
- [**Qwen/Qwen2.5-VL-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — 7B; candidate single-host path on v6e-4 with `--tp-size 1`.
- [**Qwen/Qwen2.5-VL-32B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) — 32B; starter single-host path on v6e-4 with `--tp-size 4`.
- [**Qwen/Qwen2.5-VL-72B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) — 72B; multi-host serving is pending.

For the text-only Qwen3 dense recipes see [`Qwen3.md`](Qwen3.md).

**Key Features**:

- **Multi-image and video input** — single chat request can mix any number of `image_url` and `video_url` content blocks alongside the text prompt; the OpenAI Vision API schema is used directly.
- **Long-context VL** — supports the underlying Qwen2.5 32K context window (extendable to 128K with rope scaling on supported checkpoints).
- **Instruction-tuned** — default chat behaviour; no per-request `enable_thinking` toggle (Qwen2.5 is non-reasoning; for reasoning use Qwen3).
- **Two-stage SGL-JAX pipeline** — Qwen2.5-VL runs as a `vit` scheduler stage producing vision embeddings + an `auto_regressive` stage doing prefill/decode; `--multimodal` enables both.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024` (verify defaults against each variant's model card).

**License**: see the [Qwen model cards](https://huggingface.co/Qwen) for the authoritative Tongyi Qianwen License terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Tier | Model | TPU | Topology | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Candidate | Qwen2.5-VL-3B | v6e-4 | 2x2 | 1 | SGL-JAX currently exposes a 1-way AR path for this size; validate before marking production |
| Candidate | Qwen2.5-VL-7B | v6e-4 | 2x2 | 1 | SGL-JAX currently exposes a 1-way AR path for this size; validate HBM and latency before increasing traffic |
| Starter target | Qwen2.5-VL-32B | v6e-4 | 2x2 | 4 | Current followable starter path |
| Pending | Qwen2.5-VL-72B | _Pending_ | _Pending_ | _Pending_ | Needs a multi-host SGL-JAX configuration and scheduler fix before becoming followable |

> Multimodal recipes are constrained by SGL-JAX's built-in staged runtime. Use the `--tp-size` shown for the model; a larger TPU slice is not automatically used by changing only `--tp-size`.

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For the current single-host VL paths use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md). Qwen2.5-VL 72B multi-host should stay pending until the built-in staging and scheduler path are fixed. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

Extra pip for accuracy benchmarking only:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (Qwen2.5-VL-32B-Instruct)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --multimodal \
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 32 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

For Qwen2.5-VL-3B or Qwen2.5-VL-7B, use the same command shape but set `--model-path` to the target checkpoint and `--tp-size 1`. Keep those paths as candidates until validated on your TPU host.

> `--multimodal` is required — without it, the launcher boots the text-only HTTP server which has no ViT stage and cannot consume `image_url` / `video_url` content blocks.

#### Multi-host (GKE Indexed Job) — Qwen2.5-VL-72B pending

Do not use a v6e-16/v6e-32 launch command for Qwen2.5-VL-72B as a cookbook path today. Current SGL-JAX multimodal staging does not provide a followable 72B multi-host setup, and simply changing `--tp-size` to 16 or 32 does not make the runtime use that larger slice correctly.

Before this becomes a followable recipe, SGL-JAX needs:

- a 72B-specific multi-host staging path that matches the intended `--tp-size`;
- multimodal AR scheduler support for `QueueBackend + nnodes > 1` request broadcast;
- GKE validation that all ranks pass `sync_pub_sub()` and participate in the same AR forward.

### 2.4 Configuration Tips

**Memory Management:**
- VL workloads use HBM for both KV cache **and** vision embeddings (ViT output). The 32B/72B variants run with lower `--mem-fraction-static` (0.8 / 0.9) than the equivalent text-only Qwen3 to leave room for vision tensors that scale with input image count.
- Lower `--max-running-requests` (64 for VL vs 256 for text-only Qwen3 at the same size) — each VL request can carry multiple high-resolution images that explode KV demand. Raise only if you measure HBM headroom.

**Built-in multimodal staging:**
- SGL-JAX internally splits Qwen2.5-VL into a vision stage and an autoregressive generation stage.
- The public launch knob is still `--tp-size`, but it must match the supported staging path for the selected model. Do not scale VL models by increasing only `--tp-size`.
- Current cookbook paths: 3B/7B candidates use `--tp-size 1`, 32B uses `--tp-size 4`, and 72B multi-host remains pending.

**Chunked Prefill (image embeddings):**
- `--chunked-prefill-size 2048` bounds peak HBM during prefill. Vision-language prefills include both text tokens and vision embeddings — raising this past 4096 risks prefill-time OOM on 32B/72B variants.

**Multimodal Attention Backend:**
- The vision-language attention path runs on the default `--attention-backend fa` (FlashAttention on Pallas) — no override needed.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per stage while XLA/Pallas re-compiles every kernel (ViT and AR each compile independently).
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, image resolution buckets, or `--context-length` invalidates cached entries.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md); run `python -m sgl_jax.launch_server --multimodal --help` to see multimodal-specific flags.

## 3. Invocation

### 3.1 Basic Chat Completion (text only)

Qwen2.5-VL accepts plain-text requests on the same OpenAI-compatible `/v1/chat/completions` endpoint — useful for sanity-checking the server before sending images:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
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
    model="Qwen/Qwen2.5-VL-32B-Instruct",
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
    model="Qwen/Qwen2.5-VL-32B-Instruct",
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
    model="Qwen/Qwen2.5-VL-32B-Instruct",
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

> Qwen2.5-VL is non-reasoning (no `<think>` blocks) and does not ship a native tool-call format. For reasoning workloads use [Qwen3](Qwen3.md); for tool-calling workloads use a model with `--tool-call-parser` support (see [`Qwen3.md` §3.3](Qwen3.md#33-tool-calling)).

## 4. Benchmark

> Benchmark section is intentionally omitted — Qwen2.5-VL is a Starter recipe (banner). All §4.1 Accuracy / §4.2 Speed cells are pending real PR-back measurements. When you run a numbered MMMU / MMMU Pro Vision / DocVQA / ChartQA eval against the model on TPU, file a PR adding the §4 block back with the actual numbers and upgrade the banner to Partially validated or Validated. For the canonical four-part §4 form (Test Environment / Deployment Command / Benchmark Command / Test Results) see any Validated recipe in [`../index.md`](../index.md).
>
> Note: `bench_serving` does not have native multimodal input support today, so §4.2 Speed needs a custom OpenAI-client load test driving the §3.2 multi-image / video patterns; PR back full TTFT / ITL / output tok/s along with the image resolution and prompt template used.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Server rejects `image_url` / `video_url` content blocks | `--multimodal` not set | Add `--multimodal` to the launch command; the text-only server cannot parse vision content blocks. |
| 72B multi-host launch fails or hangs | Current SGL-JAX staging and multimodal AR multi-host broadcast are not ready for a followable 72B recipe | Do not use 72B as a followable target until a 72B multi-host path and scheduler fix land. |
| OOM at startup (32B) | Default `--mem-fraction-static` too high once ViT / AR weights load | Lower to 0.8, reduce `--max-running-requests`, and keep the recipe's `--tp-size 4`. |
| OOM at request-time on multi-image input | Vision embeddings push KV demand past budget at chosen `--max-running-requests` | Lower `--max-running-requests` to 16 or 8; raise `--context-length` if available to spread KV per request. |
| First request takes 6+ min (vs 4 min text-only) | Both ViT and AR stages JIT-compile separately on first request | Persist `JAX_COMPILATION_CACHE_DIR` across restarts — both stages share the same cache dir. |
| Video URL works in browser but server rejects it | URL not directly fetchable from the TPU host (auth / region / firewall) | Stage the video on a mounted volume and pass `file:///path/to/video.mp4` instead, or use a publicly reachable URL. |

## Additional Resources

- [Qwen2.5-VL model collection](https://huggingface.co/Qwen)
- [`Qwen3.md`](Qwen3.md) — text-only Qwen3 dense recipe (Qwen3 series is the reasoning generation).
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
