---
title: "Qwen2.5-VL"
---

# Qwen2.5-VL on SGL-JAX

> **Validated recipe** — Qwen2.5-VL-32B-Instruct validated on a single TPU v7x-8 with DP4 × effective TP2, including MMMU/MMMU-Pro accuracy and a multimodal serving benchmark.

## 1. Model Introduction

[**Qwen/Qwen2.5-VL**](https://huggingface.co/Qwen) is Alibaba's second-generation Qwen vision-language family — multimodal decoders that ingest images / video frames and emit text, with the same chat interface as text-only Qwen2.5. SGL-JAX serves it through the regular autoregressive server, which activates the in-model vision encoder for multimodal requests before language-model prefill and decode.

**Variants** (pick by size):

- [**Qwen/Qwen2.5-VL-3B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B parameters; candidate single-host path on v6e-4 with `--tp-size 1`.
- [**Qwen/Qwen2.5-VL-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — 7B; candidate single-host path on v6e-4 with `--tp-size 1`.
- [**Qwen/Qwen2.5-VL-32B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) — 32B; validated on v7x-8 with `--tp-size 8 --dp-size 4`.
- [**Qwen/Qwen2.5-VL-72B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) — 72B; multi-host serving is pending.

For the text-only Qwen3 dense recipes see [Qwen3 recipe](/autoregressive/Qwen/Qwen3).

**Key Features**:

- **Multi-image and video input** — single chat request can mix any number of `image_url` and `video_url` content blocks alongside the text prompt; the OpenAI Vision API schema is used directly.
- **Long-context VL** — supports the underlying Qwen2.5 32K context window (extendable to 128K with rope scaling on supported checkpoints).
- **Instruction-tuned** — default chat behaviour; no per-request `enable_thinking` toggle (Qwen2.5 is non-reasoning; for reasoning use Qwen3).
- **In-model vision encoder** — the regular autoregressive server runs vision encoding, embedding merge, LM prefill, and decode; `--vision-encoder-parallel` selects DP or TP placement for the ViT.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=1024` (verify defaults against each variant's model card).

**License**: see the [Qwen model cards](https://huggingface.co/Qwen) for the authoritative Tongyi Qianwen License terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | `--tp-size` | Notes |
|---|---|---|---|---|
| Qwen2.5-VL-32B | **v7x-8** | `2x2x1` | 8 | Single host with 4 chips / 8 JAX devices; `--dp-size 4` gives DP4 × effective TP2. |

The validated benchmark uses data-parallel vision-encoder placement. TP vision-encoder placement was separately verified on the same topology.

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices, see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). For the current single-host VL path use [Single-host Docker template](/deployment/single-host-docker).

Extra pip for accuracy benchmarking only:

```bash
pip install 'evalscope[app,perf]==1.5.1'
```

### 2.3 Launch

#### Single-host — TPU v7x-8

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --device tpu \
  --tp-size 8 --dp-size 4 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --context-length 32768 --max-seq-len 32768 \
  --max-running-requests 1024 \
  --max-prefill-tokens 16384 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.9 --page-size 128 \
  --vision-encoder-parallel dp \
  --mm-io-worker-num 4 \
  --mm-processor-worker-num 16 \
  --random-seed 0 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

The regular server recognizes the model's multimodal contract. The separate `--multimodal` staged runtime is used by diffusion recipes and is not needed here.

### 2.4 Configuration Tips

**Memory Management:**
- VL workloads use HBM for both KV cache **and** vision embeddings (ViT output). Leave sufficient headroom for vision tensors that scale with input image count.
- Tune `--max-running-requests` against the target image count and resolution; each VL request can carry multiple high-resolution images that increase KV and embedding demand.

**In-model multimodal serving:**
- Qwen2.5-VL uses the regular autoregressive server; `--multimodal` is not required.
- `--tp-size 8 --dp-size 4` gives effective TP2 for the language model.
- `--vision-encoder-parallel dp` load-balances images across all devices. `tp` shards the ViT attention, MLP, and merger linear weights over each replica's tensor axis.

**Chunked Prefill (image embeddings):**
- `--chunked-prefill-size 4096` bounds peak HBM during prefill. Vision-language prefills include both text tokens and vision embeddings.

**Multimodal Attention Backend:**
- The vision-language attention path runs on the default `--attention-backend fa` (FlashAttention on Pallas) — no override needed.

**Remote media URLs:**
- `image_url` / `video_url` inputs must be fetchable **from the TPU host** — a URL that loads in your browser can still fail server-side on auth / region / firewall. Stage the media on a mounted volume and pass `file:///path/to/media`, or use a publicly reachable URL.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` avoids recompiling the vision and autoregressive kernels after restart.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, image resolution buckets, or `--context-length` invalidates cached entries.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference); run `python -m sgl_jax.launch_server --help` to see the available flags.

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

Use a `video_url` content block — same schema as `image_url`. The server samples frames from the video and feeds them through the in-model vision encoder:

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

> Qwen2.5-VL is non-reasoning (no `<think>` blocks) and does not ship a native tool-call format. For reasoning workloads use [Qwen3](/autoregressive/Qwen/Qwen3); for tool-calling workloads use a model with `--tool-call-parser` support (see [`Qwen3.md` §3.3](/autoregressive/Qwen/Qwen3#3-3-tool-calling)).

## 4. Benchmark

The data below is a snapshot of Qwen2.5-VL-32B-Instruct on a single TPU v7x-8. Accuracy and performance use separate context sizes, recorded in their respective test environments.

### 4.1 Accuracy — MMMU and MMMU-Pro Vision

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-8, single host, 4 chips / 8 JAX devices |
| Model | `Qwen/Qwen2.5-VL-32B-Instruct` |
| Parallelism | DP4 × effective TP2 (`--tp-size 8 --dp-size 4`) |
| Precision | BF16 model, vision, and KV; no quantization |
| Context / max sequence | 32768 / 32768 |
| Evaluator | EvalScope 1.5.1, OpenAI-compatible API |
| Generation | `max_tokens=8192`, temperature 0, seed 42 |
| Evaluation batch size | 24 |

**Deployment Command** — use the [§2.3 v7x-8 launch command](/autoregressive/Qwen/Qwen2.5-VL#2-3-launch).

**Benchmark Command**

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    api_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
    eval_type="openai_api",
    datasets=["mmmu", "mmmu_pro"],
    dataset_hub="huggingface",
    dataset_args={
        "mmmu": {"dataset_id": "MMMU/MMMU"},
        "mmmu_pro": {
            "dataset_id": "MMMU/MMMU_Pro",
            "extra_params": {"dataset_format": "vision"},
        },
    },
    eval_batch_size=24,
    generation_config={
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False,
    },
    seed=42,
    work_dir="./outputs/qwen25vl32b_vlm",
)

run_task(task_cfg=task_cfg)
```

**Test Results**

| Dataset | Metric | Samples | Score |
|---|---|---:|---:|
| MMMU Val | Mean Accuracy | 900 | **62.67%** |
| MMMU-Pro Vision | Mean Accuracy | 1,730 | **47.75%** |

### 4.2 Speed — single multimodal workload

> **Multimodal throughput row.** The workload submits 1,000 requests at an unbounded request rate. Each request contains 1,024 random source-text tokens and one random 512×512 JPEG, averages 1,097.23 text tokens plus 326.00 vision tokens after chat templating, and generates 500 output tokens. The run uses one warmup request and flushes the cache before measurement.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-8, single host, 4 chips / 8 JAX devices |
| Model | `Qwen/Qwen2.5-VL-32B-Instruct` |
| Parallelism | DP4 × effective TP2 (`--tp-size 8 --dp-size 4`) |
| Precision | BF16 model, vision, and KV; no quantization |
| Context / max sequence | 2048 / 2048 |
| Preprocessing | 4 I/O workers, 16 processor workers |
| Cache | Radix Cache disabled; cache flushed before each case |
| Traffic | 1,000 requests, request rate `inf`, 500 output tokens |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code --skip-server-warmup \
  --device tpu --tp-size 8 --dp-size 4 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --context-length 2048 --max-seq-len 2048 \
  --max-running-requests 1024 \
  --max-prefill-tokens 16384 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.9 --page-size 128 \
  --disable-radix-cache --vision-encoder-parallel dp \
  --mm-io-worker-num 4 --mm-processor-worker-num 16 \
  --random-seed 0 --host 0.0.0.0 --port 30000
```

**Benchmark Command**

```bash
python -m sgl_jax.bench_serving \
  --backend sglang-oai-chat \
  --host 127.0.0.1 --port 30000 \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --tokenizer Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset-name image \
  --num-prompts 1000 \
  --random-input-len 1024 --random-output-len 500 \
  --random-range-ratio 1.0 \
  --image-count 1 --image-resolution 512x512 \
  --image-format jpeg --image-content random \
  --request-rate inf \
  --seed 0 --warmup-requests 1 --flush-cache --output-details
```

**Test Results**

| Metric | Result |
|---|---:|
| Successful requests | 1,000 |
| Avg text input tokens / request | 1,097.23 |
| Avg vision input tokens / request | 326.00 |
| Avg total input tokens / request | 1,423.24 |
| Output tokens / request | 500 |
| Mean TTFT | 19,145.39 ms |
| Median TTFT | 15,025.80 ms |
| P99 TTFT | 55,411.70 ms |
| Duration | 66.641 s |
| Request throughput | 15.006 req/s |
| Input token throughput | 21,356.74 tok/s |
| Output token throughput | 7,502.88 tok/s |
| Total token throughput | 28,859.62 tok/s |
| Mean TPOT | 69.91 ms |
| Median TPOT | 74.01 ms |
| P99 TPOT | 99.35 ms |
| Median E2E latency | 52,066.95 ms |
| P99 E2E latency | 66,011.26 ms |

This is a saturated burst workload, so TTFT includes scheduler queueing in addition to multimodal preprocessing, vision encoding, embedding merge, and LM prefill.

## Additional Resources

- [Qwen2.5-VL model collection](https://huggingface.co/Qwen)
- [Qwen3 recipe](/autoregressive/Qwen/Qwen3) — text-only Qwen3 dense recipe (Qwen3 series is the reasoning generation).
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
