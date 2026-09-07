---
title: "Qwen3-VL"
---

# Qwen3-VL on SGL-JAX

> **Validated recipe** — Qwen3-VL-32B-Instruct validated on a single TPU v7x-8 with DP4 × effective TP2, including MMMU/MMMU-Pro Vision accuracy and a single-image multimodal serving benchmark.

## 1. Model Introduction

[**Qwen/Qwen3-VL-32B-Instruct**](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) is a 32B dense vision-language instruction model that accepts text, images, and video and generates text. SGL-JAX serves it through the regular autoregressive server: multimodal requests pass through media loading, the in-model vision encoder, vision/text embedding merge, language-model prefill, and decode.

The recipe validates `Qwen/Qwen3-VL-32B-Instruct`. Other Qwen3-VL sizes and Thinking variants require their own memory sizing and validation.

For text-only Qwen3 dense models, see the [Qwen3 recipe](/autoregressive/Qwen/Qwen3).

**Key Features**:

- **Image and video understanding** — OpenAI-compatible chat requests can mix visual content with text prompts.
- **Visual agent and OCR capabilities** — the model family targets visual interaction, document understanding, multilingual OCR, and spatial reasoning.
- **Long-context multimodal input** — long text and visual token sequences share the same serving context budget.
- **In-model vision encoder** — `--vision-encoder-parallel` selects data- or tensor-parallel placement for the ViT.

**Recommended Generation Parameters**: choose parameters for the target task. The accuracy recipe in §4.1 uses deterministic decoding with `temperature=0`, `max_tokens=8192`, and seed 42.

**License**: see the [Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | `--tp-size` | Notes |
|---|---|---|---|---|
| Qwen3-VL-32B | **v7x-8** | `2x2x1` | 8 | Single host with 4 chips / 8 JAX devices; `--dp-size 4` gives DP4 × effective TP2. |

The validated path uses data-parallel vision-encoder placement and full BF16 model, vision tower, and KV cache precision without quantization.

See [TPU topology reference](/base/tpu-topology-reference) for TPU generation details. For other slices, see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per the [Install guide](/get_started/install). For the current single-host VL path, use the [Single-host Docker template](/deployment/single-host-docker).

The validated source revision is `41bcb3f8f4b155853b468f9adc0a1c6f73888fb7` on `feat/qwen3-vl`.

Extra packages for multimodal processing and accuracy benchmarking:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'evalscope[app,perf]==1.5.1'
```

### 2.3 Launch

#### Single-host — TPU v7x-8

This 32K configuration is used for the accuracy run in §4.1:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --skip-server-warmup \
  --device tpu \
  --tp-size 8 --dp-size 4 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --context-length 32768 --max-seq-len 32768 \
  --max-running-requests 1024 \
  --max-prefill-tokens 16384 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.8 --page-size 128 \
  --disable-radix-cache \
  --vision-encoder-parallel dp \
  --mm-io-worker-num 4 \
  --mm-processor-worker-num 2 \
  --random-seed 0 \
  --host 0.0.0.0 --port 30000
```

The regular server recognizes the model's multimodal contract. The separate `--multimodal` staged runtime used by diffusion recipes is not needed.

### 2.4 Configuration Tips

**Memory Management:**

- VL workloads use HBM for KV cache and vision embeddings. Leave headroom for visual tensors that scale with image count and resolution.
- Accuracy uses `--mem-fraction-static 0.8` with a 32K context. The 2K throughput run uses `0.9`; do not copy that value blindly to longer-context workloads.
- Tune `--max-running-requests` against the target visual input distribution.

**In-model multimodal serving:**

- Qwen3-VL uses the regular autoregressive server; `--multimodal` is not required.
- `--tp-size 8 --dp-size 4` is a global device mesh and gives effective TP2 inside each of four DP replicas.
- `--vision-encoder-parallel dp` load-balances images across DP ranks.

**Chunked Prefill and vision buckets:**

- `--chunked-prefill-size 4096` bounds peak HBM during prefill. The budget includes both text and vision embeddings.
- A 512×512 benchmark image produces processor tensors with shape `(1024, 1536)` before the model's vision-token merge. The default precompiled vision patch buckets cover this shape; `1024` is not the only runtime bucket or a model-wide image limit.

**Multimodal workers:**

- The validated Qwen3-VL configuration uses 4 I/O workers and 2 processor workers.
- I/O workers load and decode media; processor workers run the Hugging Face processor. Increasing either value should be driven by host-side profiling and workload shape.

**Remote media URLs:**

- `image_url` and `video_url` inputs must be fetchable from the TPU host. For private media, use a controlled object-store URL, a base64 data URL, or a mounted local file.

**Compilation Cache Hygiene:**

- Set `JAX_COMPILATION_CACHE_DIR` to reuse compiled vision, prefill, and decode programs after restart.
- Cache keys include full shapes and relevant serving flags. Changes to context length, parallelism, page size, or visual shapes can compile new programs.

For complete flag definitions, see [Launch flags reference](/base/launch-flags-reference); run `python -m sgl_jax.launch_server --help` for the current CLI.

## 3. Invocation

### 3.1 Basic Chat Completion (text only)

Qwen3-VL accepts text-only requests on the same OpenAI-compatible `/v1/chat/completions` endpoint. This is useful for checking server health before sending images:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### 3.2 Multimodal Input

Vision-language input uses the OpenAI Vision API schema. Each `messages[i].content` is a list of content blocks that mixes visual inputs and text.

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
                "image_url": {
                    "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
            },
            {"type": "text", "text": "Describe the main content of this image."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=messages,
    temperature=0,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

#### Multi-Image

Stack multiple `image_url` blocks in the same content list, followed by the prompt:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}},
            {"type": "text", "text": "Compare these images and describe what changed."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=messages,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

#### Video

Use a `video_url` block for video input:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
            {"type": "text", "text": "Describe this video in three bullet points."},
        ],
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=messages,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

> **Validated scope:** the invocation interface supports text, images, and video, but the performance benchmark in this recipe validates only single-image requests containing prompt text and one 512×512 JPEG. Multi-image, video, Qwen3-VL Thinking variants, and tool-calling are outside this validation.

## 4. Benchmark

The data below is a snapshot of Qwen3-VL-32B-Instruct on a single TPU v7x-8 at source commit `41bcb3f8f4b155853b468f9adc0a1c6f73888fb7`. Accuracy and performance use separate context sizes, recorded in their respective environments.

### 4.1 Accuracy — MMMU and MMMU-Pro Vision

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-8, single host, 4 chips / 8 JAX devices |
| Model | `Qwen/Qwen3-VL-32B-Instruct` |
| Parallelism | DP4 × effective TP2 (`--tp-size 8 --dp-size 4`) |
| Precision | BF16 model, vision, and KV; no quantization |
| Context / max sequence | 32768 / 32768 |
| Evaluator | EvalScope 1.5.1, OpenAI-compatible API |
| Dataset source | Hugging Face `MMMU/MMMU` and `MMMU/MMMU_Pro` (`vision`) |
| Generation | `max_tokens=8192`, temperature 0, seed 42 |
| Evaluation batch size | 24 (MMMU); 256 (MMMU-Pro Vision) |

**Deployment Command** — use the [§2.3 v7x-8 launch command](/autoregressive/Qwen/Qwen3-VL#2-3-launch).

**Benchmark Command**

```python
import os

from evalscope import TaskConfig, run_task

dataset_args = {
    "mmmu": {"dataset_id": "MMMU/MMMU"},
    "mmmu_pro": {
        "dataset_id": "MMMU/MMMU_Pro",
        "extra_params": {"dataset_format": "vision"},
    },
}

for dataset, batch_size in (("mmmu", 24), ("mmmu_pro", 256)):
    task_cfg = TaskConfig(
        model="Qwen/Qwen3-VL-32B-Instruct",
        api_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=[dataset],
        dataset_hub="huggingface",
        dataset_args={dataset: dataset_args[dataset]},
        eval_batch_size=batch_size,
        generation_config={
            "max_tokens": 8192,
            "temperature": 0.0,
            "stream": False,
        },
        seed=42,
        work_dir=os.path.join("./outputs/qwen3vl32b_vlm", dataset),
    )
    run_task(task_cfg=task_cfg)
```

**Test Results**

| Dataset | Metric | Samples | Score |
|---|---|---:|---:|
| MMMU Val | Mean Accuracy | 900 | **72.22%** |
| MMMU-Pro Vision | Mean Accuracy | 1,730 | **62.43%** |

### 4.2 Speed — single multimodal workload

> **Multimodal throughput row.** The workload submits 1,000 requests at an unbounded request rate. Each request contains 1,024 random source-text tokens and one random 512×512 JPEG, averages 1,086.25 text tokens plus 258 vision tokens after chat templating, and generates 500 output tokens. The run uses one warmup request and flushes the cache before measurement.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-8, single host, 4 chips / 8 JAX devices |
| Model | `Qwen/Qwen3-VL-32B-Instruct` |
| Source | `feat/qwen3-vl` at `41bcb3f8f4b155853b468f9adc0a1c6f73888fb7` |
| Parallelism | DP4 × effective TP2 (`--tp-size 8 --dp-size 4`) |
| Precision | BF16 model, vision, and KV; no quantization |
| Context / max sequence | 2048 / 2048 |
| Preprocessing | 4 I/O workers, 2 processor workers |
| Cache | Radix cache disabled; cache flushed before the measured case |
| Traffic | 1,000 requests, request rate `inf`, 500 output tokens |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code --skip-server-warmup \
  --device tpu --tp-size 8 --dp-size 4 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --context-length 2048 --max-seq-len 2048 \
  --max-running-requests 1024 \
  --max-prefill-tokens 16384 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.9 --page-size 128 \
  --disable-radix-cache --vision-encoder-parallel dp \
  --mm-io-worker-num 4 --mm-processor-worker-num 2 \
  --random-seed 0 --host 0.0.0.0 --port 30000
```

**Benchmark Command**

```bash
python -m sgl_jax.bench_serving \
  --backend sglang-oai-chat \
  --host 127.0.0.1 --port 30000 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --tokenizer Qwen/Qwen3-VL-32B-Instruct \
  --dataset-name image \
  --num-prompts 1000 \
  --random-input-len 1024 \
  --random-output-len 500 \
  --random-range-ratio 1.0 \
  --image-count 1 \
  --image-resolution 512x512 \
  --image-format jpeg \
  --image-content random \
  --request-rate inf \
  --seed 0 \
  --warmup-requests 1 \
  --flush-cache --output-details
```

**Test Results**

| Metric | Result |
|---|---:|
| Successful requests | 1,000 |
| Avg text input tokens / request | 1,086.25 |
| Avg vision input tokens / request | 258.00 |
| Avg total input tokens / request | 1,344.25 |
| Output tokens / request | 500 |
| Mean TTFT | 16,636.34 ms |
| Median TTFT | 13,818.60 ms |
| P99 TTFT | 54,075.23 ms |
| Duration | 64.332 s |
| Request throughput | 15.544 req/s |
| Input token throughput | 20,895.61 tok/s |
| Output token throughput | 7,772.22 tok/s |
| Total token throughput | 28,667.82 tok/s |
| Mean TPOT | 73.94 ms |
| Median TPOT | 77.37 ms |
| P99 TPOT | 100.26 ms |
| Median E2E latency | 52,606.81 ms |
| P99 E2E latency | 63,657.17 ms |

This is a saturated burst workload, so TTFT includes scheduler queueing in addition to media processing, vision encoding, embedding merge, and language-model prefill.

## Additional Resources

- [Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3 recipe](/autoregressive/Qwen/Qwen3) — text-only Qwen3 dense recipe.
- [Qwen2.5-VL recipe](/autoregressive/Qwen/Qwen2.5-VL) — previous-generation vision-language serving recipe.
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting)
