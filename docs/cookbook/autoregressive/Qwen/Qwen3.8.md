---
title: "Qwen3.8-27B"
---

# Qwen3.8-27B on SGL-JAX

> **Validated recipe** — BF16 text-only chat, reasoning, tool calling, GSM8K and MMLU accuracy, and a fixed-shape serving workload have been validated on TPU v6e-4.

## 1. Model Introduction

[**Qwen/Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B) is a 27B-parameter dense vision-language model built on the Qwen3.5 hybrid architecture. Its language model contains 64 decoder layers arranged as 48 Gated DeltaNet layers and 16 full-attention layers. SGL-JAX currently serves the language model through its existing Qwen3.5 dense implementation.

**Key Features**:

- **Dense hybrid architecture**: 27B parameters with a repeating three-linear-attention-to-one-full-attention layout.
- **Single-host deployment**: The roughly 56 GB BF16 checkpoint fits on one TPU v6e-4 with tensor parallelism 4.
- **Hybrid reasoning**: Thinking is enabled by default and can be disabled per request with `chat_template_kwargs.enable_thinking`.
- **Preserved thinking**: The model's chat template accepts `preserve_thinking` for multi-turn reasoning history.
- **Tool calling**: OpenAI-compatible tool calls use the existing `qwen3_coder` parser.
- **Long context**: The checkpoint natively supports up to 262,144 tokens.

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
- Thinking-off: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`.

**Current Scope**: This recipe covers BF16 text generation only. Image and video input, MTP speculative decoding, FP8, context extension beyond 262,144 tokens, and Qwen3.8-2.4T-A95B are not currently validated in SGL-JAX.

**License**: see the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Qwen3.8-27B | **v6e-4** | 2x2 | 4 | 4 | BF16 text-only; single host. |

See [TPU topology reference](../../base/tpu-topology-reference.md) for TPU generation and topology details. For other slices, see [Adapting to other topologies](../../base/tpu-topology-reference.md#adapting-to-other-topologies).

### 2.2 Environment

Install per the [Install guide](../../get_started/install.md) and use the [Single-host Docker template](../../deployment/single-host-docker.md) for the container setup.

Install the OpenAI Python client and accuracy-evaluation dependency:

```bash
pip install openai evalscope==1.11.1
```

<a id="deployment-launch"></a>

### 2.3 Launch

#### Single-host — TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 512 \
  --page-size 64 \
  --max-running-requests 16 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --download-dir /tmp \
  --random-seed 3 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

On the first launch, JAX worker precompilation can take several minutes even with
`--skip-server-warmup`. The HTTP port opens only after precompilation finishes.
From a second shell on the same TPU VM, wait for readiness before sending requests:

```bash
until curl -fsS http://127.0.0.1:30000/health; do
  echo "Waiting for the server to finish precompiling..."
  sleep 10
done
```

### 2.4 Configuration Tips

**Memory Management:**

- `--mem-fraction-static 0.8` is the validated starting point for BF16 on v6e-4. Lower it if model initialization or long-prefill workloads exhaust device memory.
- `--chunked-prefill-size 512` is conservative and bounds transient prefill memory. Increase it only after measuring available headroom.
- `--max-running-requests 16` is the configured upper bound. The recurrent-state pool capped the validated run at 12 concurrent requests; check the startup summary for the effective value on other configurations.
- Hybrid recurrent-state models require either `--disable-radix-cache` or `--enable-unified-radix-tree`. This recipe uses the validated legacy path with radix caching and overlap scheduling disabled.

**Hybrid Reasoning:**

- `--reasoning-parser qwen3` separates generated reasoning into `reasoning_content` while leaving the final answer in `content`.
- Thinking is enabled by default. Set `chat_template_kwargs.enable_thinking` to `false` for a direct answer.
- `chat_template_kwargs.preserve_thinking` is accepted by the model chat template. Qwen3.8's separate `reasoning_effort` API field is not currently exposed by SGL-JAX.

**Tool Calling:**

- `--tool-call-parser qwen3_coder` parses Qwen3.8's tool-call format into OpenAI-compatible `tool_calls`.
- Keep both parser flags enabled when a workload combines reasoning and tools.

**Architecture Reuse and Feature Limits:**

- The checkpoint declares `model_type: "qwen3_5"` and `Qwen3_5ForConditionalGeneration`, so it resolves to the existing Qwen3.5 dense model implementation without a separate Qwen3.8 runtime class.
- The current text-only path loads the language-model weights and intentionally skips the vision tower and bundled MTP weights.
- The native 262,144-token context length has not been exercised end to end by this recipe. Do not enable 1M-context YaRN scaling without separate correctness and memory validation.

**Compilation Cache Hygiene:**

- Set `JAX_COMPILATION_CACHE_DIR` to avoid recompiling the same XLA/Pallas programs after each restart.
- Changing tensor parallelism, page size, chunked-prefill size, or context length produces different compiled shapes and cache entries.

For full flag definitions and defaults, see the [Launch flags reference](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL and native `/generate` patterns, see [Basic API usage](../../base/basic-api-usage.md). The following OpenAI-compatible request disables thinking for a concise response:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    max_tokens=256,
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(response.choices[0].message.content)
```

Observed on TPU v6e-4 with the deployment configuration in §2.3:

```text
The capital of France is **Paris**.
```

### 3.2 Reasoning

Thinking is enabled by default. This streaming example makes the setting explicit and prints reasoning separately from the final answer:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "user", "content": "Solve step by step: what is 15% of 240?"}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048,
    stream=True,
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    },
)

for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

On TPU v6e-4, this request streamed reasoning separately and returned the correct
final answer: **36**.

### 3.3 Tool Calling

Launch the server with `--tool-call-parser qwen3_coder`, as shown in §2.3, and provide tool schemas through the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(tool_call.function.name, tool_call.function.arguments)
else:
    print(message.content)
```

Observed parsed tool call on TPU v6e-4:

```text
get_weather {"location": "Beijing"}
```

## 4. Benchmark

> Benchmark data is a snapshot from the tested build and is not refreshed on every release.

### 4.1 Accuracy — GSM8K

GSM8K is the canonical accuracy benchmark used by the existing model cookbooks. This run used 200 examples and the official Qwen3.8 thinking-mode sampling parameters.

**Deployment Command** — same as [§2.3 Single-host](../../autoregressive/Qwen/Qwen3.8.md#deployment-launch).

**Benchmark Command**

```bash
evalscope eval \
  --model Qwen/Qwen3.8-27B \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 200 \
  --generation-config '{"chat_template_kwargs":{"enable_thinking":true},"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"max_tokens":32768}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|---:|---:|
| Qwen3.8-27B | gsm8k | AverageAccuracy | main | 200 | 0.975 |

EvalScope reported an average latency of 11.295 seconds and average output
throughput of 54.13 tokens/s across these evaluation requests. Because the run
used `--limit 200`, treat this as a sampled integration result rather than a
formal full-dataset evaluation.

### 4.2 Additional Validation — MMLU

The initial integration run used the repository's 100-example MMLU evaluator. This result remains useful as loader and generation evidence, but it is separate from the canonical cookbook GSM8K benchmark.

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|---:|---:|
| Qwen3.8-27B | MMLU | Accuracy | sampled evaluation | 100 | 0.900 |

The integration run used BF16 on v6e-4 with TP4 from the same tested build.
SGL-JAX loaded all 851 language-model tensors without missing or unexpected
language weights. Vision and MTP tensors were intentionally excluded by the
text-only path.

### 4.3 Speed — single workload

The following result uses the §2.3 v6e-4 configuration after server-side JAX
precompilation. The benchmark itself used no warm-up requests, 100 fixed-length
random prompts, 512 input tokens, 128 requested output tokens, and request
concurrency 8:

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --dataset-name random \
  --num-prompts 100 \
  --random-input 512 \
  --random-output 128 \
  --max-concurrency 8 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --tokenizer Qwen/Qwen3.8-27B
```

**Test Results**

```text
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 8
Successful requests:                     100
Benchmark duration (s):                  32.67
Total input tokens:                      51200
Total input text tokens:                 51200
Total generated tokens:                  12800
Total generated tokens (retokenized):    12682
Total cached tokens:                     0
Cache hit rate:                          0.0000
Request throughput (req/s):              3.06
Input token throughput (tok/s):          1567.07
Output token throughput (tok/s):         391.77
Peak output token throughput (tok/s):    456.00
Peak concurrent requests:                16
Total token throughput (tok/s):          1958.84
Concurrency:                             7.72
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2521.53
Median E2E Latency (ms):                 2531.10
P90 E2E Latency (ms):                    2539.32
P99 E2E Latency (ms):                    2543.01
---------------Time to First Token----------------
Mean TTFT (ms):                          169.93
Median TTFT (ms):                        153.21
P99 TTFT (ms):                           303.33
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.52
Median TPOT (ms):                        18.47
P99 TPOT (ms):                           19.67
---------------Inter-Token Latency----------------
Mean ITL (ms):                           18.52
Median ITL (ms):                         17.53
P95 ITL (ms):                            17.86
P99 ITL (ms):                            18.52
Max ITL (ms):                            280.76
==================================================
```

These figures describe this exact synthetic workload; they are not a general
throughput claim for other prompt lengths, concurrency levels, or cache modes.

## Additional Resources

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3 recipe](../../autoregressive/Qwen/Qwen3.md) — dense Qwen3 reasoning and tool-calling examples.
- [Qwen3-MoE recipe](../../autoregressive/Qwen/Qwen3-MoE.md) — expert-parallel Qwen deployment.
- [TPU topology reference](../../base/tpu-topology-reference.md)
- [Launch flags reference](../../base/launch-flags-reference.md)
- [Cross-recipe troubleshooting](../../deployment/troubleshooting.md)
