---
title: "Qwen3.8-Flash-Next"
---

# Qwen3.8-Flash-Next on SGL-JAX

> **Validated recipe** — BF16 text-only serving with the N-gram embedding enabled, MMLU and GPQA-Diamond accuracy, and two fixed-shape serving workloads have been validated on TPU v7x-8.

## 1. Model Introduction

[**Qwen/Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a sparse mixture-of-experts model: a 125B-parameter backbone with 6B parameters active per token, a 51B-parameter N-gram embedding table, and a 4B-parameter MTP module. It keeps Qwen3.5's hybrid layer schedule — 48 decoder layers, of which every fourth (12 in total) is full attention and the rest Gated DeltaNet — and changes three things around it. SGL-JAX serves it as `Qwen4ExpForConditionalGeneration`.

**Key Features**:

- **Hyper Connections**: the residual path carries 4 parallel streams, and every block is wrapped in a learned mix/combine pair that also owns the normalization.
- **Qwen Sparse Attention (QSA)**: the full-attention layers attend only to the key blocks a lightweight indexer selects, within a budget of 2,048 tokens per query, in both prefill and decode. Served by `--attention-backend qsa_sparse`.
- **N-gram embedding (PLE)**: one Gated DeltaNet layer adds an embedding looked up from hashed 2- and 3-gram token contexts. The table stays in host memory and only the rows a batch needs reach the TPU.
- **MoE**: 512 routed experts with 10 active per token, plus one shared expert.
- **Hybrid reasoning**: thinking is enabled by default and can be disabled per request with `chat_template_kwargs.enable_thinking`.
- **Long context**: the checkpoint natively supports up to 262,144 tokens.

**Recommended Generation Parameters** (thinking-on, the default): `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`. See the model card for the thinking-off settings.

**Current Scope**: This recipe covers BF16 text generation only. Image and video input, MTP speculative decoding, FP8, prefix caching, overlap scheduling, multi-host deployment, and contexts beyond 69,632 tokens are not currently validated in SGL-JAX.

**License**: see the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | **v7x-8** | 2x2x1 | 4 | 8 | BF16 text-only; `--ep-size 8`; single host. About 30 GiB of weights per JAX device. |

A v7x chip is two JAX devices, so `--tp-size 8` spans the four chips of one host. See [TPU topology reference](../../base/tpu-topology-reference.md) for TPU generation and topology details.

### 2.2 Environment

Install per the [Install guide](../../get_started/install.md) and use the [Single-host Docker template](../../deployment/single-host-docker.md) for the container setup.

The host needs room for two things besides the TPU weights:

- **Disk**: the BF16 checkpoint is about 360 GB (335 GiB). Download it to local disk before the first launch.
- **Host memory**: the N-gram table is 320,001,536 rows × 160 BF16 values, 95.4 GiB, and is read into host memory at startup. Keep at least 128 GiB of host memory free for the server process.

Install the OpenAI Python client:

```bash
pip install openai
```

<a id="deployment-launch"></a>

### 2.3 Launch

#### Single-host — TPU v7x-8

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3.8-Flash-Next \
  --device tpu \
  --dtype bfloat16 \
  --tp-size 8 \
  --ep-size 8 \
  --attention-backend qsa_sparse \
  --context-length 69632 \
  --page-size 64 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --max-recurrent-state-size 64 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --precompile-bs-paddings 16 64 \
  --precompile-token-paddings 16 64 512 1024 \
  --random-seed 3 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

A cold start took about 14 minutes on the validated host: about 7 minutes to load the weights, under a minute to read the N-gram table, and about 6 minutes of JAX precompilation. The HTTP port opens only after precompilation finishes. From a second shell on the same TPU VM, wait for readiness before sending requests:

```bash
until curl -fsS http://127.0.0.1:30000/health; do
  echo "Waiting for the server to finish precompiling..."
  sleep 10
done
```

A complete load logs `WeightLoader summary: consumed=1163, skipped=495, missing=0, unexpected=0` followed by `N-gram table: 128/128 shards`.

### 2.4 Configuration Tips

**Memory and Batch Size:**

- `--mem-fraction-static 0.8` is the validated starting point for BF16 on v7x-8.
- Set `--max-recurrent-state-size` to the same value as `--max-running-requests`. Given only the latter, the server reserves part of the recurrent-state pool for snapshots and admits fewer requests than requested. With `--ep-size 8` the fused MoE kernel needs a batch of at least 16, so a reduced limit of 16 is rejected at startup.
- `--context-length 69632` leaves room for 65,536 output tokens after the longest GPQA-Diamond prompt. Lower it when long outputs are not needed.

**Attention and Scheduling:**

- `--attention-backend qsa_sparse` is required: the 12 full-attention layers are QSA. See [Attention backend](../../../features/attention_backend.md#qwen-sparse-attention-qsa_sparse).
- The N-gram lookup hashes the real token ids on the host before each step, so the server turns overlap scheduling off for this model and rejects speculative decoding. `--disable-overlap-schedule` only makes this explicit.
- Hybrid recurrent-state models require either `--disable-radix-cache` or `--enable-unified-radix-tree`. This recipe uses the validated path with radix caching disabled.

**Serving Without the N-gram Embedding:**

- `--json-model-override-args '{"text_config": {"ple_layer_ids": []}}'` serves the model without its N-gram layer: no table is read, and the load summary becomes `consumed=1157, skipped=501, missing=0, unexpected=0`. Use it to rule the N-gram path in or out when debugging. It changes the model's outputs, so do not use it for evaluation.
- `--load-format dummy` installs no N-gram table, so it needs the same override.

**Architecture Reuse and Feature Limits:**

- The checkpoint declares `model_type: "qwen4_exp"` and `Qwen4ExpForConditionalGeneration`. It reuses the Qwen3.5 weight loader, Gated DeltaNet and MoE blocks.
- The text-only path intentionally skips the vision tower and the bundled MTP weights.
- The native 262,144-token context length has not been exercised end to end by this recipe.

**Compilation Cache Hygiene:**

- Set `JAX_COMPILATION_CACHE_DIR` to avoid recompiling the same XLA/Pallas programs after each restart.
- Changing tensor or expert parallelism, page size, chunked-prefill size, context length, or the precompile paddings produces different compiled shapes and cache entries.

For full flag definitions and defaults, see the [Launch flags reference](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL and native `/generate` patterns, see [Basic API usage](../../base/basic-api-usage.md). The following OpenAI-compatible request disables thinking for a concise response:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-Flash-Next",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

<!-- TODO(v7x run): paste the observed output. -->

### 3.2 Reasoning

Thinking is enabled by default. This streaming example uses the recommended thinking-on parameters and prints reasoning separately from the final answer:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-Flash-Next",
    messages=[
        {"role": "user", "content": "Solve step by step: what is 15% of 240?"}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
    stream=True,
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
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

<!-- TODO(v7x run): confirm reasoning streams separately and record the final answer. -->

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
    model="Qwen/Qwen3.8-Flash-Next",
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

<!-- TODO(v7x run): paste the observed parsed tool call. -->

## 4. Benchmark

> Benchmark data is a snapshot from the tested build and is not refreshed on every release.

**Deployment Command** — same as [§2.3 Single-host](../../autoregressive/Qwen/Qwen3.8-Flash-Next.md#deployment-launch), without the two parser flags.

### 4.1 Accuracy — GPQA-Diamond

GPQA-Diamond is the one model-card benchmark the repository's evaluator also runs. Each run covers all 198 questions with thinking on and the recommended thinking-on parameters, and caps the output at 65,536 tokens.

**Benchmark Command**

```bash
python3 test/srt/run_eval.py \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3.8-Flash-Next \
  --eval-name gpqa \
  --num-threads 64 \
  --enable-thinking true \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repetition-penalty 1.0 \
  --max-tokens 65536 \
  --seed 17
```

**Test Results**

| Model | Dataset | Runs | Per-run accuracy | Median | Model card |
|:---|:---|---:|:---|---:|---:|
| Qwen3.8-Flash-Next | GPQA-Diamond | 5 | 0.909 / 0.904 / 0.894 / 0.894 / 0.914 | **0.904** | 0.917 |

TPU sampling under concurrent batching is not reproducible run to run, so five runs (seeds 17, 23, 41, 59 and 97) were taken and the median reported. In each run, 6 to 11 answers reached the 65,536-token cap; every question that produced no extractable answer was one of them, which cost 2.0 to 3.5 points per run.

### 4.2 Additional Validation — MMLU

The repository's 100-example MMLU smoke (`TestQwen38FlashNextModel.test_mmlu_smoke` in `test/srt/test_qwen3_5_models.py`, thinking on) scored 0.91, against the test's 0.70 threshold. Serving without the N-gram embedding also scored 0.91.

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|---:|---:|
| Qwen3.8-Flash-Next | MMLU | Accuracy | sampled evaluation | 100 | 0.910 |

### 4.3 Speed — two workloads

The following results use the §2.3 v7x-8 configuration after server-side JAX precompilation. The benchmark itself used no warm-up requests, fixed-length random prompts, 128 requested output tokens, and request concurrency 8:

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
  --tokenizer Qwen/Qwen3.8-Flash-Next
```

**Test Results** — 512 input tokens, 100 prompts

```text
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 8
Successful requests:                     100
Benchmark duration (s):                  53.81
Total input tokens:                      51200
Total input text tokens:                 51200
Total generated tokens:                  12800
Total generated tokens (retokenized):    12595
Total cached tokens:                     0
Cache hit rate:                          0.0000
Request throughput (req/s):              1.86
Input token throughput (tok/s):          951.58
Output token throughput (tok/s):         237.89
Peak output token throughput (tok/s):    320.00
Peak concurrent requests:                16
Total token throughput (tok/s):          1189.47
Concurrency:                             7.73
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4161.62
Median E2E Latency (ms):                 4184.36
P90 E2E Latency (ms):                    4212.05
P99 E2E Latency (ms):                    4213.89
---------------Time to First Token----------------
Mean TTFT (ms):                          671.83
Median TTFT (ms):                        572.20
P99 TTFT (ms):                           1001.07
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.48
Median TPOT (ms):                        28.36
P99 TPOT (ms):                           32.10
---------------Inter-Token Latency----------------
Mean ITL (ms):                           27.49
Median ITL (ms):                         25.19
P95 ITL (ms):                            26.62
P99 ITL (ms):                            30.14
Max ITL (ms):                            867.10
==================================================
```

**Test Results** — 8,192 input tokens, 32 prompts (`--num-prompts 32 --random-input 8192`, otherwise the same)

| Median TTFT (ms) | Median TPOT (ms) | Output throughput (tok/s) | Total throughput (tok/s) |
|---:|---:|---:|---:|
| 12123.50 | 83.13 | 45.17 | 2935.73 |

These figures describe these exact synthetic workloads; they are not a general throughput claim for other prompt lengths, concurrency levels, or cache modes.

## Additional Resources

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-27B recipe](../../autoregressive/Qwen/Qwen3.8.md) — the dense Qwen3.8 model on v6e-4, with the same reasoning and tool-calling parsers.
- [Qwen3-MoE recipe](../../autoregressive/Qwen/Qwen3-MoE.md) — expert-parallel Qwen deployment.
- [Attention backend](../../../features/attention_backend.md) — the `qsa_sparse` backend.
- [TPU topology reference](../../base/tpu-topology-reference.md)
- [Launch flags reference](../../base/launch-flags-reference.md)
- [Cross-recipe troubleshooting](../../deployment/troubleshooting.md)
