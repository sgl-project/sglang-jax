---
title: "Qwen3"
---

# Qwen3-8B / Qwen3-32B on SGL-JAX

## 1. Model Introduction

[**Qwen/Qwen3-8B**](https://huggingface.co/Qwen/Qwen3-8B) (8B) and [**Qwen/Qwen3-32B**](https://huggingface.co/Qwen/Qwen3-32B) (32B) are Alibaba's dense decoder LLMs from the Qwen3 series — strong general-purpose models with hybrid reasoning support, deployable on a single TPU v6e-4 host. SGL-JAX serves both with tensor parallelism. For the Qwen3 MoE variants (30B-A3B / 235B-A22B) see [`Qwen3-MoE.md`](Qwen3-MoE.md).

**Key Features**:

- **Dense, single-host friendly**: Both 8B and 32B fit on TPU v6e-4 with `bfloat16`. No multi-host complexity for typical serving.
- **Hybrid Reasoning**: Supports thinking-on (default) and thinking-off via `chat_template_kwargs.enable_thinking` per-request.
- **Tool Calling**: OpenAI-compatible tool/function calling supported.
- **Long Context**: 128K context window.
- **Production-validated benchmarks**: §4.1 below has measured throughput vs vLLM on the same hardware.

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+`.
- Thinking-off (instant): `temperature=0.7`, `top_p=0.8`, `max_tokens=512`.

**License**: see [Qwen model cards](https://huggingface.co/Qwen) for authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|
| Qwen3-8B | v6e-4 | 2x2 | 4 | 4 | Single host; ~16 GB BF16 weights |
| Qwen3-32B | v6e-4 | 2x2 | 4 | 4 | Single host; ~64 GB BF16 weights — fits with `--mem-fraction-static 0.8` |

Both fit on a single v6e-4 host with `bfloat16`. See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4

The same launch command works for both 8B and 32B — only `MODEL_NAME` changes:

```bash
MODEL_NAME="Qwen/Qwen3-8B"  # or "Qwen/Qwen3-32B"

JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path ${MODEL_NAME} \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --download-dir /tmp \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Multi-host

Not needed — both 8B and 32B fit single-host on v6e-4.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.8` is conservative for 32B with `--max-running-requests 256`. Raise to 0.85–0.9 for 8B to admit more concurrent decodes.
- `--download-dir /tmp` keeps HuggingFace weights cache on tmpfs for fast reload across restarts.

**Throughput vs Latency Tradeoffs:**
- `--page-size 128` is benchmark-tuned (vs default `1`). Larger page reduces page-table overhead at high concurrency but uses more KV per request. Default `1` is more flexible for low-concurrency mixed traffic.
- `--chunked-prefill-size 2048` splits long prefills into 2K-token chunks for predictable HBM. Raise to 4096 if you have HBM headroom; lower to 1024 if prefill OOM.
- `--max-running-requests 256` is the concurrent decode cap. Throughput plateaus around this; raising further mainly increases queue depth.

**Prefix Caching:**
- `--disable-radix-cache` disables RadixAttention prefix caching. Set this **only for benchmarks** that compare against engines without prefix caching (so measurements are apples-to-apples). For production, leave it off (default) — it gives a free latency win on repeated prefixes.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, `--chunked-prefill-size`, or `--context-length` invalidates cached entries.

For full flag definitions and defaults see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Explain mixture-of-experts in 2 sentences."}]
  }'
```

Python OpenAI client equivalent:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Explain mixture-of-experts in 2 sentences."}],
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (thinking-on default, thinking-off optional)

Qwen3 is a hybrid reasoning model: thinking-on is the default; turn it off per-request via `chat_template_kwargs`. Launch the server with `--reasoning-parser qwen3` so the API splits `reasoning_content` from `content`:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Thinking-on (default) — streaming with separated reasoning/content

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    stream=True,
)

thinking_started = False
content_started = False
for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not thinking_started:
            print("=============== Thinking =================", flush=True)
            thinking_started = True
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        if thinking_started and not content_started:
            print("\n=============== Content =================", flush=True)
            content_started = True
        print(delta.content, end="", flush=True)
print()
```

**Output Example:**

```text
=============== Thinking =================
The user wants 15% of 240. Convert 15% to a decimal: 15% = 0.15.
Then multiply: 0.15 × 240 = 36.
=============== Content =================

15% of 240 is **36**.
```

#### Thinking-off (instant answer)

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

**Output Example:**

```text
The capital of France is Paris.
```

To see the full set of `--reasoning-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

### 3.3 Tool Calling

Launch with `--tool-call-parser qwen25` (compatible with Qwen3 tool-call format) plus `--reasoning-parser qwen3` if you also want thinking. Append these flags to the §2.3 launch command.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    tool_choice="auto",
    stream=True,
)

thinking_started = False
tool_calls_accumulator = {}
for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not thinking_started:
            print("=============== Thinking =================", flush=True)
            thinking_started = True
        print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "tool_calls") and delta.tool_calls:
        if thinking_started:
            print("\n=============== Content =================\n", flush=True)
            thinking_started = False
        for tc in delta.tool_calls:
            acc = tool_calls_accumulator.setdefault(tc.index, {"name": None, "arguments": ""})
            if tc.function:
                if tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function.arguments:
                    acc["arguments"] += tc.function.arguments
    if delta.content:
        print(delta.content, end="", flush=True)

for idx, tc in sorted(tool_calls_accumulator.items()):
    print(f"🔧 Tool Call: {tc['name']}")
    print(f"   Arguments: {tc['arguments']}")
print()
```

**Output Example:**

```text
=============== Thinking =================
User wants Beijing weather. I should call get_weather with location="Beijing".
Defaulting to celsius (common in China).
=============== Content =================

🔧 Tool Call: get_weather
   Arguments: {"location": "Beijing", "unit": "celsius"}
```

#### Handling Tool Call Results (multi-turn)

```python
import json

def get_weather(location, unit="celsius"):
    return f"22°{unit[0].upper()} and sunny"

first_idx = sorted(tool_calls_accumulator.keys())[0]
first_call = tool_calls_accumulator[first_idx]
args = json.loads(first_call["arguments"])
tool_result = get_weather(**args)

messages = [
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": first_call["name"], "arguments": first_call["arguments"]},
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
]

final = client.chat.completions.create(model="Qwen/Qwen3-8B", messages=messages)
# Thinking-on hybrid models may place text in reasoning_content; print both to avoid None.
print("Reasoning:", final.choices[0].message.reasoning_content)
print("Content:  ", final.choices[0].message.content)
```

**Output Example:**

```text
Reasoning: The tool returned 22°C and sunny. Present this naturally.
Content:   The weather in Beijing is currently 22°C and sunny.
```

To see the full set of `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release. The full archived ISL × OSL × batch matrix and chart images live in [`../../performance/qwen3_benchmark.md`](../../../performance/qwen3_benchmark.md) as a release-notes-style report.

### 4.1 Speed — SGL-JAX vs vLLM

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | Qwen/Qwen3-8B and Qwen/Qwen3-32B (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax `main-10f32e49ab19f54fa393a2564c0ea0b6a78bc967` (2025-09-12); vLLM `main-5931b7e5d9acd4fd9eb42d56086c379fa2e2014e` |

Methodology: TTFT measured at `output_len=1` to isolate first-token latency; ITL / throughput measured at `output_len=1024`. Workload sweeps input lengths 1024 / 4096 / 8192 tokens × output lengths 1 / 1024 tokens × concurrency 8 / 16 / 32 / 64 / 128 / 256.

**Deployment Command** — same as [§2.3 Single-host](#single-host-docker--tpu-v6e-4) for the SGL-JAX side, with `--disable-radix-cache` added to keep apples-to-apples comparison with vLLM (which doesn't use prefix caching in this run). For the vLLM baseline:

```bash
MODEL_NAME="Qwen/Qwen3-8B"  # or "Qwen/Qwen3-32B"
vllm serve "${MODEL_NAME}" \
  --download_dir /tmp --swap-space 16 --disable-log-requests \
  --tensor_parallel_size 4 --trust-remote-code \
  --max-model-len 9216 --no-enable-prefix-caching
```

**Benchmark Command** — bash driver that sweeps (ISL × OSL × concurrency):

```bash
#!/bin/bash
set -e
if [ -z "$1" ]; then
  echo "Usage: $0 <engine> [model]"; echo "engine: sgl-jax or vllm"; exit 1
fi
backend=${1}
MODEL_NAME=${2:-Qwen/Qwen3-8B}  # or pass Qwen/Qwen3-32B as 2nd arg
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
  for output_seq_len in "${output_seq_lens[@]}"; do
    for max_concurrency in "${max_concurrencies[@]}"; do
      num_prompts=$((num_prompts_per_concurrency * max_concurrency))
      python3 -m sgl_jax.bench_serving \
        --backend ${backend} \
        --dataset-name random \
        --num-prompts ${num_prompts} \
        --random-input ${input_seq_len} \
        --random-output ${output_seq_len} \
        --max-concurrency ${max_concurrency} \
        --random-range-ratio 1 \
        --warmup-requests 0 \
        --tokenizer "${MODEL_NAME}"
    done
  done
done
```

Run against both engines:

```bash
chmod +x benchmark.sh
./benchmark.sh sgl-jax Qwen/Qwen3-8B
./benchmark.sh vllm Qwen/Qwen3-8B
```

**Test Results** (selected representative cells — see [the benchmark report](../../../performance/qwen3_benchmark.md) for the full ISL × OSL × batch matrix)

Qwen3-8B:

| ISL/OSL | Batch | TTFT (ms) SGL_JAX | TTFT (ms) vLLM | ITL (ms) SGL_JAX | ITL (ms) vLLM | Out tok/s SGL_JAX | Out tok/s vLLM |
|---|---|---|---|---|---|---|---|
| 1024/1024 | 64  | 940.87   | 1346.38  | 11.11 | 18.34 | 5296.60 | 3043.58 |
| 1024/1024 | 256 | 3793.50  | 5496.67  | 30.00 | 56.81 | 7571.84 | 4051.08 |
| 4096/1024 | 64  | 4108.43  | 6594.65  | 21.13 | 42.20 | 2528.79 | 1304.04 |
| 8192/1024 | 64  | 9797.87  | 16565.67 | 31.98 | 58.86 | 1458.77 | 873.82  |

Qwen3-32B:

| ISL/OSL | Batch | TTFT (ms) SGL_JAX | TTFT (ms) vLLM | ITL (ms) SGL_JAX | ITL (ms) vLLM | Out tok/s SGL_JAX | Out tok/s vLLM |
|---|---|---|---|---|---|---|---|
| 1024/1024 | 64  | 2864.06  | 3734.00  | 29.48 | 42.45  | 1977.45 | 1391.29 |
| 1024/1024 | 256 | 11500.61 | 14985.60 | 34.27 | 68.14  | 2122.98 | 1652.79 |
| 4096/1024 | 64  | 12329.34 | 16108.16 | 35.32 | 73.64  | 785.30  | 422.50  |
| 8192/1024 | 64  | 28849.51 | 36082.81 | 33.43 | 142.77 | 435.25  | 199.05  |

SGL-JAX wins consistently on this hardware across all measured cells: ~1.5–2.2× output throughput, ~1.4–2.0× faster TTFT, ~1.6–2.4× lower ITL.

### 4.2 Accuracy

_Not measured in this benchmark run._ Run `evalscope` against the launched server using the four-section pattern from [`Qwen.md` §4.2](Qwen.md#42-accuracy--gsm8k) (Test Environment → Deployment Command → Benchmark Command → Test Results) if you need accuracy numbers.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup with Qwen3-32B | `--mem-fraction-static 0.8` still too high for 32B + large `--max-running-requests` | Lower `--max-running-requests` to 128, or `--mem-fraction-static` to 0.75. Verify `--tp-size 4` matches v6e-4 chip count. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker). |
| Throughput plateaus below benchmark numbers | RadixAttention prefix caching helping (or not) | If reproducing the §4.1 vLLM comparison, ensure `--disable-radix-cache` is set. For production, leave it off — it's a free win on repeated prefixes. |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser qwen25` to the launch command. |

## Additional Resources

- [Qwen Model Cards](https://huggingface.co/Qwen)
- [`../../performance/qwen3_benchmark.md`](../../../performance/qwen3_benchmark.md) — full benchmark report with charts.
- [`Qwen.md`](Qwen.md) — first-generation Qwen-7B-Chat recipe.
- [`Qwen3-MoE.md`](Qwen3-MoE.md) — Qwen3 MoE variants (30B-A3B / 235B-A22B).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
