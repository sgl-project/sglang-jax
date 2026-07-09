---
title: "GLM-4.5"
---

# GLM-4.5 MoE on SGL-JAX

> **Validated recipe** — GLM-4.5-Air (106B) validated on TPU v6e-32 with sglang-jax 0.1.0; sanity + GSM8K pass, and §4.2 now includes the recommended v7x-4 high-throughput `bench_serving` row with the historical v6e-32 baseline kept as context. Pin to sglang-jax 0.1.0+ — earlier builds have a stale `q_proj` weight-transpose mapping that fails at first prefill.

## 1. Model Introduction

[**zai-org/GLM-4.5-Air**](https://huggingface.co/zai-org/GLM-4.5-Air) is Zhipu AI's GLM-4.5-Air — a 106B total / 12B activated MoE decoder with hybrid reasoning support and native tool calling; multi-host on v6e-32.

**Recommended Generation Parameters**:

- General: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.
- Reasoning (thinking-on): `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+`.

**License**: see the [GLM-4.5-Air model card](https://huggingface.co/zai-org/GLM-4.5-Air) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| GLM-4.5-Air (106B) | **v7x-4** | 2x2x1 | 1 | 4 chips / 8 devices | 8 | 8 | Recommended throughput recipe in §4.2. v7x exposes 2 JAX devices/chip. |
| GLM-4.5-Air (106B) | **v6e-32** | 4x8 | 8  | 32 | 32 | 32 | This is the slice we measured on. BF16 ~210 GB. |

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). Multi-host required — use [GKE Indexed Job launcher](/deployment/gke-indexed-job) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](/deployment/skypilot).

### 2.3 Launch

#### Multi-host — TPU v6e-32

Use [GKE Indexed Job launcher](/deployment/gke-indexed-job) with `<JOB>=glm-4-5-air`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path zai-org/GLM-4.5-Air \
  --trust-remote-code \
  --tp-size 32 --ep-size 32 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

> `--moe-backend epmoe` is mandatory for GLM-4.5-Air. The fused Pallas backend requires `moe_intermediate_size % 512 == 0`; GLM-4.5-Air's `moe_intermediate_size=1408` fails that alignment and crashes at startup (`tile_n` divisibility assert).

For temporary v6e experiments, advanced users can adapt [SkyPilot launcher](/deployment/skypilot) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- GLM-4.5-Air → `--moe-backend epmoe` (mandatory; `moe_intermediate_size=1408 % 512 ≠ 0` crashes fused).

**Memory Management:**
- `--mem-fraction-static 0.9` for GLM-4.5-Air on v6e-32. Drop by 0.02 if you hit OOM at startup with high `--max-running-requests`.

**Reasoning + Tool Calling (GLM-4.5 parsers):**
- Add `--reasoning-parser glm45` to expose `reasoning_content` separately from `content`.
- Add `--tool-call-parser glm45` to parse the GLM-4.5 tool-call format into OpenAI-compatible `tool_calls`.
- See §3.2 / §3.3 for the streaming Python client + Handling Tool Call Results pattern.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's nodes to amortize compilation.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](/base/basic-api-usage).

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (thinking streaming)

GLM-4.5 uses the `glm45` reasoning parser. Append `--reasoning-parser glm45` to the §2.3 launch command, then stream `reasoning_content` separately from `content`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air",
    messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
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

For non-streaming requests, the field is on `response.choices[0].message.reasoning_content`.

### 3.3 Tool Calling

GLM-4.5 uses the `glm45` tool-call parser (same key as the reasoning parser). Append `--tool-call-parser glm45` to the §2.3 launch command. Pass `tools=[...]` per the OpenAI function-calling schema:

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
    model="zai-org/GLM-4.5-Air",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message
for tc in (msg.tool_calls or []):
    print(f"🔧 Tool Call: {tc.function.name}")
    print(f"   Arguments: {tc.function.arguments}")
```

#### Handling Tool Call Results (multi-turn)

After the model returns a tool call, run the function locally and send the result back as a `tool` role message so the model can produce a natural-language answer:

```python
import json

def get_weather(location, unit="celsius"):
    return f"22°{unit[0].upper()} and sunny"

first_call = response.choices[0].message.tool_calls[0]
args = json.loads(first_call.function.arguments)
tool_result = get_weather(**args)

messages = [
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": first_call.function.name,
                "arguments": first_call.function.arguments,
            },
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
]

final = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air",
    messages=messages,
)
# On thinking-on hybrid models, the final response may put text in reasoning_content
# alongside (or instead of) content — print both to avoid misleading None output.
print("Reasoning:", final.choices[0].message.reasoning_content)
print("Content:  ", final.choices[0].message.content)
```

To run reasoning and tool-calling together, pass both flags (`--reasoning-parser glm45 --tool-call-parser glm45`) and use the streaming pattern from §3.2 — `delta.reasoning_content`, `delta.content`, and `delta.tool_calls` will all appear on the same stream.

To see the full set of `--reasoning-parser` / `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-32 (Air, 8 nodes × 4 chips) |
| Model | zai-org/GLM-4.5-Air (BF16) |
| Tensor Parallelism | 32 |
| Expert Parallelism | 32 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3](/autoregressive/GLM/GLM-4.5#2-3-launch).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model zai-org/GLM-4.5-Air \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --limit 200 \
  --generation-config temperature=0,max_tokens=4096
```

> GLM-4.5 is a **hybrid reasoning** model — keep `max_tokens` ≥ 4096 so the `<think>...</think>` trace plus the final answer both fit; truncating mid-reasoning crashes accuracy.

Recommended additional datasets: MMLU, GPQA Diamond, AIME 2025.

**Test Results** — GLM-4.5-Air on v6e-32 (sglang-jax 0.1.0):

| Model | Dataset | Limit | Score |
|:---|:---|:---|:---|
| GLM-4.5-Air | gsm8k main | 200 | **0.955** |

### 4.2 Speed

> **High-throughput v7x-4 row.** This cookbook row uses fixed-length random requests (ISL=1024, OSL=1024), `max_concurrency=128`, 384 prompts, `random_range_ratio=1`, `seed=42`, and no warmup requests. Radix cache is disabled and DP scheduling uses `round_robin`, so the result is throughput-oriented and not prefix-cache dependent.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-4 (1 node x 4 chips, 8 JAX devices) |
| Model | zai-org/GLM-4.5-Air (real BF16 weights) |
| Tensor Parallelism | 8 |
| Expert Parallelism | 8 |
| Tested build | origin/main (`2d97c787f712f715784216f7c414a4f477ea8218`) |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path /models/GLM-4.5-Air \
  --trust-remote-code \
  --tp-size 8 --ep-size 8 \
  --moe-backend epmoe \
  --dtype bfloat16 \
  --context-length 32768 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --disable-radix-cache \
  --dp-schedule-policy round_robin \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/GLM-4.5-Air \
  --tokenizer /models/GLM-4.5-Air \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 384 --max-concurrency 128 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

| ISL | OSL | Max concurrency | Prompts | Input tok/s | Output tok/s | Peak output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Duration (s) | OK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 128 | 384 | 4013.55 | 4013.55 | 4992.00 | 3126.11 | 28.84 | 97.97 | 384 |

> Historical v6e-32 baseline: `1024/1024/c16`, 100 prompts, 1076.76 output tok/s. The v7x-4 row above uses fewer chips and is the recommended throughput-oriented recipe.

## Additional Resources

- [GLM-4.5-Air model card](https://huggingface.co/zai-org/GLM-4.5-Air)
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
