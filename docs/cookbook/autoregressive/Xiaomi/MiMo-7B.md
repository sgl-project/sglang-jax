---
title: "MiMo-7B"
---

# MiMo-7B on SGL-JAX

> **Partially validated recipe** — MiMo-7B-RL has TPU v6e-4 speed and GSM8K results. MiMo-7B-Base and MiMo-7B-SFT validation are still pending.

## 1. Model Introduction

[**XiaomiMiMo/MiMo-7B**](https://huggingface.co/XiaomiMiMo) is Xiaomi's 7B-parameter dense decoder model trained with reasoning-oriented objectives — built on the Qwen 2 base architecture. Fits comfortably on a single TPU v6e-4 host.

**Variants** (pick by training objective):

- [**XiaomiMiMo/MiMo-7B-Base**](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base) — base pre-trained.
- [**XiaomiMiMo/MiMo-7B-SFT**](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT) — supervised fine-tuned for instruction following.
- [**XiaomiMiMo/MiMo-7B-RL**](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) — RL-tuned for reasoning; default choice for chain-of-thought workloads.

For the larger Xiaomi MoE models, see [`MiMo-V2-Flash.md`](MiMo-V2-Flash.md) and [`MiMo-V2.5-Pro.md`](MiMo-V2.5-Pro.md) — these are different architectures (256-expert MoE with hybrid attention), not just larger MiMo-7B variants.

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` for RL/SFT variants (give room for reasoning).

**License**: see the [HuggingFace model card](https://huggingface.co/XiaomiMiMo) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | MiMo-7B (any variant) | v6e-4 | 2x2 | 4 | 4 | BF16 weights ~14 GB — fits with headroom; lowest-cost single-host serving |
| Recommended production | MiMo-7B (any variant) | v6e-8 | 2x4 | 8 | 8 | More HBM headroom for higher `--max-running-requests` and longer reasoning outputs on RL variant |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-7B-RL \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `MiMo-7B-Base` or `MiMo-7B-SFT` as needed.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.88` is the TPU default. Raise to `0.9` for dedicated serving / higher concurrency.

**Tool Calling:**
- MiMo-7B shares the `mimo` tool-call parser format with MiMo-V2.5-Pro. Add `--tool-call-parser mimo` when using the OpenAI tools API. See [`MiMo-V2.5-Pro.md` §3.3](MiMo-V2.5-Pro.md#33-tool-calling) for the request/response pattern.

**Reasoning (RL / SFT variants):**
- Pass `extra_body={"chat_template_kwargs": {"enable_thinking": true}}` per-request to unlock chain-of-thought outputs (verify support per checkpoint via model card).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="XiaomiMiMo/MiMo-7B-RL"` (or `MiMo-7B-Base` / `MiMo-7B-SFT`) with the §1 recommended sampling parameters; for thinking + content streaming see §3.2, for tool calling see §3.3.

### 3.2 Reasoning (thinking-on default, thinking-off optional)

MiMo-7B uses the `mimo` reasoning parser. Append `--reasoning-parser mimo` to the §2.3 launch command. Thinking-on is the default; turn it off per-request via `chat_template_kwargs`.

#### Thinking-on (default) — streaming with separated reasoning/content

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-7B-RL",
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

#### Thinking-off (instant answer)

```python
response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-7B-RL",
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

### 3.3 Tool Calling

MiMo-7B uses the `mimo` tool-call parser (same key as the reasoning parser). Append `--tool-call-parser mimo` to the §2.3 launch command. Pass `tools=[...]` per the OpenAI function-calling schema:

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
    model="XiaomiMiMo/MiMo-7B-RL",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    tool_choice="auto",
    stream=True,
)

tool_calls_accumulator = {}
for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "tool_calls") and delta.tool_calls:
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

To see the full set of `--reasoning-parser` / `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — single-workload latency baseline.**

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | XiaomiMiMo/MiMo-7B-RL (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax `fe092bf` (2026-05-22) |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4).

**Benchmark Command**

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
  --tokenizer XiaomiMiMo/MiMo-7B-RL
```

**Test Results**

```text
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 8
Successful requests:                     100
Benchmark duration (s):                  27.64
Total input tokens:                      51200
Total input text tokens:                 51200
Total generated tokens:                  12800
Total generated tokens (retokenized):    12789
Request throughput (req/s):              3.62
Input token throughput (tok/s):          1852.20
Output token throughput (tok/s):         463.05
Peak output token throughput (tok/s):    484.00
Peak concurrent requests:                12
Total token throughput (tok/s):          2315.26
Concurrency:                             7.83
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2165.35
Median E2E Latency (ms):                 2204.56
P90 E2E Latency (ms):                    2205.63
P99 E2E Latency (ms):                    2264.84
---------------Time to First Token----------------
Mean TTFT (ms):                          1116.00
Median TTFT (ms):                        1155.78
P99 TTFT (ms):                           1216.67
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.26
Median TPOT (ms):                        8.26
P99 TPOT (ms):                           8.36
---------------Inter-Token Latency----------------
Mean ITL (ms):                           8.26
Median ITL (ms):                         8.26
P95 ITL (ms):                            8.40
P99 ITL (ms):                            8.64
Max ITL (ms):                            38.99
==================================================
```

### 4.2 Accuracy — GSM8K

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | XiaomiMiMo/MiMo-7B-RL (BF16) |
| Tensor Parallelism | 4 |
| Reasoning Parser | `mimo` (thinking-on per-request via `chat_template_kwargs.enable_thinking=true`) |
| Tested build | sglang-jax `fe092bf` (2026-05-22) |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4) plus `--reasoning-parser mimo --tool-call-parser mimo`.

**Benchmark Command**

```bash
evalscope eval \
  --model XiaomiMiMo/MiMo-7B-RL \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 500 \
  --generation-config '{"chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 4096}'
```

Recommended additional datasets for reasoning variants: AIME 2025, MATH.

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| MiMo-7B-RL | gsm8k | AverageAccuracy | main | 500 | 0.920 |

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser mimo` to the launch command. |
| No `reasoning_content` in response on RL/SFT variant | `--reasoning-parser` not set, or `enable_thinking` not passed | Add `--reasoning-parser mimo` to launch; pass `extra_body={"chat_template_kwargs":{"enable_thinking":true}}` per request. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [MiMo-7B family on HuggingFace](https://huggingface.co/XiaomiMiMo)
- [`MiMo-V2-Flash.md`](MiMo-V2-Flash.md) and [`MiMo-V2.5-Pro.md`](MiMo-V2.5-Pro.md) — larger Xiaomi MoE models (different architecture).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
