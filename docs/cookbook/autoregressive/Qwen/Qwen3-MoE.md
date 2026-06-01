---
title: "Qwen3-MoE"
---

# Qwen3-MoE on SGL-JAX

> **Validated recipe** — Qwen3-30B-A3B validated on TPU v6e-16 with sglang-jax 0.1.0; see §4 for measured numbers.

## 1. Model Introduction

[**Qwen/Qwen3-30B-A3B**](https://huggingface.co/Qwen/Qwen3-30B-A3B) is Alibaba's MoE variant of the Qwen3 series — a sparse mixture-of-experts decoder with 30B total / 3B activated parameters and the same hybrid reasoning + tool-call format as dense Qwen3; multi-host on v6e-16.

For the dense Qwen3 variants (8B / 32B) see [`Qwen3.md`](Qwen3.md).

**Key Features**:

- **Compact MoE size**: 30B-A3B (3B active) — multi-host on v6e-16, lower compute budget than dense 32B at similar quality.
- **Hybrid Reasoning**: thinking-on (default) and thinking-off via `chat_template_kwargs.enable_thinking` per-request — use `--reasoning-parser qwen3` to expose `reasoning_content` (§3.2).
- **OpenAI-compatible tool calling**: `--tool-call-parser qwen25` exposes `tool_calls` on the response — full streaming + multi-turn examples in §3.3.
- **MoE backend selection matters**: 30B-A3B has `moe_intermediate_size=768` (not multiple of 512) — **must** use `--moe-backend epmoe` (§2.4).
- **Production-validated**: GSM8K **0.980** thinking-on on TPU v6e-16 (§4.1); ~2.8 req/s and ~1.5K output tok/s under random 1K→1K at concurrency 16 (§4.2).

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+`.
- Thinking-off (instant): `temperature=0.7`, `top_p=0.8`, `max_tokens=512`.

**License**: see the [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) for authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B   | v6e-16 | 4x4 | 4  | 16 | 16 | 16 | BF16 ~60 GB  |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). For multi-host launches use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

Qwen3-30B-A3B is multi-host only.

#### Multi-host (GKE Indexed Job) — TPU v6e-16 (Qwen3-30B-A3B)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=qwen3-moe`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

> Note: 30B-A3B uses `--moe-backend epmoe`, not `fused`. The fused MoE kernel requires `intermediate_size % 512 == 0`; Qwen3-30B-A3B's per-expert FFN inner dim is 768 (not a multiple of 512), so launching with `--moe-backend fused` raises `ValueError: Expected intermediate_size=768 to be aligned to bf=512`. See §2.4 MoE Backend.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` is the throughput-optimal choice at EP ≥ 16, **but it requires the per-expert FFN intermediate size to be a multiple of 512** (`fused_moe/v1/kernel.py` block-factor alignment).
- Qwen3-30B-A3B has `moe_intermediate_size=768` which is **not** aligned, so it must use `--moe-backend epmoe` even at EP=16.
- For EP ≤ 8 use `epmoe` regardless.

**Memory Management:**
- `--mem-fraction-static 0.9` is appropriate for the 30B-A3B config.
- Lower by 0.02 increments if you hit OOM at startup with high `--max-running-requests`.

**Hybrid Reasoning / Tool Calling:**
- Qwen3-MoE shares the hybrid reasoning (`--reasoning-parser qwen3`) and tool-call (`--tool-call-parser qwen25`) format with dense Qwen3. Append both flags to the §2.3 launch command to expose `reasoning_content` and OpenAI-compatible `tool_calls` on the response.
- Full streaming Python examples (thinking-on + thinking-off, streaming tool-call accumulator, multi-turn Handling Tool Call Results) live in [`Qwen3.md` §3.2](Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) and [§3.3](Qwen3.md#33-tool-calling) — substitute the Qwen3-30B-A3B model path and the §2.3 launch flags above.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale. Default `1` is much slower at high concurrency.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters the cache is per-node. Mount a shared PVC to amortize compilation across all nodes.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). For thinking + content streaming see §3.2, for tool calling see §3.3.

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP; thinking-off baseline):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (thinking-on default, thinking-off optional)

Qwen3-MoE inherits Qwen3's hybrid-reasoning format. Append `--reasoning-parser qwen3` to the §2.3 launch command. Thinking-on is the default; turn it off per-request via `chat_template_kwargs`.

#### Thinking-on (default) — streaming with separated reasoning/content

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
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
The user wants 15% of 240. I should compute this directly:
15% means 15 / 100 = 0.15.
0.15 × 240 = 36.
Let me double-check: 10% of 240 is 24, 5% of 240 is 12, so 15% = 24 + 12 = 36. ✓
=============== Content =================

15% of 240 is **36**.
```

#### Thinking-off (instant answer)

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

### 3.3 Tool Calling

Qwen3-MoE uses the same `qwen25` tool-call parser as the dense Qwen3 models. Append `--tool-call-parser qwen25` to the §2.3 launch command (combine with `--reasoning-parser qwen3` if you also want thinking).

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
    model="Qwen/Qwen3-30B-A3B",
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

**Output Example:**

```text
🔧 Tool Call: get_weather
   Arguments: {"location": "Beijing", "unit": "celsius"}
```

#### Handling Tool Call Results (multi-turn)

After the model returns a tool call, run the function locally and send the result back as a `tool` role message so the model can produce a natural-language answer:

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
            "function": {
                "name": first_call["name"],
                "arguments": first_call["arguments"],
            },
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
]

final = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=messages,
)
# On thinking-on hybrid models, the final response may put text in reasoning_content
# alongside (or instead of) content — print both to avoid misleading None output.
print("Reasoning:", final.choices[0].message.reasoning_content)
print("Content:  ", final.choices[0].message.content)
```

**Output Example:**

```text
Reasoning: The weather tool returned 22°C and sunny — a pleasant day.
I should present this clearly to the user.
Content:   It's currently 22°C and sunny in Beijing.
```

To see the full set of `--reasoning-parser` / `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 |
| Model | Qwen/Qwen3-30B-A3B (BF16) |
| Tensor Parallelism | 16 |
| Expert Parallelism | 16 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-16-qwen3-30b-a3b).

**Benchmark Command** — example for GSM8K (with thinking-on for reasoning):

```bash
evalscope eval \
  --model Qwen/Qwen3-30B-A3B \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --generation-config '{"chat_template_kwargs": {"enable_thinking": true}, "temperature": 0.7, "top_p": 0.95}'
```

**Test Results**

| Dataset | Subset | Samples | Score | Notes |
|---|---|---|---|---|
| gsm8k | main | 200 | **0.980** | thinking-on, `temperature=0.7`, `top_p=0.95`, `max_tokens=8192` (Qwen3-30B-A3B) |

### 4.2 Speed

> **Layout B — measured baseline.** TPU v6e-16 (4 nodes × 4 chips, TP=16, EP=16), sglang-jax 0.1.0. sgl-jax-only; no vLLM-on-TPU comparison.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | Qwen/Qwen3-30B-A3B (BF16, MoE A3B) |
| Tensor Parallelism | 16 |
| Expert Parallelism | 16 |
| Tested build | sglang-jax 0.1.0 |

**Benchmark Command**

```bash
python3 -m sgl_jax.bench_serving \
  --backend sglang \
  --model Qwen/Qwen3-30B-A3B \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

```
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  35.53
Total input tokens:                      50561
Total generated tokens:                  52444
Request throughput (req/s):              2.81
Input token throughput (tok/s):          1423.18
Output token throughput (tok/s):         1476.18
Peak output token throughput (tok/s):    1744.00
Total token throughput (tok/s):          2899.37
Mean E2E Latency (ms):                   5223.28
Mean TTFT (ms):                          75.77
Mean TPOT (ms):                          9.88
Median TPOT (ms):                        9.97
Mean ITL (ms):                           9.83
==================================================
```

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: Expected intermediate_size=768 to be aligned to bf=512` at startup (30B-A3B) | `--moe-backend fused` requires `moe_intermediate_size % 512 == 0`; Qwen3-30B-A3B has 768 | Switch to `--moe-backend epmoe`. See §2.4 MoE Backend. |
| OOM at startup | `--mem-fraction-static` too high | Lower by 0.02. Verify `--tp-size 16` matches v6e-16 chip count (4 × 4 = 16). |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser qwen25`. |
| No `reasoning_content` in response | `--reasoning-parser` not set | Add `--reasoning-parser qwen3`. |
| Multi-node hang at init | `--dist-init-addr` unreachable | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [`Qwen3.md`](Qwen3.md) — dense Qwen3-8B / 32B recipe (same reasoning/tool-call format).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
