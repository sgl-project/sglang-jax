---
title: "GLM-4.5"
---

# GLM-4.5 MoE on SGL-JAX

> **Validated recipe** — GLM-4.5-Air (106B) validated on TPU v6e-32 with sglang-jax 0.1.0; sanity + GSM8K + bench all pass. If using a pre-0.1.0 build see §5 Troubleshooting.

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
| GLM-4.5-Air (106B) | v6e-32 | 4x8 | 8  | 32 | 32 | 32 | BF16 ~210 GB |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

GLM-4.5-Air is multi-host only.

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (GLM-4.5-Air)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=glm-4-5-air`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

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

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

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

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md).

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

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-32-glm-45-air).

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

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.2](../Qwen/Qwen3.md#42-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the GLM-4.5 checkpoint, remove the vLLM half).

**Test Results** — GLM-4.5-Air, Layout B (`bench_serving` random 1024→1024, N=100, max-concurrency 16), v6e-32 + `--moe-backend epmoe`, sglang-jax 0.1.0:

```
============ Serving Benchmark Result ============
Backend:                  sgl-jax
Successful requests:      100
Benchmark duration (s):   95.10
Request throughput:       1.05 req/s
Input throughput:         1076.76 tok/s
Output throughput:        1076.76 tok/s
Total throughput:         2153.51 tok/s
Mean E2E Latency (ms):    14229.21
Mean TTFT (ms):           576.77
Mean TPOT (ms):           13.35
==================================================
```

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Startup assert `tile_n` divisibility failure (GLM-4.5-Air) | `--moe-backend fused` with `moe_intermediate_size=1408 % 512 ≠ 0` | Use `--moe-backend epmoe` for GLM-4.5-Air (mandatory). |
| `dot_general` contracting shape mismatch in `q_proj` during first prefill | Pre-0.1.0 build with stale weight transpose mappings | Upgrade to sglang-jax 0.1.0; the transpose fix is merged. |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser glm45` to the launch command. |
| No `reasoning_content` in response | `--reasoning-parser` not set | Add `--reasoning-parser glm45` to launch. |
| OOM at startup (GLM-4.5-Air) | `--mem-fraction-static 0.9` too high for this slice | Lower to 0.88. Verify `--tp-size 32` matches v6e-32 chip count (4 × 8 = 32). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [GLM-4.5-Air model card](https://huggingface.co/zai-org/GLM-4.5-Air)
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
