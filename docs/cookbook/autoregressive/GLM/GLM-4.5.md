---
title: "GLM-4.5"
---

# GLM-4.5 MoE on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**zai-org/GLM-4.5**](https://huggingface.co/zai-org) is Zhipu AI's GLM-4.5 series — MoE decoder models with hybrid reasoning support and native tool calling. Two released sizes share the same architecture and parsers.

**Variants** (pick by size):

- [**zai-org/GLM-4.5**](https://huggingface.co/zai-org/GLM-4.5) — 355B total / 32B activated; multi-host on v6e-64.
- [**zai-org/GLM-4.5-Air**](https://huggingface.co/zai-org/GLM-4.5-Air) — 106B total / 12B activated; multi-host on v6e-32.

For the newer GLM-5 family with DeepSeek-style sparse attention see [`GLM-5.md`](GLM-5.md).

**Recommended Generation Parameters**:

- General: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.
- Reasoning (thinking-on): `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+`.

**License**: see the [GLM-4.5 model card](https://huggingface.co/zai-org/GLM-4.5) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| GLM-4.5-Air (106B) | v6e-32 | 4x8 | 8  | 32 | 32 | 32 | BF16 ~210 GB |
| GLM-4.5 (355B)     | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~710 GB |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required at both sizes — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

GLM-4.5 is multi-host only at both released sizes.

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (GLM-4.5-Air)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=glm-4-5-air`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path zai-org/GLM-4.5-Air \
  --trust-remote-code \
  --tp-size 32 --ep-size 32 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (GLM-4.5)

Use `<JOB>=glm-4-5`, `<TOPOLOGY>=8x8`, `parallelism: 16`, and `completions: 16`; change the launch flags above to:

```text
  --tp-size 64 --ep-size 64 \
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` is the right choice at the EP ≥ 16 sizes above. Switch to `epmoe` only at EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.92` for dedicated multi-host serving. Drop to `0.9` if you hit OOM at startup with high `--max-running-requests`.

**Reasoning + Tool Calling (GLM-4.5 parsers):**
- Add `--reasoning-parser glm45` to expose `reasoning_content` separately from `content`.
- Add `--tool-call-parser glm45` to parse the GLM-4.5 tool-call format into OpenAI-compatible `tool_calls`.
- The streaming Python client pattern from [`Qwen3.md` §3.2](../Qwen/Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) / [§3.3](../Qwen/Qwen3.md#33-tool-calling) applies directly — only the parser names change.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's nodes to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="zai-org/GLM-4.5-Air"` (or `GLM-4.5`) with the §1 recommended sampling parameters.

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

To run reasoning and tool-calling together, pass both flags (`--reasoning-parser glm45 --tool-call-parser glm45`) and use the streaming pattern from §3.2 — `delta.reasoning_content`, `delta.content`, and `delta.tool_calls` will all appear on the same stream.

To see the full set of `--reasoning-parser` / `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the GLM-4.5 checkpoint, remove the vLLM half).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-32 (Air) / v6e-64 (GLM-4.5) |
| Model | zai-org/GLM-4.5-Air or GLM-4.5 (BF16) |
| Tensor Parallelism | 32 / 64 |
| Expert Parallelism | 32 / 64 |
| Tested build | _Pending_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-32-glm-45-air).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model zai-org/GLM-4.5-Air \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, AIME 2025.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. `epmoe` is for EP ≤ 8. |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser glm45` to the launch command. |
| No `reasoning_content` in response | `--reasoning-parser` not set | Add `--reasoning-parser glm45` to launch. |
| OOM at startup (GLM-4.5) | `--mem-fraction-static 0.92` too high for this slice | Lower to 0.9. Verify `--tp-size 64` matches v6e-64 chip count (8 × 8 = 64). |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [GLM-4.5 model collection](https://huggingface.co/zai-org)
- [`GLM-5.md`](GLM-5.md) — newer GLM-5 family with DeepSeek-style sparse attention.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
