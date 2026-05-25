---
title: "Qwen3-MoE"
---

# Qwen3-MoE on SGL-JAX

> **Starter recipe** — derived from the HuggingFace model card; not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.

## 1. Model Introduction

[**Qwen/Qwen3-MoE**](https://huggingface.co/Qwen) is Alibaba's MoE variant of the Qwen3 series — sparse mixture-of-experts decoders with the same hybrid reasoning + tool-call format as dense Qwen3. Two released sizes: 30B-A3B (3B active) and 235B-A22B (22B active).

**Variants** (pick by size):

- [**Qwen/Qwen3-30B-A3B**](https://huggingface.co/Qwen/Qwen3-30B-A3B) — 30B total / 3B activated; multi-host on v6e-16.
- [**Qwen/Qwen3-235B-A22B**](https://huggingface.co/Qwen/Qwen3-235B-A22B) — 235B total / 22B activated; multi-host on v6e-64.

For the dense Qwen3 variants (8B / 32B) see [`Qwen3.md`](Qwen3.md).

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+`.
- Thinking-off (instant): `temperature=0.7`, `top_p=0.8`, `max_tokens=512`.

**License**: see the [Qwen model cards](https://huggingface.co/Qwen) for authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B   | v6e-16 | 4x4 | 4  | 16 | 16 | 16 | BF16 ~60 GB  |
| Qwen3-235B-A22B | v6e-64 | 8x8 | 16 | 64 | 64 | 64 | BF16 ~470 GB |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). For multi-host launches use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

Qwen3-MoE is multi-host only at both released sizes.

#### Multi-host (GKE Indexed Job) — TPU v6e-16 (Qwen3-30B-A3B)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=qwen3-moe`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend fused \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (Qwen3-235B-A22B)

Use `<TOPOLOGY>=8x8`, `parallelism: 16`, and `completions: 16`; change the launch flags above to:

```text
  --tp-size 64 --ep-size 64 \
  --mem-fraction-static 0.92 \
```

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend fused` is the recommended choice at EP ≥ 16 (both 30B and 235B configs above). Switch to `epmoe` only when EP ≤ 8.

**Memory Management:**
- `--mem-fraction-static 0.9` is appropriate for the 30B-A3B config; raise to `0.92` for 235B-A22B to make room for the much larger weight set.
- Lower by 0.02 increments if you hit OOM at startup with high `--max-running-requests`.

**Hybrid Reasoning / Tool Calling:**
- Qwen3-MoE shares the hybrid reasoning (`--reasoning-parser qwen3`) and tool-call (`--tool-call-parser qwen25`) format with dense Qwen3. Append both flags to the §2.3 launch command to expose `reasoning_content` and OpenAI-compatible `tool_calls` on the response.
- Full streaming Python examples (thinking-on + thinking-off, streaming tool-call accumulator, multi-turn Handling Tool Call Results) live in [`Qwen3.md` §3.2](Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) and [§3.3](Qwen3.md#33-tool-calling) — substitute the Qwen3-MoE model path and the §2.3 launch flags above.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale. Default `1` is much slower at high concurrency.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- On multi-node clusters the cache is per-node. Mount a shared PVC to amortize compilation across all nodes.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="Qwen/Qwen3-30B-A3B"` (or `Qwen/Qwen3-235B-A22B`) with the §1 recommended sampling parameters; for thinking + content streaming see §3.2, for tool calling see §3.3.

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

To see the full set of `--reasoning-parser` / `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — methodology + command template.** No measured numbers yet; PR back full `============ Serving Benchmark Result ============` blocks from `bench_serving` to upgrade to Validated.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the Qwen3-MoE checkpoint, remove the vLLM half if not comparing).

**Test Results** — _Pending._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (30B-A3B) / v6e-64 (235B-A22B) |
| Model | Qwen/Qwen3-30B-A3B or Qwen3-235B-A22B (BF16) |
| Tensor Parallelism | 16 / 64 |
| Expert Parallelism | 16 / 64 |
| Tested build | _Pending_ |

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

Recommended additional datasets: AIME 2025, MATH, GPQA Diamond.

**Test Results** — _Pending. Run and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. `epmoe` is for EP ≤ 8. |
| OOM at startup (235B-A22B) | `--mem-fraction-static` too high | Lower to 0.9. Verify `--tp-size 64` matches v6e-64 chip count (8 × 8 = 64). |
| Tool calls return empty arguments | `--tool-call-parser` not set | Add `--tool-call-parser qwen25`. |
| No `reasoning_content` in response | `--reasoning-parser` not set | Add `--reasoning-parser qwen3`. |
| Multi-node hang at init | `--dist-init-addr` unreachable | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [Qwen3 model collection](https://huggingface.co/Qwen)
- [`Qwen3.md`](Qwen3.md) — dense Qwen3-8B / 32B recipe (same reasoning/tool-call format).
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
