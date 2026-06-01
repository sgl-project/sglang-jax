---
title: "MiMo-V2-Flash"
---

# MiMo-V2-Flash on SGL-JAX

> **Partially validated recipe** — TPU v7x-8 configuration sweep and TPU v6e-16 smoke benchmark results are available. The full workload matrix and current pinned accuracy rerun are still pending.

## 1. Model Introduction

[**XiaomiMiMo/MiMo-V2-Flash**](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash), with **309B total parameters and 15B activated parameters**, is Xiaomi's inference-centric Mixture-of-Experts model designed to maximize decoding efficiency for real-world serving workloads — enabling flexible throughput/latency tradeoffs across different hardware. SGL-JAX serves it on TPU v6e and v7x with tensor + expert parallelism.

**Key Features**:

- **Hybrid Attention Architecture**: Interleaves Sliding Window Attention (SWA) and Global Attention (GA) with a **5:1 SWA:GA ratio** and an aggressive 128-token window. Reduces KV cache storage by **~6×** while preserving long-context quality via a learnable attention sink bias.
- **Multi-Token Prediction (MTP)**: Self-distilled MTP head **triples decode throughput** on supported deployments. Native MTP draft model is shipped with the checkpoint.
- **Natively FP8 Quantized**: Weights ship in FP8 — no extra quantization flag needed. ~20 GB/chip footprint.
- **Long Context**: Pre-trained at native 32K sequence length on 27T tokens; serving supports up to **256K context** (`--context-length 262144`).
- **Hybrid Reasoning**: Supports thinking-on (default) and thinking-off via `chat_template_kwargs.enable_thinking` per-request.
- **Agentic Capabilities**: Post-trained with on-policy distillation and large-scale agentic RL — strong on SWE-Bench and complex tool-use tasks.

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=0.8`, `top_p=0.95`, `max_tokens=32768+` (give room for thinking chain).
- Thinking-off (instant): `temperature=0.7`, `top_p=0.95`, `max_tokens=4096`.

**License**: see the [HuggingFace model card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Chips per node | Nodes | Total chips | `--tp-size` | `--dp-size` | `--ep-size` | `--moe-backend` | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **v7x-8** (minimum, dev) | 1 host × 4 chips | 4 | 1 | 4 chips → 8 JAX devices | 8 | 2 | 8 | `epmoe` | v7x exposes 2 JAX devices/chip; `--tp-size` counts devices not chips |
| **v6e-16** (recommended, production) | 4x4 | 4 | 4 | 16 | 16 | 4 | 16 | `fused` | Multi-host required |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation / HBM / device-per-chip reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md) and use one of the launcher templates from [`../deployment/`](../../deployment/). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

Extra pip for accuracy benchmarking only:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Single-host (Docker) — TPU v7x-8

Boot a TPU container per [`../deployment/single-host-docker.md`](../../deployment/single-host-docker.md), then:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 8 --dp-size 2 --ep-size 8 \
  --moe-backend epmoe \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.25 \
  --max-running-requests 128 \
  --attention-backend fa \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Multi-host (GKE Indexed Job) — TPU v6e-16 (4 nodes)

The launch command is the same on every node — only `${NODE_RANK}` and `${MASTER_ADDR}` vary. `${NODE_RANK}` ranges from `0` to `3`.

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 16 --dp-size 4 --ep-size 16 \
  --moe-backend fused \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 2048 \
  --dtype bfloat16 --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.20 \
  --max-running-requests 128 \
  --attention-backend fa \
  --skip-server-warmup \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

**Launcher** — wrap the above into GKE:

- **GKE Indexed Job + headless Service** — adapt [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md). Differences from the template: `<JOB>=mimo-v2-flash`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `<N>=4`, `<MODEL_PATH>=XiaomiMiMo/MiMo-V2-Flash`, `<HTTP_PORT>=30000`, plus the launch flags above. `${NODE_RANK}` comes from `${JOB_COMPLETION_INDEX}`.

For an end-to-end GKE manifest with the same template applied, see [`MiMo-V2.5-Pro.md` §2.3 Multi-host](MiMo-V2.5-Pro.md#23-launch) — substitute the model path and TP/DP/EP from above. For temporary v6e experiments, advanced users can adapt [`../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.95` works because MiMo-V2-Flash weights are ~20 GB/chip in FP8 — fits with headroom. Drop to 0.90 if the host shares the TPU with other processes.
- `--context-length 262144` is the model's native 256K. Lower to e.g. `131072` to free KV pool budget for higher `--max-running-requests`.

**SWA Pool Sizing (hybrid attention):**
- This recipe uses `--swa-full-tokens-ratio 0.25` on v7x-8 and `0.20` on v6e-16.
- The flag is a **per-layer** ratio `swa_tokens_per_layer / full_tokens_per_layer` — **not** a pool fraction. Default `0.8` means each SWA layer gets 80% as many KV tokens as each full layer.
- MiMo runs SWA:GA at 5:1, so values 0.15–0.25 shift the total KV pool toward the (smaller number of) full-attention layers, which carry the bulk of KV demand.
- Observation point: server logs `swa token usage` / `full token usage`. If SWA hits OOM, **raise** the ratio; if full hits OOM, **lower** it.

**MoE Backend Selection:**
- This recipe uses `--moe-backend epmoe` on single-host v7x-8 and `--moe-backend fused` on multi-host v6e-16.
- At EP ≤ 8 (single host) `epmoe` wins by ~18–26% on long-context throughput (see §4.2 sweep). At EP ≥ 16 the fused Pallas kernel wins.
- The fused MoE tuned-config table covers the EP=8 shapes (server logs report `Using tuned block config` for the precompiled buckets), so the gap is not a tuner-coverage issue — it's the kernel design balance at small EP.

**Chunked Prefill Tuning:**
- `--chunked-prefill-size 4096` on v7x-8, `2048` on v6e-16. Splits long prefills to bound peak HBM during prefill.
- Raise to `8192` for shorter TTFT on long prompts (if v7x HBM allows); lower to `1024` if prefill-time OOM.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles every kernel.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, `--chunked-prefill-size`, or `--context-length` invalidates cached entries. Use a fresh cache dir per tuning experiment to avoid stale-cache cross-pollution.

For full flag definitions and defaults see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="XiaomiMiMo/MiMo-V2-Flash"` and replace `127.0.0.1` with your rank-0 internal IP, with the §1 recommended sampling parameters; for thinking + content streaming see §3.2, for tool calling see §3.3.

### 3.2 Reasoning (thinking-on default, thinking-off optional)

MiMo-V2-Flash is a hybrid reasoning model: thinking-on is the default; turn it off per-request via `chat_template_kwargs`. Launch the server with `--reasoning-parser mimo` so the API splits `reasoning_content` from `content`:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2-Flash \
  --trust-remote-code \
  --reasoning-parser mimo \
  --tp-size 8 --dp-size 2 --ep-size 8 --moe-backend epmoe \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.25 \
  --max-running-requests 128 \
  --attention-backend fa \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

#### Thinking-on (default) — streaming with separated reasoning/content

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2-Flash",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
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
Assume for contradiction sqrt(2) = p/q in lowest terms.
Then 2 q^2 = p^2, so p^2 is even, hence p is even.
Write p = 2k. Substituting: 2 q^2 = 4 k^2, so q^2 = 2 k^2 → q is even.
Both p and q being even contradicts "lowest terms".
=============== Content =================

Therefore √2 cannot be written as a ratio of integers — it is irrational. ∎
```

#### Thinking-off (instant answer)

```python
response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2-Flash",
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

Launch with both `--reasoning-parser mimo` and `--tool-call-parser mimo`. The launch command differs from §2.3 only by these two flags — append them to the §2.3 single-host or multi-host command.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

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
    model="XiaomiMiMo/MiMo-V2-Flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
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

    # Thinking (if hybrid reasoning is active)
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not thinking_started:
            print("=============== Thinking =================", flush=True)
            thinking_started = True
        print(delta.reasoning_content, end="", flush=True)

    # Tool calls — accumulate by index (a single tool call's name/args spans
    # multiple delta chunks; tool_call.index identifies which call this fragment
    # belongs to when multiple parallel calls are streamed)
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
The user is asking about the weather in Tokyo. I should call the get_weather
function with location="Tokyo". The unit isn't specified — I'll default to
celsius, which is the common unit in Japan.
=============== Content =================

🔧 Tool Call: get_weather
   Arguments: {"location": "Tokyo", "unit": "celsius"}
```

#### Handling Tool Call Results (multi-turn)

After the model returns a tool call, run the function locally and send the result back as a `tool` role message so the model can produce a natural-language answer:

```python
import json

def get_weather(location, unit="celsius"):
    return f"22°{unit[0].upper()} and sunny"

# Reconstruct the tool call from the accumulator above
first_idx = sorted(tool_calls_accumulator.keys())[0]
first_call = tool_calls_accumulator[first_idx]
args = json.loads(first_call["arguments"])
tool_result = get_weather(**args)

messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"},
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
    model="XiaomiMiMo/MiMo-V2-Flash",
    messages=messages,
)
# On thinking-on hybrid models, the final response may put text in reasoning_content
# alongside (or instead of) content — print both to avoid misleading None output.
print("Reasoning:", final.choices[0].message.reasoning_content)
print("Content:  ", final.choices[0].message.content)
```

**Output Example:**

```text
Reasoning: The weather tool returned 22°C and sunny — a comfortable spring day.
I should present this clearly to the user.
Content:   It's currently 22°C and sunny in Tokyo.
```

To see the full set of `--tool-call-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release. New numbers are added via new PRs; older numbers stay as historical records of that build.

### 4.1 Accuracy — GSM8K

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | XiaomiMiMo/MiMo-V2-Flash (FP8) |
| Tensor Parallelism | 16 |
| Data Parallelism | 4 |
| Expert Parallelism | 16 |
| Reasoning Parser | `mimo` |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 Multi-host](#multi-host-gke-indexed-job--tpu-v6e-16-4-nodes), plus `--reasoning-parser mimo`.

**Benchmark Command**

```bash
evalscope eval \
  --model XiaomiMiMo/MiMo-V2-Flash \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 32 \
  --generation-config '{"max_tokens": 32768, "chat_template_kwargs": {"enable_thinking": true}}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score | Tested build |
|:---|:---|:---|:---|:---|:---|:---|
| MiMo-V2-Flash | gsm8k | AverageAccuracy | main | 1319 | 0.9401 | (pre-pin) |
| MiMo-V2-Flash | gsm8k | AverageAccuracy | main | 200 | 0.9750 | sglang-jax 0.1.0 (smoke run, `max_tokens 8192`, no `--reasoning-parser` flag) |

### 4.2 Speed — single-workload configuration sweep

> **Layout F — single-workload configuration sweep.** One fixed cell (ISL=16384, OSL=1024, concurrency=64, 256 prompts), varying `--moe-backend` × `--chunked-prefill-size` × `--swa-full-tokens-ratio` × `--mem-fraction-static` on v7x-8.

This recipe uses a **single-workload configuration sweep**: one fixed ISL/OSL/concurrency cell, varying `--moe-backend` × `--chunked-prefill-size` × `--swa-full-tokens-ratio` × `--mem-fraction-static` to pick the best v7x-8 single-host configuration.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-8 (single host, 4 chips × 2 devices) |
| Model | XiaomiMiMo/MiMo-V2-Flash (FP8) |
| Tensor Parallelism | 8 |
| Expert Parallelism | 8 |
| Data Parallelism | 1 (for this sweep only) |
| Tested build | sglang-jax 0.1.0 (run pre-dates pin convention; approximately late 2025) |

**Workload**: 256 prompts, ISL=16384, OSL=1024, concurrency=64 (single fixed cell).

**Deployment Command** — same shape as [§2.3 Single-host](#single-host-docker--tpu-v7x-8), with `--dp-size 1` and the sweep dimensions (`--moe-backend` / `--chunked-prefill-size` / `--swa-full-tokens-ratio` / `--mem-fraction-static`) varied per row.

**Benchmark Command**

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --dataset-name random \
  --num-prompts 256 \
  --random-input 16384 \
  --random-output 1024 \
  --max-concurrency 64 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --tokenizer XiaomiMiMo/MiMo-V2-Flash
```

**Test Results — MoE backend × prefill × SWA × mem sweep on v7x-8**

| moe-backend | chunked-prefill-size | swa-full-tokens-ratio | mem-fraction-static | Output tok/s | Median ITL |
|---|---|---|---|---|---|
| **epmoe** | 4096 | 0.25 | 0.95 | **480.04** | 37.84 ms |
| epmoe | 2048 | 0.20 | 0.90 | 467.32 | 36.90 ms |
| fused | 2048 | 0.20 | 0.90 | 396.75 | 38.19 ms |
| fused | 4096 | 0.25 | 0.95 | 382.32 | 42.45 ms |

The fused MoE tuned-config table covers the EP=8 shapes (server logs report `Using tuned block config` for the precompiled buckets), so the gap is not a tuner-coverage issue — it reflects the kernel design balance at small EP.

**Multi-host v6e-16 (4 nodes × 4 chips, `--tp-size 16 --dp-size 4 --ep-size 16 --moe-backend fused`)** — sglang-jax 0.1.0, 100 prompts, ISL=1024, OSL=1024, concurrency=16:

| Output tok/s | Peak output tok/s | Mean TTFT | Mean TPOT | Median TPOT |
|---|---|---|---|---|
| 1034.44 | 1216.00 | 1093.50 ms | 13.29 ms | 13.33 ms |

**Other workload cells**: _Pending_ — additional v6e-16 (ISL, OSL, concurrency) combinations not yet measured. PR the full `============ Serving Benchmark Result ============` block from `bench_serving` when measured.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Weights don't fit | Lower `--mem-fraction-static` to 0.90; verify `--tp-size` matches device count (8 for v7x-8, 16 for v6e-16). |
| SWA pool exhaustion at runtime | Too much concurrent decode demand on SWA layers | Lower `--max-running-requests`, or raise `--swa-full-tokens-ratio`. Observe `swa token usage` in server logs. |
| Full-attention pool exhaustion | Long full-attention KV demand exceeds budget | Lower `--swa-full-tokens-ratio` (shifts pool toward full layers), or shorten `--context-length`. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker; PVC in GKE). |
| Multi-node hang at `jax.distributed.initialize` | `--dist-init-addr` unreachable | Verify rank-0 IP and port reachable from all nodes; check firewall on the JAX init port. |
| `fused` MoE backend slower than `epmoe` on single-host v7x-8 | Expected — see §4.2 measurements | Use `--moe-backend epmoe` for EP ≤ 8; `fused` is the right pick at EP ≥ 16. |

## Additional Resources

- [MiMo-V2-Flash Model Card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [`MiMo-V2.5-Pro.md`](MiMo-V2.5-Pro.md) — larger sibling model, same architectural family.
- [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md)
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
