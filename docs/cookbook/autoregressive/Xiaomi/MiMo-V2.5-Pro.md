---
title: "MiMo-V2.5-Pro"
---

# MiMo-V2.5-Pro on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, thinking-on output correct, GSM8K accuracy 97.5% (200 examples, see §4.1), `bench_serving` numbers in §4.2. TPU v7x-16 is a supported alternative hardware path (same launch shape, lower HBM-per-chip pressure); v6e-64 is the validation target for this recipe and v7x reruns are not required to keep the ✅ status.

## 1. Model Introduction

[**XiaomiMiMo/MiMo-V2.5-Pro**](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) is Xiaomi's large-scale inference-centric Mixture-of-Experts model with hybrid attention and natively FP8-quantized weights, designed for long-context reasoning at production throughput. SGL-JAX serves it on TPU v6e and v7x with tensor + expert parallelism + sharded attention.

**Key Features**:

- **Hybrid Attention Architecture**: Sliding Window Attention (SWA) interleaved with full-attention layers; separate KV cache pools with automatic eviction. Long-context efficiency without giving up global modeling.
- **Multi-Token Prediction (MTP)**: Self-distilled NEXTN-style draft head ships with the checkpoint — boosts decode throughput in latency-sensitive serving.
- **Natively FP8 Quantized**: Weights ship in FP8, dequantized at load time. No `--quantization` flag needed.
- **Long Context**: Up to **256K context window** (`--context-length 262144`).
- **Reasoning + Tool Use**: Hybrid reasoning (thinking-on default); supports OpenAI-compatible tool calling. Post-trained for agentic workflows.

**Recommended Generation Parameters**:

- Thinking-on (default): `temperature=1.0`, `top_p=0.95`, `max_tokens=131072` (give room for long reasoning chains).
- Tool calling: `temperature=0.7`, `max_tokens=4096`.

**License**: see the [HuggingFace model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Tier | TPU | Topology | Chips per node | Nodes | Total chips | `--tp-size` | `--dp-size` | `--ep-size` | `--moe-backend` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Primary production | **v6e-64** | `4x4x4` | 4 | 16 | 64 | 64 | 8 | 64 | `fused` | ✅ validated | Primary validation target. v6e is 1:1 chip↔device; lower HBM per chip — see §2.4 SWA Pool Sizing for tradeoff. GSM8K + bench_serving in §4. |
| Alternative production | **v7x-16** | `2x2x4` | 4 | 4 | 16 | 32 | 4 | 32 | `fused` | alternative | Supported alternative hardware; same launch shape. v7x exposes 2 JAX devices/chip → 16 × 2 = 32; attention TP = `tp_size/dp_size` = 8. Historical AIME 2025 reference numbers in §4.1; not rerun on current build (not required for ✅). |

MiMo-V2.5-Pro ships a single supported deployment class — multi-host MoE on either v6e-64 (primary) or v7x-16 (alternative); single-host configurations are not supported. All nodes must be in the same TPU slice and reach each other on the JAX init port (`5000` by default) and the TPU process port (`8471`).

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation / HBM / device-per-chip reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md) and use one of the launcher templates from [`../../deployment/`](../../deployment/). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

The `jax0.8.1-rev1` image is what SGL-JAX's GKE launcher and advanced SkyPilot path use; pinning it keeps the JAX runtime in lockstep with the SGL-JAX `[tpu]` extras.

Extra pip for accuracy benchmarking only:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

MiMo-V2.5-Pro is multi-host only. Run the same command on every node; only `${NODE_RANK}` and `${MASTER_ADDR}` vary across nodes.

#### Multi-host (GKE Indexed Job) — TPU v7x-16 (4 nodes, `2x2x4`)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5-Pro \
  --trust-remote-code \
  --tp-size 32 --dp-size 4 --ep-size 32 \
  --moe-backend fused \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.25 \
  --max-running-requests 512 \
  --attention-backend fa \
  --skip-server-warmup \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

`${NODE_RANK}` ranges from `0` to `3`.

#### Multi-host — TPU v6e-64 (16 nodes, `4x4x4`)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5-Pro \
  --trust-remote-code \
  --tp-size 64 --dp-size 8 --ep-size 64 \
  --moe-backend fused \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 --mem-fraction-static 0.92 \
  --swa-full-tokens-ratio 0.15 \
  --max-running-requests 512 \
  --attention-backend fa \
  --skip-server-warmup \
  --nnodes 16 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

`${NODE_RANK}` ranges from `0` to `15`. Compared with v7x-16, this lowers `--mem-fraction-static` to `0.92` and `--swa-full-tokens-ratio` to `0.15` because v6e has less HBM per chip — the lower SWA ratio shifts the smaller KV pool toward full-attention layers.

For the GKE Indexed Job + headless Service manifest pattern that wraps both launch commands, see [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) — fill in `<JOB>=mimo-v25-pro`, `<ACCELERATOR>=tpu7x` (v7x) or `tpu-v6e-slice` (v6e), `<TOPOLOGY>=2x2x4` / `4x4x4`, `<N>=4` / `16`, and paste the launch flags above into `<LAUNCH_FLAGS>`. For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md); the default SkyPilot template is v6e-only, so use GKE for v7x.

### 2.4 Configuration Tips

**Memory Management:**
- `--context-length 262144` is the model's native 256K. For most production traffic, **lowering to 128K (`131072`) is safe** and frees KV budget for higher `--max-running-requests`.
- `--mem-fraction-static 0.95` (v7x) / `0.92` (v6e) — v6e has less HBM per chip, so the conservative value avoids host fragmentation killing the launch.

**SWA Pool Sizing (hybrid attention):**
- This recipe uses `--swa-full-tokens-ratio 0.25` on v7x-16 and `0.20`–`0.15` on v6e-64.
- The flag is a **per-layer** ratio `swa_tokens_per_layer / full_tokens_per_layer` — **not** a pool fraction. Default `0.8` means each SWA layer gets 80% as many KV tokens as each full layer.
- Lower the ratio when full-attention KV is the bottleneck (each full layer gets relatively more); raise when SWA layers are saturating first.
- Observation point: server logs `swa token usage` / `full token usage`. If SWA hits OOM, **raise** the ratio; if full hits OOM, **lower** it.

**MoE Backend Selection:**
- `--moe-backend fused` is the right pick for this recipe (EP ≥ 16) — fused Pallas kernel wins on large EP shapes.
- For smaller single-host MoE setups (EP ≤ 8), `epmoe` actually wins.

**Speculative Decoding (NEXTN / MTP):**
- MiMo-V2.5-Pro ships an MTP draft head; enable speculative decoding via:
  ```
  --speculative-algorithm NEXTN
  --speculative-draft-model-path <draft-checkpoint>
  --speculative-num-steps 3
  --speculative-num-draft-tokens 4
  --disable-overlap-schedule
  ```
- Pin the draft checkpoint from the MiMo model card. `--disable-overlap-schedule` is mandatory — speculative + overlap scheduler are mutually exclusive (the server validates this at startup).
- Mainly latency-sensitive scenarios benefit; high-concurrency throughput often does not.

**Chunked Prefill Tuning:**
- `--chunked-prefill-size 4096` bounds peak HBM during prefill. Raise to `8192` on v7x for shorter TTFT on long prompts (if HBM allows); lower to `2048` on v6e if prefill-time OOM occurs.
- Setting `-1` disables chunking entirely — **not recommended** at 256K context.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles every kernel.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, `--chunked-prefill-size`, or `--context-length` invalidates cached entries. Give each tuning experiment its own cache dir to avoid stale-cache misses across runs.

For full flag definitions and defaults see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). For thinking + content streaming see §3.2, for tool calling see §3.3.

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP; tool-calling sampling baseline — for long thinking-on chains raise `max_tokens`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2.5-Pro",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.7,
    max_tokens=4096,
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (thinking-on default, thinking-off optional)

MiMo-V2.5-Pro is a hybrid reasoning model: thinking-on is the default; turn it off per-request via `chat_template_kwargs`. Launch the server with `--reasoning-parser mimo` so the API splits `reasoning_content` from `content`:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5-Pro \
  --trust-remote-code \
  --reasoning-parser mimo \
  --tp-size 32 --dp-size 4 --ep-size 32 \
  --moe-backend fused \
  --page-size 256 --context-length 262144 \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.25 \
  --max-running-requests 512 \
  --attention-backend fa \
  --skip-server-warmup \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

#### Thinking-on (default) — streaming with separated reasoning/content

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2.5-Pro",
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
    model="XiaomiMiMo/MiMo-V2.5-Pro",
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

Launch with both `--reasoning-parser mimo` and `--tool-call-parser mimo`. The launch command differs from §2.3 only by these two flags — append them to the §2.3 multi-host command.

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
    model="XiaomiMiMo/MiMo-V2.5-Pro",
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
The user asked about Tokyo weather. I should call get_weather with location="Tokyo".
The unit isn't specified — Japan uses celsius, I'll go with that.
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
    model="XiaomiMiMo/MiMo-V2.5-Pro",
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

### 4.1 Accuracy — GSM8K (thinking enabled)

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | XiaomiMiMo/MiMo-V2.5-Pro (FP8) |
| Tensor Parallelism | 64 |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Reasoning Parser | `mimo` |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 Multi-host (v6e-64)](#multi-host--tpu-v6e-64-16-nodes-4x4x4), plus `--reasoning-parser mimo`.

**Benchmark Command**

```bash
evalscope eval \
  --model XiaomiMiMo/MiMo-V2.5-Pro \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 200 \
  --timeout 6000000 \
  --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":8192,"chat_template_kwargs":{"enable_thinking":true}}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| MiMo-V2.5-Pro | gsm8k | AverageAccuracy | main | 200 | **0.975** |

> Historical AIME 2025 results on **TPU v7x-16** (different hardware): AIME2025-I AveragePass@1 = 0.8667 (15 problems), AIME2025-II = 1.0000 (15), OVERALL = 0.9334 (30). Build pre-dates the pin convention. Kept here as a reference for the v7x path until that path is re-validated on a current build.

### 4.2 Speed

> **Layout F — single-workload sweep (one data point).** Standard chat (ISL=1000, OSL=1000), `max_concurrency=16`, 80 prompts, `seed=42`. Future PRs can add reasoning-typical workloads (long OSL) and concurrency sweeps. Do **not** set `--reasoning-parser mimo` for throughput benchmarks (the parser adds per-token CPU work that distorts raw token rates).

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | XiaomiMiMo/MiMo-V2.5-Pro (FP8) |
| Tensor Parallelism | 64 |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 Multi-host (v6e-64)](#multi-host--tpu-v6e-64-16-nodes-4x4x4), without `--reasoning-parser`.

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/MiMo-V2.5-Pro \
  --tokenizer /models/MiMo-V2.5-Pro \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1000 --random-output-len 1000 \
  --num-prompts 80 --max-concurrency 16 \
  --seed 42
```

**Test Results**

```text
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  81.53
Total input tokens:                      37205
Total generated tokens:                  38314
Request throughput (req/s):              0.98
Input token throughput (tok/s):          456.31
Output token throughput (tok/s):         469.91
Peak output token throughput (tok/s):    688.00
Peak concurrent requests:                20
Total token throughput (tok/s):          926.22
Concurrency:                             13.68
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   13939.08
Median E2E Latency (ms):                 13182.43
P90 E2E Latency (ms):                    24648.15
P99 E2E Latency (ms):                    29268.58
---------------Time to First Token----------------
Mean TTFT (ms):                          466.86
Median TTFT (ms):                        289.31
P99 TTFT (ms):                           1515.35
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.66
Median TPOT (ms):                        28.54
P99 TPOT (ms):                           37.07
---------------Inter-Token Latency----------------
Mean ITL (ms):                           28.19
Median ITL (ms):                         23.52
P95 ITL (ms):                            23.95
P99 ITL (ms):                            264.43
Max ITL (ms):                            1290.76
==================================================
```

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup on v6e-64 | `--mem-fraction-static 0.95` too aggressive for v6e's smaller per-chip HBM | Use `0.92` on v6e-64 (already the value in §2.3); also lower `--swa-full-tokens-ratio` to 0.15 to shift KV pool to full-attention layers. |
| SWA pool exhaustion (`swa token usage` near 100%) | Long generation traffic outgrows SWA per-layer budget | Raise `--swa-full-tokens-ratio` to 0.30+ or lower `--max-running-requests`. |
| Full-attention pool exhaustion (`full token usage` near 100%) | Long-context full-attention KV outgrows budget | Lower `--swa-full-tokens-ratio` (shifts pool toward full layers) or shorten `--context-length`. |
| First request takes ~4 min on each new launch | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker; PVC in GKE). Don't share a cache dir across recipes with different `--page-size` / `--tp-size` / etc. |
| Speculative decoding refuses to start | Overlap scheduler conflict | Add `--disable-overlap-schedule` — required whenever `--speculative-algorithm` is set. |
| Multi-node hang at `jax.distributed.initialize` | `${MASTER_ADDR}` unreachable from non-rank-0 nodes | Verify rank-0 IP + port `5000` reachable from all nodes; check firewall and headless Service DNS resolution (GKE) or `sky status -a` (SkyPilot). |
| `Mismatched TPU process count` at first step | `TPU_PROCESS_ADDRESSES` length ≠ `--nnodes` | `echo $TPU_PROCESS_ADDRESSES | tr ',' '\n' | wc -l` should equal `${NNODES}`. GKE manifest hardcodes 4 entries — make sure you didn't apply the v7x-16 manifest to a v6e-64 job. |

## Additional Resources

- [MiMo-V2.5-Pro Model Card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)
- [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md)
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) — primary multi-host launcher.
- [`../../deployment/skypilot.md`](../../deployment/skypilot.md) — advanced v6e experiment alternative.
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
