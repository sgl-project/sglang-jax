---
title: "DeepSeek R1"
---

# DeepSeek R1 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, reasoning_content streams correctly, GSM8K accuracy 98.0% (50 examples, thinking-on). §4.2 includes a v7x-16 launch and `bench_serving` template; throughput numbers are intentionally omitted until the recipe has a release-quality performance datapoint.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) is DeepSeek's 671B / 37B-activated reasoning-tuned MoE — RL-trained on long chain-of-thought for math, code, and graduate-level reasoning. The model emits `<think>` blocks that SGL-JAX exposes as `reasoning_content` via the `deepseek-r1` parser. The official checkpoint uses FP8 block-wise weights (`block_size=128`); `--dtype bfloat16` controls runtime compute/output dtype, not BF16 weight residency. Multi-host serving required.

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — 256 routed experts and 1 shared expert per MoE layer; first 3 layers are dense MLP. See §2.4 for the backend choice.
- **FP8 block-quant compatibility** — the per-rank `out_dim` of the shared expert `gate_proj` / `up_proj` must be **strictly greater than** `block_size_out=128`. This forces the v6e-64 mesh shape and is why `--dp-size 8` (effective tensor axis 8) is recommended over `--dp-size 4` (tensor axis 16, which collides with the block size — see §2.4).
- Reasoning surface needs `--reasoning-parser deepseek-r1` at launch — see [§3.2](/autoregressive/DeepSeek/DeepSeek-R1#3-2-reasoning-thinking-enabled-streaming) for the streaming pattern.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+` (give room for thinking).

**License**: see the [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips / JAX devices | `--tp-size` | `--dp-size` | Tensor axis | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|---|
| **v7x-16** | 2x2x4 | 4 | 16 chips / 32 devices | 32 | 4 | 8 | 32 | Launch and benchmark template in §4.2. v7x exposes 2 JAX devices/chip; keep tensor axis 8 for the FP8 block-quant and MLA layout. |
| **v6e-64** | 8x8 | 16 | 64 | 64 | 8 | 8 | 64 | This is the slice we measured on. `dp=8` required for FP8 shared-expert block-quant compatibility (`2048/8=256 > 128 = block_size`); `dp=4` silently collapses. Dense MLP block-quant scale grid `(144, 56)` further requires `144 % tensor == 0`, so tensor=8 is the only working option. HBM is tight at `dp=8`; see §2.4 Memory Management. |

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). Multi-host required — use [GKE Indexed Job launcher](/deployment/gke-indexed-job) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](/deployment/skypilot).

For evaluation, additionally install `evalscope` in the client environment:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host — TPU v6e-64

Use [GKE Indexed Job launcher](/deployment/gke-indexed-job) with `<JOB>=deepseek-r1`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16` (transient GKE control-plane blips happen; a non-zero backoff lets the job survive). Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
  --tp-size 64 --dp-size 8 --ep-size 64 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 1024 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup
```

Mount a shared `JAX_COMPILATION_CACHE_DIR` on the same PVC as the model weights — first-time compile is ~4 minutes total (EXTEND ~70 s + DECODE ~3 min); subsequent restarts with the same mesh shape skip almost all of that.

For temporary v6e experiments, advanced users can adapt [SkyPilot launcher](/deployment/skypilot) with the same launch flags.

### 2.4 Configuration Tips

**Reasoning Parser:**
- `--reasoning-parser deepseek-r1` is **required** for R1 — without it, the model's `<think>` content stays inline in `content` instead of being split into `reasoning_content`. See [§3.2](/autoregressive/DeepSeek/DeepSeek-R1#3-2-reasoning-thinking-enabled-streaming) for the streaming pattern.

**Tensor/Data Mesh Layout:**
- Mesh shape is `Mesh(data=dp_size, tensor=tp_size/dp_size)`. On v6e-64 with `--tp-size 64 --dp-size 8`, the tensor axis is **8**.
- Choose `--dp-size` so that the per-rank shared-expert `out_dim = moe_intermediate_size(2048) / tensor_axis` is **strictly greater than `block_size_out=128`**. At `tensor=16` (i.e., `dp=4`) you hit `2048/16 = 128` exactly, which trips the block-wise quantized matmul kernel's documented "accuracy collapse" regime (the `epmoe` path asserts; the `fused` path silently emits garbage tokens). At `tensor=8` (`dp=8`) you get 256 > 128, which is correct.
- The dense MLP block-quant scale `(144, 56)` for `gate_proj`/`up_proj` (first 3 layers) further requires `144 % tensor == 0`. Tensor axes 1/2/4/8/16 all satisfy this; tensor=32/64 do not. Combined with the shared-expert constraint above, **tensor=8 (i.e., `--dp-size 8`) is the only working option on v6e-64**.

**MoE Backend:**
- Use `--moe-backend epmoe` as the validated default for R1 at the current sglang-jax 0.1.0 build. EPMoE adds an "offline EPMoE scale → GMM layout" conversion step at load time and is slightly slower to load than `fused`, but it carries the accuracy-guard assertion that the `fused` kernel path is missing. The `fused` backend is known to produce collapsed greedy output at `dp=4` due to the shared-expert collapse described above.
- Despite the historical hint that "epmoe is only for EP ≤ 8," it runs correctly at EP=64 on v6e-64 — the hint is a throughput recommendation, not a correctness limit.

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.
- `--page-size 128` is **mandatory** for the MLA backend; smaller values trigger a startup assertion in the MLA pager.

**Memory Management:**
- HBM is genuinely tight at `dp=8` because attention/dense weights replicate 8x across DP groups (vs 4x at `dp=4`). The current settings (`--mem-fraction-static 0.88 --chunked-prefill-size 1024 --max-running-requests 64`) leave just enough headroom for the EXTEND precompile peak (`bs=64, tokens=8192`). Do **not** raise `--chunked-prefill-size` past 1024 or `--max-running-requests` past 64 without first measuring HBM headroom; the previous `chunked=2048` setting OOMed by ~440 MB.
- The official DeepSeek-R1 checkpoint is FP8. Do **not** add `--quantization fp8`; keep `--dtype bfloat16` for runtime compute dtype. FP8 auto-detection is driven by HF `quantization_config.quant_method == "fp8"`.

**Reasoning-specific tuning:**
- Reasoning outputs are 2-10x longer than chat completions. Set client-side `max_tokens >= 4096` (R1 single-shot answers regularly use 2k-3k tokens for thinking before the final response).
- For accuracy benchmarks (AIME / MATH / GPQA / LiveCodeBench), use `max_tokens >= 8192` to avoid truncation mid-trace.
- `--max-running-requests 64` is conservative for reasoning; raise to 128 only after measuring HBM headroom — reasoning workloads grow KV cache per active request faster than chat.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first request blocks ~4 min per node and is repeated on every restart.
- Mount a shared PVC at the cache directory to amortize compilation across all 16 nodes and across pod restarts. Mesh shape (`data × tensor`) is part of the cache key; changing `--dp-size` invalidates the cache.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](/base/basic-api-usage). For thinking + content streaming see §3.2.

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP; give `max_tokens` room for the thinking trace):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "If a train travels 60 km in 45 minutes, what is its average speed in km/h?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (thinking-enabled streaming)

R1 emits a thinking block before the final answer. Launch with `--reasoning-parser deepseek-r1` (see §2.4), then stream both `reasoning_content` and `content` deltas:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Solve step by step: What is 15% of 240?"}],
    temperature=0.6,
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

**Output Example** (shape; actual reasoning trace will vary):

```text
=============== Thinking =================
To find 15% of 240, convert 15% to 0.15 and multiply: 0.15 × 240 = 36.
Let me verify: 10% of 240 is 24, 5% is 12, so 15% = 24 + 12 = 36. ✓
=============== Content =================

15% of 240 is **36**.
```

For non-streaming requests, the field appears on `response.choices[0].message.reasoning_content` and `response.choices[0].message.content`.

> R1 does not ship with a native tool-call format. For tool-call workloads, see the **Parser key reference** in [Parser key reference](/autoregressive#parser-key-reference) for the list of cookbook recipes with tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy — GSM8K (thinking-on)

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | deepseek-ai/DeepSeek-R1 (official FP8 block-wise checkpoint; runtime dtype bfloat16) |
| Tensor Parallelism | 64 (effective tensor axis 8 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Reasoning Parser | deepseek-r1 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3](/autoregressive/DeepSeek/DeepSeek-R1#2-3-launch).

**Benchmark Command**

```bash
evalscope eval \
  --model /models/DeepSeek-R1 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 50 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| DeepSeek-R1 | gsm8k | AverageAccuracy | main | 50 | 0.980 |

> Recommended primary datasets where R1's reasoning advantage shows: **AIME 2025**, **MATH**, **GPQA Diamond**, **LiveCodeBench**. PR back results when you run them.

### 4.2 Speed

> **v7x-16 benchmark template.** This recipe provides the launch and benchmark command shape, but does not publish a throughput result yet. That keeps the page useful as a working deployment recipe without turning a still-tunable datapoint into a release headline.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-16 (4 nodes x 4 chips, 32 JAX devices) |
| Model | deepseek-ai/DeepSeek-R1 (real FP8 weights; runtime dtype bfloat16) |
| Tensor Parallelism | 32 (tensor axis 8 via `--dp-size 4`) |
| Data Parallelism | 4 |
| Expert Parallelism | 32 |
| Tested build | origin/main (`2d97c787f712f715784216f7c414a4f477ea8218`) |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path /models/DeepSeek-R1 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
  --tp-size 32 --dp-size 4 --ep-size 32 \
  --moe-backend epmoe \
  --dtype bfloat16 \
  --context-length 32768 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.88 \
  --page-size 128 \
  --max-running-requests 128 \
  --attention-backend fa \
  --disable-radix-cache \
  --dp-schedule-policy round_robin \
  --skip-server-warmup \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

**Workload** — `bench_serving` with `--dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 384 --max-concurrency 128 --random-range-ratio 1 --seed 42 --warmup-requests 0`.

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/DeepSeek-R1 \
  --tokenizer /models/DeepSeek-R1 \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 384 --max-concurrency 128 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0
```

**Published Results**

Throughput numbers are intentionally omitted from this recipe for now. Use the command above to validate local deployments; publish a result row only after the configuration is tuned enough to be representative of the release target.

## Additional Resources

- [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
