---
title: "DeepSeek R1"
---

# DeepSeek R1 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax `de29d9f0` (2026-05-26): server starts, reasoning_content streams correctly, GSM8K accuracy 98.0% (50 examples, thinking-on), `bench_serving` numbers in §4.2. TPU v7x path is still a starter target.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) is DeepSeek's reasoning-tuned derivative of V3 — RL-trained on long chain-of-thought for math, code, and graduate-level reasoning. The model emits `<think>` blocks that SGL-JAX exposes as `reasoning_content` via the `deepseek-r1` parser. Same 671B / 37B-activated MoE backbone as V3, same FP8 block-wise checkpoint format. Multi-host serving required.

For the V3 non-reasoning base see [`DeepSeek-V3.md`](DeepSeek-V3.md). For the V2 generation see [`DeepSeek-V2.md`](DeepSeek-V2.md).

**Architectural notes**:

- Same architecture as V3 (MLA + 256 routed experts + 1 shared expert + first 3 dense MLP layers, FP8 block-wise `block_size=128`). All V3 sharding / mesh-shape constraints apply unchanged — see [`DeepSeek-V3.md` §2.4](DeepSeek-V3.md#24-configuration-tips).
- Reasoning surface needs `--reasoning-parser deepseek-r1` at launch — see [§3.2](#32-reasoning-thinking-enabled-streaming) for the streaming pattern.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+` (give room for thinking).

**License**: see the [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips / JAX devices | `--tp-size` | `--dp-size` | Tensor axis | `--ep-size` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| v6e-64 | 8x8 | 16 | 64 | 64 | 8 | 8 | 64 | ✅ validated | Same mesh constraints as V3 — `dp=8` required for FP8 shared-expert block-quant compatibility. See [`DeepSeek-V3.md` §2.4](DeepSeek-V3.md#24-configuration-tips). |
| v7x-8 | 2x4 | 2 | 8 chips / 16 devices | 16 | 1 | 16 | 16 | 🚧 starter | Not yet validated end-to-end for R1. |

V6e-64 is the minimum slice that fits the official FP8 checkpoint plus runtime overhead. See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

For evaluation, additionally install `evalscope` in the client environment:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-64

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=deepseek-r1`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16` (transient GKE control-plane blips happen; a non-zero backoff lets the job survive). Put these model-specific flags into `<LAUNCH_FLAGS>`:

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

Identical to the V3 launch flags except for `--model-path` and the added `--reasoning-parser deepseek-r1`.

#### Multi-host (GKE Indexed Job) — TPU v7x-8 (starter)

Use `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=2x4`, `parallelism: 2`, and `completions: 2`; change the launch flags above to:

```text
  --tp-size 16 --dp-size 1 --ep-size 16 \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
```

Not yet validated end-to-end — open a PR with measured numbers when you run it.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The default SkyPilot template is v6e-only; use GKE for v7x.

### 2.4 Configuration Tips

**Reasoning Parser:**
- `--reasoning-parser deepseek-r1` is **required** for R1 — without it, the model's `<think>` content stays inline in `content` instead of being split into `reasoning_content`. See [§3.2](#32-reasoning-thinking-enabled-streaming) for the streaming pattern.

**Mesh / MoE / Memory (same as V3):**
- The mesh shape, MoE backend, FP8 block-quant constraints, and HBM management for R1 are **identical to V3**. Rather than duplicate them, read [`DeepSeek-V3.md` §2.4](DeepSeek-V3.md#24-configuration-tips) — the rationale (`dp=8` for shared-expert compatibility, `epmoe` for the accuracy assertion, `chunked=1024` for HBM headroom) applies unchanged.

**Reasoning-specific tuning:**
- Reasoning outputs are 2-10x longer than chat completions. Set client-side `max_tokens >= 4096` (R1 single-shot answers regularly use 2k-3k tokens for thinking before the final response).
- For accuracy benchmarks (AIME / MATH / GPQA / LiveCodeBench), use `max_tokens >= 8192` to avoid truncation mid-trace.
- `--max-running-requests 64` is conservative for reasoning; raise to 128 only after measuring HBM headroom — reasoning workloads grow KV cache per active request faster than chat.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first request blocks ~4 min per node. Use a different cache dir than V3 (e.g., `/models/jit_cache_ds_r1`) so the two models do not interfere.
- Mesh shape (`data × tensor`) is part of the cache key; the R1 cache is reusable across pod restarts as long as `--dp-size` does not change.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="deepseek-ai/DeepSeek-R1"` with the §1 recommended sampling parameters; for thinking + content streaming see §3.2.

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

> R1 does not ship with a native tool-call format. For tool-call workloads choose a model with `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

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
| Tested build | sglang-jax `de29d9f0` (2026-05-26) |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-64).

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

> Recommended primary datasets where R1's reasoning advantage shows vs V3: **AIME 2025**, **MATH**, **GPQA Diamond**, **LiveCodeBench**. PR back results when you run them.

### 4.2 Speed

> **Layout F — single-workload sweep (one data point).** Same workload as V3 §4.2 for direct comparison (ISL=1000, OSL=1000, `max_concurrency=16`, 80 prompts, `seed=42`). Future PRs can add reasoning-typical workloads (e.g., OSL=4096) and concurrency sweeps.

**Test Environment** — same hardware/build as §4.1.

**Workload** — `bench_serving` with `--dataset-name random --random-input-len 1000 --random-output-len 1000 --num-prompts 80 --max-concurrency 16 --seed 42`. See `2026-05-21-recipe-command-audit/deepseek-r1/bench_serving.log` for the raw output.

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/DeepSeek-R1 \
  --tokenizer /models/DeepSeek-R1 \
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
Benchmark duration (s):                  154.53
Total input tokens:                      37205
Total generated tokens:                  38314
Request throughput (req/s):              0.52
Input token throughput (tok/s):          240.76
Output token throughput (tok/s):         247.94
Peak output token throughput (tok/s):    464.00
Peak concurrent requests:                18
Total token throughput (tok/s):          488.70
Concurrency:                             13.94
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   26926.31
Median E2E Latency (ms):                 26597.51
P90 E2E Latency (ms):                    49470.68
P99 E2E Latency (ms):                    57541.02
---------------Time to First Token----------------
Mean TTFT (ms):                          1145.82
Median TTFT (ms):                        1292.52
P99 TTFT (ms):                           2607.65
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          56.05
Median TPOT (ms):                        54.35
P99 TPOT (ms):                           102.02
---------------Inter-Token Latency----------------
Mean ITL (ms):                           53.94
Median ITL (ms):                         34.51
P95 ITL (ms):                            37.43
P99 ITL (ms):                            1256.92
Max ITL (ms):                            2518.61
==================================================
```

> Total throughput is within 0.5% of V3 on the same workload (V3: 491.26 tok/s, R1: 488.70 tok/s) — expected since R1 shares V3's architecture and serving path.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Response contains raw `<think>` text instead of `reasoning_content` | `--reasoning-parser` not set | Add `--reasoning-parser deepseek-r1` to the launch command. |
| Truncated thinking trace at low `max_tokens` | R1 thinking budgets are 2k-3k tokens before the final answer; client requests with `max_tokens=512` get cut off mid-trace | Set client `max_tokens >= 4096`. For accuracy benchmarks (AIME / MATH / GPQA), use `max_tokens >= 8192`. |
| `ValueError: dimension 0 must be divisible by tensor=64` during `_shard_weight` on `model.layers.0.mlp.gate_proj.weight_scale_inv (144, 56)` | Same V3 dense MLP block-quant constraint applies to R1. | Add `--dp-size 8`. See [`DeepSeek-V3.md` §5](DeepSeek-V3.md#5-troubleshooting) for the full V3 troubleshooting table — every row applies unchanged to R1. |
| Server up but greedy outputs are a single repeating token | Same shared-expert block-quant accuracy collapse as V3 at `dp=4`. | Use `--dp-size 8`. |
| `RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm` during EXTEND precompile | Same `dp=8` HBM tightness as V3. | Drop `--chunked-prefill-size` to 1024. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` on a shared PVC across all 16 nodes and across pod restarts. Use a different cache dir than V3 so the two models do not interfere. |

For V3-shared issues (block-quant accuracy collapse, OOM tuning, `t_packing` precompile assert, GKE control-plane blip handling), see [`DeepSeek-V3.md` §5](DeepSeek-V3.md#5-troubleshooting) — the V3 table is exhaustive and applies to R1 unchanged.

## Additional Resources

- [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [`DeepSeek-V3.md`](DeepSeek-V3.md) — V3 non-reasoning base; primary reference for mesh / MoE / FP8 configuration.
- [`DeepSeek-V2.md`](DeepSeek-V2.md) — V2 / V2-Lite generation.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
