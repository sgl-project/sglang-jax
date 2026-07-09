---
title: "Ling 2.6"
---

# Ling-2.6 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, greedy + raw completion correct, GSM8K accuracy 98.5% (200 examples, see §4.1). §4.3 now includes the recommended balanced v7x-16 `bench_serving` row, with the historical v6e-64 baseline kept as context. Pin to sglang-jax 0.1.0 (or any commit that includes the channel-wise FP8 QKV split fix); earlier builds crash at weight load.

## 1. Model Introduction

[**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) is InclusionAI's 1T-parameter Ling 2.6 release — a trillion-scale MoE built on **linear / delta attention** with a **hybrid recurrent state** pool that shares HBM with the KV cache.

**Architectural distinguishers**:

- **Linear / delta attention** in place of standard softmax attention — most of the long-context benefit shows up here.
- **Hybrid recurrent state pool** — budgeted against the KV cache via `--recurrent-state-memory-ratio` (default `0.9`).

**Recommended Generation Parameters**: see the Ling-2.6 model card for authoritative defaults. As a starter: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` (give room if you enable reasoning mode).

**License**: see the [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--dp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|---|
| Ling-2.6-1T | **v6e-64** | 8x8 | 16 | 64 | 64 | 8 | 64 | This is the slice we measured on. Trillion-scale; multi-host mandatory. `dp=8` required (GLA `num_groups=8` ≤ tensor axis); `--disable-radix-cache` required (hybrid recurrent state). |
| Ling-2.6-1T | **v7x-16** | 2×2×4 | 4 | 16 | 32 | 8 | 32 | v7x exposes 2 JAX devices/chip, so `--tp-size` = 16 chips × 2 = 32. Runs the V2 fused MoE kernel (`--moe-backend fused_v2`), which cuts MoE-layer prefill latency ~53% vs V1. Same `dp=8` and `--disable-radix-cache` constraints as v6e-64. |

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). **Build pin**: use sglang-jax 0.1.0 or later — earlier builds crash at weight load on Ling-2.6's compressed-tensors FP8 QKV split. Multi-host required — use [GKE Indexed Job launcher](/deployment/gke-indexed-job). Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](/deployment/skypilot).

For evaluation, additionally install `evalscope` in the client environment:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host — TPU v6e-64

Launch the server on **every node**, varying `${NODE_RANK}` (`0..15`) and pointing all nodes at the rank-0 host via `--dist-init-addr`:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-2.6-1T \
  --trust-remote-code \
  --tp-size 64 --dp-size 8 --ep-size 64 \
  --moe-backend fused \
  --nnodes 16 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_IP}:10011 \
  --host 0.0.0.0 --port 30000 \
  --recurrent-state-memory-ratio 0.9 \
  --disable-radix-cache \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

> `--moe-backend` is tunable — `epmoe` (megablox GMM), `fused` (Pallas fused MoE V1), or `fused_v2` (Pallas fused MoE V2, double-buffered); the fused kernels require full EP (`--ep-size` = `--tp-size`). See §2.4.

On GKE, use the [GKE Indexed Job launcher](/deployment/gke-indexed-job) with `<JOB>=ling-2-6`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16`; the launcher injects `--nnodes`, `--node-rank`, `--dist-init-addr`, `--host`, and `--port`, so drop those and put the remaining model flags into `<LAUNCH_FLAGS>`. For temporary v6e experiments, advanced users can adapt the [SkyPilot launcher](/deployment/skypilot) with the same flags.

#### Multi-host — TPU v7x-16 (fused MoE V2)

On a `v7x-16` slice (`2×2×4`, 4 nodes, 16 chips → 32 JAX devices) serve Ling-2.6-1T with the V2 fused MoE kernel (`--moe-backend fused_v2`), which packs scatter, expert FFN, and gather into one Pallas call with double buffering and activation quantization — cutting MoE-layer prefill latency ~53% vs V1. Launch the server on **every node**, varying `${NODE_RANK}` (`0..3`) and pointing all nodes at the rank-0 host via `--dist-init-addr`:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path inclusionAI/Ling-2.6-1T \
  --trust-remote-code \
  --tp-size 32 --dp-size 8 --ep-size 32 \
  --moe-backend fused_v2 \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_IP}:10011 \
  --host 0.0.0.0 --port 30000 \
  --page-size 256 \
  --context-length 262144 \
  --chunked-prefill-size 2048 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --max-running-requests 512 \
  --dp-schedule-policy round_robin \
  --attention-backend fa \
  --disable-radix-cache \
  --skip-server-warmup \
  --log-level info
```

> `--moe-backend` is tunable — `epmoe` (megablox GMM), `fused` (Pallas fused MoE V1), or `fused_v2` (Pallas fused MoE V2, double-buffered); the fused kernels require full EP (`--ep-size` = `--tp-size`). See §2.4.

All nodes must sit in the same TPU slice and reach each other on the `--dist-init-addr` port (`10011` here, plus the handful of ports just above it that JAX derives for coordination — up to `dist_init_port + 6`) and the TPU process port (`8471`). The `dp=8` / `--disable-radix-cache` constraints from §2.4 apply here too. Beyond the v7x device count (`tp = ep = 32`) and `--moe-backend fused_v2`, this path also retunes several serving knobs vs. v6e-64 — `--page-size 256`, `--mem-fraction-static 0.85`, `--max-running-requests 512`, `--dp-schedule-policy round_robin`, `--attention-backend fa`, and `--context-length 262144` — so treat it as its own recipe rather than a one-flag diff.

### 2.4 Configuration Tips

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.
- `--disable-radix-cache` is **required**, not optional — prefix / radix caching for hybrid recurrent models is planned but not yet shipped. The server asserts at startup if you omit the flag: `AssertionError: Hybrid recurrent state models require --disable-radix-cache`.

**Mesh / GLA Constraint:**
- The GLA linear attention layer has a per-group RMSNorm with `num_groups=8`, sharded along the "tensor" mesh axis. **Effective tensor axis must be ≤ 8.** On v6e-64 that forces `--tp-size 64 --dp-size 8` (tensor axis = `tp/dp` = 8). Setting `--dp-size 1` builds tensor=64 and JIT trace crashes with `Sharding spec ('tensor',) implies that array axis 1 is partitioned 64 times, but does not evenly divide the dimension size 8`.

**MoE Backend:**
- `--moe-backend` selects the EP MoE implementation: `epmoe` (megablox GMM), `fused` (Pallas fused MoE V1), or `fused_v2` (Pallas fused MoE V2 — double-buffered with activation quantization, ~53% lower MoE-layer prefill latency vs V1). The v6e-64 recipe uses `fused`; the v7x-16 recipe uses `fused_v2`. The fused kernels (`fused` / `fused_v2`) require **full expert parallelism** — they treat the entire `data × tensor` mesh as the EP group, so set `--ep-size` equal to the total JAX device count (`--tp-size`): 64 on v6e-64, 32 on v7x-16.

**FP8 Quantization (compressed-tensors):**
- Ling-2.6 ships compressed-tensors FP8 with `strategy="channel"` (per-output channel weight scales, dynamic per-token activation). The runtime auto-detects this — no `--quantization` flag needed. `--dtype bfloat16` controls runtime compute dtype, not weight residency.
- This uses compressed-tensors channel-wise FP8, distinct from block-wise FP8 quantization (`weight_block_size=None` in HF config). Builds before sglang-jax 0.1.0 lack the channel-wise QKV split path and crash at weight load.

**Reasoning Mode:**
- If the Ling-2.6 checkpoint emits `<think>...</think>` blocks (verify per model card; some reasoning-tuned variants do, base instruct variants do not), add `--reasoning-parser deepseek-r1` to the launch command — that's the generic `<think>` parser, since no `ling-2-6` or `bailing` parser key is registered. See §3.2 for the streaming Python client that splits `reasoning_content` from `content`.

**Context Length:**
- Default `--context-length` is the model's native 256K (`262144`); pin lower to your workload's longest prompt + output if you want more KV/recurrent slots.

**Memory Management:**
- `--mem-fraction-static 0.88` on v6e-64. The cookbook starter was `0.92` but the `dp=8` mesh's EXTEND precompile peak (`bs=256, tokens=16384`) overshoots HBM at that value by ~130 MB. Drop to `0.85` if you also raise `--chunked-prefill-size` or `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at trillion-scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- Set `JAX_COMPILATION_CACHE_DIR` to a shared path (same PVC across all nodes) to persist the JIT cache across restarts.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](/base/basic-api-usage). For thinking + content streaming see §3.2.

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="inclusionAI/Ling-2.6-1T",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=2048,
)
print(resp.choices[0].message.content)
```

### 3.2 Reasoning (if supported by the checkpoint)

Ling-2.6 emits `<think>...</think>` blocks; reuse the generic `deepseek-r1` reasoning parser. Append `--reasoning-parser deepseek-r1` to the §2.3 launch command (see §2.4 Reasoning Mode). If your checkpoint supports the per-request thinking toggle, set `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`; stream `reasoning_content` separately from `content`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="inclusionAI/Ling-2.6-1T",
    messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    temperature=0.7,
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

For non-streaming requests, the field appears on `response.choices[0].message.reasoning_content`. To see the full set of `--reasoning-parser` keys available in your build, run `python -m sgl_jax.launch_server --help`.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy — GSM8K

**Deployment Command** — launch a server per [§2.3 Launch](/autoregressive/InclusionAI/Ling-2.6#2-3-launch).

**Benchmark Command**

```bash
evalscope eval \
  --model inclusionAI/Ling-2.6-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 200 \
  --generation-config '{"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| Ling-2.6-1T | gsm8k | AverageAccuracy | main | 200 | **0.985** |

> Recommended additional datasets: AIME 2025, GPQA Diamond (reasoning); MMLU (general); RULER (long-context linear-attention).

### 4.2 Accuracy — AIME 2026

Sanity-checks the quantized fused-MoE serving path on competition math: AIME 2026 (`MathArena/aime_2026`, 30 problems, pass@1; extracted answers exact-matched against the reference). Point `test/srt/run_eval.py` at the served Ling-2.6-1T endpoint.

**Deployment Command** — launch a server per [§2.3 Launch](/autoregressive/InclusionAI/Ling-2.6#2-3-launch).

**Benchmark Command**

```bash
python test/srt/run_eval.py \
  --base-url http://127.0.0.1:30000 \
  --model Ling-2.6-1T \
  --eval-name aime26 \
  --num-examples 30 \
  --num-threads 16 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 32768
```

**Test Results**

| Model | Dataset | Metric | Num | Score |
|:---|:---|:---|:---|:---|
| Ling-2.6-1T | aime26 | pass@1 | 30 | **0.867** (26 / 30) |

> Zero request errors; every response terminated normally (`finish_reason=stop`, no truncation at 32768 tokens) — the four misses are reasoning errors, not generation cutoffs. The quantized fused-MoE serving path therefore preserves competition-math accuracy.

### 4.3 Speed

> **Balanced v7x-16 throughput row.** This cookbook row uses one fixed-length random workload (ISL=1024, OSL=1024), `max_concurrency=32`, 160 prompts, `random_range_ratio=1`, `seed=42`, and no warmup requests. Radix cache is disabled and DP scheduling uses `round_robin`, so the result is not prefix-cache dependent.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-16 (4 nodes x 4 chips, 32 JAX devices) |
| Model | inclusionAI/Ling-2.6-1T (real weights, compressed-tensors FP8 native; runtime dtype bfloat16) |
| Tensor Parallelism | 32 (effective tensor axis 4 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 32 |
| Tested build | origin/main (`2d97c787f712f715784216f7c414a4f477ea8218`) |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path /models/Ling-2.6-1T \
  --trust-remote-code \
  --tp-size 32 --dp-size 8 --ep-size 32 \
  --moe-backend fused_v2 \
  --context-length 32768 \
  --chunked-prefill-size 2048 \
  --page-size 256 \
  --max-running-requests 64 \
  --attention-backend fa \
  --disable-radix-cache \
  --dp-schedule-policy round_robin \
  --precompile-bs-paddings 16 32 64 \
  --precompile-token-paddings 1024 2048 4096 8192 16384 \
  --skip-server-warmup \
  --nnodes 4 --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTER_ADDR} \
  --host 0.0.0.0 --port 30000
```

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/Ling-2.6-1T \
  --tokenizer /models/Ling-2.6-1T \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 160 --max-concurrency 32 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0
```

**Test Results**

| ISL | OSL | Max concurrency | Prompts | Input tok/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Duration (s) | OK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 32 | 160 | 1145.13 | 1145.13 | 1078.91 | 26.91 | 143.08 | 160 |

> This row is throughput-oriented and keeps the full latency metrics visible for reproducibility. The earlier v6e-64 cookbook point at `1000/1000/c16` measured 128.64 output tok/s; the v7x-16 row above is the recommended current throughput recipe.

## Additional Resources

- [Ling-2.6-1T model card](https://huggingface.co/inclusionAI/Ling-2.6-1T)
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
