---
title: "Kimi-Linear"
---

# Kimi-Linear on SGL-JAX

> **Validated recipe** — empirically validated on TPU v6e-16 and TPU v6e-32 with sglang-jax 0.1.0.

## 1. Model Introduction

[**moonshotai/Kimi-Linear-48B-A3B-Instruct**](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) is Moonshot AI's linear-attention decoder built on **Kimi Delta Attention** with a hybrid recurrent state pool — a 48B MoE with 3B activated parameters, instruction-tuned. It is the only checkpoint currently released in the Kimi-Linear series.

**Key Features**:

- **Sparse MoE, 48B total / 3B activated**: Only 3B parameters fire per token — MoE economy with single-expert-class latency. Instruction-tuned chat model.
- **Kimi Delta Attention (linear attention)**: Replaces softmax attention with a linear-attention variant — prefill cost scales near-linearly with sequence length, big win on long prompts (see §3.2 long-context streaming).
- **Hybrid recurrent state pool**: Linear-attention layers carry a recurrent state alongside the KV cache; `--recurrent-state-memory-ratio` (default 0.9) budgets recurrent vs KV HBM. **Requires `--disable-radix-cache`** — prefix / radix caching for hybrid recurrent models is planned but not yet shipped.
- **Long-context throughput focus**: Designed to amortize prefill on tens-of-thousands-of-token prompts — pair with streaming output for best perceived latency.
- **Multi-host serving**: Validated on TPU v6e-16 and v6e-32; GSM8K **0.925 / 0.935** (§4.1).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024` (Kimi defaults — verify against the model card).

**License**: see the [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Kimi-Linear-48B-A3B | **v6e-16** | 4x4 | 4 | 16 | 16 | One of two slices we measured on. BF16 ~96 GB — multi-host required to fit weights + recurrent state pool. |
| Kimi-Linear-48B-A3B | **v6e-32** | 4x8 | 8 | 32 | 32 | The other slice we measured on. More HBM per active expert and larger recurrent state budget for long-prompt linear-attention workloads. |

See [TPU topology reference](../../base/tpu-topology-reference.md) for the TPU generation reference. For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](../../base/tpu-topology-reference.md#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](../../../get_started/install.md). Multi-host recommended at this size — use [GKE Indexed Job launcher](../../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](../../../deployment/skypilot.md).

### 2.3 Launch

Two slices we measured on, each as a separate launch path:

- **TPU v6e-16 (4 nodes, `4x4`)** — `--tp-size 16`.
- **TPU v6e-32 (8 nodes, `4x8`)** — `--tp-size 32`.

#### Multi-host — TPU v6e-16

Use [GKE Indexed Job launcher](../../../deployment/gke-indexed-job.md) with `<JOB>=kimi-linear`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --tp-size 16 \
  --recurrent-state-memory-ratio 0.9 \
  --disable-radix-cache \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

#### Multi-host — TPU v6e-32

Same template with `<TOPOLOGY>=4x8`, `parallelism: 8`, `completions: 8`, and `--tp-size 32`:

```bash
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --tp-size 32 \
  --recurrent-state-memory-ratio 0.9 \
  --disable-radix-cache \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

For temporary v6e experiments, advanced users can adapt [SkyPilot launcher](../../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Mandatory: `--disable-radix-cache` for hybrid recurrent state:**
- Kimi-Linear is a hybrid recurrent state model — prefix / radix caching for hybrid recurrent models is planned but not yet shipped. Until it lands, `--disable-radix-cache` is required at launch; without it the server asserts: `Hybrid recurrent state models require --disable-radix-cache`.

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.

**Memory Management:**
- `--mem-fraction-static 0.9` for dedicated multi-host serving. Drop to `0.88` (TPU default) if you hit OOM at startup with high `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill — Kimi-Linear's linear-attention benefit is most visible on long prompts.

**Multi-host launch:**
- Dispatch all ranks within seconds of each other — a >5-min stagger between ranks trips the JAX distributed service's RPC deadline (`RegisterTask DEADLINE_EXCEEDED`) on the early-arriving ranks. Fan the launch out (parallel `kubectl exec`) rather than starting ranks one by one.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's 4 nodes to amortize compilation.

For full flag definitions see [Launch flags reference](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](../../base/basic-api-usage.md). For long-context streaming see §3.2.

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

### 3.2 Long-Context Streaming

Linear attention's win is amortized prefill cost on long prompts — stream the response so first-token latency is the only thing the user waits for, then tokens arrive at steady-state TPOT (~20 ms on v6e-16, see §4.2):

```python
from openai import OpenAI
client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

with open("long_document.txt") as f:
    document = f.read()  # tens of thousands of tokens

stream = client.chat.completions.create(
    model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a careful technical reviewer."},
        {"role": "user", "content": f"Review the following document and list the top 5 issues:\n\n{document}"},
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

> Kimi-Linear-Instruct ships with a chat template but no native reasoning or tool-call format. For reasoning / tool-call workloads, pick a model with `--reasoning-parser` / `--tool-call-parser` support — see the **Parser key reference** table in [Parser key reference](../index.md#parser-key-reference) for the list of cookbook recipes with reasoning / tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) and TPU v6e-32 (8 nodes × 4 chips) |
| Model | moonshotai/Kimi-Linear-48B-A3B-Instruct (BF16) |
| Tensor Parallelism | 16 (v6e-16) / 32 (v6e-32) |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — see [§2.3](#multi-host--tpu-v6e-16) (v6e-16) or [§2.3](#multi-host--tpu-v6e-32) (v6e-32).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --limit 200 \
  --generation-config temperature=0,max_tokens=2048
```

Recommended additional datasets: MMLU, GPQA Diamond, RULER (to exercise long-context linear-attention).

**Test Results** — Kimi-Linear-48B-A3B-Instruct:

| Model | Hardware | Build | Dataset | Limit | Score |
|:---|:---|:---|:---|:---|:---|
| Kimi-Linear-48B-A3B-Instruct | TPU v6e-16 | sglang-jax 0.1.0 | gsm8k main | 200 | **0.925** |
| Kimi-Linear-48B-A3B-Instruct | TPU v6e-32 | sglang-jax 0.1.0 | gsm8k main | 200 | **0.935** |

Within the ±2.3 pp sampling-noise band at limit=200; v6e-32 doubling TP does not regress accuracy.

### 4.2 Speed

> **Layout B — single-workload latency baseline.** Measured on Kimi-Linear-48B-A3B-Instruct v6e-16 with `bench_serving` random 1024→1024, max-concurrency 16, sglang-jax 0.1.0.

**Benchmark Command** — `bench_serving` driver: `python3 -m sgl_jax.bench_serving --backend sgl-jax --model moonshotai/Kimi-Linear-48B-A3B-Instruct --tokenizer moonshotai/Kimi-Linear-48B-A3B-Instruct --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 16 --host 127.0.0.1 --port 30000`.

**Test Results** — Kimi-Linear-48B-A3B-Instruct, Layout B (`bench_serving` random 1024→1024, N=100, max-concurrency 16):

| Hardware | Build | Duration (s) | Total throughput (tok/s) | Output throughput (tok/s) | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---|---:|---:|---:|---:|---:|
| TPU v6e-16 | sglang-jax 0.1.0 | 148.28 | 1381.14 | 690.57 | 20.77 | 607.66 |
| TPU v6e-32 | sglang-jax 0.1.0 | 140.55 | 1457.14 | 728.57 | 19.72 | 526.89 |

v6e-32 delivers ~5% throughput / 13% TTFT improvement over v6e-16. Scaling is sublinear because Kimi-Linear's sparse 3B-activated MoE caps the per-token chip utilization — the larger slice's main win is HBM headroom for long-context recurrent state, not raw token throughput.

## Additional Resources

- [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [Launch flags reference](../../base/launch-flags-reference.md)
- [Cross-recipe troubleshooting](../../../deployment/troubleshooting.md) — cross-recipe generic issues.
