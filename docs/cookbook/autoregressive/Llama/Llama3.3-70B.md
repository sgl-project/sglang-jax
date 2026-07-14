---
title: "Llama 3.3 70B"
---

# Llama 3.3 70B on SGL-JAX

> **Validated recipe** — empirically validated on TPU v6e-16 with sglang-jax 0.1.0 for accuracy and historical baseline; §4.2 now includes the recommended v7x-4 high-throughput `bench_serving` row.

## 1. Model Introduction

[**meta-llama/Llama-3.3-70B-Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) is Meta's 70B dense decoder from the Llama 3.3 release — multi-host serving required at BF16.

For Llama 4 see the upstream sgl-cookbook (`Llama/Llama4.md`).

**Key Features**:

- **70B dense decoder, multi-host required**: BF16 weights ~140 GB — needs v6e-16 minimum (validated, see §4).
- **Llama 3.3 Instruct**: Instruction-tuned chat model — strong general-purpose assistant; non-reasoning, no native tool-call format.
- **Grouped-Query Attention (GQA)**: `num_kv_heads=8` shrinks the KV cache vs full MHA, leaving more HBM headroom for concurrency and longer prompts.
- **128K context window**: Supports long-document inputs out of the box; pair with `--chunked-prefill-size 2048` to bound peak HBM on long prefills.
- **Production-validated**: GSM8K **0.950** on TPU v6e-16 with sglang-jax 0.1.0 (§4.1).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.9`, `max_tokens=1024` (Llama 3 Instruct defaults).

**License**: see the [Llama model card](https://huggingface.co/meta-llama) for the authoritative Meta Llama Community License terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Llama 3.3 70B | **v7x-4** | 2x2x1 | 1 | 4 chips / 8 devices | 8 | Recommended throughput recipe in §4.2. v7x exposes 2 JAX devices/chip. |
| Llama 3.3 70B | **v6e-16** | 4x4 | 4 | 16 | 16 | This is the slice we measured on. BF16 ~140 GB — fits with `--mem-fraction-static 0.85` (~8.75 GB weights/chip + ample KV headroom). |

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices (larger v6e, v7x variants), see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). Multi-host required — use [GKE Indexed Job launcher](/deployment/gke-indexed-job) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](/deployment/skypilot).

### 2.3 Launch

#### Multi-host — TPU v6e-16

Use [GKE Indexed Job launcher](/deployment/gke-indexed-job) with `<JOB>=llama-70b`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tp-size 16 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

For temporary v6e experiments, advanced users can adapt [SkyPilot launcher](/deployment/skypilot) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Memory Management:**
- v6e-16 (TP=16): `--mem-fraction-static 0.85` validated; ~8.75 GB weights per chip leaves ~18 GB headroom for KV at 32 GB HBM.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at 70B scale.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill on long prompts.
- `--max-running-requests 256` caps concurrent decodes; lower for tighter latency tails.

**Tensor Parallelism:**
- `--tp-size 16` on v6e-16 fully shards Llama 3.3 70B's GQA `num_kv_heads=8` (tensor axis must be a divisor of 8 — 16 maps cleanly: 2 chips per KV head). All ranks must be in the same TPU slice; verify `--nnodes` matches the slice node count.
- Multi-host coordination: every rank runs the same launch command; only `${NODE_RANK}` and `${MASTER_ADDR}` vary. Dispatch all ranks within seconds of each other (a >5-min stagger trips the JAX distributed RPC deadline).
- Stale libtpu lock: if a prior multi-host launch on the same pod failed, run `rm -f /tmp/libtpu_lockfile` on every rank before relaunching — a leftover lock surfaces as `Internal error when accessing libtpu multi-process lockfile` on rank 0. Worth scripting into the launch wrapper.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- The cache is per-node; mount a shared PVC at the cache directory to amortize compilation across all 4 nodes.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](/base/basic-api-usage).

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

> Llama 3 Instruct is non-reasoning and has no native tool-call format. For those workloads, see the **Parser key reference** in [Parser key reference](/autoregressive#parser-key-reference) for the list of cookbook recipes with reasoning / tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | meta-llama/Llama-3.3-70B-Instruct (BF16) |
| Tensor Parallelism | 16 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 v6e-16](/autoregressive/Llama/Llama3.3-70B#2-3-launch).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 16 \
  --limit 200
```

Recommended additional datasets: MMLU, GPQA Diamond, IFEval.

**Test Results**

| Dataset | Subset | Samples | Score |
|---|---|---|---|
| gsm8k | main | 200 | **0.950** |

### 4.2 Speed

> **High-throughput v7x-4 row.** This cookbook row uses fixed-length random requests (ISL=1024, OSL=1024), `max_concurrency=128`, 384 prompts, `random_range_ratio=1`, `seed=42`, and no warmup requests. DP scheduling uses `round_robin`.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-4 (1 node x 4 chips, 8 JAX devices) |
| Model | meta-llama/Llama-3.3-70B-Instruct (real BF16 weights) |
| Tensor Parallelism | 8 |
| Tested build | origin/main (`2d97c787f712f715784216f7c414a4f477ea8218`) |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path /models/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 32768 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --dp-schedule-policy round_robin \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/Llama-3.3-70B-Instruct \
  --tokenizer /models/Llama-3.3-70B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 384 --max-concurrency 128 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

| ISL | OSL | Max concurrency | Prompts | Input tok/s | Output tok/s | Peak output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Duration (s) | OK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 128 | 384 | 3609.65 | 3609.65 | 4992.00 | 4796.46 | 30.78 | 108.93 | 384 |

> Historical v6e-16 baseline: `1024/1024/c16`, 100 prompts, 882.17 output tok/s, 1040.00 peak output tok/s. The v7x-4 row above uses fewer chips and is the recommended throughput-oriented recipe.

## Additional Resources

- [Llama 3.3 70B model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
