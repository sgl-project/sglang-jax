---
title: "Llama 3.1"
---

# Llama 3.1 on SGL-JAX

> **Validated recipe** — empirically validated on TPU v6e-4 with sglang-jax 0.1.0; see §4 for measured numbers. Phi-3 / InternLM3 aliases below use the same launch path but are unmeasured.

## 1. Model Introduction

[**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) is Meta's 8B dense decoder from the Llama 3.1 release — comfortable single-host fit. The same SGL-JAX runtime path also serves the Llama-compatible Phi-3 and InternLM3 8B checkpoints.

**Variants** (pick by fine-tune):

- [**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — 8B instruction-tuned; default chat choice.
- [**microsoft/Phi-3.5-mini-instruct**](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) — 3.8B Phi-3.5 variant; runs on the same Llama path.
- [**internlm/internlm3-8b-instruct**](https://huggingface.co/internlm/internlm3-8b-instruct) — 8B InternLM3 variant; runs on the same Llama path.

For the 70B size (multi-host required) see [`Llama3.3-70B.md`](Llama3.3-70B.md). For Llama 4 see the upstream sgl-cookbook (`Llama/Llama4.md`).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.9`, `max_tokens=1024` (Llama 3 Instruct defaults).

**License**: see the [Llama model card](https://huggingface.co/meta-llama) for the authoritative Meta Llama Community License terms. Phi-3 / InternLM3 follow their own model-card licenses.

## 2. Deployment

### 2.1 Hardware Matrix

| Tier | Model | TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|
| Minimum runnable | Llama 3.1 8B / Phi-3.5 / InternLM3-8B | v6e-4 | 2x2 | 4 | 4 | BF16 ~16 GB — fits with headroom; lowest-cost single-host serving |
| Recommended production | Llama 3.1 8B / Phi-3.5 / InternLM3-8B | v6e-8 | 2x4 | 8 | 8 | More HBM headroom for higher `--max-running-requests` / longer context — same single-host class, no multi-node coordination |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md) and use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Swap `--model-path` to `microsoft/Phi-3.5-mini-instruct` or `internlm/internlm3-8b-instruct` for the aliased variants.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.88` is the TPU default. Raise to `0.9` for higher concurrency on a dedicated host.

**Paging / concurrency (mandatory):**
- `--page-size 128` is **mandatory**. Without it the attention backend defaults to `page_size=1` and the `Max running requests` constraint chain collapses `Final max_running_requests` to 1 — at concurrency=16 the bench serializes and output throughput drops ~9× (validated 2026-05-25: 156 tok/s without flag → 1449 tok/s with). Same constraint applies to the Phi-3 / InternLM3 aliases.
- `--max-running-requests 64` pairs with the page-size flag; raise/lower to match your `--max-concurrency` workload.

**Tensor Parallelism:**
- `--tp-size 4` matches v6e-4's 4 chips (v6e is 1:1 chip↔device). For v6e-8 use `--tp-size 8`. Llama 3.1 8B's GQA `num_kv_heads=8` constrains tensor axis to be a divisor of 8 — values 1/2/4/8 are safe.
- For multi-host scaling (e.g., Llama 3.3 70B) see [`Llama3.3-70B.md`](Llama3.3-70B.md).

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md).

Short Python OpenAI client example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

> Llama 3 Instruct is non-reasoning and has no native tool-call format. For those workloads choose a model with `--reasoning-parser` / `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | meta-llama/Llama-3.1-8B-Instruct (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 16 \
  --limit 200
```

Recommended additional datasets: MMLU, HumanEval, IFEval.

**Test Results**

| Dataset | Subset | Samples | Score |
|---|---|---|---|
| gsm8k | main | 200 | **0.825** |

### 4.2 Speed

> **Layout B — measured baseline.** Single-host TPU v6e-4, sglang-jax 0.1.0. Numbers are sgl-jax-only; vLLM-on-TPU comparison is not run.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | meta-llama/Llama-3.1-8B-Instruct (BF16) |
| Tensor Parallelism | 4 |
| Tested build | sglang-jax 0.1.0 |

**Benchmark Command**

```bash
python3 -m sgl_jax.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 512 \
  --num-prompts 100 --max-concurrency 16 \
  --host 127.0.0.1 --port 30000
```

**Test Results**

```
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  17.82
Total input tokens:                      26497
Total generated tokens:                  25820
Request throughput (req/s):              5.61
Input token throughput (tok/s):          1486.90
Output token throughput (tok/s):         1448.91
Peak output token throughput (tok/s):    1693.00
Total token throughput (tok/s):          2935.81
Mean E2E Latency (ms):                   2574.53
Mean TTFT (ms):                          35.33
Mean TPOT (ms):                          9.94
Median TPOT (ms):                        9.84
Mean ITL (ms):                           9.87
==================================================
```

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Weights + KV exceed budget at chosen `--mem-fraction-static` | Lower to 0.85. Verify `--tp-size 4` matches v6e-4 chip count. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker). |
| Phi-3 / InternLM3 fails to load | Missing `--trust-remote-code` | Add it to the launch command; both aliases ship custom modeling code. |
| Bench shows ~150 tok/s output and ~50s TTFT at concurrency=16 | `--page-size` defaulted to 1 → `Final max_running_requests: 1` (visible in launch log), requests serialize | Add `--page-size 128 --max-running-requests 64` to the launch command. See §2.4. |

## Additional Resources

- [Llama model collection](https://huggingface.co/meta-llama)
- [Phi-3 model card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [InternLM3 model card](https://huggingface.co/internlm/internlm3-8b-instruct)
- [`Llama3.3-70B.md`](Llama3.3-70B.md) — 70B multi-host sibling.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
