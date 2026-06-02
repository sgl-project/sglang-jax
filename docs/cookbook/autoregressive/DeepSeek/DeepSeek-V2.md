---
title: "DeepSeek V2"
---

# DeepSeek V2 on SGL-JAX

> **Validated recipe** — DeepSeek-V2-Lite / V2-Lite-Chat has TPU v6e-4 speed and GSM8K results.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-V2-Lite**](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) is the single-host "Lite" tier of DeepSeek's second-generation MoE decoder built on **MLA** (Multi-head Latent Attention) — 15.7B total / 2.4B activated; minimal MoE that fits a single TPU v6e-4 host.

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — `--moe-backend` choice matters (see §2.4).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the [DeepSeek-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | v6e-4  | 2x2 | 1 | 4  | 4  | 4  | BF16 ~32 GB — single host |

See [TPU topology reference](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [install guide](../../../get_started/install.md). For V2-Lite single-host use [Single-host Docker template](../../deployment/single-host-docker.md).
The required JAX TPU container image: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` (covers v5e / v5p / v6e Trillium / v7x Ironwood).

### 2.3 Launch

#### Single-host — TPU v6e-4 (DeepSeek-V2-Lite)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp-size 4 --ep-size 4 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --page-size 64 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

> **`--page-size` mandatory for MLA**: DeepSeek's MLA backend asserts `page_size > 1` (the MLA v2 kernel packs KV slots and infers effective page size from `cache_kv.shape[1] * kv_packing`). The default `--page-size 1` will hit `AssertionError: MLA attention backend does not support page_size=1` at startup. Use 64 (or any power-of-2 ≥ 2).

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (V2-Lite).

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.

**Memory Management:**
- V2-Lite: `--mem-fraction-static 0.88` (TPU default).

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.

For full flag definitions see [Launch flags reference](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](../../base/basic-api-usage.md).

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V2-Lite-Chat",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

## 4. Benchmark

### 4.1 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 |
| Model | deepseek-ai/DeepSeek-V2-Lite-Chat (BF16) |
| Tensor Parallelism | 4 |
| Expert Parallelism | 4 |
| Tested build | sglang-jax 0.1.0 |

> **Use the `-Chat` checkpoint for accuracy eval.** The base `DeepSeek-V2-Lite` ships without a chat template; evalscope's few-shot GSM8K prompt loops indefinitely against `/v1/chat/completions` (observed 0.014 score, `finish_reason: length`). The instruct-tuned `DeepSeek-V2-Lite-Chat` has the chat template and parses `\nThe answer is X` reliably.

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-deepseek-v2-lite) but swap `--model-path` to `deepseek-ai/DeepSeek-V2-Lite-Chat`.

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, HumanEval.

**Test Results** — V2-Lite-Chat (TPU v6e-4, sglang-jax 0.1.0):

| Model | Dataset | Limit | Score |
|:---|:---|:---|:---|
| DeepSeek-V2-Lite-Chat | gsm8k | 200 | **0.685** |
| DeepSeek-V2-Lite (base, anti-pattern reference) | gsm8k | 500 | 0.014 (chat-completions endpoint loops on 4-shot prompt; do not use base for chat-completions eval) |

### 4.2 Speed

> **Layout B — single-workload latency baseline.** Measured on V2-Lite v6e-4 with `bench_serving` random 512→128, max-concurrency 8, sglang-jax 0.1.0.

**Benchmark Command** — `bench_serving` driver with `--dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 100 --max-concurrency 8 --backend sgl-jax --model deepseek-ai/DeepSeek-V2-Lite --tokenizer deepseek-ai/DeepSeek-V2-Lite --host 127.0.0.1 --port 30000`.

**Test Results** — V2-Lite (TPU v6e-4):

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Successful requests:                     100
Benchmark duration (s):                  14.65
Request throughput (req/s):              6.83
Input token throughput (tok/s):          3495.51
Output token throughput (tok/s):         873.88
Peak output token throughput (tok/s):    1022.00
Total token throughput (tok/s):          4369.38
Mean E2E Latency (ms):                   1136.32
Mean TTFT (ms):                          194.98
Mean TPOT (ms):                          7.41
==================================================
```

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau | Wrong `--moe-backend` for EP size | V2-Lite uses `epmoe` at EP=4. |
| OOM at startup (V2-Lite) | `--mem-fraction-static` too high | Lower from default 0.88 to 0.85. Verify `--tp-size 4` matches v6e-4 chip count. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts. |

## Additional Resources

- [DeepSeek-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [Launch flags reference](../../base/launch-flags-reference.md)
- [Cross-recipe troubleshooting](../../troubleshooting.md) — cross-recipe generic issues.
