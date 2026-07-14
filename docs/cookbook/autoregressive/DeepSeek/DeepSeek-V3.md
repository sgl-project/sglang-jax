---
title: "DeepSeek V3"
---

# DeepSeek V3 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, greedy output correct, GSM8K accuracy 97.5% (200 examples). §4.2 includes a v7x-16 launch and `bench_serving` template.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-V3**](https://huggingface.co/deepseek-ai/DeepSeek-V3) is DeepSeek's 671B / 37B-activated MoE flagship — built on **MLA** (Multi-head Latent Attention) with 256 routed experts and 1 shared expert per MoE layer. The official checkpoint uses FP8 block-wise weights (`block_size=128`); `--dtype bfloat16` controls runtime compute/output dtype, not BF16 weight residency. Multi-host serving is required.

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — see §2.4 for the backend choice (`epmoe` is the currently validated one at V3 scale on v6e-64; `fused` has known regression at this scale).
- **FP8 block-quant compatibility** — the per-rank `out_dim` of the shared expert `gate_proj` / `up_proj` must be **strictly greater than** `block_size_out=128`. This constraint forces the v6e-64 mesh shape and is why `--dp-size 8` (effective tensor axis 8) is recommended over `--dp-size 4` (tensor axis 16, which collides with the block size — see §2.4).
- **DSA** (DeepSeek Sparse Attention) on V3.2 — activated by model config; no extra launch flag.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V3) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips / JAX devices | `--tp-size` | `--dp-size` | Tensor axis | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|---|
| **v7x-16** | 2x2x4 | 4 | 16 chips / 32 devices | 32 | 4 | 8 | 32 | Launch and benchmark template in §4.2. v7x exposes 2 JAX devices/chip; keep tensor axis 8 for the FP8 block-quant and MLA layout. |
| **v6e-64** | 8x8 | 16 | 64 | 64 | 8 | 8 | 64 | This is the slice we measured on. `dp=8` is required for FP8 shared-expert block-quant compatibility (`2048/8=256 > 128 = block_size`); `dp=4` silently collapses. HBM is tight at `dp=8`; see §2.4 Memory Management. |

See [TPU topology reference](/base/tpu-topology-reference) for the TPU generation reference. For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](/base/tpu-topology-reference#adapting-to-other-topologies).

### 2.2 Environment

Install per [Install guide](/get_started/install). Multi-host required — use [GKE Indexed Job launcher](/deployment/gke-indexed-job) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](/deployment/skypilot).

For evaluation, additionally install `evalscope` in the client environment (any host that can reach the served `:30000` port — typically a port-forwarded client laptop):

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host — TPU v6e-64

Use [GKE Indexed Job launcher](/deployment/gke-indexed-job) with `<JOB>=deepseek-v3`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16` (transient GKE control-plane blips happen; a non-zero backoff lets the job survive). Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
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

**Tensor/Data Mesh Layout:**
- Mesh shape is `Mesh(data=dp_size, tensor=tp_size/dp_size)`. On v6e-64 with `--tp-size 64 --dp-size 8`, the tensor axis is **8**.
- Choose `--dp-size` so that the per-rank shared-expert `out_dim = moe_intermediate_size(2048) / tensor_axis` is **strictly greater than `block_size_out=128`**. At `tensor=16` (i.e., `dp=4`) you hit `2048/16 = 128` exactly, which trips the block-wise quantized matmul kernel's documented "accuracy collapse" regime (the `epmoe` path asserts; the `fused` path silently emits garbage tokens). At `tensor=8` (`dp=8`) you get 256 > 128, which is correct.
- The dense MLP block-quant scale `(144, 56)` for `gate_proj`/`up_proj` (first 3 layers) further requires `144 % tensor == 0`. Tensor axes 1/2/4/8/16 all satisfy this; tensor=32/64 do not. Combined with the shared-expert constraint above, **tensor=8 (i.e., `--dp-size 8`) is the only working option on v6e-64**.

**MoE Backend:**
- Use `--moe-backend epmoe` as the validated default for V3 at the current sglang-jax 0.1.0 build. EPMoE adds an "offline EPMoE scale → GMM layout" conversion step at load time and is slightly slower to load than `fused`, but it carries the accuracy-guard assertion that the `fused` kernel path is missing. The `fused` backend is known to produce collapsed greedy output at `dp=4` due to the shared-expert collapse described above.
- Despite the historical hint that "epmoe is only for EP ≤ 8," it runs correctly at EP=64 on v6e-64 — the hint is a throughput recommendation, not a correctness limit.

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.
- `--page-size 128` is **mandatory** for the MLA backend; smaller values trigger a startup assertion in the MLA pager.

**Memory Management:**
- HBM is genuinely tight at `dp=8` because attention/dense weights replicate 8x across DP groups (vs 4x at `dp=4`). The current settings (`--mem-fraction-static 0.88 --chunked-prefill-size 1024 --max-running-requests 64`) leave just enough headroom for the EXTEND precompile peak (`bs=64, tokens=8192`). Do **not** raise `--chunked-prefill-size` past 1024 or `--max-running-requests` past 64 without first measuring HBM headroom; the previous `chunked=2048` setting OOMed by ~440 MB.
- The official DeepSeek-V3 checkpoint is FP8. Do **not** add `--quantization fp8`; keep `--dtype bfloat16` for runtime compute dtype. FP8 auto-detection is driven by HF `quantization_config.quant_method == "fp8"`.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first request blocks ~4 min per node and is repeated on every restart.
- Mount a shared PVC at the cache directory to amortize compilation across all 16 nodes and across pod restarts. Mesh shape (`data × tensor`) is part of the cache key; changing `--dp-size` invalidates the cache.

For full flag definitions see [Launch flags reference](/base/launch-flags-reference).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [Basic API usage](/base/basic-api-usage).

Short Python OpenAI client example (replace `<rank0-ip>` with your rank-0 internal IP):

```python
from openai import OpenAI

client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

> DeepSeek V3 is non-reasoning and has no native tool-call format. For those workloads, see the **Parser key reference** in [Parser key reference](/autoregressive#parser-key-reference) for the list of cookbook recipes with reasoning / tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Accuracy — GSM8K

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | deepseek-ai/DeepSeek-V3 (official FP8 block-wise checkpoint; runtime dtype bfloat16) |
| Tensor Parallelism | 64 (effective tensor axis 8 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3](/autoregressive/DeepSeek/DeepSeek-V3#2-3-launch).

**Benchmark Command**

```bash
evalscope eval \
  --model /models/DeepSeek-V3 \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 200 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024}'
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| DeepSeek-V3 | gsm8k | AverageAccuracy | main | 200 | 0.975 |

Recommended additional datasets to PR back: MMLU, GPQA Diamond, HumanEval, LiveCodeBench.

### 4.2 Speed

> **v7x-16 benchmark template.** Fixed-length random requests (ISL=1024, OSL=1024), `max_concurrency=128`, 384 prompts, `random_range_ratio=1`, `seed=42`, and no warmup requests.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v7x-16 (4 nodes x 4 chips, 32 JAX devices) |
| Model | deepseek-ai/DeepSeek-V3 (real FP8 weights; runtime dtype bfloat16) |
| Tensor Parallelism | 32 (tensor axis 8 via `--dp-size 4`) |
| Data Parallelism | 4 |
| Expert Parallelism | 32 |
| Tested build | origin/main (`2d97c787f712f715784216f7c414a4f477ea8218`) |

**Serving Flags Used**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
  --model-path /models/DeepSeek-V3 \
  --trust-remote-code \
  --tp-size 32 --dp-size 4 --ep-size 32 \
  --moe-backend epmoe \
  --dtype bfloat16 \
  --context-length 32768 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.88 \
  --page-size 128 \
  --max-running-requests 128 \
  --attention-backend fa \
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
  --model /models/DeepSeek-V3 \
  --tokenizer /models/DeepSeek-V3 \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 384 --max-concurrency 128 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0
```

## Additional Resources

- [DeepSeek-V3 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [SGLang DeepSeek V3 guide](https://docs.sglang.io/docs/basic_usage/deepseek_v3)
- [Launch flags reference](/base/launch-flags-reference)
- [Cross-recipe troubleshooting](/deployment/troubleshooting) — cross-recipe generic issues.
