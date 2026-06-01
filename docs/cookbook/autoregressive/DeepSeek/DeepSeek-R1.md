---
title: "DeepSeek R1"
---

# DeepSeek R1 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, reasoning_content streams correctly, GSM8K accuracy 98.0% (50 examples, thinking-on), `bench_serving` numbers in §4.2. TPU v7x path is still a starter target.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) is DeepSeek's 671B / 37B-activated reasoning-tuned MoE — RL-trained on long chain-of-thought for math, code, and graduate-level reasoning. The model emits `<think>` blocks that SGL-JAX exposes as `reasoning_content` via the `deepseek-r1` parser. The official checkpoint uses FP8 block-wise weights (`block_size=128`); `--dtype bfloat16` controls runtime compute/output dtype, not BF16 weight residency. Multi-host serving required.

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — 256 routed experts and 1 shared expert per MoE layer; first 3 layers are dense MLP. See §2.4 for the backend choice.
- **FP8 block-quant compatibility** — the per-rank `out_dim` of the shared expert `gate_proj` / `up_proj` must be **strictly greater than** `block_size_out=128`. This forces the v6e-64 mesh shape and is why `--dp-size 8` (effective tensor axis 8) is recommended over `--dp-size 4` (tensor axis 16, which collides with the block size — see §2.4).
- Reasoning surface needs `--reasoning-parser deepseek-r1` at launch — see [§3.2](#32-reasoning-thinking-enabled-streaming) for the streaming pattern.

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=4096+` (give room for thinking).

**License**: see the [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips / JAX devices | `--tp-size` | `--dp-size` | Tensor axis | `--ep-size` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| v6e-64 | 8x8 | 16 | 64 | 64 | 8 | 8 | 64 | ✅ validated | `dp=8` required for FP8 shared-expert block-quant compatibility (`2048/8=256 > 128 = block_size`); `dp=4` silently collapses. Dense MLP block-quant scale grid `(144, 56)` further requires `144 % tensor == 0`, so tensor=8 is the only working option. HBM is tight at `dp=8`; see §2.4 Memory Management. |
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

Mount a shared `JAX_COMPILATION_CACHE_DIR` on the same PVC as the model weights — first-time compile is ~4 minutes total (EXTEND ~70 s + DECODE ~3 min); subsequent restarts with the same mesh shape skip almost all of that.

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

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

For full cURL + native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). For thinking + content streaming see §3.2.

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

> R1 does not ship with a native tool-call format. For tool-call workloads, see the **Parser key reference** in [`../index.md`](../index.md#parser-key-reference) for the list of cookbook recipes with tool-call parsers registered.

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

> Recommended primary datasets where R1's reasoning advantage shows: **AIME 2025**, **MATH**, **GPQA Diamond**, **LiveCodeBench**. PR back results when you run them.

### 4.2 Speed

> **Layout F — single-workload sweep (one data point).** Standard chat (ISL=1000, OSL=1000, `max_concurrency=16`, 80 prompts, `seed=42`). Future PRs can add reasoning-typical workloads (e.g., OSL=4096) and concurrency sweeps.

**Test Environment** — same hardware/build as §4.1.

**Workload** — `bench_serving` with `--dataset-name random --random-input-len 1000 --random-output-len 1000 --num-prompts 80 --max-concurrency 16 --seed 42`.

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

> R1's throughput on this workload reflects MoE + MLA + FP8 block-quant on the validated `dp=8` mesh; future PRs can add reasoning-typical workloads (OSL=4096+) for trace-heavy scenarios.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Response contains raw `<think>` text instead of `reasoning_content` | `--reasoning-parser` not set | Add `--reasoning-parser deepseek-r1` to the launch command. |
| Truncated thinking trace at low `max_tokens` | R1 thinking budgets are 2k-3k tokens before the final answer; client requests with `max_tokens=512` get cut off mid-trace | Set client `max_tokens >= 4096`. For accuracy benchmarks (AIME / MATH / GPQA), use `max_tokens >= 8192`. |
| `ValueError: dimension 0 must be divisible by tensor=64` during `_shard_weight` on `model.layers.0.mlp.gate_proj.weight_scale_inv (144, 56)` | Tensor axis too large for the dense MLP block-quant scale grid. 144 = `intermediate_size(18432) / block_size(128)`. | Add `--dp-size 8` (or another `dp` that makes `tp_size/dp_size` a divisor of 144). |
| Server up but **all outputs are a single repeating token** (e.g., "爲了爲了爲了…") | Per-rank `out_dim` of the shared expert `gate_proj`/`up_proj` equals `block_size_out=128`, hitting the block-wise quant kernel's accuracy-collapse regime. At `dp=4` on v6e-64, `2048/16 = 128`. | Use `--dp-size 8` (gives `2048/8 = 256 > 128`). The `epmoe` path will assert explicitly; the `fused` path is silent — see §2.4 MoE Backend. |
| `RuntimeError: Block-wise kernel does not support out_dim=128 with block_size_out=128 (known to cause accuracy collapse)` | Same as above, surfaced by the `epmoe` assertion. | Same fix: `--dp-size 8`. Do **not** set `allow_narrow_n_blockwise=True` — it suppresses the guard, not the bug. |
| `RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm. Used 31.68G of 31.25G hbm. Exceeded hbm capacity by ~440M.` during EXTEND precompile | At `dp=8`, the per-rank trace peak with `--chunked-prefill-size 2048` overshoots HBM. | Drop `--chunked-prefill-size` to 1024. Lowering `--max-running-requests` alone does not help — the peak is in prefill, not decode. |
| `ValueError: Expected local_num_tokens=1 to be aligned to t_packing=2` | Using `--moe-backend fused` at low effective per-rank token count during decode precompile. | Switch to `--moe-backend epmoe` (current recommended default), or raise `--max-running-requests` until `(max / dp_size) / (ep_size / dp_size) >= t_packing`. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port (default 5000 in the cookbook manifest) is open between nodes. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` on a shared PVC across all 16 nodes and across pod restarts (mesh-shape-keyed; safe across `backoffLimit` retries). |
| GKE control-plane blip evicts all 16 pods mid-run (`kube-root-ca.crt not registered` / `gcsfuse.csi.storage.gke.io not found`) | Transient kube-system flap tainted nodes with NoExecute; default `backoffLimit: 0` collapsed the Job. | Set `backoffLimit: 16` (or higher) in the GKE Indexed Job manifest. Pods get replacements and the server comes back; JIT cache hit keeps recovery time short. |

## Additional Resources

- [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
