---
title: "Ling 2.6"
---

# Ling-2.6 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, greedy + raw completion correct, GSM8K accuracy 98.5% (200 examples, see §4.1), `bench_serving` numbers in §4.2. Pin to sglang-jax 0.1.0 (or any commit that includes the channel-wise FP8 QKV split fix); earlier builds crash at weight load. TPU v7x-16 path is still a starter target.

## 1. Model Introduction

[**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) is InclusionAI's 1T-parameter Ling 2.6 release — a trillion-scale MoE built on **linear / delta attention** with a **hybrid recurrent state** pool that shares HBM with the KV cache. Smaller siblings (e.g. `Ling-2.6-flash`) are released under the same [InclusionAI HF collection](https://huggingface.co/inclusionAI).

**Architectural distinguishers**:

- **Linear / delta attention** in place of standard softmax attention — most of the long-context benefit shows up here.
- **Hybrid recurrent state pool** — budgeted against the KV cache via `--recurrent-state-memory-ratio` (default `0.9`).

**Variants**:

- [**inclusionAI/Ling-2.6-1T**](https://huggingface.co/inclusionAI/Ling-2.6-1T) — full trillion-scale flagship; default focus of this page.
- Smaller Ling-2.6 variants — adapt the §2.3 launch command after picking a checkpoint.

For Moonshot AI's separate linear-attention model see [`Kimi-Linear.md`](../Moonshotai/Kimi-Linear.md).

**Recommended Generation Parameters**: see the Ling-2.6 model card for authoritative defaults. As a starter: `temperature=0.7`, `top_p=0.95`, `max_tokens=2048+` (give room if you enable reasoning mode).

**License**: see the [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--dp-size` | `--ep-size` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Ling-2.6-1T | v6e-64 | 8x8 | 16 | 64 | 64 | 8 | 64 | ✅ validated | Trillion-scale; multi-host mandatory. `dp=8` required (GLA `num_groups=8` ≤ tensor axis); `--disable-radix-cache` required (hybrid recurrent state). |
| Ling-2.6-1T | v7x-16 | 4x4 | 4  | 16 | 32 | 4 | 32 | 🚧 starter | v7x exposes 2 JAX devices per chip → `--tp-size 32`. Apply same `--dp-size`/`--disable-radix-cache` deltas. Not yet validated end-to-end. |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). **Build pin**: use sglang-jax `d9c98c80` or any later commit that includes the channel-wise FP8 `[out, 1]` QKV split fix; earlier builds crash at weight load with `TypeError: 'NoneType' object is not subscriptable` on Ling-2.6 (it's an upstream gap, not a Ling-specific bug — see §5 Troubleshooting). Multi-host required — use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

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

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=ling-2-6`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path inclusionAI/Ling-2.6-1T \
  --trust-remote-code \
  --tp-size 64 --dp-size 8 --ep-size 64 \
  --moe-backend fused \
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

Mount a shared `JAX_COMPILATION_CACHE_DIR` on the same PVC as the model weights — first-time compile is ~9 minutes total (EXTEND ~7 min + DECODE ~2 min) at this build because the GLA chunk kernel has many distinct shape configurations; subsequent restarts with the same mesh shape skip almost all of that.

#### Multi-host (GKE Indexed Job) — TPU v7x-16 (starter)

Use GKE with `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Change the launch flags above to:

```text
  --tp-size 32 --dp-size 4 --ep-size 32 \
```

Keep `--disable-radix-cache` and the rest of the v6e-64 starter values. Not yet validated end-to-end — open a PR with measured numbers when you run it.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.
- `--disable-radix-cache` is **required**, not optional. The server asserts on this at startup: `AssertionError: Hybrid recurrent state models require --disable-radix-cache (prefix sharing is unsafe with recurrent state)`.

**Mesh / GLA Constraint:**
- The GLA (linear attention) `GroupRMSNorm` uses `num_groups=8` and shards `num_groups` along the "tensor" mesh axis. **Effective tensor axis must be ≤ 8.** On v6e-64 that forces `--tp-size 64 --dp-size 8` (tensor axis = `tp/dp` = 8). Setting `--dp-size 1` builds tensor=64 and JIT trace crashes with `Sharding spec ('tensor',) implies that array axis 1 is partitioned 64 times, but does not evenly divide the dimension size 8`.
- Same constraint on v7x-16: `--tp-size 32 --dp-size 4` → tensor axis = 8.

**MoE Backend:**
- `--moe-backend fused` for `--ep-size ≥ 16` (both configs above). The fused EP size = mesh `data * tensor` = 8 * 8 = 64 on v6e-64, matching `--ep-size 64`. Switch to `epmoe` only at EP ≤ 8.

**FP8 Quantization (compressed-tensors):**
- Ling-2.6 ships compressed-tensors FP8 with `strategy="channel"` (per-output channel weight scales, dynamic per-token activation). The runtime auto-detects this — no `--quantization` flag needed. `--dtype bfloat16` controls runtime compute dtype, not weight residency.
- This is **not** DeepSeek-V3 block-wise FP8 — the loader path is different (`weight_block_size=None`). Builds before `d9c98c80` lack the channel-wise QKV split path and crash at weight load.

**Reasoning Mode:**
- If the Ling-2.6 checkpoint emits `<think>...</think>` blocks (verify per model card; some reasoning-tuned variants do, base instruct variants do not), add `--reasoning-parser deepseek-r1` to the launch command — that's the generic `<think>` parser, since no `ling-2-6` or `bailing` parser key is registered. The streaming Python client from [`Qwen3.md` §3.2](../Qwen/Qwen3.md#32-reasoning-thinking-on-default-thinking-off-optional) applies directly once the parser is set.

**Context Length:**
- Default `--context-length` is the model's native 256K (`262144`); pin lower to your workload's longest prompt + output if you want more KV/recurrent slots.

**Memory Management:**
- `--mem-fraction-static 0.88` on v6e-64. The cookbook starter was `0.92` but the `dp=8` mesh's EXTEND precompile peak (`bs=256, tokens=16384`) overshoots HBM at that value by ~130 MB. Drop to `0.85` if you also raise `--chunked-prefill-size` or `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at trillion-scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first request blocks ~9 min per node (GLA chunk kernel ships many distinct shape configurations).
- Mount a shared PVC across the cluster's nodes to amortize compilation. Mesh shape (`data × tensor`) is part of the cache key; changing `--dp-size` invalidates the cache.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="inclusionAI/Ling-2.6-1T"` with the §1 recommended sampling parameters; for thinking + content streaming see §3.2.

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

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | inclusionAI/Ling-2.6-1T (compressed-tensors FP8 native; runtime dtype bfloat16) |
| Tensor Parallelism | 64 (effective tensor axis 8 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 Multi-host (v6e-64)](#multi-host-gke-indexed-job--tpu-v6e-64).

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

### 4.2 Speed

> **Layout F — single-workload sweep (one data point).** Standard chat (ISL=1000, OSL=1000), `max_concurrency=16`, 80 prompts, `seed=42`. Future PRs can add long-context (OSL=4096+) and concurrency sweeps to validate the GLA long-context advantage.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | inclusionAI/Ling-2.6-1T (compressed-tensors FP8 native; runtime dtype bfloat16) |
| Tensor Parallelism | 64 (effective tensor axis 8 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Tested build | sglang-jax 0.1.0 |

**Deployment Command** — same as [§2.3 Multi-host (v6e-64)](#multi-host-gke-indexed-job--tpu-v6e-64).

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/Ling-2.6-1T \
  --tokenizer /models/Ling-2.6-1T \
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
Benchmark duration (s):                  297.83
Total input tokens:                      37205
Total generated tokens:                  38314
Request throughput (req/s):              0.27
Input token throughput (tok/s):          124.92
Output token throughput (tok/s):         128.64
Peak output token throughput (tok/s):    192.00
Peak concurrent requests:                18
Total token throughput (tok/s):          253.56
Concurrency:                             13.40
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   49902.98
Median E2E Latency (ms):                 50129.75
P90 E2E Latency (ms):                    91094.50
P99 E2E Latency (ms):                    102596.07
---------------Time to First Token----------------
Mean TTFT (ms):                          897.32
Median TTFT (ms):                        845.06
P99 TTFT (ms):                           1603.57
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          104.01
Median TPOT (ms):                        105.22
P99 TPOT (ms):                           125.02
---------------Inter-Token Latency----------------
Mean ITL (ms):                           102.54
Median ITL (ms):                         88.50
P95 ITL (ms):                            89.27
P99 ITL (ms):                            581.03
Max ITL (ms):                            1293.91
==================================================
```

> At the same workload (ISL=1000, OSL=1000, c=16), DeepSeek-V3 hits 491 tok/s and MiMo-V2.5-Pro hits 926 tok/s on the same v6e-64 hardware. Ling-2.6-1T's lower throughput reflects its much larger total parameter count (1T vs 671B / 309B) plus the GLA recurrent state pool overhead — decode is bound by the linear attention chunk kernel.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TypeError: 'NoneType' object is not subscriptable` in `weight_utils.py:_split_qkv_weight` | Build pre-dates the channel-wise FP8 QKV split fix; checkpoint has `weight_block_size=None` (compressed-tensors `strategy="channel"`). | Pin sglang-jax to `d9c98c80` (primatrix `docs/cookbook-migration` branch) or any later commit that lands this fix. |
| `AssertionError: Hybrid recurrent state models require --disable-radix-cache` at startup | Missing `--disable-radix-cache`. | Add `--disable-radix-cache` to the launch flags — it's mandatory for any hybrid recurrent state model, not optional. |
| `ValueError: ... axis 1 is partitioned 64 times, but does not evenly divide the dimension size 8` from `group_rmsnorm.py` during JIT trace | Effective tensor axis (`tp_size / dp_size`) > GLA `num_groups=8`. | Set `--dp-size` such that `tp_size / dp_size <= 8`. On v6e-64 use `--dp-size 8`; on v7x-16 use `--dp-size 4`. |
| `RESOURCE_EXHAUSTED: ... Used 31.37G of 31.25G hbm. Exceeded hbm capacity by ~130M` during EXTEND precompile | `--mem-fraction-static 0.92` overshoots HBM at `dp=8` mesh trace peak. | Drop `--mem-fraction-static` to `0.88` (current default). For more headroom also lower `--chunked-prefill-size` to 1024 or `--max-running-requests` to 128. |
| OOM at startup | Recurrent state + KV exceed budget | Lower `--recurrent-state-memory-ratio` (e.g. to 0.7) and/or `--mem-fraction-static` to 0.85. |
| Long-prompt requests stall | KV cache exhausted before recurrent state | Lower `--recurrent-state-memory-ratio` to give the KV cache more headroom. |
| MoE throughput plateau at EP ≥ 16 | Wrong `--moe-backend` | Switch to `--moe-backend fused`. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~9 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` on a shared PVC; the GLA chunk kernel ships many distinct shape configurations so cold compile is slower than dense models. |

## Additional Resources

- [Ling-2.6 model card](https://huggingface.co/inclusionAI/Ling-2.6-1T)
- [InclusionAI HF collection](https://huggingface.co/inclusionAI) — sibling checkpoints.
- [`Kimi-Linear.md`](../Moonshotai/Kimi-Linear.md) — Moonshot AI's separate linear-attention model.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
