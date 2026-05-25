---
title: "Kimi-Linear"
---

# Kimi-Linear on SGL-JAX

> **Validated recipe** — empirically validated on TPU v6e-16 (sglang-jax `b2daa46d`, 2026-05-25). Multi-host scaling to v6e-32 still pending.

## 1. Model Introduction

[**moonshotai/Kimi-Linear**](https://huggingface.co/moonshotai) is Moonshot AI's linear-attention decoder series built on **Kimi Delta Attention** with a hybrid recurrent state pool. The currently released checkpoint is a 48B MoE with 3B activated parameters.

**Variants**:

- [**moonshotai/Kimi-Linear-48B-A3B-Instruct**](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) — 48B total / 3B activated, instruction-tuned.

For other linear-attention models in the cookbook see [`Ling-2.6.md`](../InclusionAI/Ling-2.6.md) (InclusionAI's trillion-scale linear-attention MoE).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024` (Kimi defaults — verify against the model card).

**License**: see the [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter target)

| Tier | Model | TPU | Topology | Nodes | Chips | `--tp-size` | Notes |
|---|---|---|---|---|---|---|---|
| Minimum runnable | Kimi-Linear-48B-A3B | v6e-16 | 4x4 | 4 | 16 | 16 | BF16 ~96 GB — multi-host required to fit weights + recurrent state pool |
| Recommended production | Kimi-Linear-48B-A3B | v6e-32 | 4x8 | 8 | 32 | 32 | More HBM per active expert and larger recurrent state budget for long-prompt linear-attention workloads |

See [`../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../../get_started/install.md). Multi-host recommended at this size — use [`../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-16

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=kimi-linear`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

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

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**Mandatory: `--disable-radix-cache` for hybrid recurrent state:**
- Kimi-Linear is a hybrid recurrent state model — radix prefix sharing is unsafe with the recurrent state pool. Without `--disable-radix-cache` startup asserts: `Hybrid recurrent state models require --disable-radix-cache (prefix sharing is unsafe with recurrent state)`.

**Recurrent State Pool (linear-attention specific):**
- `--recurrent-state-memory-ratio 0.9` (default) budgets the recurrent state pool against the KV cache. The recurrent pool gets `available * ratio / (1 + ratio)` of free HBM.
- Lower the ratio (e.g. `0.5`) if KV cache is your bottleneck — long prompts with small recurrent state benefit from more KV.
- `--max-recurrent-state-size` (unset by default — auto) caps recurrent state slots across DP ranks; set only when you need a hard ceiling.

**Memory Management:**
- `--mem-fraction-static 0.9` for dedicated multi-host serving. Drop to `0.88` (TPU default) if you hit OOM at startup with high `--max-running-requests`.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill — Kimi-Linear's linear-attention benefit is most visible on long prompts.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Mount a shared PVC across the cluster's 4 nodes to amortize compilation.

For full flag definitions see [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="moonshotai/Kimi-Linear-48B-A3B-Instruct"` and replace `127.0.0.1` with your rank-0 internal IP, with the §1 recommended sampling parameters; for long-context streaming see §3.2.

### 3.2 Long-Context Streaming

Linear attention's win is amortized prefill cost on long prompts — stream the response so first-token latency is the only thing the user waits for, then tokens arrive at steady-state TPOT (~20 ms on v6e-16, see §4.1):

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

> Kimi-Linear-Instruct ships with a chat template but no native reasoning or tool-call format. For reasoning / tool-call workloads, use a model with `--reasoning-parser` / `--tool-call-parser` support — see [`Qwen3.md` §3.2 / §3.3](../Qwen/Qwen3.md) or [`MiMo-V2.5-Pro.md` §3.2 / §3.3](../Xiaomi/MiMo-V2.5-Pro.md).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — single-workload latency baseline.** Measured on Kimi-Linear-48B-A3B-Instruct v6e-16 with `bench_serving` random 1024→1024, max-concurrency 16, build `b2daa46d`.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to `moonshotai/Kimi-Linear-48B-A3B-Instruct`, remove the vLLM half).

**Test Results** — Kimi-Linear-48B-A3B-Instruct (TPU v6e-16):

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Successful requests:                     100
Benchmark duration (s):                  148.28
Request throughput (req/s):              0.67
Input token throughput (tok/s):          690.57
Output token throughput (tok/s):         690.57
Peak output token throughput (tok/s):    832.00
Total token throughput (tok/s):          1381.14
Mean E2E Latency (ms):                   21854.88
Mean TTFT (ms):                          607.66
Mean TPOT (ms):                          20.77
==================================================
```

v6e-32 multi-host: _Pending — run and PR back._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-16 (4 nodes × 4 chips) |
| Model | moonshotai/Kimi-Linear-48B-A3B-Instruct (BF16) |
| Tensor Parallelism | 16 |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | sglang-jax `b2daa46d` (2026-05-25) |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-16).

**Benchmark Command** — example for GSM8K:

```bash
evalscope eval \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8
```

Recommended additional datasets: MMLU, GPQA Diamond, RULER (to exercise long-context linear-attention).

**Test Results** — Kimi-Linear-48B-A3B-Instruct (TPU v6e-16, sglang-jax `b2daa46d`):

| Model | Dataset | Limit | Score |
|:---|:---|:---|:---|
| Kimi-Linear-48B-A3B-Instruct | gsm8k main | 200 | **0.925** |

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AssertionError: Hybrid recurrent state models require --disable-radix-cache` at startup | Radix prefix sharing incompatible with recurrent state pool | Pass `--disable-radix-cache` (mandatory for Kimi-Linear) — see §2.4. |
| `RegisterTask DEADLINE_EXCEEDED` (5 min) on multi-host launch | Ranks dispatched too far apart — the JAX distributed service has a 5-min RPC deadline | Dispatch all ranks within seconds of each other (parallel `kubectl exec`); a >5-min stagger between ranks fails registration on the early-arriving ranks. |
| OOM at startup | Recurrent state + KV exceed budget | Lower `--recurrent-state-memory-ratio` (e.g. to 0.7) and/or `--mem-fraction-static` to 0.88. |
| Long-prompt requests stall | KV cache exhausted before recurrent state | Lower `--recurrent-state-memory-ratio` to give the KV cache more headroom. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [`Ling-2.6.md`](../InclusionAI/Ling-2.6.md) — InclusionAI's trillion-scale linear-attention MoE.
- [`../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
