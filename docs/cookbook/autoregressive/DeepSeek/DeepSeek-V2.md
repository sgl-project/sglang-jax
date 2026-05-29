---
title: "DeepSeek V2"
---

# DeepSeek V2 on SGL-JAX

> **Partially validated recipe** — DeepSeek-V2-Lite / V2-Lite-Chat has TPU v6e-4 speed and GSM8K results. **DeepSeek-V2 (236B) on v6e-32 is validated for sanity + bench but requires a one-line `gate.py` patch** as of `d9c98c80` (upstream bug: `routed_scaling_factor` is nested inside the `renormalize` branch, so V2's `norm_topk_prob=False` + `routed_scaling_factor=16.0` combination silently skips scaling and produces garbage output without the patch). The patch and full audit trail: [`../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md`](../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md). Until that fix lands upstream, apply the in-manifest patch shown there before deploying V2 full.

## 1. Model Introduction

[**deepseek-ai/DeepSeek-V2**](https://huggingface.co/deepseek-ai/DeepSeek-V2) is DeepSeek's second-generation MoE decoder built on **MLA** (Multi-head Latent Attention). The V2 generation ships in two sizes — a single-host "Lite" tier and a 236B multi-host flagship.

**Variants** (pick by size):

- [**deepseek-ai/DeepSeek-V2-Lite**](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) — 15.7B total / 2.4B activated; minimal MoE that fits single-host.
- [**deepseek-ai/DeepSeek-V2**](https://huggingface.co/deepseek-ai/DeepSeek-V2) — 236B total / 21B activated; multi-host on v6e-32.

For the 671B V3 flagship see [`DeepSeek-V3.md`](DeepSeek-V3.md). For the reasoning-tuned R1 see [`DeepSeek-R1.md`](DeepSeek-R1.md).

**Architectural notes**:

- **MLA** — uses the FlashAttention Pallas MLA kernel by default; no extra flag needed.
- **MoE with shared + routed experts** — `--moe-backend` choice matters (see §2.4).

**Recommended Generation Parameters**: `temperature=0.6`, `top_p=0.95`, `max_tokens=1024`.

**License**: see the [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V2) for the authoritative DeepSeek license terms.

## 2. Deployment

### 2.1 Hardware Matrix (starter targets)

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | v6e-4  | 2x2 | 1 | 4  | 4  | 4  | BF16 ~32 GB — single host |
| DeepSeek-V2      | v6e-32 | 4x8 | 8 | 32 | 32 | 32 | BF16 ~470 GB |

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). For V2-Lite single-host use [`../../deployment/single-host-docker.md`](../../deployment/single-host-docker.md). For V2 multi-host, use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4 (DeepSeek-V2-Lite)

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

> **`--page-size` mandatory for MLA**: DeepSeek's MLA backend asserts `page_size > 1` (the MLA v2 kernel packs KV slots and infers effective page size from `cache_kv.shape[1] * kv_packing`). The default `--page-size 1` will hit `AssertionError: MLA attention backend does not support page_size=1` at startup. Use 64 (or any power-of-2 ≥ 2). Same constraint applies to DeepSeek-R1 / V3.

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (DeepSeek-V2)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=deepseek-v2`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=4x8`, `parallelism: 8`, and `completions: 8`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path deepseek-ai/DeepSeek-V2 \
  --trust-remote-code \
  --tp-size 32 --ep-size 32 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 256 \
  --skip-server-warmup
```

> **Requires the `gate.py` patch as of `d9c98c80`** (one-line dedent moving `routed_scaling_factor *=` outside the `if self.renormalize:` block — DS-V2 is the only validated DeepSeek config with `norm_topk_prob=False` and `routed_scaling_factor > 1`). Apply via the in-manifest python heredoc shown in [`../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/manifest.yaml`](../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/manifest.yaml). Until upstream lands the fix.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend:**
- `--moe-backend epmoe` for `--ep-size ≤ 8` (V2-Lite).
- `--moe-backend epmoe` for V2 full on v6e-32 (after applying the gate.py patch — see banner). The fused backend requires `total_tokens % (ep_size × t_packing == 64)` at EP=32, which forces `--max-running-requests 64` and caps concurrency; epmoe has no such alignment constraint and is the path validated in `2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/`.
- `--moe-backend fused` for V3 / R1 on v6e-64 (validated separately — different `norm_topk_prob=True` config doesn't hit the gate.py bug).

**Upstream gate.py bug (V2-specific):**
- DS-V2 has `norm_topk_prob=False` and `routed_scaling_factor=16.0`. The current `gate.py:104-108` nests `routed_scaling_factor *=` inside `if self.renormalize:`, so the 16× scaling is silently skipped. Symptom: server boots clean, every prompt returns degenerate token loops (`，\n\n`, `& & &`). Fix: move the scaling block outside the renormalize branch (one-line dedent). See [`../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md`](../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md).

**MLA:**
- DeepSeek's MLA runs on the default `--attention-backend fa` (FlashAttention Pallas) — no override needed.

**Memory Management:**
- V2-Lite: `--mem-fraction-static 0.88` (TPU default).
- V2: start at `0.92` for dedicated multi-host serving; drop to `0.9` if OOM at startup.

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at MoE scale.
- `--chunked-prefill-size 2048` bounds peak HBM during long-prompt prefill.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min per node.
- Multi-node clusters: mount a shared PVC at the cache directory to amortize compilation.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="deepseek-ai/DeepSeek-V2"` (or `DeepSeek-V2-Lite`) with the §1 recommended sampling parameters.

> DeepSeek V2 (and V2-Lite) is non-reasoning and has no native tool-call format. For reasoning use [`DeepSeek-R1.md`](DeepSeek-R1.md); for tool-call workloads choose a model with `--tool-call-parser` support (e.g., [Qwen3](../Qwen/Qwen3.md), [MiMo-V2.5-Pro](../Xiaomi/MiMo-V2.5-Pro.md)).

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build`; not refreshed on every release.

### 4.1 Speed

> **Layout B — single-workload latency baseline.** Measured on V2-Lite v6e-4 with `bench_serving` random 512→128, max-concurrency 8, build `de29d9f0`.

**Benchmark Command** — adapt the driver from [`Qwen3.md` §4.1](../Qwen/Qwen3.md#41-speed--sgl-jax-vs-vllm) (swap `MODEL_NAME` to the DeepSeek-V2 checkpoint, remove the vLLM half).

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

V2 multi-host: Layout B (random 1024→1024, N=100, c=16) on v6e-32, build `d9c98c80` + gate.py patch, `--moe-backend epmoe`:

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Max request concurrency:                 16
Successful requests:                     100
Benchmark duration (s):                  142.03
Request throughput (req/s):              0.70
Input token throughput (tok/s):          720.96
Output token throughput (tok/s):         720.96
Peak output token throughput (tok/s):    912.00
Total token throughput (tok/s):          1441.92
Mean E2E Latency (ms):                   21075.12
Mean TTFT (ms):                          1537.69
Mean TPOT (ms):                          19.10
==================================================
```

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (V2-Lite) / v6e-32 (V2) |
| Model | deepseek-ai/DeepSeek-V2-Lite-Chat or DeepSeek-V2-Chat (BF16) |
| Tensor Parallelism | 4 (Lite) / 32 (V2) |
| Expert Parallelism | 4 (Lite) / 32 (V2) |
| Tested build | sglang-jax `de29d9f0` (2026-05-24) |

> **Use the `-Chat` checkpoint for accuracy eval.** The base `DeepSeek-V2-Lite` ships without a chat template; evalscope's few-shot GSM8K prompt loops indefinitely against `/v1/chat/completions` (observed 0.014 score, `finish_reason: length`). The instruct-tuned `DeepSeek-V2-Lite-Chat` has the chat template and parses `\nThe answer is X` reliably.

**Deployment Command** — same as [§2.3](#single-host-docker--tpu-v6e-4-deepseek-v2-lite) but swap `--model-path` to `deepseek-ai/DeepSeek-V2-Lite-Chat` (V2-Lite tier) or `deepseek-ai/DeepSeek-V2-Chat` (V2 multi-host).

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

**Test Results** — V2-Lite-Chat (TPU v6e-4, sglang-jax `fe092bf`):

| Model | Dataset | Limit | Score |
|:---|:---|:---|:---|
| DeepSeek-V2-Lite-Chat | gsm8k | 200 | **0.685** |
| DeepSeek-V2-Lite (base, anti-pattern reference) | gsm8k | 500 | 0.014 (chat-completions endpoint loops on 4-shot prompt; do not use base for chat-completions eval) |

V2 multi-host accuracy: _Pending — run on v6e-32 and PR back._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| MoE throughput plateau (V2) | Wrong `--moe-backend` for EP size | Use `--moe-backend epmoe` for V2 full on v6e-32 (validated path, no alignment quirks); `--moe-backend fused` requires `--max-running-requests 64` (caps concurrency). V2-Lite uses `epmoe` at EP=4. |
| V2 full @ v6e-32: every prompt returns garbage (`& & & ...`, `，\n\n\n...`, token salad) | Upstream `gate.py` bug — `routed_scaling_factor` nested inside `if self.renormalize:`; V2's `norm_topk_prob=False` + `routed_scaling_factor=16.0` makes the 16× scaling silently skipped | Patch `gate.py:104-108` to move scaling outside the renormalize branch. See [`../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md`](../../2026-05-21-recipe-command-audit/deepseek-v2-32-r2-epmoe-patched/NOTES.md) for the in-manifest patch heredoc. |
| V2 full @ v6e-32, `fused`: `num_tokens=N not aligned to ep_size=32` or `local_num_tokens=K not aligned to t_packing=2` | Fused EP MoE kernel requires `total_tokens % (ep_size × 2) == 0` (= 64 at EP=32) | Pin `--max-running-requests 64` (sanity passes, but caps concurrency vs `epmoe` which auto-caps to 102 from MLA). |
| OOM at startup (V2) | `--mem-fraction-static 0.92` too high | Lower to 0.9. Verify `--tp-size 32` matches v6e-32 chip count. |
| Multi-node hang at init | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP and that the chosen port is open. |
| First request takes ~4 min per node | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR`; mount a shared PVC across nodes for amortized compilation. |

## Additional Resources

- [DeepSeek-V2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V2)
- [DeepSeek-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [`DeepSeek-V3.md`](DeepSeek-V3.md) — 671B V3 flagship.
- [`DeepSeek-R1.md`](DeepSeek-R1.md) — reasoning-tuned V3 derivative.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
