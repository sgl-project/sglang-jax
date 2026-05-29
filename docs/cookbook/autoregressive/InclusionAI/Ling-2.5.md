---
title: "Ling 2.5"
---

# Ling 2.5 on SGL-JAX

> **🚫 Not feasible on TPU v6e-64 with current sglang-jax (2026-05-28 audit).** Empirical attempt on sglang-jax `d9c98c80` with `--quantization fp8 --quantization-config-path fp8_bailing.yaml` consistently OOMs during JIT trace (`RESOURCE_EXHAUSTED: Attempting to allocate 128.00M. That was not possible. There are 87.93M free.`) — the result is the same after sweeping `--mem-fraction-static` 0.88→0.82, `--max-running-requests` 64→16, and `--chunked-prefill-size` 2048→1024. Root cause: `fp8_bailing.yaml` excludes MoE expert weight paths (`mlp.experts.wi_0|wi_1|wo`), so the bulk of the model stays BF16 even after on-the-fly quant, leaving an effective on-chip footprint of ~1.5-1.7 TB that pushes activation peaks past available HBM. See §5 Troubleshooting and `../../2026-05-21-recipe-command-audit/ling-2-5-1t/NOTES.md` for the full iteration history.
>
> **v6e-64 is blocked until one of these is true**:
> 1. The `fp8_bailing.yaml` quant scope is extended to cover `wi_0/wi_1/wo` (validate accuracy doesn't regress).
> 2. InclusionAI ships a pre-quantized FP8 / INT4 Ling-2.5-1T (they did this for Ling-2.6).
> 3. The workload runs on a higher-HBM platform (v6e-128, v7x-16+).
>
> **What was verified before the OOM** (worth keeping in this recipe as forward references):
> - Build pin `d9c98c80` correctly handles compressed-tensors channel-wise FP8 QKV split.
> - `--dp-size 8` mesh works (GLA `num_groups=8` constraint same as Ling-2.6-1T).
> - `--disable-radix-cache` requirement enforced as expected.
> - `--quantization fp8 --quantization-config-path fp8_bailing.yaml` activates correctly; the issue is *scope* of quant, not failure to activate.
> - Loader emits `No file found for weight: model.layers.{7,15,23,31,...}.attention.o_proj.weight` (every 8th layer = the MLA layers in the 1:7 MLA + Linear Attention hybrid). May be a Ling-2.5-specific name-mapping gap on top of the OOM. Not investigated further once OOM became reproducible.

## 1. Model Introduction

[**inclusionAI/Ling-2.5-1T**](https://huggingface.co/inclusionAI/Ling-2.5-1T) and [**inclusionAI/Ring-2.5-1T**](https://huggingface.co/inclusionAI/Ring-2.5-1T) are InclusionAI's trillion-parameter Ling 2.5 generation:

- **Ling-2.5-1T** — non-reasoning / instant model; 1T total parameters, 63B active parameters, 256K context extendable to 1M with YaRN.
- **Ring-2.5-1T** — thinking / reasoning model on the same Ling 2.5 architecture; 1T total parameters, 128K context extendable to 256K with YaRN.

SGL-JAX serves this family through `BailingMoeV2_5ForCausalLM` (`python/sgl_jax/srt/models/bailing_moe_linear.py`). The architecture is a hybrid linear-attention MoE path: Ling 2.5 upgrades the Ling 2.0 GQA stack to a **1:7 MLA + Lightning Linear Attention** mix, reducing KV-cache pressure for long-context decode while keeping full-attention layers for expressiveness.

**Runtime architecture fields to check before launch:**

- `architectures`: must include `BailingMoeV2_5ForCausalLM`.
- `model_type`: usually `bailing_hybrid`.
- MoE: 256 experts, 8 experts per token in the default SGL-JAX config path.
- Attention: `layers_block_type` selects `linear_attention` vs `attention` / `full_attention`; MLA layers use the DeepSeek-style MLA patch path.

For the newer 2.6 generation see [`Ling-2.6.md`](Ling-2.6.md).

**Recommended Generation Parameters**:

- Ling-2.5-1T: use model-card defaults first; starter values `temperature=0.7`, `top_p=0.95`, `max_tokens=4096`.
- Ring-2.5-1T: reasoning workloads usually need larger budgets; starter values `temperature=0.6`, `top_p=0.95`, `max_tokens=8192+`.

**License**: see each HuggingFace model card for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| Model | TPU | Topology | Nodes | Chips | `--tp-size` | `--dp-size` | `--ep-size` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Ling-2.5-1T | v6e-64 | 8x8 | 16 | 64 | 64 | 8 | 64 | 🚫 not feasible | BF16 1T (2 TB on disk) + W8A8 on-the-fly via `fp8_bailing.yaml` still leaves MoE expert weights BF16 → ~1.5-1.7 TB on-chip → OOM during JIT trace. See banner + §5. |
| Ling-2.5-1T | v7x-16 | 4x4 | 4 | 16 (32 devices) | 32 | 4 | 32 | 🚧 starter | v7x-16 has 96 GiB per JAX device = 3072 GB total HBM. Expected to fit BF16 1T headroom, but not yet validated end-to-end. Keep `--dp-size 4`, `--disable-radix-cache`, and the FP8 quant flags for symmetry. |
| Ring-2.5-1T | v6e-64 | 8x8 | 16 | 64 | 64 | 8 | 64 | 🚫 not feasible | Same arch as Ling-2.5-1T plus `--reasoning-parser deepseek-r1`. Inherits the same HBM blocker. |

> v6e has 32 GiB per JAX device. v7x exposes 2 JAX devices per chip with 96 GiB per JAX device, so `tpu-v7x-16` is 32 JAX devices. See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md).

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). **Build pin**: use sglang-jax `d9c98c80` (`primatrix/docs/cookbook-migration`) or later — adds the channel-wise FP8 `[out, 1]` QKV split fix needed for any compressed-tensors-style weight scale layout. For multi-host serving, use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md).

| Hardware Platform | Docker Image |
|---|---|
| TPU v6e (Trillium) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood) | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

For evaluation, additionally install `evalscope` in the client environment:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (Ling-2.5-1T)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=ling-2-5`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path inclusionAI/Ling-2.5-1T \
  --trust-remote-code \
  --tp-size 64 --dp-size 8 --ep-size 64 \
  --moe-backend fused \
  --quantization fp8 \
  --quantization-config-path fp8_bailing.yaml \
  --disable-radix-cache \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 64 \
  --skip-server-warmup
```

Mount a shared `JAX_COMPILATION_CACHE_DIR` on the same PVC as the model weights. Cold compile is ~9-12 min (1T params + GLA chunk kernel shape sweep).

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (Ring-2.5-1T)

Ring-2.5 is the thinking variant. Use the same topology and FP8/mesh/cache settings, swap the model path, and add the generic `<think>` parser (no `ring`/`ling` parser key is registered):

```text
  --model-path inclusionAI/Ring-2.5-1T \
  --reasoning-parser deepseek-r1 \
```

#### TPU v7x-16 candidate

For v7x-16, use GKE with `<ACCELERATOR>=tpu7x`, `<TOPOLOGY>=4x4`, `parallelism: 4`, and `completions: 4`. Change the parallelism flags to:

```text
  --tp-size 32 --dp-size 4 --ep-size 32 \
```

Keep `--disable-radix-cache` and `--quantization fp8 --quantization-config-path fp8_bailing.yaml`. v7x-16's larger HBM-per-device might let you drop the FP8 quant flag and run BF16 directly, but that's not yet validated.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**HBM fit — `--quantization fp8` is mandatory on v6e-64:**
- BF16 1T checkpoint = ~2 TB on disk. v6e-64 total HBM = 64 × 32 GiB = 2048 GiB. Without quantization, weights alone consume ~99% of HBM, leaving nothing for KV cache, recurrent state, or compile-time intermediates → load-time OOM.
- `--quantization fp8 --quantization-config-path fp8_bailing.yaml` quantizes non-MoE linear and MoE expert weights to FP8 on load (with MoE gates / shared expert variants / `logits_processor` left full-precision per `python/sgl_jax/srt/utils/quantization/configs/fp8_bailing.yaml`). Effective weight footprint drops to ~1 TB → ~16 GiB/chip, leaving ~16 GiB for KV + recurrent state + activations.
- Do **not** drop the `--dtype bfloat16` flag — it controls runtime compute/activation dtype, separate from the FP8 weight quant.

**Mesh / GLA Constraint:**
- The GLA (linear attention) `GroupRMSNorm` uses `num_groups=8` and shards `num_groups` along the "tensor" mesh axis. **Effective tensor axis must be ≤ 8.** On v6e-64 that forces `--tp-size 64 --dp-size 8` (tensor axis = `tp/dp` = 8). Setting `--dp-size 1` builds tensor=64 and the first forward pass crashes with `Sharding spec ('tensor',) implies that array axis 1 is partitioned 64 times, but does not evenly divide the dimension size 8`. Same constraint as Ling-2.6-1T.
- Same constraint on v7x-16: `--tp-size 32 --dp-size 4` → tensor axis = 8.

**Hybrid recurrent state:**
- `--disable-radix-cache` is **required**, not optional. The server asserts on startup: `AssertionError: Hybrid recurrent state models require --disable-radix-cache (prefix sharing is unsafe with recurrent state)`.
- `--recurrent-state-memory-ratio` default `0.9` is fine to start. Lower (e.g. `0.7`) only if KV cache becomes the bottleneck on long-prompt traffic.

**MoE backend and EP sizing:**
- Use `--moe-backend fused` for the 1T variants. The fused EP group = mesh `data * tensor` = 8 * 8 = 64 on v6e-64, matching `--ep-size 64`.
- Confirm `num_experts % ep_size == 0`. The default config path has 256 experts, so EP 64 and EP 32 both divide cleanly.

**Memory management:**
- Start at `--mem-fraction-static 0.88` (not 0.92). The `dp=8` mesh EXTEND precompile peak overshoots HBM at 0.92 on Ling-2.6-1T (same arch); the same applies here.
- Keep `--max-running-requests 64` for the first run. Raise only after measuring HBM headroom from server logs.

**Reasoning parser:**
- Ling-2.5-1T is the instant variant and should not need a reasoning parser.
- Ring-2.5-1T is the thinking variant. Start with `--reasoning-parser deepseek-r1` because no `ring` / `ling` parser key is registered in SGL-JAX today.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first compile blocks ~9-12 min per node (1T params + GLA chunk kernel shape sweep).
- Mount a shared PVC across the cluster's nodes to amortize compilation. Mesh shape (`data × tensor`) is part of the cache key; changing `--dp-size` invalidates the cache.

For full flag definitions see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

See [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Use `model="inclusionAI/Ling-2.5-1T"` (instant) or `inclusionAI/Ring-2.5-1T` (thinking) with the §1 recommended sampling parameters; for Ring-2.5 reasoning streaming see §3.2.

### 3.2 Reasoning (Ring-2.5)

Launch Ring-2.5 with `--reasoning-parser deepseek-r1`, then stream `reasoning_content` separately from final `content`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="inclusionAI/Ring-2.5-1T",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

### 3.3 Tool Calling

The Ling-2.5 model card advertises agent/tool-calling strength, but SGL-JAX currently has no Ling-specific `--tool-call-parser` key documented in the cookbook. Do not claim native tool-calling until a parser key and end-to-end response shape are validated. For tool-calling examples with known parser support, see [`../Qwen/Qwen3.md`](../Qwen/Qwen3.md) or [`../Xiaomi/MiMo-V2.5-Pro.md`](../Xiaomi/MiMo-V2.5-Pro.md).

## 4. Benchmark

> Benchmark data below is a placeholder. Add real `bench_serving` and `evalscope` output before upgrading this recipe from Starter.

### 4.1 Speed

> **Layout F — single-workload sweep (one data point).** Standard chat (ISL=1000, OSL=1000), `max_concurrency=16`, 80 prompts, `seed=42`. Same workload as DeepSeek-V3 / MiMo-V2.5-Pro / Ling-2.6-1T §4.2 for cross-model comparison.

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/Ling-2.5-1T \
  --tokenizer /models/Ling-2.5-1T \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1000 --random-output-len 1000 \
  --num-prompts 80 --max-concurrency 16 \
  --seed 42
```

**Test Results** — _Pending. Numbers will land here once the v6e-64 run completes._

### 4.2 Accuracy

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | inclusionAI/Ling-2.5-1T (BF16 checkpoint, runtime FP8 W8A8 via `fp8_bailing.yaml`) |
| Tensor Parallelism | 64 (effective tensor axis 8 via `--dp-size 8`) |
| Data Parallelism | 8 |
| Expert Parallelism | 64 |
| Recurrent State Memory Ratio | 0.9 |
| Tested build | _Pending — pin `d9c98c80` on `primatrix/docs/cookbook-migration`_ |

**Deployment Command** — same as [§2.3](#multi-host-gke-indexed-job--tpu-v6e-64-ling-25-1t).

**Benchmark Command** — example for GSM8K (Ling-2.5 instant variant):

```bash
evalscope eval \
  --model inclusionAI/Ling-2.5-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 200 \
  --generation-config '{"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048}'
```

For Ring-2.5, run reasoning-heavy datasets with larger output budgets:

```bash
evalscope eval \
  --model inclusionAI/Ring-2.5-1T \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime_2025 \
  --eval-batch-size 4 \
  --generation-config '{"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}'
```

**Test Results** — _Pending. Numbers will land here once the v6e-64 run completes._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RESOURCE_EXHAUSTED: Attempting to allocate 128.00M. That was not possible. There are 87.93M free` during JIT trace, after weight load completes | On v6e-64 with on-the-fly W8A8: `fp8_bailing.yaml` excludes MoE expert paths (`mlp.experts.wi_0|wi_1|wo`), so the bulk of MoE weights stay BF16 → effective ~1.5-1.7 TB on-chip → activation peaks have no HBM room. Reproduced with `mem-fraction-static` 0.88→0.82, `max-running-requests` 64→16, `chunked-prefill-size` 2048→1024 — number stays 128M-needed / 87M-free (the JIT trace buffer size is shape-fixed, not flag-tunable here). | (1) Extend `fp8_bailing.yaml` to cover `wi_0/wi_1/wo` (audit accuracy doesn't regress); (2) wait for a pre-quantized FP8 / INT4 Ling-2.5 release; (3) move to higher-HBM hardware (v7x-16+). |
| `No file found for weight: model.layers.{7,15,23,31,...}.attention.o_proj.weight` during weight load (every 8th layer) | Ling-2.5 uses a 1:7 MLA + Linear Attention hybrid; MLA layers store the output projection under a different key than what the loader expects (sglang-jax's BailingMoeV2_5 path expects `attention.o_proj.weight`). | Investigate the checkpoint vs the loader's expected key map. Likely fix is in `models/bailing_moe_linear.py` or the loader's HF→JAX key mapping for MLA layers. Not investigated end-to-end during this audit because the HBM OOM blocks the run regardless. |
| `TypeError: 'NoneType' object is not subscriptable` in `weight_utils.py:_split_qkv_weight` | Build pre-dates the channel-wise FP8 QKV split fix; the `--quantization fp8` path ships per-channel `[out, 1]` scales that the old code can't split. | Pin sglang-jax to `d9c98c80` (`primatrix/docs/cookbook-migration`) or later. |
| `AssertionError: Hybrid recurrent state models require --disable-radix-cache` at startup | Missing `--disable-radix-cache`. | Add `--disable-radix-cache` to the launch flags. Required for `bailing_hybrid` regardless of variant. |
| `ValueError: ... axis 1 is partitioned 64 times, but does not evenly divide the dimension size 8` from `group_rmsnorm.py` during JIT trace | Effective tensor axis (`tp_size / dp_size`) > GLA `num_groups=8`. | Set `--dp-size` so that `tp_size / dp_size <= 8`. On v6e-64 use `--dp-size 8`; on v7x-16 use `--dp-size 4`. |
| Model architecture is unsupported at startup | HF config `architectures` does not resolve to `BailingMoeV2_5ForCausalLM`. | Verify the model card and `config.json` match. Use a registered architecture override only after checking the model card. |
| Response contains raw `<think>` text for Ring-2.5 | Missing reasoning parser. | Add `--reasoning-parser deepseek-r1`. |
| MoE throughput plateau | EP mesh does not match fused MoE assumptions. | With `--moe-backend fused`, EP = mesh `data * tensor`. On the v6e-64 recipe that's 8 * 8 = 64, which must equal `--ep-size 64`. Always check `num_experts % ep_size == 0`. |
| First request takes ~9-12 min per node | Empty JIT cache (large kernel shape sweep for GLA chunk forward). | Persist `JAX_COMPILATION_CACHE_DIR` on a shared PVC across all nodes and restarts. |
| GKE control-plane blip evicts all 16 pods mid-run | Transient kube-system flap tainted nodes; default `backoffLimit: 0` collapsed the Job. | Set `backoffLimit: 16` in the GKE Indexed Job manifest. Pods get replacements and the server comes back; JIT cache hit keeps recovery time short. |

## Additional Resources

- [Ling-2.5-1T model card](https://huggingface.co/inclusionAI/Ling-2.5-1T)
- [Ring-2.5-1T model card](https://huggingface.co/inclusionAI/Ring-2.5-1T)
- [`Ling-2.6.md`](Ling-2.6.md) — newer Ling 2.6 generation.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
