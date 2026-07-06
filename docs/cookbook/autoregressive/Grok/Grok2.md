---
title: "Grok-2"
---

# Grok-2 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, sanity output correct, `bench_serving` numbers in §4.1. **Accuracy intentionally omitted** — Grok-2 is a base model (no chat template; see §1) and the cookbook follows design §6.F: base model recipes skip §4.1 Accuracy (chat-format datasets via `/v1/chat/completions` mis-extract on base models — see §3.1 for the underlying chat-template + evalscope-extractor interaction). **Cookbook used `--moe-backend epmoe`** because the fused MoE backend fails to init on this small-EP large-mesh layout (8 experts on 64 chips) (see §2.4). **Grok-2 architecture**: `config.json` declares `num_local_experts=8 num_experts_per_tok=2` under `Grok1ForCausalLM` — i.e. **MoE with 8 experts, 2 active per token** (not dense). Launch flags below assume MoE (`--ep-size 8 --moe-backend epmoe`).

## 1. Model Introduction

[**xai-org/grok-2**](https://huggingface.co/xai-org/grok-2) is xAI's open-weight Grok-2 release — a **~269B-parameter MoE base model — not instruction-tuned** (8 experts, 2 active per token; ~70B active parameters) under the `Grok1ForCausalLM` runtime. xAI did not release a Grok-2-Chat / Grok-2-Instruct sibling; the HuggingFace repo has no `chat_template` in `tokenizer_config.json`. SGL-JAX serves it multi-host on TPU v6e-64 (validated below) with tensor + expert parallelism. The primary user-facing deployment path is GKE Indexed Job; SkyPilot is an advanced v6e experiment alternative.

**Key Features**:

- **~269B MoE / ~70B active** — 8 experts, 2 active per token (25% active fraction); served via SGL-JAX `Grok1ForCausalLM` runtime. The validated v6e-64 path uses `--moe-backend epmoe`.
- **Base model, not chat-tuned** — has no chat template; use the raw `/v1/completions` endpoint (see [§3.1](#31-basic-text-completion-base-model)), not `/v1/chat/completions`. xAI did not release a chat / instruct variant.
- **Pre-sharded TP=8 safetensors checkpoint** — files named `pytorch_model-NNNNN-TP-{000..007}.safetensors`. The 8 per-expert/per-shard files imply the checkpoint expects **TP to be a multiple of 8** when serving (matches `--ep-size 8` for the MoE experts).
- **GQA attention** — `num_attention_heads=64`, `num_key_value_heads=8` → 8 KV heads (sharding constraint: tensor axis must divide 8).
- **Long context** — `max_position_embeddings=131072` (128K tokens native).
- **Open weights** — community-licensed for self-hosted serving.

**Recommended Generation Parameters** (base-model text completion): `temperature=0.7`, `top_p=0.8`, `top_k=20`. xAI's reference chat playground also adds `presence_penalty=0.5`, but that is a chat-stack default tuned for multi-turn conversation; do not apply it to raw `/v1/completions` calls — it penalizes reusing prompt tokens and degrades structured outputs (math problems' variables, code identifiers, etc.).

**Tokenizer note**: Grok-2 weights ship **without a tokenizer file** — use the community tokenizer `alvarobartt/grok-2-tokenizer` via `--tokenizer-path`. See [Community tokenizer card](https://huggingface.co/alvarobartt/grok-2-tokenizer).

**License**: see the [HuggingFace model card](https://huggingface.co/xai-org/grok-2) for the authoritative xAI Community License terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | `--moe-backend` | Notes |
|---|---|---|---|---|---|---|---|
| **v6e-64** | 8x8 | 16 | 64 | 64 | 8 | `epmoe` | This is the slice we measured on. 64-chip slice; ~8.4 GB weights per chip, plenty of room for KV / activations. `--ep-size 8` matches the 8 experts (and the pre-sharded TP-{000..007} file layout). |

**Pre-sharded TP=8 constraint**: the checkpoint files are named `pytorch_model-NNNNN-TP-{000..007}.safetensors` — `--tp-size` must be a multiple of 8 so the loader can map each pre-shard onto a contiguous device slice. v6e-64 (`tp=64=8×8`) satisfies this.

For other slices (larger v6e, v7x variants, scaled-down configs), see [Adapting to other topologies](../../base/tpu-topology-reference.md#adapting-to-other-topologies) — the `--tp-size = chip_count × devices_per_chip` and `tp_size % 8 == 0` rules carry over directly.

### 2.2 Environment

Install per [Install guide](../../../get_started/install.md). **Build pin**: use sglang-jax 0.1.0 or later. For multi-host serving, use [GKE Indexed Job launcher](../../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [SkyPilot launcher](../../../deployment/skypilot.md).

The community tokenizer is downloaded on first launch — no extra pip needed beyond standard install. For evaluation, additionally install `evalscope`:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

#### Multi-host — TPU v6e-64

Use [GKE Indexed Job launcher](../../../deployment/gke-indexed-job.md) with `<JOB>=grok-2`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

```bash
  --model-path /models/grok-2 \
  --trust-remote-code \
  --tokenizer-path alvarobartt/grok-2-tokenizer \
  --tp-size 64 --ep-size 8 \
  --moe-backend epmoe \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 \
  --page-size 128 \
  --max-running-requests 128 \
  --random-seed 3 \
  --skip-server-warmup
```

Mount a shared `JAX_COMPILATION_CACHE_DIR` on the same PVC as the model weights — first cold compile is ~5-10 min, much faster than 1T-class models since the MoE kernel shape sweep is smaller.

For temporary v6e experiments, advanced users can adapt [SkyPilot launcher](../../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend and EP Sizing:**
- Grok-2 is MoE with 8 experts (2 active per token). Use `--ep-size 8` to align with the pre-sharded TP=8 checkpoint layout.
- `--moe-backend epmoe` is the validated backend on the v6e-64 path. The fused backend currently fails to initialize on this small-EP large-mesh layout; revisit if fused MoE support changes.
- `--ep-size` must divide `num_local_experts (=8)` evenly. EP 1 / 2 / 4 / 8 are valid; 16+ would mis-shard the expert dim.

**Mesh / TP Constraints:**
- Pre-sharded TP=8 checkpoint files force `--tp-size` to be a multiple of 8. v6e-64 → tp=64, v6e-32 → tp=32, v7x-16 → tp=32.
- On v6e-64 with default `--dp-size 1`, mesh is `(data=1, tensor=64)` and `tensor / ep_size = 8` becomes the within-EP TP shard count.
- GQA `num_kv_heads=8` means the attention KV head dim is sharded over 8 — combined with `--ep-size 8`, all the dimensional constraints converge on 8.

**Memory Management:**
- `--mem-fraction-static 0.88` is the safe default. At ~8.4 GB / chip weights on v6e-64, plenty of HBM is free for KV cache and activations — but the JIT trace can spike, so don't push to 0.95 without measuring.
- `--max-running-requests 128` is a balanced starter; raise if throughput plateaus and HBM stays low in `kv usage` logs.
- `--download-dir` was set to `/dev/shm` in earlier starters for fast cold load, but the validated path serves weights directly from the PVC at `/models/grok-2` (no `--download-dir` needed when `--model-path` already points at local storage).

**Throughput vs Latency:**
- `--page-size 128` reduces KV page-table overhead at this scale. Default `1` is much slower at high concurrency on similarly sized models.
- `--chunked-prefill-size 2048` bounds peak HBM during prefill. Larger values (4096) reduce TTFT on long prompts but risk prefill-time OOM.

**Tokenizer:**
- Without `--tokenizer-path alvarobartt/grok-2-tokenizer`, the server fails at startup since Grok-2 weights ship without a tokenizer file.
- The tokenizer requires an HF token if your network has rate limits — set `HF_TOKEN` env if needed.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR` is mandatory — without it, first request blocks ~5-10 min per node (smaller than 1T-class models but still non-trivial).
- On multi-node clusters, mount a shared PVC at the cache directory so all 16 nodes share warmups. Mesh shape (`data × tensor`) is part of the cache key; changing `--tp-size` or `--ep-size` invalidates the cache.

For full flag definitions and defaults see [Launch flags reference](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Text Completion (base model)

For the standard cURL / Python `requests` / OpenAI client / native `/generate` patterns see [Basic API usage](../../base/basic-api-usage.md). Grok-2 is a base model with no chat template — use the raw `/v1/completions` endpoint, not `/v1/chat/completions`. Replace `127.0.0.1` with your rank-0 internal IP:

```bash
curl -X POST http://<rank0-ip>:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai-org/grok-2",
    "prompt": "The capital city of France is",
    "max_tokens": 16,
    "temperature": 0,
    "top_k": 1
  }'
```

Python OpenAI client equivalent:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<rank0-ip>:30000/v1", api_key="EMPTY")

resp = client.completions.create(
    model="xai-org/grok-2",
    prompt="The capital city of France is",
    max_tokens=16,
    temperature=0,
)
print(resp.choices[0].text)
```

> **Why not `/v1/chat/completions`?** Grok-2 has no chat template — sending `messages` either fails (no template registered) or wraps the prompt in a community-grafted template the model wasn't trained on. The community-template path looks superficially OK on single-turn sanity prompts but silently degrades accuracy on chat-format eval datasets: the model doesn't emit EOS at the end of a short answer and continues in-context with self-generated follow-ups (per design §6.F base models skip §4 Accuracy entirely).

> Grok-2 has no hybrid reasoning or native tool-calling format. For those workloads, see the **Parser key reference** in [Parser key reference](../index.md#parser-key-reference) for the list of cookbook recipes with reasoning / tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.
>
> Accuracy section is omitted by design — see the banner + §1 for why base models skip it (per design §6.F).

### 4.1 Speed — single workload (low-concurrency latency baseline)

> **Layout F — single-workload sweep (one data point).** Standard chat (ISL=1000, OSL=1000), `max_concurrency=16`, 80 prompts, `seed=42`.

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-64 (16 nodes × 4 chips) |
| Model | xai-org/grok-2 (BF16) |
| Tensor Parallelism | 64 |
| Expert Parallelism | 8 |
| MoE Backend | `epmoe` |
| Tested build | sglang-jax 0.1.0 |

**Benchmark Command**

```bash
PYTHONPATH=/tmp/sglang-jax/python python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --model /models/grok-2 \
  --tokenizer alvarobartt/grok-2-tokenizer \
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
Benchmark duration (s):                  540.78
Total input tokens:                      37205
Total generated tokens:                  38314
Request throughput (req/s):              0.15
Input token throughput (tok/s):          68.80
Output token throughput (tok/s):         70.85
Peak output token throughput (tok/s):    96.00
Peak concurrent requests:                18
Total token throughput (tok/s):          139.65
Concurrency:                             12.89
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   87163.46
Median E2E Latency (ms):                 85736.22
P90 E2E Latency (ms):                    157043.93
P99 E2E Latency (ms):                    176535.69
---------------Time to First Token----------------
Mean TTFT (ms):                          756.62
Median TTFT (ms):                        632.52
P99 TTFT (ms):                           1765.84
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          185.63
Median TPOT (ms):                        181.85
P99 TPOT (ms):                           266.39
---------------Inter-Token Latency----------------
Mean ITL (ms):                           180.80
Median ITL (ms):                         174.79
P95 ITL (ms):                            175.14
P99 ITL (ms):                            461.90
Max ITL (ms):                            1270.39
==================================================
```

> Grok-2 throughput on this v6e-64 mesh is bottlenecked by small-EP MoE underutilization: with only 8 experts on 64 chips, the `--moe-backend epmoe` fallback (forced because `fused` crashes on this mesh, see §2.4) leaves most chips idle per token. This is a known limitation of the current fused MoE backend assumes large-EP; Grok-2's 8-expert layout sits below that assumption.

## Additional Resources

- [Grok-2 Model Card](https://huggingface.co/xai-org/grok-2)
- [Community tokenizer](https://huggingface.co/alvarobartt/grok-2-tokenizer)
- [GKE Indexed Job launcher](../../../deployment/gke-indexed-job.md) — primary multi-host launcher template.
- [SkyPilot launcher](../../../deployment/skypilot.md) — advanced v6e experiment alternative.
- [Launch flags reference](../../base/launch-flags-reference.md)
- [Cross-recipe troubleshooting](../../../deployment/troubleshooting.md) — cross-recipe generic issues.
