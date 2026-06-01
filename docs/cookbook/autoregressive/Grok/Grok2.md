---
title: "Grok-2"
---

# Grok-2 on SGL-JAX

> **Validated recipe** — TPU v6e-64 path validated on sglang-jax 0.1.0: server starts, sanity output correct, `bench_serving` numbers in §4.1. **Accuracy intentionally omitted** — Grok-2 is a base model (no chat template; see §1) and the cookbook follows design §6.F: base model recipes skip §4.1 Accuracy (chat-format datasets via `/v1/chat/completions` mis-extract on base models — see §5 troubleshooting for the underlying chat-template + evalscope-extractor interaction). **Cookbook used `--moe-backend epmoe`** because the fused MoE backend fails to init on this small-EP large-mesh layout (8 experts on 64 chips) (see §5). **Grok-2 architecture**: `config.json` declares `num_local_experts=8 num_experts_per_tok=2` under `Grok1ForCausalLM` — i.e. **MoE with 8 experts, 2 active per token** (not dense). Launch flags below assume MoE (`--ep-size 8 --moe-backend epmoe`).

## 1. Model Introduction

[**xai-org/grok-2**](https://huggingface.co/xai-org/grok-2) is xAI's open-weight Grok-2 release — a **~269B-parameter MoE base model — not instruction-tuned** (8 experts, 2 active per token; ~70B active parameters) under the `Grok1ForCausalLM` runtime. xAI did not release a Grok-2-Chat / Grok-2-Instruct sibling; the HuggingFace repo has no `chat_template` in `tokenizer_config.json`. SGL-JAX serves it multi-host on TPU v6e-32 (starter) or v6e-64 (validated below) with tensor + expert parallelism. The primary user-facing deployment path is GKE Indexed Job; SkyPilot is an advanced v6e experiment alternative.

**Key Features**:

- **~269B MoE / ~70B active** — 8 experts, 2 active per token (25% active fraction); served via SGL-JAX `Grok1ForCausalLM` runtime with `--moe-backend fused` (or `epmoe` for small EP).
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

| Tier | TPU | Topology | Nodes | Chips | `--tp-size` | `--ep-size` | `--moe-backend` | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Recommended production | **v6e-64** | 8x8 | 16 | 64 | 64 | 8 | `fused` | 🧪 in validation | Primary validation target. 64-chip slice; ~8.4 GB weights per chip, plenty of room for KV / activations. `--ep-size 8` matches the 8 experts (and the pre-sharded TP-{000..007} file layout). |
| Minimum runnable | **v6e-32** | 4x8 | 8 | 32 | 32 | 8 | `fused` | 🚧 starter | Smaller v6e slice; same flags but `--tp-size 32`. Tighter HBM but should fit BF16. Not validated end-to-end yet. |
| Alternative | **v7x-16** | 4x4 | 4 | 16 (32 devices) | 32 | 8 | `fused` | 🚧 starter | v7x exposes 2 JAX devices per chip → 16 × 2 = 32. Same `--tp-size 32` shape as v6e-32 but with more HBM headroom. Not validated end-to-end yet. |

**Pre-sharded TP=8 constraint**: the checkpoint files are named `pytorch_model-NNNNN-TP-{000..007}.safetensors` — `--tp-size` must be a multiple of 8 so the loader can map each pre-shard onto a contiguous device slice. v6e-64 (`tp=64=8×8`), v6e-32 (`tp=32=8×4`), and v7x-16 (`tp=32=8×4` over 32 JAX devices) all satisfy this.

See [`../../base/tpu-topology-reference.md`](../../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../../get_started/install.md`](../../../get_started/install.md). **Build pin**: use sglang-jax 0.1.0 or later. For multi-host serving, use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) as the primary user-facing path. Advanced users running temporary v6e experiments can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md). The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

The community tokenizer is downloaded on first launch — no extra pip needed beyond standard install. For evaluation, additionally install `evalscope`:

```bash
pip install evalscope==0.17.1
```

### 2.3 Launch

Grok-2 is multi-host only; cannot fit single-host.

#### Multi-host (GKE Indexed Job) — TPU v6e-64 (16 nodes)

Use [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) with `<JOB>=grok-2`, `<ACCELERATOR>=tpu-v6e-slice`, `<TOPOLOGY>=8x8`, `parallelism: 16`, `completions: 16`, and `backoffLimit: 16`. Put these model-specific flags into `<LAUNCH_FLAGS>`:

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

#### Multi-host (GKE Indexed Job) — TPU v6e-32 (8 nodes, starter)

Change `<TOPOLOGY>=4x8`, `parallelism: 8`, `completions: 8`, and adjust the parallelism flags:

```text
  --tp-size 32 --ep-size 8 \
```

The other flags stay identical. v6e-32 is the smallest slice the pre-sharded TP=8 layout fits cleanly on; v6e-16 doesn't satisfy `tp_size % 8 == 0` constraints cleanly without dp.

For temporary v6e experiments, advanced users can adapt [`../../deployment/skypilot.md`](../../deployment/skypilot.md) with the same launch flags. The model recipe does not require users to run repository-local SkyPilot helper scripts.

### 2.4 Configuration Tips

**MoE Backend and EP Sizing:**
- Grok-2 is MoE with 8 experts (2 active per token). Use `--ep-size 8` to align with the pre-sharded TP=8 checkpoint layout.
- `--moe-backend fused` is correct for EP=8: `intermediate_size=32768`, and `32768 % 512 == 0` satisfies the fused kernel alignment constraint. `--moe-backend epmoe` is the safe fallback at this EP size if the fused path regresses.
- `--ep-size` must divide `num_local_experts (=8)` evenly. EP 1 / 2 / 4 / 8 are valid; 16+ would mis-shard the expert dim.

**Mesh / TP Constraints:**
- Pre-sharded TP=8 checkpoint files force `--tp-size` to be a multiple of 8. v6e-64 → tp=64, v6e-32 → tp=32, v7x-16 → tp=32.
- With `--moe-backend fused`, the fused kernel maps mesh `data * tensor` to the EP group. On v6e-64 with default `--dp-size 1`, mesh is `(data=1, tensor=64)` and `tensor / ep_size = 8` becomes the within-EP TP shard count.
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

For full flag definitions and defaults see [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Text Completion (base model)

For the standard cURL / Python `requests` / OpenAI client / native `/generate` patterns see [`../../base/basic-api-usage.md`](../../base/basic-api-usage.md). Grok-2 is a base model with no chat template — use the raw `/v1/completions` endpoint, not `/v1/chat/completions`. Replace `127.0.0.1` with your rank-0 internal IP:

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

> **Why not `/v1/chat/completions`?** Grok-2 has no chat template — sending `messages` either fails (no template registered) or wraps the prompt in a community-grafted template the model wasn't trained on. The community-template path looks superficially OK on single-turn sanity prompts but silently degrades accuracy on chat-format eval datasets: the model doesn't emit EOS at the end of a short answer and continues in-context with self-generated follow-ups (see §5 troubleshooting; per design §6.F base models skip §4 Accuracy entirely).

> Grok-2 has no hybrid reasoning or native tool-calling format. For those workloads, see the **Parser key reference** in [`../index.md`](../index.md#parser-key-reference) for the list of cookbook recipes with reasoning / tool-call parsers registered.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.
>
> Accuracy section is omitted by design — see banner + §5 for why base models skip it (per design §6.F).

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

> Grok-2 throughput on this v6e-64 mesh is bottlenecked by small-EP MoE underutilization: with only 8 experts on 64 chips, the `--moe-backend epmoe` fallback (forced because `fused` crashes on this mesh, see §5) leaves most chips idle per token. This is a known limitation of the current fused MoE backend assumes large-EP; Grok-2's 8-expert layout sits below that assumption.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `--moe-backend fused` crashes at init with a sharding-spec `ValueError` on the (data, tensor) mesh | fused MoE backend assumes the expert axis aligns with the combined (data × tensor) mesh; this small-EP / large-mesh layout (8 experts on 64 chips) doesn't satisfy that | Use `--moe-backend epmoe` (current cookbook default) — slower than fused for large-EP models but is the only working choice on this slice. |
| `/v1/chat/completions` returns empty / nonsense, or evalscope chat-format datasets (GSM8K few-shot / MT-Bench) score 10-30% despite sanity prompts working | Grok-2 is a base model — `tokenizer_config.json` has no `chat_template`. Sending `messages` either fails outright or wraps the prompt in a community-grafted template the model wasn't trained on; without that training the model fails to emit EOS at the "right" place and continues in-context with self-generated Q→A chains. evalscope's "last number in output" extractor then grabs a number from the model's self-generated follow-up question instead of the actual answer, marking correct cases as failed. | Use `/v1/completions` with raw prompts (see [§3.1](#31-basic-text-completion-base-model)) and **completion-style datasets** (MMLU / HellaSwag / BBH / ARC). For GSM8K specifically, use `python -m sglang.test.few_shot_gsm8k --num-questions 200` which constructs a raw 5-shot prompt and stops on `\nQuestion:`. **Never** run `evalscope ... --api-url http://.../v1/chat/completions` against a base model — full rule in [`cookbook-recipe-design.md` §6.F](../../cookbook-recipe-design.md). |
| Server outputs garbage / silent accuracy collapse | `--ep-size` missing or wrong; Grok-2 is MoE (8 experts, 2 active), not dense. Without `--ep-size 8 --moe-backend epmoe` (or `fused` once the upstream fused crash is fixed) the loader will route experts incorrectly. | Always set `--ep-size 8 --moe-backend epmoe`. The cookbook §2.3 template is correct; reject any "dense Grok-2" recipe variant. |
| Weight loader fails to align pre-sharded TP files | `--tp-size` not a multiple of 8. The checkpoint is pre-sharded `pytorch_model-NNNNN-TP-{000..007}.safetensors` so the loader expects to slice along 8 TP partitions. | Use `--tp-size` ∈ {8, 16, 24, 32, 40, ..., 64}. On v6e-64 use 64; on v6e-32 use 32; on v6e-16 (if attempted) needs dp to avoid breaking the constraint. |
| Tokenizer load fails at startup | `--tokenizer-path` missing or unreachable | Add `--tokenizer-path alvarobartt/grok-2-tokenizer`; if HF rate-limited, set `HF_TOKEN` env. |
| Multi-node hang at `jax.distributed.initialize` | `--dist-init-addr` unreachable from non-rank-0 nodes | Verify the rank-0 internal IP / pod DNS (`<JOB>-0.<JOB>-headless-svc:5000` on GKE) and that the chosen port is open between nodes. |
| OOM at startup | `--mem-fraction-static` too high relative to available HBM | Drop to `0.85`. Verify `--tp-size` matches the chip count expected by the slice. |
| MoE throughput plateau / very slow decode (~190 ms TPOT) | Inherent to small-EP MoE on large mesh. With 8 experts on 64 chips, only 16 chips fully utilized per token; epmoe path doesn't have a fused kernel. | This is a known limitation pending upstream fix for small-EP shapes. Until then, expect ~140 tok/s total throughput on v6e-64. |
| Slow cold start (~5-10 min per node) on every launch | JIT cache empty (`JAX_COMPILATION_CACHE_DIR` not persisted across launches) | Mount a shared PVC at the cache directory across all nodes. Mesh shape is part of the cache key; changing `--tp-size`/`--ep-size` invalidates it. |
| GKE control-plane blip evicts pods mid-run | Default `backoffLimit: 0` collapses the Job on transient node taint | Set `backoffLimit: 16` in the GKE Indexed Job manifest. Replacements pick up the warm JIT cache. |

## Additional Resources

- [Grok-2 Model Card](https://huggingface.co/xai-org/grok-2)
- [Community tokenizer](https://huggingface.co/alvarobartt/grok-2-tokenizer)
- [`../../deployment/gke-indexed-job.md`](../../deployment/gke-indexed-job.md) — primary multi-host launcher template.
- [`../../deployment/skypilot.md`](../../deployment/skypilot.md) — advanced v6e experiment alternative.
- [`../../base/launch-flags-reference.md`](../../base/launch-flags-reference.md)
- [`../../troubleshooting.md`](../../troubleshooting.md) — cross-recipe generic issues.
