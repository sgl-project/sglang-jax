---
title: "Launch Flags Reference"
---

# Launch Flag Reference

Cookbook-relevant flags for `python -m sgl_jax.launch_server`, grouped by what you're tuning. All defaults and choices come from `python/sgl_jax/srt/server_args.py` (function `ServerArgs.add_cli_args`) — if this page disagrees with that file, the code wins.

> **For the full list**, run `python -m sgl_jax.launch_server --help` against the same checkout you're deploying. This page only covers the ~30 flags that appear in cookbook recipes.

## 0. Where the entrypoint lives

`launch_server.py` is a thin wrapper; all CLI parsing happens in `ServerArgs.from_cli()` which calls `ServerArgs.add_cli_args(parser)`. To grep:

```bash
grep -n 'add_argument' python/sgl_jax/srt/server_args.py
```

## 1. Model & runtime basics

| Flag | Default | Notes |
|---|---|---|
| `--model-path` / `--model` | required | Local folder or HuggingFace repo id. |
| `--trust-remote-code` | `False` | Required for models that ship custom modeling code (Qwen, MiMo, Bailing, ...). Almost every cookbook recipe sets this. |
| `--tokenizer-path` | = `model-path` | Override only when the tokenizer lives separately (e.g. Grok-2: `alvarobartt/grok-2-tokenizer`). |
| `--dtype` | `auto` | `auto` / `bfloat16` / `float16` / `float32` / `half` / `float`. Cookbook recipes always pin `bfloat16` on TPU. |
| `--device` | auto-detect (`tpu` on TPU hosts) | Rarely needed; `__post_init__` resolves to `tpu` when JAX sees TPUs. |
| `--download-dir` | `None` | HuggingFace cache dir. Grok-2 recipe pins `/dev/shm` for speed; benchmarks use `/tmp`. |
| `--random-seed` | `42` (post-init) | Affects sampling determinism. |

## 2. Parallelism

| Flag | Default | Notes |
|---|---|---|
| `--tensor-parallel-size` / `--tp-size` | `1` | Total JAX devices across all nodes. See [`tpu-topology-reference.md`](tpu-topology-reference.md) for the v7x 2-devices-per-chip rule. |
| `--data-parallel-size` / `--dp-size` | `1` | DP factor for the **attention** path only. Attention TP becomes `tp_size / dp_size`. MoE layers still run with full `ep_size`. |
| `--ep-size` | `1` | Expert parallelism. Typically `--ep-size == --tp-size` for MoE models. |

## 3. KV cache & sequence length

| Flag | Default | Notes |
|---|---|---|
| `--context-length` | `None` (model config) | Max context. MiMo recipes pin `262144` (256K). |
| `--max-seq-len` | `4096` | Per-request max generated sequence length. |
| `--page-size` | `1` | Tokens per KV page. Cookbook MoE recipes use `256` (required by SWA pool eviction in MiMo); benchmarks use `128`. |
| `--chunked-prefill-size` | `None` (→ `4096`) | Tokens per prefill chunk. `-1` disables chunking. |
| `--max-prefill-tokens` | `16384` | Cap on prefill batch tokens. |
| `--max-running-requests` | `None` | Concurrent decode bound. `128` for v6e-16, `512` for v6e-64 / v7x-16. |
| `--swa-full-tokens-ratio` | `0.8` | **Per-layer** KV-token ratio: `swa_tokens_per_layer / full_tokens_per_layer` (independent of how many SWA vs full layers the model has). E.g. `0.5` → each SWA layer gets half the KV tokens of each full layer. MiMo recipes pin much smaller values (0.15–0.25); see those recipes for the empirically tuned numbers. |
| `--disable-radix-cache` | `False` | Disables RadixAttention prefix sharing. Set in benchmarks where prefix caching would skew results. |

## 4. Memory

| Flag | Default | Notes |
|---|---|---|
| `--mem-fraction-static` | `None` (→ `0.88` on TPU) | HBM fraction for weights + KV cache. 0.92–0.95 typical for dedicated serving. |
| `--kv-cache-dtype` | `auto` | `auto` / `fp8_e5m2` / `fp8_e4m3` / `bf16`. Cookbook recipes don't override today. |

## 5. Attention & MoE backends

| Flag | Default | Choices | Notes |
|---|---|---|---|
| `--attention-backend` | `fa` | `native` / `fa` / `fa_mha` | `fa` = FlashAttention on Pallas (MHA / MLA). `fa_mha` forces MHA path for MLA models. |
| `--moe-backend` | `epmoe` | `epmoe` / `fused` / `auto` | Scale-dependent. At EP≤8 (single host v7x-8) `epmoe` wins on MiMo-V2-Flash; at EP≥16 (multi-node) `fused` wins. See [MiMo-V2-Flash recipe](../autoregressive/Xiaomi/MiMo-V2-Flash.md) for measured numbers. |

## 6. Networking & multi-node

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind to `0.0.0.0` for any non-localhost client. |
| `--port` | `30000` | HTTP server port. MiMo recipes use `30271`; Qwen/Grok recipes use `30000`. |
| `--nnodes` | `1` | Total node count. |
| `--node-rank` | `0` | Per-node rank, `0..nnodes-1`. SkyPilot exposes this as `${SKYPILOT_NODE_RANK}`, GKE Indexed Job as `${JOB_COMPLETION_INDEX}`. |
| `--dist-init-addr` | `None` | `host:port` of the rank-0 node for `jax.distributed` rendezvous. Conventional port: `5000` (MiMo) or any unused. |
| `--dist-timeout` | `None` | `jax.distributed.initialize` timeout. |

## 7. Lifecycle & observability

| Flag | Default | Notes |
|---|---|---|
| `--skip-server-warmup` | `False` | Skip warmup. Cookbook recipes universally set this — saves ~1 min, the JIT cache covers the cost. |
| `--log-level` | `info` | `debug` for kernel diagnostic, `info` is fine in prod. |
| `--enable-metrics` | `False` | Prometheus metrics endpoint. |

## 8. Reasoning / Tool calling parsers

| Flag | Default | Choices | Notes |
|---|---|---|---|
| `--reasoning-parser` | `None` | `deepseek-r1` / `qwen3` / `mimo` / `kimi` / `glm45` | Splits `<think>` blocks into `reasoning_content` on the OpenAI-compatible response (`ReasoningParser.DetectorMap` in `python/sgl_jax/srt/reasoning_parser.py`). |
| `--tool-call-parser` | `None` | `qwen25` / `qwen3_coder` / `mimo` / `glm47` / `glm45` | Parses tool/function-call output into `tool_calls` (`FunctionCallParser.ToolCallParserEnum` in `python/sgl_jax/srt/function_call/function_call_parser.py`). |

**Parser → recipe mapping** (current cookbook coverage):

| Parser | Cookbook recipes |
|---|---|
| `mimo` (reasoning + tool) | [`mimo-v2.5-pro.md`](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md) · [`mimo-v2-flash.md`](../autoregressive/Xiaomi/MiMo-V2-Flash.md) · [`mimo-7b.md`](../autoregressive/Xiaomi/MiMo-7B.md) |
| `deepseek-r1` (reasoning) | [`deepseek-v3.md`](../autoregressive/DeepSeek/DeepSeek-V3.md) (R1 / V3.2-Speciale) |
| `glm45` (reasoning + tool) | [`glm4-moe.md`](../autoregressive/GLM/GLM-4.5.md) (GLM-4.5 / 4.6) |
| `glm47` (tool) | [`glm5-moe.md`](../autoregressive/GLM/GLM-5.md) (GLM-5 family) |
| `qwen3` (reasoning), `qwen25` / `qwen3_coder` (tool) | _no Qwen recipe currently sets these — pick by model card on a per-checkpoint basis_ |
| `kimi` (reasoning) | _no Kimi recipe currently sets this_ |

For complete request/response examples see [`mimo-v2.5-pro.md` §5.2](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md#52-reasoning-thinking-enabled) (reasoning streaming) and [§5.3](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md#53-tool-calling) (tool calling).

## 9. Compilation cache (environment, not a flag)

Not a CLI flag, but every cookbook recipe sets it in the same place:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server ...
```

Without this the first request blocks for ~4 minutes while XLA/Pallas re-compiles every kernel. With it, subsequent restarts skip recompilation entirely. The path must be writable and persistent across restarts (e.g. host volume mount in Docker / GKE).

## 10. Flags intentionally omitted from this page

- **LoRA** (`--enable-lora`, `--lora-paths`, ...) — no cookbook recipe touches LoRA today.
- **Speculative decoding** (`--speculative-algorithm`, ...) — no cookbook recipe today.
- **Grammar / structured output** (`--grammar-backend`, ...) — covered by feature docs in `basic_usage/features/`.
- **Quantization** (`--quantization`) — TPU recipes use natively quantized checkpoints (FP8 for MiMo) without re-quantization flags.
- **Staged multimodal runtime flags** — registered by `MultimodalServerArgs.add_cli_args`; currently used by autoregressive VL recipes and diffusion recipes.

Run `--help` to see them in context.
