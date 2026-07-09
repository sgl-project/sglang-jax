# Server Arguments

This page is the top-level entry point for launch-time configuration. It groups the most common `python -m sgl_jax.launch_server` arguments by the serving decision they control.

For the complete list and exact defaults, run:

```bash
python -m sgl_jax.launch_server --help
```

The implementation source of truth is `python/sgl_jax/srt/server_args.py`.

## Common Launch Commands

Single-host TPU serving:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --host 0.0.0.0 \
  --port 30000
```

Multi-host serving adds distributed placement flags:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path <MODEL_PATH> \
  --trust-remote-code \
  --tp-size <TOTAL_JAX_DEVICES> \
  --nnodes <NODE_COUNT> \
  --node-rank <THIS_NODE_RANK> \
  --dist-init-addr <RANK0_HOST:PORT> \
  --host 0.0.0.0 \
  --port 30000
```

For reusable Docker, GKE, and SkyPilot launchers, see [Deployment](../deployment/index.md). For model-specific values, use the matching cookbook recipe.

## Model and Tokenizer

| Argument | Purpose |
|---|---|
| `--model-path`, `--model` | Local path or Hugging Face repo id for the model weights. |
| `--tokenizer-path` | Override the tokenizer location when it differs from the model path. |
| `--tokenizer-mode` | Select `auto` fast-tokenizer usage or force `slow` tokenization. |
| `--tokenizer-backend` | Select the tokenizer implementation, such as Hugging Face or `fastokens`. |
| `--skip-tokenizer-init` | Skip tokenizer initialization; requests must provide `input_ids`. |
| `--load-format` | Select checkpoint loading format, such as `auto`, `safetensors`, `dummy`, `gguf`, or `layered`. |
| `--model-loader-extra-config` | Pass load-format-specific JSON config through to the selected model loader. |
| `--trust-remote-code` | Allow model-specific code from the model repository. Common for Qwen-derived and MiMo recipes. |
| `--context-length` | Override the model context length from config. |
| `--max-seq-len` | Alias for setting the maximum model sequence length. |
| `--is-embedding` | Serve a CausalLM as an embedding model. |
| `--revision` | Pin a branch, tag, or commit for model loading. |
| `--model-impl` | Select SGLang-native, Transformers, or automatic model implementation. |
| `--model-layer-nums` | Load only the first N model layers, mainly for profiling or bring-up. |
| `--json-model-override-args` | Patch selected model config fields, such as `architectures`, before loading. |
| `--multimodal` | Use the multimodal server argument class and multimodal HTTP entrypoint. |

## HTTP and Request Handling

| Argument | Purpose |
|---|---|
| `--host` | HTTP bind host. Use `0.0.0.0` for non-local clients. |
| `--port` | HTTP server port. |
| `--skip-server-warmup` | Skip server warmup. Useful for quick smoke tests, risky for latency-sensitive deployment. |
| `--warmups` | Run named warmup functions before serving requests. |
| `--stream-interval` | Control token interval for streaming responses. |
| `--stream-output` | Return output as disjoint streamed segments. |
| `--grammar-backend` | Select the backend for structured output constraints. |
| `--constrained-json-whitespace-pattern` | Customize allowed JSON whitespace for llguidance constrained decoding. |
| `--constrained-json-disable-any-whitespace` | Force compact JSON output for llguidance constrained decoding. |
| `--api-key` | Require an API key for the OpenAI-compatible server. |
| `--served-model-name` | Override the model name returned by `/v1/models`. |
| `--file-storage-path` | Backend storage path for uploaded files. |
| `--reasoning-parser` | Parse reasoning-model outputs with a supported detector. |
| `--tool-call-parser` | Parse tool-call outputs for OpenAI-compatible function/tool calling. |
| `--preferred-sampling-params` | JSON sampling defaults returned by `/get_model_info`. |
| `--allow-auto-truncate` | Truncate over-length requests instead of returning an error. |
| `--enable-tokenizer-batch-encode` | Batch tokenization for multi-text requests when inputs are not multimodal or pre-tokenized. |

## Precision, Quantization, and Dtype

| Argument | Purpose |
|---|---|
| `--dtype` | Model weight and activation dtype: `auto`, `half`, `float16`, `bfloat16`, `float`, or `float32`. |
| `--dtype-config` | JSON string or JSON file path for submodule-specific dtype overrides. |
| `--quantization` | Weight quantization method. Supported choices include `awq`, `fp8`, `gptq`, `bitsandbytes`, and related variants. |
| `--quantization-config-path` | YAML quantization config path or built-in config name. |
| `--quantization-param-path` | JSON scale file for quantized KV cache paths. |
| `--kv-cache-dtype` | KV cache dtype: `auto`, `fp8_e5m2`, `fp8_e4m3`, or `bf16`. |

`--dtype-config` is an advanced escape hatch. It only affects model code paths that explicitly consume `DtypeConfig`; model-level `--dtype` remains the normal user-facing control.

## Parallelism and Distributed Serving

| Argument | Purpose |
|---|---|
| `--tensor-parallel-size`, `--tp-size` | Total tensor-parallel JAX devices. |
| `--data-parallel-size`, `--dp-size` | Data parallelism factor for supported execution paths. |
| `--dp-schedule-policy` | DP rank assignment policy. If unset, radix-cache serving uses `cache_aware`; `--disable-radix-cache` and Pathways PD use `min_running_queue`. Explicit choices are `cache_aware`, `shape_aware`, `min_running_queue`, and `round_robin`; use non-default choices as workload-specific tuning overrides. |
| `--ep-size` | Expert parallelism size for MoE models. |
| `--ep-num-redundant-experts` | Add redundant physical experts for EP load balancing. |
| `--ep-dispatch-algorithm` | Expert dispatch algorithm: `static`, `dynamic`, or `fake`. |
| `--enable-sequence-parallel` | Enable sequence parallelism. |
| `--nnodes` | Number of serving nodes. |
| `--node-rank` | Rank of the current node. |
| `--dist-init-addr` | Rank-0 rendezvous address for distributed initialization. |
| `--dist-timeout` | Distributed initialization timeout. |

For TPU topology math, see the cookbook `base/tpu-topology-reference.md` page.

## Memory and Scheduling

| Argument | Purpose |
|---|---|
| `--mem-fraction-static` | Fraction of memory reserved for model weights and cache pools. Lower it when startup or prefill OOMs. |
| `--max-running-requests` | Maximum number of active decode requests. |
| `--max-total-tokens` | Explicit token pool size; normally derived from memory limits. |
| `--chunked-prefill-size` | Prefill chunk size. `-1` disables chunked prefill. |
| `--enable-mixed-chunk` | Allow prefill and decode requests to share a batch when chunked prefill is enabled. |
| `--max-prefill-tokens` | Maximum prefill tokens admitted into a batch. |
| `--disable-overlap-schedule` | Disable scheduler/model-worker overlap. |
| `--schedule-policy` | Request scheduling policy. |
| `--schedule-conservativeness` | How conservatively the scheduler admits requests. |
| `--page-size` | KV page size. |
| `--swa-full-tokens-ratio` | Sliding-window/full-attention KV token ratio for hybrid attention models. |
| `--recurrent-state-memory-ratio` | Recurrent-state memory budget ratio for hybrid recurrent models such as Kimi-Linear. |
| `--max-recurrent-state-size` | Explicit total recurrent-state slot count across DP ranks. |
| `--disable-hybrid-swa-memory` | Disable the hybrid sliding-window/full-attention memory optimization. |
| `--disable-radix-cache` | Disable prefix cache reuse. This also makes the unset DP schedule policy resolve to `min_running_queue`; Pathways PD follows the same default because it uses chunk cache internally. |
| `--enable-unified-radix-tree` | Use unified radix tree mode. |
| `--hicache-storage` | Enable HiCache host KV offload with `none`, keep it off with `disable`, or reserve `file` for future file-backed storage. |
| `--hicache-ratio` | Size the HiCache host pool relative to the device KV pool. |
| `--hicache-write-through-threshold` | Minimum prefix hit count before backing a node up to host memory. |
| `--hicache-write-policy` | Select HiCache write-through, selective write-through, or write-back behavior. |

For deeper scheduler behavior, see [Scheduler](../architecture/03-scheduler.md) and [KV Cache](../architecture/07-kv-cache.md).

## Runtime, Logging, and Metrics

| Argument | Purpose |
|---|---|
| `--device` | Select the serving device, or leave unset for auto-detection. |
| `--device-indexes` | Restrict mesh construction to specific local device indexes. |
| `--random-seed` | Seed used by server-side sampling paths. |
| `--download-dir` | Hugging Face model download/cache directory. |
| `--sleep-on-idle` | Lower CPU usage while the server is idle. |
| `--watchdog-timeout` | Crash the server if a forward batch hangs beyond the timeout. |
| `--log-level`, `--log-level-http` | Control application and HTTP server logging verbosity. |
| `--log-requests`, `--log-requests-level` | Log request metadata and optionally sampling parameters or payloads. |
| `--crash-dump-folder` | Dump recent requests to a directory before crash handling. |
| `--enable-metrics` | Enable Prometheus metrics. |
| `--enable-metrics-for-all-schedulers` | Record scheduler metrics for all TP ranks instead of TP rank 0 only. |
| `--bucket-time-to-first-token`, `--bucket-inter-token-latency`, `--bucket-e2e-request-latency` | Customize latency histogram buckets for metrics. |
| `--decode-log-interval` | Control the decode-batch logging interval. |
| `--enable-request-time-stats-logging` | Emit per-request timing statistics in logs. |
| `--kv-events-config` | Enable NVIDIA Dynamo KV event publishing from a JSON config. |
| `--enable-cache-report` | Include cached-token counts in OpenAI-compatible usage details. |

## JIT and Kernel Backends

| Argument | Purpose |
|---|---|
| `--precompile-token-paddings` | Token buckets used for prefill JIT precompile. |
| `--precompile-bs-paddings` | Batch-size buckets used for decode JIT precompile. |
| `--disable-precompile` | Skip startup precompilation. Runtime JIT can still happen on first unseen shape. |
| `--attention-backend` | Attention backend: `fa`, `fa_mha`, or `native`. |
| `--moe-backend` | MoE backend: `epmoe`, `fused`, `fused_v2`, or `auto`. |
| `--disable-jax-allreduce-metadata` | Use the Pallas DMA allgather fallback for fused EP-MoE metadata. |
| `--enable-nan-detection` | Enable NaN detection for debugging. |

For the compiled model path, see [Global JIT Compile](global_jit_compile.md). For backend selection, see [Attention Backend](attention_backend.md).

## Feature-Specific Flags

| Area | Arguments |
|---|---|
| Speculative decoding | `--speculative-algorithm`, `--speculative-draft-model-path`, and related draft-token controls. |
| LoRA | `--enable-lora`, `--lora-paths`, `--max-loras-per-batch`, `--max-lora-rank`, `--lora-target-modules`, `--enable-static-lora`. |
| RL/control loops | `--enable-engine-loop-run-forever-daemon`. |
| MoE introspection | `--enable-return-routed-experts`; requests must also set `return_routed_experts=True`. |
| PD disaggregation | `--pd-disaggregation` for single-controller Pathways mode, or `--disaggregation-mode` plus the `--disaggregation-*` role and transfer flags for split prefill/decode roles. |
| Multimodal | `--multimodal`, plus model-specific stage config handled by the multimodal server argument class. |

Specialized feature flags should stay documented here at the launch level and in the relevant architecture page when they affect runtime internals.
