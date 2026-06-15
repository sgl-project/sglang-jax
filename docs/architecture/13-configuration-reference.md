# Configuration Reference

## Module Overview

This document consolidates all configuration entrypoints in sglang-jax: `ServerArgs`, `GlobalConfig` (runtime global constants), `ModelConfig` (model configuration), `QuantizationConfig` (quantization configuration), `LoadConfig` (load configuration), and environment variables.

Core files involved:

- `server_args.py` — `ServerArgs`, `PortArgs`
- `sgl_jax/global_config.py` — `GlobalConfig`
- `configs/model_config.py` — `ModelConfig`
- `configs/quantization_config.py` — `QuantizationConfig`
- `configs/load_config.py` — `LoadConfig`, `LoadFormat`

## Prerequisite Reading

- [01-architecture-overview](01-architecture-overview.md) — System overview

---

## Configuration Priority

Before diving into the individual config classes, here is the system's overall configuration priority (high to low):

1. **CLI arguments** — Parsed via `argparse`, highest priority
2. **`__post_init__` derived defaults** — Fills `None` fields (e.g., `mem_fraction_static`)
3. **Environment variables** — `GlobalConfig` fields and runtime feature flags
4. **Dataclass field defaults** — Static defaults from class definition
5. **HuggingFace model configuration** — `config.json` values, overridable via `json_model_override_args`

**Quantization config priority**:

1. `--quantization-config-path` (user-provided YAML)
2. HuggingFace `quantization_config` (model-shipped)
3. `None` (no quantization)

**Context length priority**:

1. `--context-length` (user-specified; must be `<= derived value` or set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=True`)
2. Value derived from HuggingFace config
3. Fallback `2048`

---

## 13.1 ServerArgs

`ServerArgs` (`server_args.py`) is the main configuration for system startup.

### 13.1.0 High-Frequency Fields At a Glance

The table below summarizes the most commonly used launch parameters, recommended for newcomers; see later sections for detailed groupings and defaults:

| Field | Default | Description |
|------|--------|------|
| `model_path` | (required) | HuggingFace Hub ID or local weights path |
| `tp_size` | `1` | Tensor parallel degree, determines the width of the mesh `"tensor"` axis |
| `dp_size` | `1` | Data parallel degree; the same Scheduler partitions requests and batches by DP rank |
| `ep_size` | `1` | Expert parallel degree, only takes effect for MoE models |
| `page_size` | `1` | KV cache page size; 16 is recommended for MHA/GQA, MLA requires `>1` |
| `mem_fraction_static` | `0.88` | Fraction of device memory reserved for KV cache |
| `max_seq_len` | `4096` | Maximum sequence length the model can process |
| `chunked_prefill_size` | `4096` | Maximum tokens per prefill chunk |
| `max_running_requests` | auto-derived | Concurrency limit, constrained by KV cache / pool capacity |
| `attention_backend` | `"fa"` | Attention backend selection (`fa` / `fa_mha` / `native`) |
| `moe_backend` | `"epmoe"` | MoE backend selection (`epmoe` / `fused` / `auto`) |

### 13.1.1 Model and Tokenizer

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `model_path` | `str` | (required) | HuggingFace Hub ID or local path |
| `tokenizer_path` | `str \| None` | `None` | Tokenizer path (defaults to `model_path`) |
| `tokenizer_mode` | `str` | `"auto"` | `"auto"` / `"slow"` |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer init, pass `input_ids` directly |
| `load_format` | `str` | `"auto"` | Load format (`auto` / `safetensors` / `dummy` / `remote` etc.) |
| `model_loader_extra_config` | `str` | `"{}"` | Extra model load config (JSON) |
| `trust_remote_code` | `bool` | `False` | Allow loading custom model code from the Hub |
| `context_length` | `int \| None` | `None` | Override the model's max context length |
| `is_embedding` | `bool` | `False` | Use a CausalLM as an embedding model |
| `revision` | `str \| None` | `None` | Model revision (branch / tag / commit ID) |
| `model_impl` | `str` | `"auto"` | `"auto"` / `"sglang"` / `"transformers"` |

### 13.1.2 HTTP Service

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `host` | `str` | `"127.0.0.1"` | HTTP server listen address |
| `port` | `int` | `30000` | HTTP server port |
| `skip_server_warmup` | `bool` | `False` | Skip startup warmup |
| `warmups` | `str \| None` | `None` | Comma-separated custom warmup function names |

### 13.1.3 Quantization and Dtype

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `dtype` | `str` | `"auto"` | Model dtype (`auto` / `bfloat16` / `float32`, etc.) |
| `quantization` | `str \| None` | `None` | Quantization scheme name (`fp8` / `awq` / `gptq` / `w8a8_int8`, etc.) |
| `quantization_param_path` | `str \| None` | `None` | Path to KV cache scaling factors JSON |
| `quantization_config_path` | `str \| None` | `None` | Path to quantization config YAML |
| `kv_cache_dtype` | `str` | `"auto"` | KV cache dtype (`auto` / `fp8_e5m2` / `fp8_e4m3` / `bf16`) |

### 13.1.4 Memory and Scheduling

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `mem_fraction_static` | `float \| None` | `None` | Static memory fraction (GPU/TPU default `0.88`, CPU default `0.5/process_count`) |
| `max_running_requests` | `int \| None` | `None` | Max concurrent requests |
| `max_total_tokens` | `int \| None` | `None` | Max KV-cache tokens (auto-computed) |
| `max_prefill_tokens` | `int` | `16384` | Max tokens per prefill batch |
| `chunked_prefill_size` | `int \| None` | `None` | Chunked prefill size (default `4096`, `-1` disables) |
| `enable_mixed_chunk` | `bool` | `False` | Mix prefill and decode in the same batch |
| `schedule_policy` | `str` | `"fcfs"` | Scheduling policy (`lpm` / `random` / `fcfs` / `dfs-weight`) |
| `schedule_conservativeness` | `float` | `1.0` | Scheduling conservativeness (higher = more conservative) |
| `page_size` | `int` | `1` | KV cache page size (tokens/page) |
| `swa_full_tokens_ratio` | `float` | `0.8` | Ratio of SWA-layer KV tokens to full layers |
| `recurrent_state_memory_ratio` | `float` | `0.9` | Memory ratio between recurrent state and KV cache for hybrid recurrent models (e.g., Kimi-Linear); `state_budget = available * ratio / (1 + ratio)`, used only when `max_recurrent_state_size` is unset and either radix cache is enabled or `max_running_requests` is unset |
| `max_recurrent_state_size` | `int \| None` | `None` | Total recurrent-state slots across all DP ranks for hybrid models; resolution priority: (1) explicit setting, (2) `max_running_requests` when `--disable-radix-cache`, (3) derived from `recurrent_state_memory_ratio` and available HBM; must be divisible by `dp_size` when set explicitly |
| `disable_hybrid_swa_memory` | `bool` | `False` | Disable hybrid SWA memory optimization |

**`mem_fraction_static` default 0.88**: Reserves as much device memory as possible for KV cache while leaving roughly 12% for JAX runtime overhead (XLA compilation cache, intermediate compute buffers, DMA staging area). Setting it too high causes XLA OOM; setting it too low wastes KV cache capacity. 0.88 is the empirically safe upper bound.

**`chunked_prefill_size` default 4096**: A balance point between prefill throughput and decode latency. Too-large chunks monopolize the MXU and cause TPOT (time per output token) jitter for decode requests; too-small chunks increase kernel launch count and KV cache update frequency. On TPU v5/v6, 4096 keeps decode P99 latency in an acceptable range in practice.

**`page_size` default 1**: Token-level allocation, no internal fragmentation — every token occupies exactly one slot. Page sizes > 1 may leave the last portion of a page unfilled (internal fragmentation), but reduce page management overhead and scatter/gather frequency. **The MLA backend does not support `page_size=1`** — `MLAAttentionBackend.__init__` asserts and rejects (the MLA v2 kernel infers the effective page size from `cache_kv.shape[1] * kv_packing`, and `page_size=1` conflicts with the allocator/metadata semantics); when MLA is enabled you must explicitly set `--page-size 16` or another value greater than 1. This constraint applies only to the MLA path; MHA/GQA may keep the default 1.

### 13.1.5 Runtime Options

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `device` | `str \| None` | `None` | Device type (auto-detected, defaults to `"tpu"`) |
| `device_indexes` | `list[int] \| None` | `None` | Device indices used by the mesh |
| `tp_size` | `int` | `1` | Tensor parallelism degree |
| `dp_size` | `int` | `1` | Data parallelism degree; the Scheduler partitions requests and batches by DP rank inside the same process |
| `dp_schedule_policy` | `str` | `"min_running_queue"` | DP rank assignment policy (`min_running_queue` / `round_robin`) |
| `ep_size` | `int` | `1` | Expert parallelism degree |
| `ep_num_redundant_experts` | `int` | `0` | Number of redundant experts for EP load balancing |
| `ep_dispatch_algorithm` | `str \| None` | `None` | EP dispatch mode (`static` / `dynamic` / `fake`) |
| `enable_sequence_parallel` | `bool` | `False` | Enable sequence parallel: row-parallel Linear outputs are reduce-scattered along the `"tensor"` axis (only takes effect when the Linear explicitly declares `output_scatter_dimension` and `should_scatter()` triggers; `models/grok.py` is currently wired up) |
| `stream_interval` | `int` | `1` | Number of streaming buffer tokens |
| `stream_output` | `bool` | `False` | Output as disjoint segments |
| `random_seed` | `int \| None` | `None` | Random seed (default `42`) |
| `watchdog_timeout` | `float` | `300` | Forward batch hang timeout (seconds) |
| `dist_timeout` | `int \| None` | `None` | `jax.distributed` initialization timeout |
| `download_dir` | `str \| None` | `None` | HuggingFace download directory |
| `sleep_on_idle` | `bool` | `False` | Lower CPU usage when idle |
| `constrained_json_whitespace_pattern` | `str \| None` | `None` | JSON whitespace regex (llguidance) |
| `constrained_json_disable_any_whitespace` | `bool` | `False` | Force compact JSON |

**`tp_size` / `dp_size`**: `tp_size` denotes the model tensor-parallel configuration of the total parallel device count, and `dp_size` denotes the number of Data Parallel ranks partitioned within the same Scheduler. The execution-side mesh shape is `(dp_size, tp_size // dp_size)`, with axis names `(data, tensor)`. Therefore the actual TP width on the attention side is `tp_size // dp_size`.

**`dp_schedule_policy`**: Controls which DP rank a new request is assigned to. `min_running_queue` selects the rank with the fewest currently running requests; `round_robin` cycles through ranks.

**`enable_sequence_parallel`**: Defaults to `False`. When set to `True`, `srt/utils/parallel_utils.py::should_scatter()` decides whether a specific row-parallel Linear actually performs reduce-scatter — only when the layer explicitly declares `output_scatter_dimension`, the per-device slice is ≥ `global_config.tpu_scatter_min_local_size`, and divisibility holds, does it take effect; otherwise it falls back to the original partition spec automatically. Currently wired up in `models/grok.py`; other models will only be affected after explicitly setting `output_scatter_dimension` on their Linear layers.

### 13.1.6 Logging

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `log_level` | `str` | `"info"` | Global log level |
| `log_level_http` | `str \| None` | `None` | HTTP server log level |
| `log_requests` | `bool` | `False` | Log all requests |
| `log_requests_level` | `int` | `0` | Request log verbosity (0=metadata, 1=+params, 2=+partial I/O, 3=full) |
| `crash_dump_folder` | `str \| None` | `None` | Crash dump directory |
| `show_time_cost` | `bool` | `False` | Show custom-mark elapsed time |
| `decode_log_interval` | `int` | `40` | Decode batch logging interval |
| `enable_request_time_stats_logging` | `bool` | `False` | Per-request timing stats |
| `bucket_time_to_first_token` | `list[float] \| None` | `None` | TTFT histogram buckets |
| `bucket_inter_token_latency` | `list[float] \| None` | `None` | ITL histogram buckets |
| `bucket_e2e_request_latency` | `list[float] \| None` | `None` | E2E latency histogram buckets |

### 13.1.7 API-related

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `api_key` | `str \| None` | `None` | API key authentication |
| `served_model_name` | `str \| None` | `None` | Externally served model name (defaults to `model_path`) |
| `file_storage_path` | `str` | `"sglang_storage"` | File storage path |
| `enable_cache_report` | `bool` | `False` | Return cached token count |
| `reasoning_parser` | `str \| None` | `None` | Reasoning model parser |
| `tool_call_parser` | `str \| None` | `None` | Tool call parser |
| `kv_events_config` | `str \| None` | `None` | KV event configuration (JSON) |

### 13.1.8 Kernel and Backend

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `attention_backend` | `str \| None` | `"fa"` | `"native"` / `"fa"` / `"fa_mha"` |
| `moe_backend` | `str` | `"epmoe"` | `"epmoe"` / `"fused"` / `"auto"` |
| `disable_jax_allreduce_metadata` | `bool` | `False` | Disable the pure-JAX allreduce metadata path for fused EP-MoE, falling back to the Pallas DMA-based allgather (only for performance baselining/debugging; the JAX path is recommended by default) |
| `grammar_backend` | `str \| None` | `None` | `"llguidance"` / `"none"` (default `"llguidance"`) |
| `max_seq_len` | `int` | `4096` | Maximum sequence length |
| `precompile_token_paddings` | `list[int] \| None` | `None` | Token padding bucket list |
| `precompile_bs_paddings` | `list[int] \| None` | `None` | Batch size padding bucket list |
| `disable_precompile` | `bool` | `False` | Disable JIT precompilation |

**`max_seq_len` default 4096**: This value caps the upper bound of JIT-precompiled token padding buckets — JAX must compile a separate HLO for each distinct input shape, and `max_seq_len` limits the bucket count to control compile time and memory. The default 4096 covers most chat scenarios; long-context use cases require manually increasing it.

### 13.1.9 Speculative Decoding

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `speculative_algorithm` | `str \| None` | `None` | CLI choices: `"EAGLE"` / `"EAGLE3"` / `"NEXTN"` / `"STANDALONE"`; the current `SpeculativeAlgorithm.from_string` only recognizes `EAGLE` / `EAGLE3` / `STANDALONE` (and `None`), so passing `NEXTN` raises `KeyError` in `from_string` — it is a reserved value not yet wired into the runtime |
| `speculative_draft_model_path` | `str \| None` | `None` | Path to draft model weights |
| `speculative_draft_model_revision` | `str \| None` | `None` | Draft model revision |
| `speculative_num_steps` | `int` | `4` | Draft model generation steps |
| `speculative_eagle_topk` | `int` | `5` | Top-K branches per step |
| `speculative_num_draft_tokens` | `int` | `4` | Draft tokens per step |
| `speculative_accept_threshold_single` | `float` | `1.0` | Single-token acceptance threshold |
| `speculative_accept_threshold_acc` | `float` | `1.0` | Cumulative acceptance threshold |

### 13.1.10 LoRA

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `enable_lora` | `bool \| None` | `None` | Enable LoRA (auto-enabled when `lora_paths` is set) |
| `max_lora_rank` | `int \| None` | `None` | Max LoRA rank (auto-inferred) |
| `lora_target_modules` | `set \| list \| None` | `None` | Target modules (`"all"` = all supported modules) |
| `lora_paths` | `dict \| list \| None` | `None` | Adapter paths (`name=path` / `path` / dict) |
| `max_loaded_loras` | `int \| None` | `None` | Max adapters in memory |
| `max_loras_per_batch` | `int` | `8` | Max LoRAs per batch |
| `lora_eviction_policy` | `str` | `"lru"` | Eviction policy |
| `enable_static_lora` | `bool \| None` | `None` | Static LoRA (mutually exclusive with `enable_lora`) |
| `lora_scaling` | `float \| None` | `None` | Static LoRA scaling (`alpha/rank`) |

### 13.1.11 Other Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `enable_engine_loop_run_forever_daemon` | `bool \| None` | `None` | Engine loop persistent run (under `# For engine` section in source) |
| `multimodal` | `bool` | `False` | Enable multimodal server |
| `disable_radix_cache` | `bool` | `False` | Disable prefix cache |
| `allow_auto_truncate` | `bool` | `False` | Auto-truncate over-long requests |
| `disable_overlap_schedule` | `bool` | `False` | Disable CPU/GPU overlap scheduling |
| `enable_precision_tracer` | `bool` | `False` | Precision tracer (disables chunked prefill) |
| `enable_deterministic_sampling` | `bool` | `False` | Deterministic sampling |
| `enable_nan_detection` | `bool` | `False` | NaN detection |
| `use_sort_for_toppk_minp` | `bool` | `False` | Use `jnp.sort` for Top-K / Top-P |
| `model_layer_nums` | `int \| None` | `None` | Override the number of model layers |
| `json_model_override_args` | `str` | `"{}"` | JSON-formatted model config overrides |

### 13.1.12 Distributed

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `dist_init_addr` | `str \| None` | `None` | Distributed init address (`host:port`) |
| `nnodes` | `int` | `1` | Number of nodes |
| `node_rank` | `int` | `0` | Current node rank |

### 13.1.13 Expert Balance

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `enable_return_routed_experts` | `bool` | `False` | Return routed expert info |
| `enable_expert_balance_debug` | `bool` | `False` | Expert balance debug stats |
| `expert_balance_segment_counter` | `int` | `100` | Balance stats segment size |
| `expert_balance_output_file` | `str \| None` | `None` | Balance stats output CSV |
| `init_expert_location` | `str` | `"trivial"` | Initial expert mapping (`"trivial"` or file path) |
| `enable_expert_distribution_recorder` | `bool` | `False` | EPLB distribution recorder |
| `expert_distribution_recorder_buffer_size` | `int` | `100` | Recorder buffer steps |
| `expert_distribution_recorder_output_file` | `str \| None` | `None` | Distribution output `.npy` file |

### 13.1.14 `__post_init__` Auto-behaviors

| Behavior | Description |
|------|------|
| `tokenizer_path` default | `None` → `model_path` |
| `device` resolution | Inferred from `JAX_PLATFORMS` env var; defaults to `"tpu"` |
| `served_model_name` default | `None` → `model_path` |
| `random_seed` default | `None` → `42` |
| `mem_fraction_static` default | GPU/TPU → `0.88`, CPU → `0.5 / process_count` |
| `chunked_prefill_size` default | `None` → `4096` |
| GGUF auto-detection | If GGUF files are detected, `load_format` and `quantization` are set to `"gguf"` |
| Remote URL detection | Auto-set `load_format = "remote"` |
| Precision tracer | Disable chunked prefill (set to `-1`) |
| Multimodal mode | Auto-disable radix cache |
| `grammar_backend` default | `None` → `"llguidance"` |
| Multi-node | Force `device_indexes = None` |

---

## 13.2 GlobalConfig

`GlobalConfig` (`global_config.py`) — Runtime global constants singleton:

```python
global_config = GlobalConfig()  # Module-level singleton
```

| Constant | Env var | Default | Description |
|------|----------|--------|------|
| `verbosity` | — | `0` | Log verbosity (0=silent, 2=output final text) |
| `default_init_new_token_ratio` | `SGLANG_INIT_NEW_TOKEN_RATIO` | `0.7` | Initial new-token reservation ratio |
| `default_min_new_token_ratio_factor` | `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR` | `0.14` | Minimum ratio factor |
| `default_new_token_ratio_decay_steps` | `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS` | `600` | Decay steps |
| `retract_decode_steps` | — | `20` | Retract decay steps |
| `flashinfer_workspace_size` | `FLASHINFER_WORKSPACE_SIZE` | `384 MiB` | FlashInfer workspace size |
| `skip_special_tokens_in_output` | — | `True` | Strip special tokens from output |
| `spaces_between_special_tokens_in_out` | — | `True` | Add spaces between special tokens |
| `enable_precache_with_tracing` | — | `True` | Frontend interpreter optimization |
| `enable_parallel_encoding` | — | `True` | Frontend parallel encoding |

---

## 13.3 PortArgs

`PortArgs` (`server_args.py`) defines inter-process ZMQ IPC channels:

| Field | Description |
|------|------|
| `tokenizer_ipc_name` | Tokenizer ↔ Detokenizer |
| `scheduler_input_ipc_name` | Scheduler (rank 0) receives Tokenizer input |
| `detokenizer_ipc_name` | Detokenizer receives Scheduler output |
| `pub_sub_addr` | Multi-node broadcast address (TCP, `None` for single node) |
| `pub_sub_sync_addr` | Multi-node sync address (TCP, `None` for single node) |
| `rpc_ipc_name` | Engine ↔ Scheduler RPC |
| `metrics_ipc_name` | Scheduler → Metrics |

`init_new()` — Uses `tempfile.NamedTemporaryFile(delete=False)` to create unique IPC socket paths. In multi-node mode, `pub_sub_addr` and `pub_sub_sync_addr` use TCP (`dist_init_host:port+5/+6`).

---

## 13.4 ModelConfig

`ModelConfig` (`configs/model_config.py`) encapsulates the full model configuration.

### Core Enums

| Enum | Values | Description |
|------|---|------|
| `AttentionArch` | `MLA=1`, `MHA=2` | Attention architecture type |
| `ModelImpl` | `AUTO`, `SGLANG`, `TRANSFORMERS` | Model backend choice |
| `MoEBackend` | `EPMOE`, `FUSED`, `AUTO` | MoE compute strategy (AUTO: single device → FUSED, multi-device → EPMOE) |

### Core Fields

| Field | Source | Description |
|------|------|------|
| `model_path` | User input | Model path |
| `hf_config` | `get_config()` | HuggingFace PretrainedConfig |
| `hf_text_config` | `get_hf_text_config()` | Text sub-config (multimodal models) |
| `context_len` | Derived (see below) | Effective context length |
| `head_dim` | `hf_text_config` | Per-head dimension |
| `v_head_dim` | `hf_text_config` | Value head dimension (may differ in MLA) |
| `attention_arch` | Default `MHA` | Attention architecture (MLA is set by the model itself) |
| `num_attention_heads` | `hf_text_config` | Query head count |
| `num_key_value_heads` | `hf_text_config` | KV head count |
| `hidden_size` | `hf_text_config` | Hidden dimension |
| `num_hidden_layers` | `hf_text_config` | Number of layers (overridable via `model_layer_nums`) |
| `vocab_size` | `hf_text_config` | Vocabulary size |
| `dtype` | `_get_and_verify_dtype()` | Resolved JAX dtype |
| `sliding_window` | `hf_text_config` | SWA window size |
| `quantization_config` | `_resolve_quantization_config()` | Quantization config |
| `ep_size` | Externally set | Expert parallel degree |
| `moe_backend` | Auto-selected | MoE backend |

### Context Length Derivation

```text
1. Base derivation get_context_length(hf_text_config):
   Priority: max_sequence_length > seq_length > max_seq_len
           > model_max_length > max_position_embeddings
   → Apply rope_scaling factor (if any)
   → Fallback 2048

2. User override:
   user value <= derived → use user value
   user value > derived → requires SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=True (default True)
```

### `_apply_model_specific_config()`

Uses `ModelRegistry.resolve_model_cls()` to find the model class, then calls `model_cls.patch_model_config(self)`. This allows model implementations to override `attention_arch`, `head_dim`, etc. (e.g., DeepSeek V3 self-configures as MLA).

### `from_server_args()` Factory Method

Builds `ModelConfig` from `ServerArgs`. Mappings:

```text
model_path           ← server_args.model_path
trust_remote_code    ← server_args.trust_remote_code
revision             ← server_args.revision
context_length       ← server_args.context_length
model_override_args  ← server_args.json_model_override_args
dtype                ← server_args.dtype
quantization         ← server_args.quantization
quantization_config_path ← server_args.quantization_config_path
moe_backend          ← server_args.moe_backend
...
```

### TP Support Methods

| Method | Description |
|------|------|
| `get_total_num_kv_heads()` | Get total KV head count (compatible with multiple model formats) |
| `get_num_kv_heads(tp_size)` | Per-device KV head count |
| `needs_kv_head_replication(tp_size)` | Replication needed when `tp_size > total_kv_heads` |
| `get_num_kv_head_replicas(tp_size)` | Number of replicas per original head |
| `configure_for_tensor_parallel(tp_size)` | Configure TP, adjust KV head count |
| `validate_tensor_parallel_config(tp_size)` | Validate that attention heads divide evenly by TP |
| `get_kv_padding_strategy()` | GQA uses `"replicate"`, MHA uses `"zero"` |

### Draft Model Architecture Rewrite

When `is_draft_model=True`:

| Original architecture | Rewritten as |
|---------|-------|
| `DeepseekV3ForCausalLM` | `DeepseekV3ForCausalLMNextN` |
| `LlamaForCausalLM` | `LlamaForCausalLMEagle3` |
| `MiMoForCausalLM` | `MiMoMTPForCausalLM` |

---

## 13.5 LoadConfig and LoadFormat

`LoadConfig` (`configs/load_config.py`):

### LoadFormat Enum

| Value | Description |
|----|------|
| `AUTO` | Auto-detect |
| `SAFETENSORS` | Safetensors format |
| `PT` | PyTorch format |
| `DUMMY` | Empty weights (debug/profiling) |
| `GGUF` | GGUF quantized format |
| `JAX` | Native JAX format |
| `REMOTE` | Remote loading |
| `LAYERED` | Layered loading |
| `BITSANDBYTES` | BitsAndBytes quantization |
| `SHARDED_STATE` | Sharded state dict |
| `MISTRAL` | Mistral-specific format |
| `NPCACHE` | NumPy cache format |

### LoadConfig Fields

| Field | Default | Description |
|------|--------|------|
| `load_format` | `AUTO` | Load format |
| `download_dir` | `None` | Download directory (defaults to HF cache) |
| `sub_dir` | `None` | Sub-directory |
| `model_loader_extra_config` | `{}` | Extra config (JSON or dict) |
| `model_class` | `None` | Multimodal model class selection |
| `ignore_patterns` | `["original/**/*"]` | File patterns to ignore |
| `decryption_key_file` | `None` | Weights decryption key file |

---

## 13.6 QuantizationConfig

See [11-quantization](11-quantization.md#112-quantizationconfig).

---

## 13.7 Environment Variable Reference

### Runtime Configuration

| Env var | Default | Description |
|----------|--------|------|
| `JAX_PLATFORMS` | `""` | Device platform selection (`tpu` / `cpu`, etc.) |
| `JAX_COMPILATION_CACHE_DIR` | — | JIT compilation cache directory |
| `SGLANG_ENABLE_DETERMINISTIC_SAMPLING` | `"0"` | Deterministic sampling |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `"True"` | Allow overriding longer context length |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | — | When `"0"`, skip blocking non-zero-rank child processes |

### Memory and Scheduling

| Env var | Default | Description |
|----------|--------|------|
| `SGLANG_INIT_NEW_TOKEN_RATIO` | `0.7` | Initial new-token reservation ratio |
| `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR` | `0.14` | Minimum ratio factor |
| `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS` | `600` | Decay steps |
| `FLASHINFER_WORKSPACE_SIZE` | `384 MiB` | FlashInfer workspace |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | `"4096"` | Clip max_new_tokens estimate |
| `SGLANG_CI_SMALL_KV_SIZE` | — | Override KV cache size (CI tests) |
| `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` | `"32"` | In-batch prefix cache threshold |

### Model Loading

| Env var | Default | Description |
|----------|--------|------|
| `SGLANG_USE_MODELSCOPE` | `"false"` | Use ModelScope instead of HuggingFace |
| `HF_TOKEN` | — | HuggingFace auth token |

### Logging and Diagnostics

| Env var | Default | Description |
|----------|--------|------|
| `SGLANG_LOGGING_CONFIG_PATH` | — | Custom logging config JSON path |
| `DISABLE_OPENAPI_DOC` | `"false"` | Disable OpenAPI docs |
| `SGLANG_RECORD_STEP_TIME` | `"false"` | Record per-step elapsed time |
| `SGLANG_MOE_QUANT_STATS` | `"0"` | MoE quantization stats |
| `SGLANG_GRAMMAR_TIMEOUT` | `300` | Grammar compile timeout (seconds) |
| `LLGUIDANCE_LOG_LEVEL` | `"1"` | llguidance log level |

### Profiling and Debugging

| Env var | Default | Description |
|----------|--------|------|
| `SGLANG_JAX_PROFILER_DIR` | `"/tmp"` | JAX profiler output directory |
| `ENABLE_MEMORY_PROFILING` | `"0"` | Enable memory profiling |
| `SGL_MEMORY_OUTPUT_DIR` | `"memory_profiles"` | Memory profile output directory |
| `MEMORY_PROFILING_LAYERS` | `"4"` | Profile layer count |
| `SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER` | `"false"` | Kernel log recorder |
| `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` | `"false"` | Disable TP memory imbalance check |
| `SGLANG_JAX_ENABLE_CACHE_MISS_CHECK` | `"false"` | JIT cache miss check |
| `PALLAS_INTERPRET` | `""` | Pallas interpret mode |

### Testing / CI

| Env var | Default | Description |
|----------|--------|------|
| `SGLANG_JAX_IS_IN_CI` | `"false"` | CI environment flag |
| `SGLANG_TEST_MAX_RETRY` | `"1"(CI) / "0"(local)` | Max test retry count |
| `SGLANG_TEST_RETRACT` | `"false"` | Test retract behavior |
| `SGLANG_RUN_SLOW_TESTS` | `"0"` | Enable slow tests |
| `ENABLE_JAX_TRACE` | `"1"` | Enable JAX tracing |

### Speculative Decoding

| Env var | Default | Description |
|----------|--------|------|
| `SIMULATE_ACC_LEN` | — | Simulate accept length (testing) |
| `SIMULATE_ACC_METHOD` | `"multinomial"` | Simulated acceptance method |
| `RETURN_ORIGINAL_LOGPROB` | `"false"` | Return original log probability |

### Data Dump

| Env var | Default | Description |
|----------|--------|------|
| `DUMP_LAST_LAYER_LOGITS_FILENAMES` | — | Dump last-layer logits |
| `DUMP_TOPK_IDS_FILEINFO` | — | Dump Top-K token IDs |
| `SGL_FORCE_SHUTDOWN` | `"false"` | Force-shutdown the server |

### Multimodal

| Env var | Default | Description |
|----------|--------|------|
| `VIDEO_MAX_PIXELS` | `128000*28*28*0.9` | Video total-pixel cap |
| `SGLANG_WAIT_TIMEOUT` | `"4"(text) / "600"(mm)` | Async operation timeout |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | `20` | Health check timeout |

---

## Key Interfaces At a Glance

| Interface | Location | Description |
|------|------|------|
| `ServerArgs` | `server_args.py` | System startup configuration |
| `ServerArgs.__post_init__()` | `server_args.py` | Auto-fill and validation of arguments |
| `PortArgs` | `server_args.py` | ZMQ IPC address definitions (7 channels) |
| `PortArgs.init_new()` | `server_args.py` | Create temporary IPC sockets |
| `GlobalConfig` | `global_config.py` | Runtime global constants singleton |
| `ModelConfig` | `configs/model_config.py` | Model config wrapper |
| `ModelConfig.from_server_args()` | `configs/model_config.py` | Build from ServerArgs |
| `ModelConfig._resolve_quantization_config()` | `configs/model_config.py` | Three-tier quantization config resolution |
| `ModelConfig._apply_model_specific_config()` | `configs/model_config.py` | Model self-config hook |
| `AttentionArch` | `configs/model_config.py` | Attention architecture enum (`MLA` / `MHA`) |
| `ModelImpl` | `configs/model_config.py` | Model backend enum (`AUTO` / `SGLANG` / `TRANSFORMERS`) |
| `MoEBackend` | `configs/model_config.py` | MoE strategy enum (`EPMOE` / `FUSED` / `AUTO`) |
| `LoadConfig` | `configs/load_config.py` | Load configuration |
| `LoadFormat` | `configs/load_config.py` | Load format enum (12 values) |
| `QuantizationConfig` | `configs/quantization_config.py` | Quantization configuration |
