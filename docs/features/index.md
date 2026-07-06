# Features

Feature docs describe runtime capabilities, operational behavior, and implementation-specific design notes.

The stable feature pages live directly under `docs/features/`. Design/RFC material that belongs to a feature has been moved under `docs/features/design/` so readers do not need to jump between separate `features` and `design` trees.

## Runtime features

| Page | Focus |
|---|---|
| [Attention backend](attention_backend.md) | Attention backend abstraction and extension points. |
| [Chunked prefill](chunked_prefill.md) | Splitting long prefill into smaller chunks. |
| [Dtype config](dtype_config.md) | Runtime precision configuration. |
| [Dynamic continuous batching](dynamic_continuous_batching.md) | Request batching behavior. |
| [Global JIT compile](global_jit_compile.md) | Compilation cache and startup flow. |
| [Integrate with Tunix](integrate_with_tunix.md) | Tunix integration notes. |
| [Partial rollout](partial_rollout.md) | Partial rollout behavior. |
| [Quantization](quantization.md) | Quantized model serving support. |
| [Radix cache](radix_cache.md) | Prefix KV cache reuse. |
| [Return routed experts](return_routed_experts.md) | MoE routed expert return path. |
| [Run in Pathways](run_in_pathways.md) | Pathways execution notes. |
| [Speculative decoding](speculative_decoding.md) | Draft/verify decoding support. |
| [Structured output](structured_output.md) | Constrained output support. |

## Design details

| Page | Focus |
|---|---|
| [Data parallelism](design/data_parallelism.md) | DP scheduling and KV cache implications. |
| [LoRA](design/lora.md) | Multi-LoRA serving design. |
| [MLA](design/mla.md) | MLA attention backend and cache design. |
| [PD disaggregation](design/pd_disaggregation.md) | Prefill/decode split architecture. |
| [SWA eviction and LRU](design/swa_eviction_and_lru_strategy.md) | Hybrid full/SWA cache eviction. |
| [Bailing MoE linear attention](design/bailing_moe_linear_attention.md) | Linear attention module RFC. |
