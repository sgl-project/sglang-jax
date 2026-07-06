# Features

Feature docs describe runtime capabilities, operational behavior, and user-visible limitations. Deeper implementation details live in [Architecture](../architecture/index.md).

| Page | Focus |
|---|---|
| [Attention backend](attention_backend.md) | Attention backend abstraction and extension points. |
| [Chunked prefill](chunked_prefill.md) | Splitting long prefill into smaller chunks. |
| [Dtype config](dtype_config.md) | Runtime precision configuration. |
| [Dynamic continuous batching](dynamic_continuous_batching.md) | Request batching behavior. |
| [Global JIT compile](global_jit_compile.md) | Compilation cache and startup flow. |
| [Partial rollout](partial_rollout.md) | Partial rollout behavior. |
| [Quantization](quantization.md) | Quantized model serving support. |
| [Radix cache](radix_cache.md) | Prefix KV cache reuse. |
| [Return routed experts](return_routed_experts.md) | MoE routed expert return path. |
| [Run in Pathways](run_in_pathways.md) | Pathways execution notes. |
| [Speculative decoding](speculative_decoding.md) | Draft/verify decoding support. |
| [Structured output](structured_output.md) | Constrained output support. |
