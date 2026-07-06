# Architecture

The architecture section explains the current SGL-JAX runtime from the top-level serving flow down to scheduler, executor, model, attention, KV cache, and multimodal internals.

Status: this section is structurally complete enough to publish in the Sphinx tree, but it still needs a technical review against the latest runtime before being treated as authoritative.

## Reading order

1. Start with [System Overview](01-architecture-overview.md).
2. Follow the numbered subsystem docs from entrypoints through configuration.
3. Use [Project Core Structure](project-core-structure.md) as the codebase map while reading implementation.

## Pages

| Page | Focus |
|---|---|
| [01 System Overview](01-architecture-overview.md) | End-to-end runtime positioning. |
| [02 Entrypoints and Tokenization](02-entrypoints-and-tokenization.md) | Request entry, tokenization, and API flow. |
| [03 Scheduler](03-scheduler.md) | Request scheduling and batching. |
| [04 Model Executor](04-model-executor.md) | JAX execution, mesh, and compiled forward flow. |
| [05 Models](05-models.md) | Model registration and model stack. |
| [06 Layers and Attention](06-layers-and-attention.md) | Layer and attention implementations. |
| [07 KV Cache](07-kv-cache.md) | KV cache memory and prefix reuse. |
| [08 Pallas Kernels](08-pallas-kernels.md) | Custom kernel implementation. |
| [09 Speculative Decoding](09-speculative-decoding.md) | Draft/verify architecture. |
| [10 LoRA](10-lora.md) | LoRA runtime architecture. |
| [11 Quantization](11-quantization.md) | Quantization architecture. |
| [12 Multimodal](12-multimodal.md) | Multimodal extension model. |
| [13 Configuration Reference](13-configuration-reference.md) | Runtime configuration reference. |
| [Project Core Structure](project-core-structure.md) | Repository and module map. |
