# SGL-JAX: High-Performance LLM Inference on JAX/TPU

SGL-JAX is a high-performance, JAX-based inference engine for Large Language Models (LLMs), specifically optimized for Google TPUs. It is engineered from the ground up to deliver exceptional throughput and low latency for the most demanding LLM serving workloads.

The engine incorporates state-of-the-art techniques to maximize hardware utilization and serving efficiency, making it ideal for deploying large-scale models in production on TPUs.

[![Pypi](https://img.shields.io/badge/pypi-sglang--jax-orange.svg)](https://pypi.org/project/sglang-jax) [![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/sgl-project/sglang-jax?tab=Apache-2.0-1-ov-file#readme)

## Key Features

- **High-Throughput Continuous Batching**: Implements a sophisticated scheduler that dynamically batches incoming requests, maximizing TPU utilization and overall throughput.
- **Optimized KV Cache with Radix Tree**: Utilizes a Radix Tree for KV cache management (conceptually similar to PagedAttention), enabling memory-efficient prefix sharing between requests and significantly reducing computation for prompts with common prefixes.
- **FlashAttention Integration**: Leverages a high-performance FlashAttention kernel for faster and more memory-efficient attention calculations, crucial for long sequences.
- **Tensor Parallelism**: Natively supports tensor parallelism to distribute large models across multiple TPU devices, enabling inference for models that exceed the memory of a single accelerator.
- **OpenAI-Compatible API**: Provides a drop-in replacement for the OpenAI API, allowing for seamless integration with a wide range of existing clients, SDKs, and tools (e.g., LangChain, LlamaIndex).
- **Native Qwen Support**: Includes first-class, optimized support for the Qwen model family, including recent Mixture-of-Experts (MoE) variants.

## Architecture Overview

SGL-JAX operates on a distributed architecture designed for scalability and performance:

1.  **HTTP Server**: The entry point for all requests, compatible with the OpenAI API standard.
2.  **Scheduler**: The core of the engine. It receives requests, manages prompts, and schedules token generation in batches. It intelligently groups requests to form optimal batches for the model executor.
3.  **TP Worker (Tensor Parallel Worker)**: A set of distributed workers that host the model weights, distributed via tensor parallelism. They execute the forward pass for the model.
4.  **Model Runner**: Manages the actual JAX-based model execution, including the forward pass, attention computation, and KV cache operations.
5.  **Radix Cache**: A global, memory-efficient KV cache that is shared across all requests, enabling prefix reuse and reducing the memory footprint.

---

## Getting Started

- [Install SGL-JAX](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Quick Start](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [Benchmark and Profiling](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)

## Documentation

For more features and usage details, please read the documents in the [`docs`](https://github.com/sgl-project/sglang-jax/tree/main/docs) directory.

## Supported Models

SGL-JAX is designed for easy extension to new model architectures. It currently provides first-class, optimized support for:

-   **Qwen**
-   **Qwen 2**
-   **Qwen 2 MOE**
-   **Qwen 3**
-   **Qwen 3 MoE**
-   **Llama**
-   **Bailing MoE**

## Performance and Benchmarking

For detailed performance evaluation and to run the benchmarks yourself, please see the scripts located in the `benchmark/` and `python/sgl_jax/` directories (e.g., `bench_serving.py`).

## Testing

The project includes a comprehensive test suite to ensure correctness and stability. To run the full suite of tests:

```bash
python test/srt/run_suite.py
```

## Contributing

Contributions are welcome! If you would like to contribute, please feel free to open an issue to discuss your ideas or submit a pull request.

Before contributing, please read our [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
 for setup instructions, coding standards, and contribution workflow.

You can also join our community on Slack to discuss ideas, get help, or collaborate with other contributors:
👉 Join the [SGL-JAX Slack](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X)
