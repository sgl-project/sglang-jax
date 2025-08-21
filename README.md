# SGL-JAX: High-Performance LLM Inference on JAX/TPU

SGL-JAX is a high-performance, JAX-based inference engine for Large Language Models (LLMs), specifically optimized for Google TPUs. It is engineered from the ground up to deliver exceptional throughput and low latency for the most demanding LLM serving workloads.

The engine integrates state-of-the-art techniques to maximize hardware utilization and serving efficiency, making it an ideal solution for deploying large-scale models in production with TPU.

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

## Quick Start

Follow these steps to get a model server up and running.

### 1. Installation

First, clone the repository and install the necessary dependencies. It is recommended to do this in a virtual environment.

```bash
git clone https://github.com/your-org/sgl-jax.git
cd sgl-jax/python
pip install -e .
```

### 2. Launch the Server

You can launch the OpenAI-compatible API server using the `sgl_jax.launch_server` module.

```bash
# Example: Launching a server for Qwen1.5-7B-Chat
python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen1.5-7B-Chat \
    --tp-size 4 \
    --port 8000 \
    --host 0.0.0.0
```

**Key Arguments**:
*   `--model-path`: The path to the model on the Hugging Face Hub or a local directory.
*   `--tp-size`: The number of TPU devices to use for tensor parallelism.
*   `--port`: The port for the API server.
*   `--host`: The host address to bind the server to.

### 3. Send a Request

Once the server is running, you can interact with it using any OpenAI-compatible client, such as `curl` or the `openai` Python library.

#### Using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen1.5-7B-Chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, what is JAX?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Using the `openai` Python client:

```python
import openai

# Point the client to the local server
client = openai.OpenAI(
    api_key="your-api-key",  # Can be any string
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
  model="Qwen/Qwen1.5-7B-Chat",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, what is JAX?"}
  ]
)

print(response.choices[0].message.content)
```

## Documentation

For more features and usage details, please read the documents in the [`docs`](./docs/) directory.

## Supported Models

SGL-JAX is designed for easy extension to new model architectures. It currently provides first-class, optimized support for:

-   **Qwen**
-   **Qwen 3**
-   **Qwen 3 MoE**

## Performance and Benchmarking

Performance is a core focus of SGL-JAX. The engine is continuously benchmarked to ensure high throughput and low latency. For detailed performance evaluation and to run the benchmarks yourself, please see the scripts located in the `benchmark/` and `python/sgl_jax/` directories (e.g., `bench_serving.py`).

## Testing

The project includes a comprehensive test suite to ensure correctness and stability. To run the full suite of tests:

```bash
cd test/srt
python run_suite.py
```

## Contributing

Contributions are welcome! If you would like to contribute, please feel free to open an issue to discuss your ideas or submit a pull request.
