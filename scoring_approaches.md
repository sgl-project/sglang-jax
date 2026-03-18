# Scoring API Approaches in SGLang JAX

This document outlines the two primary approaches for the scoring API (`/v1/score`) in SGLang JAX, focusing on how they handle multi-item requests (e.g., a single query matched against multiple candidate items).

## 1. The Packed Approach (V1 / Baseline)

The traditional approach involves packing the query and all candidate items into a single, massive 1D sequence.

### Technical Implementation

*   **Sequence Structure:** `[Query] [Delimiter] [Item 1] [Delimiter] [Item 2] ...`
*   **Position Embeddings:** Requires custom logic (`_build_multi_item_extend_positions`) to reset RoPE positions at each delimiter. For example, the first token of `Item 2` gets `position = len(Query)`, not its absolute 1D index.
*   **Attention Masking:** Uses custom attention masking (either dense $O(N^2)$ masks or segment-based logic) to enforce that `Item K` can attend to the `Query` but **cannot** attend to `Item K-1` or any other items.
*   **Score Extraction:** The system must scan the massive `input_token_ids_logprobs` list to find the indices immediately following the delimiter tokens to extract the relevant scores for each item.

### Drawbacks

*   **Quadratic Overhead:** Dense custom masks scale quadratically with the total sequence length.
*   **Compilation Pressure:** The unique sequence lengths and custom mask shapes can trigger frequent JAX re-compilations.
*   **Kernel Efficiency:** Standard attention kernels are highly optimized for batched sequences; custom segment-based masks can sometimes be slower than running a clean batch.

## 2. Fast Path V2 (Prefill + Extend)

The "Fast path V2" is an optimized execution path that leverages the native batching and KV-caching capabilities of the engine, bypassing the overhead of the standard request queue and the complexities of the packed approach.

### Technical Implementation

1.  **Explicit Prefix Caching:** It first sends a pure prefill request (`max_new_tokens=0`) for the `query_tokens` and retrieves a `cache_handle` representing the RadixCache node for that prefix (`_prefill_and_cache`).
2.  **Bypassing the Queue:** It sends a special `ScoreFromCacheReqInput` directly to the scheduler via an RPC communicator. This request contains the `cache_handle` and a 2D list of all `items` to be scored.
3.  **Synchronous Chunking:** Inside the scheduler, the items are batched into chunks based on `items_per_step` (e.g., 64).
4.  **Direct Batch Execution:** For each chunk, the scheduler completely bypasses the normal scheduling loop. It explicitly constructs temporary `Req` objects pointing to the `cache_handle`, builds a `ScheduleBatch`, and immediately forces a forward pass (`_run_score_from_cache_v2_chunk`).
5.  **Immediate Cleanup:** After the forward pass completes, it extracts the target token probabilities directly from the logits and immediately frees the request slots and KV cache tokens used by the chunk.

### Comparison

| Feature | Packed Approach (1D) | Fast Path V2 (2D Batch) |
| :--- | :--- | :--- |
| **Data Layout** | `[Query] [D] [Item1] [D] [Item2] ...` | `[[Query, Item1], [Query, Item2], ...]` |
| **Position Embeddings** | **Manual Resets:** Custom logic to reset RoPE positions at each delimiter. | **Standard:** Standard linear positions `[0, 1, 2, ...]` for each sequence in the batch. |
| **Attention Masking** | **Custom Masking:** Requires a custom mask to isolate items. | **Standard Causal:** Uses the default causal mask. Items are naturally isolated by the batch dimension. |
| **KV Cache Use** | **Single Sequence:** The entire packed sequence is treated as one long request in the KV cache. | **Prefix Sharing:** Items share the exact same physical KV cache blocks for the `Query` prefix via the `cache_handle`, but are independent thereafter. |
| **Complexity** | High: Requires custom JAX kernels or complex masking metadata to bypass standard causality. | Low: Uses the engine's standard "Extend" mode, just triggered synchronously by the scheduler. |

## Configuration & Usage

### Launching the Server with Hybrid Fast Path V2

To enable the optimized Hybrid Fast Path V2 approach, use the following server launch command. This configuration is tuned for TPU v6e to maximize throughput while ensuring stability during JAX kernel compilation.

```bash
python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --device tpu --tp-size 1 \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --chunked-prefill-size -1 \
  --multi-item-scoring-delimiter 128001 \
  --multi-item-mask-impl auto \
  --multi-item-segment-fallback-threshold 0 \
  --enable-scoring-cache \
  --multi-item-enable-prefill-extend \
  --multi-item-extend-batch-size 8 \
  --multi-item-enable-score-from-cache-v2 \
  --multi-item-score-from-cache-v2-items-per-step 500 \
  --multi-item-hybrid-pack-len 4096 \
  --multi-item-prefill-extend-cache-timeout 300 \
  --max-running-requests 8 \
  --max-total-tokens 65536 \
  --precompile-bs-paddings 1 4 8 \
  --precompile-token-paddings 1024 4096 16384 \
  --page-size 64 --log-level=info
```

## Benchmarking

A benchmarking client script is available at `benchmark.py` to evaluate the performance of the scoring API. It measures throughput in terms of **Requests per Second (RPS)** and **Items per Second (IPS)** using concurrent requests.

### Key Features
- **Concurrent Execution:** Uses `ThreadPoolExecutor` to simulate multiple simultaneous clients.
- **Dynamic Payload Generation:** Generates dummy text of configurable lengths for queries and items.
- **Warm-up Phase:** Executes a single request to ensure the server is ready before measuring performance.
- **Metrics:** Reports Total Time, Success Rate, RPS, and IPS.

### Configuration
You can tune the benchmark by modifying the constants at the top of `benchmark.py`:
- `NUM_REQUESTS`: Total number of API calls to execute.
- `NUM_ITEMS`: Number of candidate items per request.
- `QUERY_TOKENS`: Approximate token length of the query prefix.
- `ITEM_TOKENS`: Approximate token length of each item extension.
- `CONCURRENCY`: Number of parallel threads/clients.

### Running the Benchmark
Ensure the server is running, then execute the script:
```bash
python benchmark.py
```

## Reference Files

To learn more about these implementations from scratch, refer to the following files in the codebase:

1.  **Entry Point & Routing:**
    *   `python/sgl_jax/srt/entrypoints/openai/serving_score.py`: Handles the `/v1/score` HTTP request.
    *   `python/sgl_jax/srt/managers/tokenizer_manager.py`: Contains `score_request` and `score_prefill_extend`, which route the request to the appropriate strategy.
2.  **Fast Path V2 (Scheduler & Batching):**
    *   `python/sgl_jax/srt/managers/scheduler.py`: Look for `score_from_cache_v2`, `_run_score_from_cache_v2_chunk`, and `_run_score_from_cache_v2_chunk_label_only`. This is where the queue is bypassed and the batch is constructed.
    *   `python/sgl_jax/srt/managers/schedule_batch.py`: Look for `ModelWorkerBatch` construction to see how the 2D batch is prepared for the model.
3.  **Packed Approach (Masking & Positions):**
    *   `python/sgl_jax/srt/managers/schedule_batch.py`: Look for `_build_multi_item_extend_positions` to see how RoPE positions are manually reset.
    *   `python/sgl_jax/srt/layers/attention/flashattention_backend.py`: Look for `is_multi_item_scoring` and `MULTI_ITEM_MASK_MODE_SEGMENT` to see how custom attention masks are built to isolate items within the single 1D sequence.
    *   `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`: Look for how scores are extracted from the `input_token_ids_logprobs` after the delimiter tokens.
4.  **Data Structures:**
    *   `python/sgl_jax/srt/managers/io_struct.py`: Defines `ScoreFromCacheReqInput`, `ScoreFromCacheReqOutput`, and the `is_multi_item_scoring` flags on base requests.