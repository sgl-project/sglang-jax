# Fast Path V2 (Prefill + Extend) for Scoring API

## Overview
The "Fast Path V2" is an optimized execution path that leverages the native batching and KV-caching capabilities of the engine to execute the scoring API (`/v1/score`). It bypasses the sequence packing overhead of the traditional 1D packed approach by efficiently sharing the query prefix in the KV cache and batching candidate items synchronously. This bypasses the overhead of the standard request queue and complexities of manual attention masking.

## Detailed Execution Steps

### 1. Endpoint and Routing (`serving_score.py` & `tokenizer_manager.py`)
- The request first arrives at the HTTP handler for `/v1/score` (`OpenAIServingScore._handle_non_streaming_request`).
- It is passed down to `TokenizerManager.score_request()`, which identifies the request and routes it to the specialized `prefill+extend` strategy (`TokenizerManager.score_prefill_extend()`).

### 2. Explicit Prefix Caching
Instead of evaluating the query alongside the candidate items right away, the `score_prefill_extend` method explicitly caches the query.
- It sends a pure prefill request (`max_new_tokens=0`) to the engine with the `query_tokens`.
- It awaits a `cache_handle` from `_prefill_and_cache`, representing the `RadixCache` node containing the prefilled KV blocks.

```python
# Pseudo-code from TokenizerManager.score_prefill_extend
async def score_prefill_extend(self, query_tokens, item_tokens_list, label_token_ids, apply_softmax):
    # Step 1: Prefill the query only and get the cache handle
    cache_handle = await self._prefill_and_cache(query_tokens)
    
    # Step 2: Invoke the fastpath v2 by directly communicating with the scheduler
    if getattr(self.server_args, "multi_item_enable_score_from_cache_v2", False):
        fastpath_out = await self._score_from_cache_fastpath_v2(
            cache_handle=cache_handle,
            items=item_tokens_list,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax
        )
        return fastpath_out.scores
```

### 3. Bypassing the Queue via Communicator
Rather than enqueueing hundreds or thousands of individual items into the standard multi-turn generation loop, the `TokenizerManager` directly communicates with the scheduler using a dedicated RPC endpoint (`score_from_cache_v2_communicator`).
It packs the items into a specific `ScoreFromCacheReqInput` payload.

```python
# Pseudo-code from io_struct.py
@dataclass
class ScoreFromCacheReqInput(BaseReq):
    cache_handle: str = ""
    items_2d: list[list[int]] = field(default_factory=list) # 2D array of items
    label_token_ids: list[int] = field(default_factory=list)
    apply_softmax: bool = False
    items_per_step: int = 64 # Configuration for synchronous batching chunk size
```

### 4. Direct Batch Execution in the Scheduler (`scheduler.py`)
Inside `Scheduler.score_from_cache_v2()`, the scheduler receives this payload and processes it **outside** of the main scheduling loop.

1. **Cache Lookup**: It retrieves the `RadixCache` node via the provided `cache_handle` and extracts the relevant KV cache blocks representing the query prefix.
2. **Synchronous Chunking**: To prevent Out-Of-Memory (OOM) errors and to fit within fixed batch sizes, the 2D array of items (`items_2d`) is split into chunks of `items_per_step`.
3. **Chunk Execution**: For each chunk, the scheduler prepares an explicit `ScheduleBatch` and forces a forward pass via `_run_score_from_cache_v2_chunk()` or `_run_score_from_cache_v2_chunk_label_only()`.

```python
# Pseudo-code from Scheduler._run_score_from_cache_v2_chunk
def _run_score_from_cache_v2_chunk(self, cache_handle, chunk_items, label_token_ids, apply_softmax, ...):
    # Step 4.a: Construct temporary `Req` objects pointing directly to the cache
    reqs = self._build_score_from_cache_v2_chunk_reqs(cache_handle, chunk_items, ...)
    
    # Step 4.b: Initialize a temporary ScheduleBatch
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        tree_cache=self.tree_cache,
        ...
    )
    
    # Step 4.c: Explicitly prepare the batch for an "Extend" phase and run it
    batch.prepare_for_extend()
    result = self.run_batch(batch)
    
    # Step 4.d: Extract target log probabilities directly from the logits
    logprob_vals = result.logits_output.next_token_token_ids_logprobs_val
    logprob_idxs = result.logits_output.next_token_token_ids_logprobs_idx
    
    # Reconstruct the scores specific to the provided `label_token_ids`
    scores = extract_target_scores(logprob_vals, logprob_idxs, label_token_ids, apply_softmax)
    
    return scores
```

### 5. Immediate Cleanup
After the forward pass of a chunk completes and the scores are extracted, the scheduler automatically disposes of the temporary `ScheduleBatch`. 
- The temporary request slots are freed.
- The extended KV cache tokens allocated solely for the candidate items in this chunk are immediately de-allocated.
- This prevents memory leaks while preserving the original query's KV cache via the `cache_handle`.

## Benefits vs. Packed Approach
1. **No Custom Attention Masking:** Items are completely isolated by the batch dimension (`[Batch, SeqLength]`), so standard causal attention masks work perfectly.
2. **Reduced JAX Compilation Pressure:** By executing through `items_per_step` chunks, the system maintains a fixed batch size rather than continuously creating custom variable-length 1D sequences.
3. **Zero Queue Congestion:** Candidate items do not crowd the main generation queue, reducing latency variability for other concurrent tasks.
4. **Standard Position Embeddings:** Each sequence in the 2D batch starts at a logical positional index `[0, 1, 2, ...]`. There is no need to implement manual resets for RoPE positions at delimiters as required in the packed approach.