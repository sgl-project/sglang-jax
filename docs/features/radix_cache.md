# Radix Cache

Radix Cache automatically reuses Key-Value (KV) cache across LLM inference requests to improve performance and reduce costs. The implementation is based on the original SGLang project: https://github.com/sgl-project/sglang.

## Goals

Many LLM use cases involve repeated prompt patterns:
- Chat systems with reused system prompts
- Few-shot learning with repeated examples
- Multi-turn conversations with shared context

Without prefix caching, identical token sequences are recomputed each time, wasting computation and increasing latency.

## Design

Radix cache uses a **radix tree** data structure to efficiently store and reuse cached KV states.

A radix tree is a compressed trie where each node can represent multiple tokens rather than single characters. When processing requests:

1. **First request**: `"You are helpful assistant."` → Creates root node A with KV cache
2. **Second request**: `"You are helpful assistant. Hello!"` → Reuses node A, extends with new node B
3. **Third request**: `"You are helpful assistant. What's weather?"` → Reuses node A, extends with new node C

The tree structure allows automatic prefix detection - any request starting with `"You are helpful assistant."` will reuse the cached computation from node A, significantly reducing inference time and memory usage.

Key components:
- **Tree nodes** store token sequences as keys and KV cache indices as values
- **Prefix matching** finds the longest shared sequence between requests
- **Reference counting** (`lock_ref`) protects active cache entries from eviction
- **Heap-based eviction** removes unlocked leaf nodes when memory is full

## Implementation

### Core RadixCache Operations

1. **`match_prefix(token_ids)`** - Find cached KV indices for matching prefixes
   - Called in `init_next_round_input` when new requests arrive
   - Returns cached indices and tree nodes for reuse

2. **`cache_unfinished_req(req)`** - Cache partial request progress
   - Called in `process_batch_result_prefill` after each extend/chunked prefill
   - Updates radix tree with newly computed tokens during processing

3. **`cache_finished_req(req)`** - Cache completed requests
   - Called in `process_batch_result_decode` when `req.finished()`
   - Stores complete token sequences for future prefix matching

4. **`evict(num_tokens)`** - Free memory when needed
   - Uses min-heap to select leaf nodes by `last_access_time`
   - Only evicts nodes with `lock_ref == 0`

The radix tree is stored on CPU while KV data uses TPU memory pools.

## Usage

Radix Cache is **enabled by default**. To disable it:

```bash
python -m sgl_jax.launch_server --disable-radix-cache
```
