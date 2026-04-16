# SWA Eviction Strategy

## 1. Dual-Pool Architecture

Hybrid models (e.g., MiMo-V2-Flash with 9 full-attention + 39 SWA layers) require two separate KV cache pools:

- **Full Pool** — serves full-attention layers, retains all historical KV data, never evicted per-request.
- **SWA Pool** — serves sliding-window-attention layers, bounded by the window size `W`, with slots outside the window subject to eviction.

These pools are linked via `full_to_swa_index_mapping`, a numpy array that maps a full-pool index to its corresponding SWA-pool index. A mapping value of 0 means the SWA slot has been freed.

### Key Terms

| Term | Definition |
|------|-----------|
| `swa_evicted_seqlen` | Per-request watermark. SWA slots in `[0, swa_evicted_seqlen)` have already been freed. |
| `cache_protected_len` | Slots in `[0, cache_protected_len)` belong to the radix tree; per-request eviction cannot reclaim them. |
| `tombstone` | A tree node where full KV data is retained but the SWA KV data has been freed. Only relevant in RadixCache mode. |

### Memory Layout Example

For MiMo-V2-Flash on TPU v6e-16 (TP=16, 1 KV head, head_dim=192+128=320, bf16):

```
Full pool:  104,704 tokens x 9 FA layers  x 640 bytes/token = ~603 MB
SWA pool:   83,840 tokens x 39 SWA layers x 640 bytes/token = ~2.1 GB
SWA held per request: ~256 tokens (sliding_window=128 + page_size=128 alignment)
```

## 2. Two Cache Modes

| Mode | Activation | Prefix Sharing | SWA Eviction |
|------|-----------|----------------|--------------|
| **ChunkCache** | `--disable-radix-cache` | No. Each request owns all its KV slots. | Per-request, proactive (`_evict_swa`). |
| **SWARadixCache** | Default (radix cache enabled) | Yes (radix tree). KV data shared across requests. | Tree-level, reactive (LRU-based, triggered by allocation pressure). |

> **Current status**: ChunkCache mode is fully implemented and tested. SWARadixCache basic operations (insert/match/evict/lock) work, but advanced tree-level SWA eviction (tombstone strategies) is planned for a future PR.

## 3. Allocator: `SWATokenToKVPoolAllocator`

The allocator maintains two independent sub-allocators (one for each pool) and the mapping array.

### Allocation

```
alloc(need_size):
  1. Check both pools have capacity >= need_size
  2. Allocate full_indices from full pool
  3. Allocate swa_indices from SWA pool
  4. Update mapping: full_to_swa_index_mapping[full_indices] = swa_indices
  5. Return full_indices (SWA indices are transparent to callers)
```

For paged mode (`page_size > 1`), `alloc_extend` and `alloc_decode` follow the same pattern but use page-level allocation with **atomic rollback**: if SWA allocation fails after full allocation succeeds, the full pages are rolled back to prevent partial allocation.

### Freeing

- **`free(indices)`**: Frees both full and SWA pools. Called on request completion.
- **`free_swa(indices)`**: Frees only the SWA pool side. Called during per-request SWA eviction. Looks up mapping, frees non-zero entries, zeroes the mapping.
- **`count_swa_mapped(indices)`**: Counts how many indices still have active SWA mappings. Used for bookkeeping adjustments.

## 4. Per-Request SWA Eviction (`_evict_swa`)

This function frees SWA slots that fall outside the sliding window from a request's `req_to_token` buffer.

### Algorithm

```
_evict_swa(req, pre_len, sliding_window_size, page_size):
  1. new_evicted = max(req.swa_evicted_seqlen, pre_len - sliding_window_size)
  2. If page_size > 1: align new_evicted down to page boundary
  3. If new_evicted <= req.swa_evicted_seqlen: return (nothing to evict)
  4. Read full-pool indices from req_to_token[swa_evicted_seqlen : new_evicted]
  5. Count actual SWA slots to free (count_swa_mapped)
  6. free_swa(those indices)
  7. If SWARadixCache: adjust swa_protected_size by -num_freed
  8. Update req.swa_evicted_seqlen = new_evicted
```

### Example

With `sliding_window=128`, `page_size=256`, and `seqlen=2049`:

```
new_evicted = max(0, 2049 - 128) = 1921
page-aligned: (1921 // 256) * 256 = 1792
Free SWA slots in [0, 1792), retain [1792, 2049) within the window.
```

## 5. Extend Phase Behavior

### 5.1 ChunkCache — Proactive Eviction

With overlap scheduling enabled, `maybe_evict_swa` is gated by `extend_batch_idx`:

- `extend_batch_idx < 2`: eviction **skipped** (previous extend batch may still be executing on device).
- `extend_batch_idx >= 2`: eviction proceeds with `pre_len` reduced by `chunked_prefill_size`.

This creates a one-chunk safety delay: chunk N+1 evicts chunk N-1's outdated SWA cache.

**Example** (8K tokens, chunk_size=2048, sliding_window=128, page_size=256, overlap enabled):

| Chunk | `extend_batch_idx` | Action | SWA slots freed |
|-------|-------------------|--------|-----------------|
| 1 | 0 | Skipped | — |
| 2 | 1 | Skipped | — |
| 3 | 2 | `pre_len=4096-2048=2048`, evicts `[0, 1792)` | 1792 |
| 4 | 3 | `pre_len=6144-2048=4096`, evicts `[1792, 3840)` | 2048 |

Without overlap scheduling, there is no `extend_batch_idx` gate and no `pre_len` adjustment:

| Chunk | Action | SWA slots freed |
|-------|--------|-----------------|
| 1 | `pre_len=0`, nothing to evict | — |
| 2 | `pre_len=2048`, evicts `[0, 1792)` | 1792 |
| 3 | `pre_len=4096`, evicts `[1792, 3840)` | 2048 |

### 5.2 SWARadixCache — Deferred to Tree

In RadixCache mode, `_evict_swa` is **skipped during extend**. Instead, `cache_unfinished_req` hands ownership of slots to the tree, and `cache_protected_len` advances to cover all allocated positions.

The trade-off: prefill SWA slots remain allocated until pool pressure triggers tree-level eviction — simpler ownership model, but less SWA-efficient during prefill.

## 6. Decode Phase Behavior

Both ChunkCache and SWARadixCache run `_evict_swa` during decode, with these controls:

### Eviction Interval

```python
evict_interval = max(min(sliding_window_size, page_size), 1)
```

- When `page_size >= sliding_window_size`: evict every step.
- When `page_size < sliding_window_size`: evict every `page_size` steps (no partial-page eviction).
- The `max(..., 1)` guard prevents `x % 1 == 1` (which is always false) from silently disabling eviction.

### Overlap Safety

- **`decode_batch_idx == 0`**: eviction **skipped** — the previous decode batch may still be reading those SWA pages on device.
- **`decode_batch_idx > 0`**: eviction triggers on `decode_batch_idx % evict_interval == 1`.

### SWARadixCache Difference

In RadixCache mode, eviction starts from `cache_protected_len` (not position 0), because slots in `[0, cache_protected_len)` belong to the radix tree and may be shared across requests.

## 7. Tree-Level Eviction (SWARadixCache — Future)

> **Note**: This section describes the planned design for tree-level SWA eviction. The basic `SWARadixCache` class (insert/match/evict/lock) is implemented, but the advanced tombstone strategies below are targeted for a future PR.

Triggered by allocation pressure when pool space falls below what is needed.

### Phase 1 — Full Eviction (delete leaf nodes)

Both full and SWA KV data are freed. The node is removed from the tree entirely.

If deleting a leaf causes its parent to become a childless tombstone, cascade deletion occurs — both parent and child are removed.

### Phase 2 — SWA-Only Eviction (tombstone internal nodes)

Only SWA KV data is freed. Full KV data is kept to enable prefix matching. The node becomes a tombstone.

Tombstoning proceeds in LRU order from the least recently used node:

1. Initially, all nodes hold full + SWA data.
2. Round 1: The oldest (LRU) node becomes a tombstone (full only).
3. Round 2: The next LRU node also becomes a tombstone.
4. Progression continues until enough SWA tokens are reclaimed.

### Dual LRU Lists

Two LRU lists track eviction candidates:

- **`full_lru_list`**: All non-root nodes. Used by Phase 1 to find the LRU leaf for full eviction.
- **`swa_lru_list`**: Non-root, non-tombstone nodes only. Used by Phase 2 to find the LRU node for SWA-only eviction. Tombstone nodes are removed from this list; revived nodes are re-inserted at MRU position.

On a prefix match, the deepest matched node and all its ancestors are reset to MRU in both lists, keeping frequently accessed prefixes fresh.

## 8. Summary

| Aspect | ChunkCache (current) | SWARadixCache (future) |
|--------|---------------------|----------------------|
| Extend SWA eviction | Proactive, per chunk | Skipped; tree handles it |
| Decode SWA eviction | From position 0 | From `cache_protected_len` |
| Prefill SWA waste | Low (bounded by 2-chunk delay) | Higher (until tree-level Phase 2 evicts) |
| Prefix reuse | None | Yes (via radix tree) |
| SWA reclamation | Per-request via `_evict_swa` | Tree Phase 2 (reactive, LRU-based) |
| Overlap safety | `extend_batch_idx < 2`, `decode_batch_idx == 0` | Same guards apply |
