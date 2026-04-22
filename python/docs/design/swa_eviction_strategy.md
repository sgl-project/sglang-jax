# SWA Eviction Strategy

## 1. Overview

Hybrid models (e.g., MiMo-V2-Flash with 9 full-attention + 39 SWA layers) mix full-attention and sliding-window-attention (SWA) layers. Full-attention layers retain all historical KV data, while SWA layers only need the most recent `W` tokens. This document describes the dual-pool KV cache architecture and eviction strategy that exploits this difference to reduce memory usage.

### Support Status

| Cache Mode | SWA Support | Description |
|-----------|-------------|-------------|
| **ChunkCache** (`--disable-radix-cache`) | **Supported** | Per-request proactive eviction. Each request owns its KV slots; SWA slots outside the window are freed during extend/decode. |
| **RadixCache** (default) | **Not supported** | RadixCache with SWA-aware eviction (tombstone strategies, dual LRU lists) is not implemented. Hybrid models must use `--disable-radix-cache`. |

## 2. Dual-Pool Architecture

Two separate KV cache pools are maintained:

| Pool | Serves | Lifecycle | Eviction |
|------|--------|-----------|----------|
| **Full Pool** | Full-attention layers | Retains all historical KV data for the request lifetime | Freed on request completion |
| **SWA Pool** | SWA layers | Only the most recent `W` tokens are needed | Proactively freed as tokens fall outside the sliding window |

These pools are linked via `full_to_swa_index_mapping`, a numpy array that maps a full-pool index to its corresponding SWA-pool index. A mapping value of 0 means the SWA slot has been freed.

### Key Term

| Term | Definition |
|------|-----------|
| `swa_evicted_seqlen` | Per-request watermark. SWA slots in `[0, swa_evicted_seqlen)` have already been freed. Monotonically increasing. |

### Memory Layout Example

For MiMo-V2-Flash on TPU v6e-16 (TP=16, 1 KV head, head_dim=192+128=320, bf16):

```
Full pool:  104,704 tokens x 9 FA layers  x 640 bytes/token = ~603 MB
SWA pool:   83,840 tokens x 39 SWA layers x 640 bytes/token = ~2.1 GB
SWA held per request: ~256 tokens (sliding_window=128 + page_size=128 alignment)
```

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

| Method | What it frees | When called |
|--------|--------------|-------------|
| `free(indices)` | Both full and SWA pools | Request completion |
| `free_swa(indices)` | SWA pool only (looks up mapping, frees non-zero entries, zeroes mapping) | Per-request SWA eviction |
| `count_swa_mapped(indices)` | Nothing (read-only) — counts indices with active SWA mapping | Bookkeeping before `free_swa` |

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
  7. Update req.swa_evicted_seqlen = new_evicted
```

### Example

With `sliding_window=128`, `page_size=256`, and `seqlen=2049`:

```
new_evicted = max(0, 2049 - 128) = 1921
page-aligned: (1921 // 256) * 256 = 1792
Free SWA slots in [0, 1792), retain [1792, 2049) within the window.
```

## 5. Extend Phase Behavior

With overlap scheduling enabled, `maybe_evict_swa` is gated by `extend_batch_idx` to prevent freeing SWA pages that a previous extend batch may still be reading on device:

| Condition | Action | Reason |
|-----------|--------|--------|
| `extend_batch_idx < 2` | Eviction **skipped** | Previous extend batch may still be executing |
| `extend_batch_idx >= 2` | Eviction proceeds with `pre_len -= chunked_prefill_size` | Safe: previous batch has completed |

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

## 6. Decode Phase Behavior

### Eviction Interval

```python
evict_interval = max(min(sliding_window_size, page_size), 1)
```

| Scenario | Interval | Rationale |
|----------|----------|-----------|
| `page_size >= sliding_window_size` | Every step | Window advances past a full page each step |
| `page_size < sliding_window_size` | Every `page_size` steps | Avoid partial-page eviction |
| `evict_interval == 1` | Every step | `max(..., 1)` guard prevents `x % 1 == 1` (always false) from disabling eviction |

### Overlap Safety

| Condition | Action | Reason |
|-----------|--------|--------|
| `decode_batch_idx == 0` | Eviction **skipped** | Previous decode batch may still be reading SWA pages on device |
| `decode_batch_idx > 0` | Eviction triggers on `decode_batch_idx % evict_interval == 1` | Safe: previous batch has completed |

## 7. Summary

| Component | Description |
|-----------|-------------|
| **Dual-pool architecture** | Full pool (all layers, all history) + SWA pool (SWA layers, window only) |
| **Index mapping** | `full_to_swa_index_mapping` translates full-pool indices to SWA-pool indices |
| **Allocation** | Atomic dual-pool alloc with rollback on SWA exhaustion |
| **Extend eviction** | Proactive per-chunk; skips first 2 chunks for overlap safety |
| **Decode eviction** | Periodic per-step; skips batch 0 for overlap safety |
| **Eviction algorithm** | `_evict_swa`: advance watermark, page-align, free SWA slots in `[old, new)` |
| **RadixCache SWA** | Not supported — hybrid models must use `--disable-radix-cache` |
