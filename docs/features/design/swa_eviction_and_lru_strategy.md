# SWA Eviction and LRU Strategy

## 0. Overview

Hybrid models (e.g. MiMo-V2-Flash with 9 full-attention + 39 SWA layers) mix
full-attention and sliding-window-attention (SWA) layers. Full-attention layers
retain all historical KV; SWA layers only need the most recent `W` tokens.
The dual-pool KV cache exploits this asymmetry to reduce memory.

### Cache Mode Support

| Cache Mode | SWA Support | Path |
|------------|-------------|------|
| **ChunkCache** (`--disable-radix-cache`) | Supported | Per-request proactive eviction (`ScheduleBatch._evict_swa`). |
| **SWARadixCache** (default) | Supported | Per-request eviction during decode delegates to `SWARadixCache.evict_req_swa`; tree-pressure eviction handles the rest (Phase 1 / Phase 2). |

## 1. Dual-Pool Architecture

```
+-----------------------------------------------------------+
|                      KV Cache Memory                      |
|                                                           |
|  +-------------------------+  +-------------------------+ |
|  |       Full Pool         |  |        SWA Pool         | |
|  |   (Full Attention lys)  |  |   (Sliding Window lys)  | |
|  |                         |  |                         | |
|  |   Grows with seqlen     |  |   Bounded by window     | |
|  |   Never evicted per-req |  |   Evicted outside window| |
|  +-------------------------+  +-------------------------+ |
|                                                           |
|  Linked by full_to_swa_index_mapping:                     |
|    full_idx --> swa_idx  (or 0 if SWA freed)              |
+-----------------------------------------------------------+
```

### Key Terms

| Term | Meaning |
|------|---------|
| `swa_evicted_seqlen` | SWA slots `[0, swa_evicted_seqlen)` have been freed |
| `protected prefix` | Prefix `[0, protected_prefix_len)` owned by the radix tree, derived from the request's locked `last_node` path; per-request eviction cannot touch it |
| `last_matched_prefix_len` | Page-aligned cached prefix length kept on the request for writeback / retract bookkeeping |
| `tombstone` | Tree node: full KV retained, SWA KV freed |

### Memory Layout Example

For MiMo-V2-Flash on TPU v6e-16 (TP=16, 1 KV head, head_dim=192+128=320, bf16):

```
Full pool:  104,704 tokens x  9 FA  layers x 640 B/token = ~603 MB
SWA  pool:   83,840 tokens x 39 SWA layers x 640 B/token = ~2.1 GB
SWA held per request: ~256 tokens (sliding_window=128 + page_size=128 alignment)
```

The SWA pool dominates per-layer cost (39 layers vs. 9), so freeing
out-of-window SWA slots aggressively is the lever the hybrid design pulls.

## 2. Allocator: `SWATokenToKVPoolAllocator`

Lives in `python/sgl_jax/srt/mem_cache/allocator.py`. Wraps two independent
sub-allocators (one per pool) and the `full_to_swa_index_mapping` array.

### Allocation (atomic across pools)

```
alloc_extend / alloc_decode:
  1. full_indices = full_attn_allocator.alloc_*(...)
     if None: return None                       # full pool exhausted
  2. swa_indices  = swa_attn_allocator.alloc_*(...)
     if None:                                   # SWA exhausted after full succeeded
        full_attn_allocator.free(full_indices)  # rollback
        return None
  3. mapping[full_indices] = swa_indices
  4. return full_indices                        # SWA indices stay private to the pool
```

The rollback is what makes the dual-pool allocation atomic: callers see
either both pools advanced or neither.

### Freeing

| Method | Frees | Called from |
|--------|-------|-------------|
| `free(indices)` | Both pools (delegates to `free_swa` for SWA side) | Request completion, retract |
| `free_swa(indices)` | SWA pool only — looks up mapping, frees entries with `mapping > 0`, then zeroes those mapping entries | Per-request SWA eviction (`_evict_swa`, `evict_req_swa`) and tombstone Phase 2 |
| `count_swa_mapped(indices)` | Read-only — counts indices whose mapping is still `> 0` | Bookkeeping before `free_swa` |

A mapping value of `0` means "SWA slot already freed for this full slot",
which is why `free_swa` must filter `> 0` before delegating to the SWA
sub-allocator.

## 3. Two Cache Modes at a Glance

```
ChunkCache (--disable-radix-cache)       SWARadixCache (radix cache enabled)
+-------------------------------+        +-------------------------------+
| No tree. Request owns all     |        | Radix tree caches KV for     |
| slots. Freed on completion.   |        | prefix reuse across reqs.    |
| No prefix sharing.            |        | Tree outlives requests.      |
+-------------------------------+        +-------------------------------+
```

## 4. Per-Request SWA Eviction

Frees SWA slots outside the sliding window from a request's `req_to_token` buffer.

Two code paths, depending on cache mode:

**ChunkCache path** (`ScheduleBatch._evict_swa`):
```
_evict_swa(req, pre_len, sliding_window_size, page_size):

  # If tree_cache is SWARadixCache, delegate to evict_req_swa (see below)
  1. Target: new_evicted = max(swa_evicted_seqlen, pre_len - sliding_window - page_size)
  2. Align:  new_evicted = page_floor(new_evicted)   [only if page_size > 1]
  3. Free:   free_swa( slots[swa_evicted_seqlen : new_evicted] )
```

**SWARadixCache path** (`SWARadixCache.evict_req_swa`):
```
evict_req_swa(req, pre_len):

  1. Derive: protected_prefix_len = _node_prefix_len(req.last_node)
  2. Clamp:  swa_evicted_seqlen = max(swa_evicted_seqlen, protected_prefix_len)
  3. Target: new_evicted = max(swa_evicted_seqlen, pre_len - sliding_window - page_size)
  4. Align:  new_evicted = page_floor(new_evicted)   [only if page_size > 1]
  5. Free:   free_swa( slots[swa_evicted_seqlen : new_evicted] )
```

Steps 1-2 are unique to the SWARadixCache path — they protect the tree-owned
prefix from being evicted. The `- page_size` safety margin in step 3 prevents
`swa_evicted_seqlen` from reaching the live leaf boundary, ensuring
`_insert_helper` can always create a non-tombstone leaf.

```
Example (sliding_window=128, page_size=256, seqlen=2049, decode pre_len=2048):

  ChunkCache path: new_evicted = 2048 - 128 - 256 = 1664
                   page_floor(1664, 256) = 1536

  0         1536         2048
  |..........|############|
   freed SWA   retained
               (window + page)
  ^                        ^
  swa_evicted=1536    seqlen=2049
```

### When does it run?

| Phase  | ChunkCache | SWARadixCache                             |
|--------|------------|-------------------------------------------|
| Extend | Yes        | **No** (skipped; `isinstance(tree_cache, ChunkCache)` gate) |
| Decode | Yes        | Yes (delegates to `evict_req_swa` with tree-derived protected prefix) |

## 5. Extend Phase Behavior

### 5.1 ChunkCache -- Proactive Eviction

With overlap scheduling, `pre_len` is shifted back by `chunked_prefill_size`
when `enable_overlap=True` and `req.is_chunked > 0`.
There is no explicit `extend_batch_idx` gate — the first two chunks naturally
produce no eviction because `pre_len - chunked_prefill_size` is too small to
exceed `sliding_window + page_size`.

```
8K tokens, chunk_size=2048, sliding_window=128, page_size=256, overlap=True

       0       2048      4096      6144      8192
       |        |         |         |         |
Chk 1: |########|                              pre_len=0-2048<0, no eviction
       alloc [0,2048)

Chk 2: |################|                     pre_len=2048-2048=0, no eviction
       alloc [0,4096)

Chk 3: |.........|################|           pre_len=4096-2048=2048
       freed     retained                     new_evicted=2048-128-256=1664
       [0,1536)                                page_floor(1664,256)=1536

Chk 4: |..................|################|  pre_len=6144-2048=4096
       freed              retained            new_evicted=4096-128-256=3712
       [0,3584)                                page_floor(3712,256)=3584

  . = freed SWA    # = retained SWA
```

Without overlap (`enable_overlap=False`), `pre_len` is not adjusted.
Each chunk directly evicts based on the full `len(req.prefix_indices)`:

```
Chk 1: |########|                              pre_len=0, nothing
Chk 2: |........|################|             pre_len=2048, evict [0,1536)
Chk 3: |.................|################|    pre_len=4096, evict [1536,3584)
```

### 5.2 SWARadixCache -- Deferred to Tree

```
8K tokens, chunk_size=2048, sliding_window=128, page_size=256

       0       2048      4096      6144      8192
       |        |         |         |         |
Chk 1: |########|
       _evict_swa: SKIPPED
       cache_unfinished_req --> tree owns [0,2048), all non-tombstone
       protected prefix = 2048 (derived from last_node)

Chk 2: |################|
       _evict_swa: SKIPPED
       cache_unfinished_req --> tree owns [0,4096), all non-tombstone
       protected prefix = 4096 (derived from last_node)

       ... chunks 3, 4 ...  protected prefix = 8192

First decode step:
  protected_prefix_len = prefix_len(last_node) = 8192
  swa_evicted = max(0, 8192) = 8192
  new_evicted = max(8192, 8192-128-256) = 8192
  --> NO EVICTION (all tree-protected)

  # = SWA in tree (non-tombstone, only freed by tree Phase 2)
```

**Trade-off**: Prefill SWA slots stay in tree until pool pressure triggers
tree-level eviction. The ownership boundary now lives inside the cache layer
instead of a request field, but prefill is still less SWA-efficient than ChunkCache.

## 6. Decode Phase Behavior

`maybe_evict_swa` runs on every decode batch, but the per-request
`_evict_swa` call is throttled by an `evict_interval` derived from
`page_size` and `sliding_window_size`:

```python
multiplier      = float(os.environ.get("SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER", "1.0"))
evict_interval  = max(page_size, int(sliding_window_size * multiplier))
evict_interval  = (evict_interval // page_size) * page_size   # snap down to a page multiple
trigger         = (req.decode_batch_idx % evict_interval == 1)
```

The `% == 1` rather than `% == 0` choice means the very first decode step
(`decode_batch_idx == 0`) does not evict — the previous extend batch may
still be reading SWA pages on device, so we defer one step. Subsequent
triggers fire on `decode_batch_idx == 1, 1 + evict_interval, 1 + 2*evict_interval, ...`.

`SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER` lets operators relax the
interval (e.g. evict less frequently to amortise host-side bookkeeping at
the cost of a larger transient SWA footprint). The default `1.0` evicts
every `sliding_window_size` decode steps.

For SWARadixCache, `_evict_swa` delegates to `evict_req_swa`, which clamps
`swa_evicted_seqlen` against the tree-derived `protected_prefix_len`
before computing the new frontier — so a long shared prefix held by the
tree blocks per-request SWA reclamation until tree Phase 2 runs.

## 7. Tree-Level Eviction (SWARadixCache)

Triggered by allocation pressure (`evict_from_tree_cache` when pool space < needed).

### Phase 1 -- Full Eviction (delete leaf nodes)

Frees **both** full + SWA KV. Removes node from tree entirely.

```
Before:                             After evicting leaf B:

root --> [A 0..2048] --> [B] leaf   root --> [A 0..2048] --> [C] leaf
                     --> [C] leaf
                                    B: full+SWA freed, node deleted

If parent becomes childless tombstone --> cascade delete:

root --> [A tombstone] --> [B] leaf       root (empty)
         only child                       A, B both deleted
```

### Phase 2 -- SWA-Only Eviction (tombstone internal nodes)

Uses `swa_lru_list` to find LRU nodes. Two sub-cases:

**Internal node** (has children): frees **only SWA** KV via `free_swa()`.
Full KV kept for prefix matching. Node becomes a tombstone, removed from
`swa_lru_list` but stays in `full_lru_list`.

**Leaf node** (no children): cannot become tombstone (leaf-must-not-be-tombstone
invariant), so it is fully deleted — frees **both** full + SWA via `free()`,
removed from both LRU lists, then cascade delete fires on parent.

```
Internal node case:

Before:                                After Phase 2 on node A:

root --> [A 0..2048] --> [B] leaf      root --> [A 0..2048] --> [B] leaf
          full+SWA   --> [C] leaf               TOMBSTONE   --> [C] leaf
                                                 full only
```

### Gradual Tombstone Progression (LRU order, head --> tail)

```
          [0..2048]    [2048..4096]   [4096..6144]   [6144..8192]

Initial:  full+SWA --> full+SWA  --> full+SWA  --> full+SWA

Round 1:  TOMBSTONE--> full+SWA  --> full+SWA  --> full+SWA
          full only

Round 2:  TOMBSTONE--> TOMBSTONE --> full+SWA  --> full+SWA
          full only    full only

Round 3:  TOMBSTONE--> TOMBSTONE --> TOMBSTONE --> full+SWA
          full only    full only     full only     ^ only tail
                                                     retains SWA
```

## 8. Tombstone Insert Logic

`_insert_helper` has two sets of tombstone branch logic:

- **Inside the while loop**: healing of **existing** tombstone nodes
- **After the while loop**: tombstone split when creating **new** nodes

### 8.1 Existing Tombstone Healing (while loop)

When insert walks through an existing tombstone node beyond
`update_kv_after_len`, it checks `swa_evicted_seqlen` against
`[node_start, node_end)`:

```
  BRANCH 1: swa_evicted <= node_start  --> revive entire node
  BRANCH 2: node_start < swa_evicted < node_end --> split, revive back half
  BRANCH 3: swa_evicted >= node_end    --> keep tombstone, free incoming value
```

Branch 1 fires during `cache_unfinished_req` (`swa_evicted_seqlen = 0`)
when `_match_prefix_helper` truncated the prefix due to tombstone safety.

Branches 2/3 fire during `cache_finished_req` when a prior request's
decode nodes have been tombstoned and the current request generated
identical decode tokens (e.g. greedy decoding with the same prefix).

### 8.2 New Node Tombstone Split (after while loop)

When insert has remaining unmatched tokens (`len(key) > 0`), it creates
new nodes.  Invariant: **leaf nodes must never be tombstone**.

```
  Case 1: swa_evicted <= total_prefix_length
          --> create non-tombstone leaf (normal path for cache_unfinished_req)

  Case 2: total_prefix_length < swa_evicted < total_prefix_length + len(key)
          --> split into tombstone prefix + non-tombstone leaf
              (normal path for cache_finished_req: decode suffix split)

  Case 3: swa_evicted >= total_prefix_length + len(key)
          --> all remaining SWA evicted, cannot create non-tombstone leaf
              free incoming value and return (defensive guard, kept safe
              by the extra `- page_size` in `_evict_swa` frontier)
```

### 8.3 Trigger Conditions

| Call site | `swa_evicted_seqlen` | Existing tombstone | New nodes |
|-----------|---------------------|--------------------|-----------|
| `cache_unfinished_req` | Always 0 | Branch 1 only | Case 1 only |
| `cache_finished_req` | >= protected prefix length | Prefill nodes: skip zone. Prior req's decode nodes: Branch 1/2/3 | Case 2 (normal), Case 3 (defensive) |

## 9. LRU Policy

```
+-----------------------------------------------------+
|                   full_lru_list                     |
|  Tracks: ALL non-root nodes                         |
|  Used by: Phase 1 (find LRU leaf for full eviction) |
|                                                     |
|  LRU <--- old nodes --- recent nodes ---> MRU       |
+-----------------------------------------------------+

+-----------------------------------------------------+
|                   swa_lru_list                      |
|  Tracks: non-root, NON-TOMBSTONE nodes only         |
|  Used by: Phase 2 (find LRU node for SWA eviction)  |
|                                                     |
|  Tombstone nodes REMOVED from this list.            |
|  Revived nodes RE-INSERTED at MRU.                  |
+-----------------------------------------------------+
```

On prefix match, the **deepest matched node** and all ancestors are
reset to MRU in both lists, keeping frequently accessed prefixes fresh.

## 10. Summary

```
+-------------------+-----------------+------------------------+
|                   | ChunkCache      | SWARadixCache          |
+-------------------+-----------------+------------------------+
| Extend SWA evict  | Proactive       | Skipped                |
|                   | (per chunk)     | (tree handles)         |
+-------------------+-----------------+------------------------+
| Decode SWA evict  | From pos 0      | From                   |
|                   |                 | protected tree prefix  |
+-------------------+-----------------+------------------------+
| Prefill SWA waste | Low (bounded)   | Higher (until          |
|                   |                 | Phase 2 evicts)        |
+-------------------+-----------------+------------------------+
| Prefix reuse      | None            | Yes (via tree)         |
+-------------------+-----------------+------------------------+
| SWA reclamation   | Per-request     | Tree Phase 2           |
|                   | _evict_swa      | (reactive, LRU)        |
+-------------------+-----------------+------------------------+
```
